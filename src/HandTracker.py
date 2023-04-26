"""Heavily inspired from depthai example"""
from typing import Tuple, Union, Any, List

import cv2
import pathlib
import depthai as dai
import numpy as np

from collections import deque
from threading import Lock

import src.MediapipeUtils as MpU
from src.FPS import FPS
from src.MediapipeUtils import HandRegion


class HandTracker:
    PALM_DETECTION_BLOB_PATH = "checkpoints/palm_detection_sh4.blob"
    LANDMARK_BLOB_PATH = "checkpoints/hand_landmark_full_sh4.blob"
    INTERNAL_FPS = 22
    INTERNAL_FRAME_HEIGHT = 640
    DEPTH_REGION_SIZE = 7
    RESOLUTION = (1920, 1080)

    PALM_DETECTION_SCORE_THRESH = 0.5
    PALM_DETECTION_NMS_SCORE_THRESH = 0.3
    LANDMARK_SCORE_THRESH = 0.5

    PALM_DETECTION_INP_LENGTH = 128
    LANDMARK_INPUT_LENGTH = 224

    def __init__(self, solo: bool = False) -> None:

        super().__init__()

        # Prediction variables
        self.solo: bool = solo
        self.use_previous_landmark: bool = False
        self.nb_hands_in_previous_frame: int = 0
        self.hands_from_landmarks: List[HandRegion] = []
        self.hands: List[HandRegion] = []

        self.handedness_avg: List[MpU.HandednessAverage] = [MpU.HandednessAverage() for _ in range(2)]
        self.anchors: np.ndarray = MpU.generate_handtracker_anchors(self.PALM_DETECTION_INP_LENGTH,
                                                                    self.PALM_DETECTION_INP_LENGTH)
        self.nb_anchors: int = self.anchors.shape[0]
        _, self.scale_nd = MpU.find_isp_scale_params(
            self.INTERNAL_FRAME_HEIGHT * self.RESOLUTION[0] / self.RESOLUTION[1], self.RESOLUTION, is_height=False)

        self.img_h: int = int(round(self.RESOLUTION[1] * self.scale_nd[0] / self.scale_nd[1]))
        self.img_w: int = int(round(self.RESOLUTION[0] * self.scale_nd[0] / self.scale_nd[1]))
        self.pad_h: int = (self.img_w - self.img_h) // 2
        self.pad_w: int = 0

        # better than queue as it'll remove the last element if the size exceeds the max len
        self.hand_hist = {"left": deque(maxlen=10), "right": deque(maxlen=10)}

        # FPS
        self.fps = FPS()

        # Pipeline
        self.device = dai.Device()
        self.pipeline = dai.Pipeline()
        self.create_pipeline()
        self.device.startPipeline(self.pipeline)

        # Queues
        # Color
        self.color_camera_out = self.device.getOutputQueue(name="color_camera_out", maxSize=1, blocking=False)
        self.image_manip_palm_detection_in = self.device.getInputQueue(name="image_manip_palm_detection_in")

        # Depth
        self.stereo_depth_out = self.device.getOutputQueue(name="stereo_depth_out", maxSize=1, blocking=False)
        self.spatial_location_calculator_config_in = self.device.getInputQueue(name="spatial_calc_config_in")
        self.spatial_data_out = self.device.getOutputQueue(name="spatial_data_out")

        # Accelerometer
        self.imu_out = self.device.getOutputQueue(name="imu_out", maxSize=1, blocking=False)

        # Neural networks
        self.pd_out = self.device.getOutputQueue(name="palm_detection_out", maxSize=4, blocking=True)

        self.lm_in = self.device.getInputQueue(name="landmark_in")
        self.lm_out = self.device.getOutputQueue(name="landmark_out", maxSize=4, blocking=True)

        # Synchronization of color_camera and spatial_data queue packets
        self._mutex = Lock()

    def create_color_camera(self) -> (dai.node.ColorCamera, dai.node.ImageManip):
        # Color camera
        color_camera = self.pipeline.createColorCamera()
        color_camera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        color_camera.setBoardSocket(dai.CameraBoardSocket.RGB)
        color_camera.setInterleaved(False)
        color_camera.setIspScale(self.scale_nd[0], self.scale_nd[1])
        color_camera.setFps(self.INTERNAL_FPS)

        # Image manipulation before inputting to palm detection
        image_manip_palm_detection = self.pipeline.createImageManip()
        image_manip_palm_detection.setMaxOutputFrameSize(self.PALM_DETECTION_INP_LENGTH ** 2 * 3)
        image_manip_palm_detection.setWaitForConfigInput(True)
        image_manip_palm_detection.inputImage.setQueueSize(1)
        image_manip_palm_detection.inputImage.setBlocking(False)

        # Setting preview config for the color camera
        color_camera.preview.link(image_manip_palm_detection.inputImage)
        color_camera.setVideoSize(self.img_w, self.img_h)
        color_camera.setPreviewSize(self.img_w, self.img_h)

        # Image manipulation for palm detection input stream
        image_manip_palm_detection_in = self.pipeline.createXLinkIn()
        image_manip_palm_detection_in.setStreamName("image_manip_palm_detection_in")
        image_manip_palm_detection_in.out.link(image_manip_palm_detection.inputConfig)

        # Color camera output stream
        color_camera_out = self.pipeline.createXLinkOut()
        color_camera_out.setStreamName("color_camera_out")
        color_camera_out.input.setQueueSize(1)
        color_camera_out.input.setBlocking(False)
        color_camera.video.link(color_camera_out.input)

        return color_camera, image_manip_palm_detection

    def create_stereo_depth(self, rgb: dai.node.ColorCamera) -> None:
        # Set camera to fixed focus for RGB/depth alignment
        calib = self.device.readCalibration()
        calib_pos = calib.getLensPosition(dai.CameraBoardSocket.RGB)
        rgb.initialControl.setManualFocus(calib_pos)

        # Left MonoCamera
        left_monocamera = self.pipeline.createMonoCamera()
        left_monocamera.setBoardSocket(dai.CameraBoardSocket.LEFT)
        left_monocamera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        left_monocamera.setFps(self.INTERNAL_FPS)

        # Right MonoCamera
        right_monocamera = self.pipeline.createMonoCamera()
        right_monocamera.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        right_monocamera.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        right_monocamera.setFps(self.INTERNAL_FPS)

        # Stereo Camera
        stereo_depth_camera = self.pipeline.createStereoDepth()
        stereo_depth_camera.initialConfig.setConfidenceThreshold(255)
        stereo_depth_camera.setRectifyEdgeFillColor(0)
        stereo_depth_camera.setExtendedDisparity(True)
        stereo_depth_camera.setSubpixel(False)
        # Must be true for RGB/depth alignment, false would be better for occlusion
        stereo_depth_camera.setLeftRightCheck(True)
        stereo_depth_camera.setDepthAlign(dai.CameraBoardSocket.RGB)


        # Link left and right monocamera to stereo depth camera
        left_monocamera.out.link(stereo_depth_camera.left)
        right_monocamera.out.link(stereo_depth_camera.right)

        # Set depth map output
        stereo_depth_out = self.pipeline.createXLinkOut()
        stereo_depth_out.setStreamName("stereo_depth_out")
        stereo_depth_out.input.setQueueSize(1)
        stereo_depth_out.input.setBlocking(False)
        stereo_depth_camera.depth.link(stereo_depth_out.input)

        # Spatial location calculator
        spatial_location_calculator = self.pipeline.createSpatialLocationCalculator()
        spatial_location_calculator.setWaitForConfigInput(True)
        spatial_location_calculator.inputDepth.setBlocking(False)
        spatial_location_calculator.inputDepth.setQueueSize(1)

        # Spatial data output
        spatial_data_out = self.pipeline.createXLinkOut()
        spatial_data_out.setStreamName("spatial_data_out")
        spatial_data_out.input.setQueueSize(1)
        spatial_data_out.input.setBlocking(False)

        # Link stereo depth output to spatial location calculator input
        stereo_depth_camera.depth.link(spatial_location_calculator.inputDepth)
        # Link spatial location calculator output to spatial data output
        spatial_location_calculator.out.link(spatial_data_out.input)

        # Input to change the spatial location calculator config
        spatial_calc_config_in = self.pipeline.createXLinkIn()
        spatial_calc_config_in.setStreamName("spatial_calc_config_in")
        spatial_calc_config_in.out.link(spatial_location_calculator.inputConfig)

    def create_accelerometer(self) -> None:
        # Create IMU
        imu = self.pipeline.createIMU()
        imu_out = self.pipeline.createXLinkOut()
        imu_out.setStreamName("imu_out")

        # Activate only accelerometer with 25Hz
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 25)
        imu.setMaxBatchReports(1)
        imu.out.link(imu_out.input)

    def create_nn(self, manip: dai.node.ImageManip) -> None:
        # Palm detection NN setup
        palm_detection_nn = self.pipeline.createNeuralNetwork()
        palm_detection_nn.setBlobPath(pathlib.Path(self.PALM_DETECTION_BLOB_PATH))
        palm_detection_nn.input.setQueueSize(1)
        palm_detection_nn.input.setBlocking(False)

        # Link modified image output to the palm detection network input
        manip.out.link(palm_detection_nn.input)

        # Link an output node to the palm detection nn output
        palm_detection_out = self.pipeline.createXLinkOut()
        palm_detection_out.setStreamName("palm_detection_out")
        palm_detection_nn.out.link(palm_detection_out.input)

        # Landmark NN setup
        landmark_nn = self.pipeline.createNeuralNetwork()
        landmark_nn.setBlobPath(pathlib.Path(self.LANDMARK_BLOB_PATH))
        landmark_nn.setNumInferenceThreads(2)

        # Create an input node for the landmark NN input
        landmark_in = self.pipeline.createXLinkIn()
        landmark_in.setStreamName("landmark_in")
        landmark_in.out.link(landmark_nn.input)

        # Create an output node for the landmark NN output
        landmark_out = self.pipeline.createXLinkOut()
        landmark_out.setStreamName("landmark_out")
        landmark_nn.out.link(landmark_out.input)

    def create_pipeline(self) -> None:
        self.pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

        print("Sensor setup....")
        color_camera, image_manip_palm_detection = self.create_color_camera()
        self.create_stereo_depth(color_camera)
        self.create_accelerometer()

        self.create_nn(image_manip_palm_detection)

        print("Pipeline creation successful.")

    def get_color_frame(self) -> np.ndarray:
        return self.color_camera_out.get().getCvFrame()

    def get_depth_frame(self) -> np.ndarray:
        return self.stereo_depth_out.get().getFrame()

    def get_camera_tilt(self) -> float:
        imu_packet = self.imu_out.get().packets[0]
        return np.rad2deg(np.arctan(imu_packet.acceleroMeter.z / (imu_packet.acceleroMeter.y + 1e-8)))

    def get_depth_at_coords(self, coords: Tuple[int, ...], size: int = None) -> np.ndarray:

        if not size:
            size = self.DEPTH_REGION_SIZE

        # Create a list of config data to query depth at each cartesian coordinate
        conf_datas = []
        for (x, y) in coords:
            rect_center = dai.Point2f(x, y)
            rect_size = dai.Size2f(size, size)

            conf_data = dai.SpatialLocationCalculatorConfigData()
            conf_data.depthThresholds.lowerThreshold = 100
            conf_data.depthThresholds.upperThreshold = 10000
            conf_data.roi = dai.Rect(rect_center, rect_size)
            conf_datas.append(conf_data)

        # Send the configs to the sptial location calculator
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.setROIs(conf_datas)

        # Hack to avoid synchronization issue if detector and hand spatial information are queried at the same time
        with self._mutex:
            self.spatial_location_calculator_config_in.send(cfg)

            # Fetch and return all queried depth coordinates [NOTE: Optimal distance is between 40cm and 6m]
            return np.array([loc.spatialCoordinates.z for loc in self.spatial_data_out.get().getSpatialLocations()])

    def pd_postprocess(self, inference: dai.NNData) -> List[HandRegion]:
        # Get scores and bboxes
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16)  # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape(
            (self.nb_anchors, 18))  # 896x18

        # Decode bboxes
        hands = MpU.decode_bboxes(self.PALM_DETECTION_SCORE_THRESH, scores, bboxes, self.anchors,
                                  scale=self.PALM_DETECTION_INP_LENGTH,
                                  best_only=self.solo)

        # If multiple hands use NMS to keep viable detections
        if not self.solo:
            hands = MpU.non_max_suppression(hands, self.PALM_DETECTION_NMS_SCORE_THRESH)[:2]

        # Preprocess the hand detections to pass to landmark detection
        MpU.detections_to_rect(hands)
        MpU.rect_transformation(hands, self.img_w, self.img_w)
        return hands

    def lm_postprocess(self, hand: HandRegion, inference: dai.NNData) -> None:
        # print(inference.getAllLayerNames())
        # The output names of the landmarks model are :
        # Identity_1 (1x1) : score
        # Identity_2 (1x1) : handedness
        # Identity_3 (1x63) : world 3D landmarks (in meters)
        # Identity (1x63) : screen 3D landmarks (in pixels)

        # Get the scores
        hand.lm_score = inference.getLayerFp16("Identity_1")[0]
        if hand.lm_score > self.LANDMARK_SCORE_THRESH:
            # Get the handedness (left or right?)
            hand.handedness = inference.getLayerFp16("Identity_2")[0]

            lm_raw = np.array(inference.getLayerFp16("Identity_dense/BiasAdd/Add")).reshape(-1, 3)
            # hand.norm_landmarks contains the normalized ([0:1]) 3D coordinates of landmarks in the
            # square rotated body bounding box
            hand.norm_landmarks = lm_raw / self.LANDMARK_INPUT_LENGTH

            # Now calculate hand.landmarks = the landmarks in the image coordinate system (in pixel)
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([(x, y) for x, y in hand.rect_points[1:]],
                           dtype=np.float32)  # hand.rect_points[0] is left bottom point and points going clockwise!
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(hand.norm_landmarks[:, :2], axis=0)
            hand.landmarks = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int32)

    def xy_to_xyz(self) -> None:
        for i, h in enumerate(self.hands):
            z = np.round(self.get_depth_at_coords(h.landmarks, size=2), 0)
            h.xyz = np.column_stack((h.landmarks, z))

    def next_frame(self) -> (np.ndarray, List[HandRegion]):
        self.fps.update()

        # If no previous landmarks, we modify the image before passing it to the palm detection NN
        if not self.use_previous_landmark:
            cfg = dai.ImageManipConfig()
            cfg.setResizeThumbnail(self.PALM_DETECTION_INP_LENGTH, self.PALM_DETECTION_INP_LENGTH)
            self.image_manip_palm_detection_in.send(cfg)

        # We get the frame with the
        frame = self.get_color_frame()
        square_frame = cv2.copyMakeBorder(frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)

        if self.use_previous_landmark:
            self.hands = self.hands_from_landmarks
        else:
            inference = self.pd_out.get()
            hands = self.pd_postprocess(inference)

            if not self.solo and self.nb_hands_in_previous_frame == 1 and len(hands) <= 1:
                self.hands = self.hands_from_landmarks
            else:
                self.hands = hands

        # Hand landmarks, send requests then receive requests
        for i, h in enumerate(self.hands):
            img_hand = MpU.warp_rect_img(h.rect_points, square_frame, self.LANDMARK_INPUT_LENGTH,
                                         self.LANDMARK_INPUT_LENGTH)
            img_hand = cv2.resize(img_hand, (self.LANDMARK_INPUT_LENGTH, self.LANDMARK_INPUT_LENGTH)).transpose(2, 0, 1)
            nn_data = dai.NNData()
            nn_data.setLayer("input_1", img_hand)
            self.lm_in.send(nn_data)

        for i, h in enumerate(self.hands):
            inference: dai.NNData = self.lm_out.get()
            self.lm_postprocess(h, inference)

        # Keep only hands with a score above threshold
        self.hands = [h for h in self.hands if h.lm_score > self.LANDMARK_SCORE_THRESH]

        # Check that 2 detected hands do not correspond to the same hand in the image
        # That may happen when one hand in the image cross another one
        # A simple method is to assure that the center of the rotated rectangles are not too close
        if len(self.hands) == 2:
            dist_rect_centers = MpU.distance(
                np.array((self.hands[0].rect_x_center_a, self.hands[0].rect_y_center_a)),
                np.array((self.hands[1].rect_x_center_a, self.hands[1].rect_y_center_a)))
            if dist_rect_centers < 5:
                # Keep the hand with higher landmark score
                if self.hands[0].lm_score > self.hands[1].lm_score:
                    self.hands = [self.hands[0]]
                else:
                    self.hands = [self.hands[1]]

        self.hands_from_landmarks = [MpU.hand_landmarks_to_rect(h) for h in self.hands]

        nb_hands: int = len(self.hands)

        if not self.use_previous_landmark or self.nb_hands_in_previous_frame != nb_hands:
            for i in range(2):
                self.handedness_avg[i].reset()

        for i in range(nb_hands):
            self.hands[i].handedness = self.handedness_avg[i].update(self.hands[i].handedness)

        if not self.solo and nb_hands == 2 and (self.hands[0].handedness - 0.5) * (self.hands[1].handedness - 0.5) > 0:
            self.hands = [self.hands[0]]  # We keep the hand with best score
            nb_hands = 1

        # Determine if we should use this frames landmarks to speed up next frames inference
        self.use_previous_landmark = True
        if nb_hands == 0:
            self.use_previous_landmark = False
        elif not self.solo and nb_hands == 1:
            self.use_previous_landmark = False

        self.nb_hands_in_previous_frame = nb_hands

        # Adjust hand landmarks based on padding
        hand: Union[HandRegion, Any]
        for hand in self.hands:
            if self.pad_h > 0:
                hand.landmarks[:, 1] -= self.pad_h
                for i in range(len(hand.rect_points)):
                    hand.rect_points[i][1] -= self.pad_h
            if self.pad_w > 0:
                hand.landmarks[:, 0] -= self.pad_w
                for i in range(len(hand.rect_points)):
                    hand.rect_points[i][0] -= self.pad_w
            hand.label = "right" if hand.handedness > 0.5 else "left"

        # Get depth map information for each landmark detected
        self.xy_to_xyz()

        # Record historical information about the last 10 hands to smooth over.
        for h in self.hands:
            self.hand_hist[h.label].append(h.xyz)

        return frame, self.hands


if __name__ == "__main__":
    import cv2

    ht = HandTracker()
    while True:
        frame = ht.get_depth_frame()
        frame = cv2.applyColorMap(frame.astype(np.uint8), cv2.COLORMAP_HOT)
        cv2.imshow("depth", frame)

        if cv2.waitKey(1) == ord('q'):
            break



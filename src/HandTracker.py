"""Heavily inspired from depthai example"""
import dis

import cv2
import pathlib
from collections import deque

import depthai as dai
import depthai.node
import numpy as np
from threading import Lock

import src.MediapipeUtils as mpu
from src.CustomFilters import WLSFilter
from src.FPS import FPS


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1)  # .flatten()


class HandTracker:
    PD_BLOB_PATH = "checkpoints/palm_detection_sh4.blob"
    LM_BLOB_PATH = "checkpoints/hand_landmark_full_sh4.blob"
    INTERNAL_FPS = 23
    INTERNAL_FRAME_HEIGHT = 640
    DEPTH_REGION_SIZE = 7
    RESOLUTION = (1920, 1080)

    PD_SCORE_THRESH = 0.5
    PD_NMS_SCORE_THRESH = 0.3
    LM_SCORE_THRESH = 0.5

    PD_INP_LENGTH = 128
    LM_INP_LENGTH = 224

    def __init__(self, solo=False):

        # Sensor variables
        super().__init__()
        self.hands_from_landmarks = None
        self.img_w, self.img_h = None, None

        # WLS filtering
        self.baseline = 75  # mm
        self.fov = 71.86
        self.disp_multiplier = None

        self.wls_filter = WLSFilter(_lambda=8000, _sigma=1.5, baseline=self.baseline, fov=self.fov)

        # Prediction variables
        self.solo = solo
        self.use_previous_landmark = False
        self.hands = None
        self.nb_hands_in_previous_frame = 0
        self.anchors = mpu.generate_handtracker_anchors(self.PD_INP_LENGTH, self.PD_INP_LENGTH)
        self.nb_anchors = self.anchors.shape[0]
        frame_size, self.scale_nd = mpu.find_isp_scale_params(
            self.INTERNAL_FRAME_HEIGHT * self.RESOLUTION[0] / self.RESOLUTION[1], self.RESOLUTION, is_height=False)
        self.img_h = int(round(self.RESOLUTION[1] * self.scale_nd[0] / self.scale_nd[1]))
        self.img_w = int(round(self.RESOLUTION[0] * self.scale_nd[0] / self.scale_nd[1]))
        self.pad_h = (self.img_w - self.img_h) // 2
        self.pad_w = 0
        self.frame_size = self.img_w
        self.crop_w = 0

        # better than queue as it'll remove the last element if the size exceeds the max len
        self.hand_hist = {"left": deque(maxlen=10), "right": deque(maxlen=10)}

        # FPS
        self.fps = FPS()

        # self.pipeline
        self.device = dai.Device()

        self.pipeline = dai.Pipeline()
        self.create_pipeline()
        self.device.startPipeline(self.pipeline)

        # Queues
        self.rgb_out = self.device.getOutputQueue(name="rgb_out", maxSize=1, blocking=False)
        self.dpt_out = self.device.getOutputQueue(name="dpt_out", maxSize=1, blocking=False)
        self.sdo_out = self.device.getOutputQueue(name="sdo_out", maxSize=2, blocking=False)
        self.imu_out = self.device.getOutputQueue(name="imu_out", maxSize=1, blocking=False)

        self.right = self.device.getOutputQueue(name="right", maxSize=4, blocking=False)
        self.disp = self.device.getOutputQueue(name="disp", maxSize=4, blocking=False)

        self.pd_in = self.device.getInputQueue(name="pd_in")
        self.pd_out = self.device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)

        self.lm_in = self.device.getInputQueue(name="lm_in")
        self.lm_out = self.device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)

        self.scc_in = self.device.getInputQueue(name="scc_in")

        self.manip_cfg = self.device.getInputQueue(name="manip_cfg")

        self.handedness_avg = [mpu.HandednessAverage() for _ in range(2)]

        # Detector Plane
        self.detector_plane = [0, 0, 0, 0]

    def create_rgb_pipeline(self) -> (dai.node.ImageManip, dai.node.ColorCamera):
        #### RGB Setup ####
        rgb = self.pipeline.createColorCamera()
        rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        rgb.setInterleaved(False)

        rgb.setIspScale(self.scale_nd[0], self.scale_nd[1])
        rgb.setFps(self.INTERNAL_FPS)

        # Image manipulation for palm detection
        manip = self.pipeline.createImageManip()
        manip.setMaxOutputFrameSize(self.PD_INP_LENGTH ** 2 * 3)
        manip.setWaitForConfigInput(True)
        manip.inputImage.setQueueSize(1)
        manip.inputImage.setBlocking(False)

        rgb.preview.link(manip.inputImage)
        rgb.setVideoSize(self.img_w, self.img_h)
        rgb.setPreviewSize(self.img_w, self.img_h)

        manip_cfg_in = self.pipeline.createXLinkIn()
        manip_cfg_in.setStreamName("manip_cfg")
        manip_cfg_in.out.link(manip.inputConfig)

        rgb_out = self.pipeline.createXLinkOut()
        rgb_out.setStreamName("rgb_out")
        rgb_out.input.setQueueSize(1)
        rgb_out.input.setBlocking(False)
        rgb.video.link(rgb_out.input)

        return rgb, manip

    def create_dpt_pipeline(self, rgb):
        #### Depth Setup ####
        # Set camera to fixed focus for RGB/depth alignment
        calib = self.device.readCalibration()
        calib_pos = calib.getLensPosition(dai.CameraBoardSocket.RGB)
        rgb.initialControl.setManualFocus(calib_pos)

        # Left MonoCamera
        left = self.pipeline.createMonoCamera()
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        left.setFps(self.INTERNAL_FPS)

        # Right MonoCamera
        right = self.pipeline.createMonoCamera()
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        right.setFps(self.INTERNAL_FPS)

        # Stereo Camera
        stro = self.pipeline.createStereoDepth()
        stro.initialConfig.setConfidenceThreshold(255)
        stro.setRectifyEdgeFillColor(0)
        stro.setLeftRightCheck(True)  # Must be true for RGB/depth alignment, false would be better for occlusion
        stro.setDepthAlign(dai.CameraBoardSocket.RGB)

        stro.setExtendedDisparity(False)
        stro.setSubpixel(False)

        # Set depth map output
        dpt = self.pipeline.createXLinkOut()
        dpt.setStreamName("dpt_out")
        dpt.input.setQueueSize(1)
        dpt.input.setBlocking(False)
        stro.depth.link(dpt.input)

        rectifiedLeft = self.pipeline.createXLinkOut()
        rectifiedLeft.setStreamName("right")
        stro.rectifiedLeft.link(rectifiedLeft.input)

        disp = self.pipeline.createXLinkOut()
        disp.setStreamName("disp")
        stro.disparity.link(disp.input)

        # Spatial Location Calculator
        slc = self.pipeline.createSpatialLocationCalculator()
        # Allows us to change ROI on the fly
        slc.setWaitForConfigInput(True)
        slc.inputDepth.setBlocking(False)
        slc.inputDepth.setQueueSize(1)

        sdo = self.pipeline.createXLinkOut()
        sdo.setStreamName("sdo_out")
        sdo.input.setQueueSize(1)
        sdo.input.setBlocking(True)

        scc_in = self.pipeline.createXLinkIn()
        scc_in.setStreamName("scc_in")

        # Link left and right monocamera to the stereo depth object
        left.out.link(stro.left)
        right.out.link(stro.right)

        # Link stereo depth to input into the spatial location calculator
        stro.depth.link(slc.inputDepth)

        slc.out.link(sdo.input)
        scc_in.out.link(slc.inputConfig)

    def create_acc_pipeline(self):
        #### Accelerometer Setup ####
        imu = self.pipeline.createIMU()
        imu_out = self.pipeline.createXLinkOut()
        imu_out.setStreamName("imu_out")

        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 25)
        imu.setMaxBatchReports(1)
        imu.out.link(imu_out.input)

    def create_nn_pipeline(self, manip):
        #### Neural Network Setup ####
        print("Neural networks setup...")

        # Palm detection setup
        pd_nn = self.pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(pathlib.Path(self.PD_BLOB_PATH))
        pd_nn.input.setQueueSize(1)
        pd_nn.input.setBlocking(False)
        manip.out.link(pd_nn.input)

        pd_in = self.pipeline.createXLinkIn()
        pd_in.setStreamName("pd_in")
        pd_in.out.link(pd_nn.input)

        pd_out = self.pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)

        # Landmark detection setup
        lm_nn = self.pipeline.createNeuralNetwork()
        lm_nn.setBlobPath(pathlib.Path(self.LM_BLOB_PATH))
        lm_nn.setNumInferenceThreads(2)

        lm_in = self.pipeline.createXLinkIn()
        lm_in.setStreamName("lm_in")
        lm_in.out.link(lm_nn.input)

        lm_out = self.pipeline.createXLinkOut()
        lm_out.setStreamName("lm_out")
        lm_nn.out.link(lm_out.input)

    def create_pipeline(self):
        self.pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

        print("Sensor setup....")
        rgb, manip = self.create_rgb_pipeline()
        self.create_dpt_pipeline(rgb)
        self.create_acc_pipeline()
        self.create_nn_pipeline(manip)

        print("Pipeline creation successful.")

    def get_rgb_frame(self):
        return self.rgb_out.get().getCvFrame()

    def get_dpt_frame(self):
        return self.dpt_out.get().getFrame()

    def get_camera_tilt(self):
        imu_packet = self.imu_out.get().packets[0]
        return np.rad2deg(np.arctan(imu_packet.acceleroMeter.z / (imu_packet.acceleroMeter.y + 1e-8)))

    def get_depth_at_coords(self, coords, size=None):

        if not size:
            size = self.DEPTH_REGION_SIZE

        conf_datas = []
        for (x, y) in coords:
            rect_center = dai.Point2f(x, y)
            rect_size = dai.Size2f(size, size)

            conf_data = dai.SpatialLocationCalculatorConfigData()
            conf_data.depthThresholds.lowerThreshold = 100
            conf_data.depthThresholds.upperThreshold = 10000
            conf_data.roi = dai.Rect(rect_center, rect_size)
            conf_datas.append(conf_data)

        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.setROIs(conf_datas)
        self.scc_in.send(cfg)

        # NOTE: Optimal distance is between 40cm and 6m
        return np.array([loc.spatialCoordinates.z for loc in self.sdo_out.get().getSpatialLocations()])

    def pd_postprocess(self, inference):
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16)  # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape(
            (self.nb_anchors, 18))  # 896x18
        # Decode bboxes
        hands = mpu.decode_bboxes(self.PD_SCORE_THRESH, scores, bboxes, self.anchors, scale=self.PD_INP_LENGTH,
                                  best_only=self.solo)
        if not self.solo:
            # Non maximum suppression (not needed if solo)
            hands = mpu.non_max_suppression(hands, self.PD_NMS_SCORE_THRESH)[:2]
        mpu.detections_to_rect(hands)
        mpu.rect_transformation(hands, self.img_w, self.img_w)
        return hands

    def lm_postprocess(self, hand, inference):
        # print(inference.getAllLayerNames())
        # The output names of the landmarks model are :
        # Identity_1 (1x1) : score
        # Identity_2 (1x1) : handedness
        # Identity_3 (1x63) : world 3D landmarks (in meters)
        # Identity (1x63) : screen 3D landmarks (in pixels)
        hand.lm_score = inference.getLayerFp16("Identity_1")[0]
        if hand.lm_score > self.LM_SCORE_THRESH:
            hand.handedness = inference.getLayerFp16("Identity_2")[0]
            lm_raw = np.array(inference.getLayerFp16("Identity_dense/BiasAdd/Add")).reshape(-1, 3)
            # hand.norm_landmarks contains the normalized ([0:1]) 3D coordinates of landmarks in the
            # square rotated body bounding box
            hand.norm_landmarks = lm_raw / self.LM_INP_LENGTH
            # hand.norm_landmarks[:,2] /= 0.4

            # Now calculate hand.landmarks = the landmarks in the image coordinate system (in pixel)
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([(x, y) for x, y in hand.rect_points[1:]],
                           dtype=np.float32)  # hand.rect_points[0] is left bottom point and points going clockwise!
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(hand.norm_landmarks[:, :2], axis=0)
            hand.landmarks = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int32)

    def xy_to_xyz(self):
        for i, h in enumerate(self.hands):
            z = np.round(self.get_depth_at_coords(h.landmarks, 2), 0)
            h.xyz = np.column_stack((h.landmarks, z))

    def next_frame(self):
        self.fps.update()
        if not self.use_previous_landmark:
            cfg = dai.ImageManipConfig()
            cfg.setResizeThumbnail(self.PD_INP_LENGTH, self.PD_INP_LENGTH)
            self.manip_cfg.send(cfg)

        frame = self.get_rgb_frame()

        if self.pad_h:
            square_frame = cv2.copyMakeBorder(frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w,
                                              cv2.BORDER_CONSTANT)
        else:
            square_frame = frame

        if self.use_previous_landmark:
            self.hands = self.hands_from_landmarks
        else:
            inference = self.pd_out.get()
            hands = self.pd_postprocess(inference)

            if not self.solo and self.nb_hands_in_previous_frame == 1 and len(hands) <= 1:
                self.hands = self.hands_from_landmarks
            else:
                self.hands = hands

        # Hand landmarks, send requests
        for i, h in enumerate(self.hands):
            img_hand = mpu.warp_rect_img(h.rect_points, square_frame, self.LM_INP_LENGTH, self.LM_INP_LENGTH)
            nn_data = dai.NNData()
            nn_data.setLayer("input_1", to_planar(img_hand, (self.LM_INP_LENGTH, self.LM_INP_LENGTH)))
            self.lm_in.send(nn_data)

        for i, h in enumerate(self.hands):
            inference = self.lm_out.get()
            self.lm_postprocess(h, inference)

        self.hands = [h for h in self.hands if h.lm_score > self.LM_SCORE_THRESH]

        # Check that 2 detected hands do not correspond to the same hand in the image
        # That may happen when one hand in the image cross another one
        # A simple method is to assure that the center of the rotated rectangles are not too close
        if len(self.hands) == 2:
            dist_rect_centers = mpu.distance(
                np.array((self.hands[0].rect_x_center_a, self.hands[0].rect_y_center_a)),
                np.array((self.hands[1].rect_x_center_a, self.hands[1].rect_y_center_a)))
            if dist_rect_centers < 5:
                # Keep the hand with higher landmark score
                if self.hands[0].lm_score > self.hands[1].lm_score:
                    self.hands = [self.hands[0]]
                else:
                    self.hands = [self.hands[1]]

        self.hands_from_landmarks = [mpu.hand_landmarks_to_rect(h) for h in self.hands]

        nb_hands = len(self.hands)

        if not self.use_previous_landmark or self.nb_hands_in_previous_frame != nb_hands:
            for i in range(2):
                self.handedness_avg[i].reset()
        for i in range(nb_hands):
            self.hands[i].handedness = self.handedness_avg[i].update(self.hands[i].handedness)

        if not self.solo and nb_hands == 2 and (self.hands[0].handedness - 0.5) * (self.hands[1].handedness - 0.5) > 0:
            self.hands = [self.hands[0]]  # We keep the hand with best score
            nb_hands = 1

        self.use_previous_landmark = True
        if nb_hands == 0:
            self.use_previous_landmark = False
        elif not self.solo and nb_hands == 1:
            self.use_previous_landmark = False

        self.nb_hands_in_previous_frame = nb_hands

        for hand in self.hands:
            # If we added padding to make the image square, we need to remove this padding from landmark coordinates
            # and from rect_points
            if self.pad_h > 0:
                hand.landmarks[:, 1] -= self.pad_h
                for i in range(len(hand.rect_points)):
                    hand.rect_points[i][1] -= self.pad_h
            if self.pad_w > 0:
                hand.landmarks[:, 0] -= self.pad_w
                for i in range(len(hand.rect_points)):
                    hand.rect_points[i][0] -= self.pad_w
            hand.label = "right" if hand.handedness > 0.5 else "left"

        self.xy_to_xyz()

        for h in self.hands:
            self.hand_hist[h.label].append(h.xyz)

        return frame, self.hands


def crop_center(image, new_width, new_height):
    height, width = image.shape[:2]
    left = int((width - new_width) / 2)
    top = int((height - new_height) / 2)
    right = int((width + new_width) / 2)
    bottom = int((height + new_height) / 2)
    return image[top:bottom, left:right]


if __name__ == "__main__":
    import cv2

    ht = HandTracker()
    while True:
        frame = ht.get_rgb_frame()
        dpt = ht.disp.get().getFrame()  # 16:9
        dispa = ht.disp.get().getFrame()  # 16:9

        right = ht.right.get().getFrame() # 4:3
        right = crop_center(right, right.shape[1], int(right.shape[1] * 9 / 16))
        right = cv2.resize(right, (dispa.shape[1], dispa.shape[0]))
        frame = ht.wls_filter.filter(dispa, right)
        frame = cv2.applyColorMap(frame.astype(np.uint8), cv2.COLORMAP_HOT)
        cv2.imshow("rgb", frame)
        cv2.imshow("disparity", dispa)
        cv2.imshow("right", right)

        if cv2.waitKey(1) == ord('q'):
            break

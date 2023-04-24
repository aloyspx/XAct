"""Heavily inspired from depthai example"""

import pathlib
import random
from collections import deque

import cv2
import depthai as dai
import numpy as np
import pyransac3d as pyransac

import src.MediapipeUtils as mpu
from src.CustomExceptions import DetectorPlaneNotFoundException
from src.FPS import FPS
from src.Visualisers import viz_matplotlib


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1)  # .flatten()


class HandTracker:
    PD_BLOB_PATH = "checkpoints/palm_detection_sh4.blob"
    LM_BLOB_PATH = "checkpoints/hand_landmark_full_sh4.blob"
    INTERNAL_FPS = 23
    INTERNAL_FRAME_HEIGHT = 640
    N_RETRIES = 10
    DEPTH_REGION_SIZE = 7
    RESOLUTION = (1920, 1080)

    PD_SCORE_THRESH = 0.5
    PD_NMS_SCORE_THRESH = 0.3
    LM_SCORE_THRESH = 0.5

    PD_INP_LENGTH = 128
    LM_INP_LENGTH = 224

    def __init__(self, solo=False):
        # Sensor variables
        self.hands_from_landmarks = None
        self.img_w, self.img_h = None, None

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

        # Pipeline
        self.device = dai.Device()
        pipeline = self.create_pipeline()
        self.device.startPipeline(pipeline)

        # Queues
        self.rgb_out = self.device.getOutputQueue(name="rgb_out", maxSize=1, blocking=False)
        self.dpt_out = self.device.getOutputQueue(name="dpt_out", maxSize=1, blocking=False)
        self.imu_out = self.device.getOutputQueue(name="imu_out", maxSize=1, blocking=False)

        self.pd_in = self.device.getInputQueue(name="pd_in")
        self.pd_out = self.device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)

        self.lm_in = self.device.getInputQueue(name="lm_in")
        self.lm_out = self.device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)

        self.scc_in = self.device.getInputQueue(name="scc_in")

        self.manip_cfg = self.device.getInputQueue(name="manip_cfg")

        self.handedness_avg = [mpu.HandednessAverage() for _ in range(2)]

        # Detector Plane
        self.detector_plane = [0, 0, 0, 0]
        self.get_detector_plane()

    def create_pipeline(self):
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

        print("Sensor setup....")
        #### RGB Setup ####
        rgb = pipeline.createColorCamera()
        rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        rgb.setInterleaved(False)

        rgb.setIspScale(self.scale_nd[0], self.scale_nd[1])
        rgb.setFps(self.INTERNAL_FPS)

        # Image manipulation for palm detection
        manip = pipeline.createImageManip()
        manip.setMaxOutputFrameSize(self.PD_INP_LENGTH ** 2 * 3)
        manip.setWaitForConfigInput(True)
        manip.inputImage.setQueueSize(1)
        manip.inputImage.setBlocking(False)

        rgb.preview.link(manip.inputImage)
        rgb.setVideoSize(self.img_w, self.img_h)
        rgb.setPreviewSize(self.img_w, self.img_h)

        manip_cfg_in = pipeline.createXLinkIn()
        manip_cfg_in.setStreamName("manip_cfg")
        manip_cfg_in.out.link(manip.inputConfig)

        rgb_out = pipeline.createXLinkOut()
        rgb_out.setStreamName("rgb_out")
        rgb_out.input.setQueueSize(1)
        rgb_out.input.setBlocking(False)
        rgb.video.link(rgb_out.input)

        #### Depth Setup ####
        # Set camera to fixed focus for RGB/depth alignment
        calib = self.device.readCalibration()
        calib_pos = calib.getLensPosition(dai.CameraBoardSocket.RGB)
        rgb.initialControl.setManualFocus(calib_pos)

        # Left MonoCamera
        left = pipeline.createMonoCamera()
        left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        left.setFps(self.INTERNAL_FPS)

        # Right MonoCamera
        right = pipeline.createMonoCamera()
        right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
        right.setFps(self.INTERNAL_FPS)

        # Stereo Camera
        dpt = pipeline.createStereoDepth()
        dpt.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
        dpt.setLeftRightCheck(True)
        dpt.setDepthAlign(dai.CameraBoardSocket.RGB)
        # Increase accuracy set to False, latency tradeoff
        dpt.setSubpixel(False)

        # Spatial Location Calculator
        slc = pipeline.createSpatialLocationCalculator()
        # Allows us to change ROI on the fly
        slc.setWaitForConfigInput(True)
        slc.inputDepth.setBlocking(False)
        slc.inputDepth.setQueueSize(1)

        sdo = pipeline.createXLinkOut()
        sdo.setStreamName("dpt_out")
        sdo.input.setQueueSize(1)
        sdo.input.setBlocking(False)

        scc_in = pipeline.createXLinkIn()
        scc_in.setStreamName("scc_in")

        # Link left and right monocamera to the stereo depth object
        left.out.link(dpt.left)
        right.out.link(dpt.right)

        # Link stereo depth to input into the spatial location calculator
        dpt.depth.link(slc.inputDepth)

        slc.out.link(sdo.input)
        scc_in.out.link(slc.inputConfig)

        #### Accelerometer Setup ####

        imu = pipeline.createIMU()
        imu_out = pipeline.createXLinkOut()
        imu_out.setStreamName("imu_out")

        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 25)
        imu.setMaxBatchReports(1)
        imu.out.link(imu_out.input)

        #### Neural Network Setup ####
        print("Neural networks setup...")

        # Palm detection setup
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(pathlib.Path(self.PD_BLOB_PATH))
        pd_nn.input.setQueueSize(1)
        pd_nn.input.setBlocking(False)
        manip.out.link(pd_nn.input)

        pd_in = pipeline.createXLinkIn()
        pd_in.setStreamName("pd_in")
        pd_in.out.link(pd_nn.input)

        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)

        # Landmark detection setup
        lm_nn = pipeline.createNeuralNetwork()
        lm_nn.setBlobPath(pathlib.Path(self.LM_BLOB_PATH))
        lm_nn.setNumInferenceThreads(2)

        lm_in = pipeline.createXLinkIn()
        lm_in.setStreamName("lm_in")
        lm_in.out.link(lm_nn.input)

        lm_out = pipeline.createXLinkOut()
        lm_out.setStreamName("lm_out")
        lm_nn.out.link(lm_out.input)

        print("Pipeline creation successful.")
        return pipeline

    def get_rgb_frame(self):
        return self.rgb_out.get().getCvFrame()

    def get_camera_tilt(self):
        imu_packet = self.imu_out.get().packets[0]
        return np.rad2deg(np.arctan(imu_packet.acceleroMeter.z / (imu_packet.acceleroMeter.y + 1e-8)))

    def get_depth_at_coords(self, coords):

        conf_datas = []
        for (x, y) in coords:
            rect_center = dai.Point2f(x, y)
            rect_size = dai.Size2f(self.DEPTH_REGION_SIZE, self.DEPTH_REGION_SIZE)

            conf_data = dai.SpatialLocationCalculatorConfigData()
            conf_data.depthThresholds.lowerThreshold = 100
            conf_data.depthThresholds.upperThreshold = 10000
            conf_data.roi = dai.Rect(rect_center, rect_size)
            conf_datas.append(conf_data)

        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.setROIs(conf_datas)
        self.scc_in.send(cfg)

        # NOTE: Optimal distance is between 40cm and 6m
        return np.array([loc.spatialCoordinates.z for loc in self.dpt_out.get().getSpatialLocations()])

    def get_detector_plane(self, num_points=512, display=False):

        for _ in range(self.N_RETRIES):
            # Sample some random 2D points and fetch their corresponding depth
            coords_2d = np.array([[random.randint(self.DEPTH_REGION_SIZE, self.img_w - self.DEPTH_REGION_SIZE),
                                   random.randint(self.DEPTH_REGION_SIZE, self.img_h - self.DEPTH_REGION_SIZE)]
                                  for _ in range(num_points)])
            depth_values = self.get_depth_at_coords(coords_2d)
            coords_3d = np.hstack((coords_2d, depth_values.reshape(-1, 1)))

            # Check that the depth has been detected
            if np.count_nonzero(depth_values) > 50:
                # Filter out points that have 0 depth
                coords_3d = coords_3d[coords_3d[:, 2] != 0]

                # Calculate the plane using RANSAC
                plane = pyransac.Plane()
                plane_equation, inlier_points = plane.fit(coords_3d, thresh=20, minPoints=num_points // 2)

                if display:
                    viz_matplotlib(plane_equation, coords_3d, inlier_points)

                self.detector_plane = plane_equation
                return

        raise DetectorPlaneNotFoundException

    def get_hand_plane(self, display=False):
        # n_frames x n_keypoints x coordinates
        planes = {}
        for key in self.hand_hist.keys():
            coords_3d = np.array(self.hand_hist[key]).reshape(-1, 3)

            # filter out points with depth not detected
            coords_3d = coords_3d[coords_3d[:, 2] != 0]

            if coords_3d.size == 0:
                continue

            plane = pyransac.Plane()
            plane_equation, inlier_points = plane.fit(coords_3d, thresh=20, minPoints=int(coords_3d.shape[0] * 0.5))

            if display:
                viz_matplotlib(plane_equation, coords_3d, inlier_points)

            planes[key] = plane_equation

        return planes

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
            z = np.round(self.get_depth_at_coords(h.landmarks), 0)
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

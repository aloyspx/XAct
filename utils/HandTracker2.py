import random
import cv2
import depthai as dai
import numpy as np

import MediapipeUtils as mpu

from CustomExceptions import TableNotFoundException


class HandTracker2:
    INTERNAL_FPS = 23
    INTERNAL_FRAME_HEIGHT = 640

    def __init__(self):
        self.device = dai.Device()

        # Pipeline
        print(self.device.getCameraSensorNames())
        pipeline = self.create_pipeline()
        self.device.startPipeline(pipeline)

        # Queues
        self.rgb_out = self.device.getOutputQueue(name="rgb_out", maxSize=1, blocking=False)
        self.dpt_out = self.device.getOutputQueue(name="dpt_out", maxSize=1, blocking=False)
        self.scc_in = self.device.getInputQueue(name="scc_in")

        x,y = 900, 900
        while True:
            im = self.rgb_out.get().getCvFrame()
            im = cv2.circle(im, (x, y), radius=1, color=(0, 0, 255), thickness=10)
            im = cv2.flip(im, 0)
            im = cv2.flip(im, 1)
            cv2.imshow("test", im)
            if cv2.waitKey(1) == ord('q'):
                break

            self.get_table_plane()



        print("")

    def create_pipeline(self):
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

        # RGB Setup
        rgb = pipeline.createColorCamera()
        rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        rgb.setInterleaved(False)
        _, scale_nd = mpu.find_isp_scale_params(self.INTERNAL_FRAME_HEIGHT * 1920 / 1080, (1920, 1080), is_height=False)
        rgb.setIspScale(scale_nd[0], scale_nd[1])
        rgb.setFps(self.INTERNAL_FPS)

        rgb_out = pipeline.createXLinkOut()
        rgb_out.setStreamName("rgb_out")
        rgb_out.input.setQueueSize(1)
        rgb_out.input.setBlocking(False)
        rgb.video.link(rgb_out.input)

        # Depth Setup

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
        # dpt.set
        dpt.setLeftRightCheck(True)
        dpt.setDepthAlign(dai.CameraBoardSocket.RGB)
        # Increase accuracy set to True, latency tradeoff
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

        return pipeline

    def get_rgb_frame(self):
        pass

    def get_depth_at_coords(self, coords, size=5):

        conf_datas = []
        for coord in coords:
            x, y = coord[0], coord[1]
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
        return np.array([loc.spatialCoordinates.z for loc in self.dpt_out.get().getSpatialLocations()])

    def get_table_plane(self, n_pts=256):
        n_retries = 0
        depth_z = list(np.zeros(n_pts))

        while depth_z == list(np.zeros(n_pts)):
            width, height = 1152, 648
            coords = [[random.randint(5, width-5), random.randint(5, height-5)] for _ in range(n_pts)]
            depth_z = self.get_depth_at_coords(coords)

            # Check that the depth has been detected
            if (depth_z == np.zeros(n_pts)).all():
                n_retries += 1
                if n_retries > 5:
                    raise TableNotFoundException

            # Find the most common depth values
            depth_z = depth_z[depth_z != 0]
            unique, count = np.unique(depth_z, return_counts=True)
            cmn_vals = unique[np.argsort(-count)][:5]


if __name__ == "__main__":
    ht2 = HandTracker2()

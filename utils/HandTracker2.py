import cv2
import depthai as dai
import MediapipeUtils as mpu
import numpy as np
import pyransac3d as pyransac
import random

from CustomExceptions import DetectorPlaneNotFoundException
from matplotlib import pyplot as plt


class HandTracker2:
    BLOB_PATH = "../checkpoints/hand_landmark_full_sh4.blob"
    INTERNAL_FPS = 23
    INTERNAL_FRAME_HEIGHT = 640
    N_RETRIES = 10
    DEPTH_REGION_SIZE = 5

    def __init__(self):
        self.device = dai.Device()

        # Pipeline
        print(self.device.getCameraSensorNames())
        pipeline = self.create_pipeline()
        self.device.startPipeline(pipeline)

        # Queues
        self.rgb_out = self.device.getOutputQueue(name="rgb_out", maxSize=1, blocking=False)
        self.dpt_out = self.device.getOutputQueue(name="dpt_out", maxSize=1, blocking=False)
        self.imu_out = self.device.getOutputQueue(name="imu_out", maxSize=1, blocking=False)
        self.scc_in = self.device.getInputQueue(name="scc_in")

        self.detector_plane = self.get_detector_plane()

    def create_pipeline(self):
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

        print("Sensor setup....")
        #### RGB Setup ####
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
        # dpt.set
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

        print("Neural Network setup...")

        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath()

        return pipeline

    def get_rgb_frame(self):
        return self.rgb_out.get().getCvFrame()

    def get_camera_tilt(self):
        imu_packet = self.imu_out.get().packets[0]
        return np.rad2deg(np.arctan(imu_packet.acceleroMeter.z / (imu_packet.acceleroMeter.y + 1e-8)))

    def get_depth_at_coords(self, coords):

        conf_datas = []
        for coord in coords:
            x, y = coord[0], coord[1]
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
        image_width, image_height = 1152, 648

        for _ in range(self.N_RETRIES):
            # Sample some random 2D points and fetch their corresponding depth
            coords_2d = np.array([[random.randint(self.DEPTH_REGION_SIZE, image_width - self.DEPTH_REGION_SIZE),
                                   random.randint(self.DEPTH_REGION_SIZE, image_height - self.DEPTH_REGION_SIZE)]
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
                    self.viz_matplotlib(plane_equation, coords_3d, inlier_points)

                return plane_equation

        raise DetectorPlaneNotFoundException

    @staticmethod
    def viz_matplotlib(plane_equation, coords, pts):
        plt.close('all')
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim([-1000, 0])

        # Plot the points
        ax.scatter(coords[pts, 0], coords[pts, 1], -coords[pts, 2], c='b', marker='o')
        outliers = list(set(list(np.arange(len(coords)))).difference(set(list(pts))))
        ax.scatter(coords[outliers, 0], coords[outliers, 1], -coords[outliers, 2], c='r', marker='x')

        # Plot the plane
        a, b, c, d = plane_equation
        xx, yy = np.meshgrid(range(0, 1152), range(0, 648))
        zz = (-a * xx - b * yy - d) * 1. / c
        ax.plot_surface(xx, yy, -zz)

        # Show the plot
        plt.show()


if __name__ == "__main__":
    ht2 = HandTracker2()
    while True:
        print(ht2.get_detector_plane(display=True))
        cv2.imshow("fff", ht2.get_rgb_frame())
        if cv2.waitKey(1) == ord('q'):
            break
    print("dfdf")

import random
import time

import numpy as np
import pyransac3d as pyransac

from PyQt5.QtCore import QThread

from src.CustomExceptions import DetectorPlaneNotFoundException
from src.HandTracker import HandTracker


class PlaneCalculatorThread(QThread):

    N_RETRIES = 10

    def __init__(self, tracker: HandTracker):
        super().__init__()
        self._run_flag = True
        self.tracker = tracker

    @staticmethod
    def scale_plane(plane):
        if plane[-2] < 0:
            plane[-2] = -plane[-2]

        return plane

    @staticmethod
    def calc_angle_between_planes(P, Q):
        assert len(P) == 4 and len(Q) == 4

        n1 = np.round(P[:-1], 1)
        n2 = np.round(Q[:-1], 1)
        ang = np.arccos(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-8))

        return np.rad2deg(ang)

    def get_detector_plane(self, num_points=512):

        for _ in range(self.N_RETRIES):
            # Sample some random 2D points and fetch their corresponding depth
            coords_2d = np.array([[random.randint(self.tracker.DEPTH_REGION_SIZE,
                                                  self.tracker.img_w - self.tracker.DEPTH_REGION_SIZE),
                                   random.randint(self.tracker.DEPTH_REGION_SIZE,
                                                  self.tracker.img_h - self.tracker.DEPTH_REGION_SIZE)]
                                  for _ in range(num_points)])

            depth_values = self.tracker.get_depth_at_coords(coords_2d)
            coords_3d = np.hstack((coords_2d, depth_values.reshape(-1, 1)))

            # Check that the depth has been detected
            if np.count_nonzero(depth_values) > 50:
                # Filter out points that have 0 depth
                coords_3d = coords_3d[coords_3d[:, 2] != 0]

                # Calculate the plane using RANSAC
                plane = pyransac.Plane()
                plane_equation, inlier_points = plane.fit(coords_3d, thresh=20, minPoints=num_points // 2,
                                                          maxIteration=50)
                self.scale_plane(plane_equation)

                self.tracker.detector_plane = plane_equation
                return

        raise DetectorPlaneNotFoundException

    def get_hand_plane(self):
        # n_frames x n_keypoints x coordinates
        planes = {}
        for key in self.tracker.hand_hist.keys():

            hist = self.tracker.hand_hist[key]
            if len(hist) == 0:
                continue

            # filter out points with depth not detected
            masked_arr = np.ma.masked_equal(hist, 0)
            coords_3d = np.mean(masked_arr, axis=0)

            plane = pyransac.Plane()
            plane_equation, inlier_points = plane.fit(coords_3d, thresh=20, minPoints=int(coords_3d.shape[0] * 0.5),
                                                      maxIteration=50)
            planes[key] = self.scale_plane(plane_equation)

        return planes

    def run(self):
        while self._run_flag:
            time.sleep(2.5)
            hand_plane = self.get_hand_plane()
            self.get_detector_plane()
            print(self.calc_angle_between_planes(hand_plane['left'], self.tracker.detector_plane))

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

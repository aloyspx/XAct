import random
import time
from PyQt5.QtCore import QThread

from src.Geometry import calc_angle_between_planes
from src.HandTracker import HandTracker


class PlaneCalculatorThread(QThread):

    def __init__(self, tracker: HandTracker) -> None:
        super().__init__()
        self._run_flag = True
        self.tracker = tracker

    def run(self) -> None:
        while self._run_flag:
            time.sleep(2.5)
            hand_plane = self.tracker.get_hand_plane()
            self.tracker.get_detector_plane()
            if "left" in hand_plane.keys():
                print("left angle : ", calc_angle_between_planes(hand_plane['left'], self.tracker.detector_plane))
            if "right" in hand_plane.keys():
                print("right angle : ", calc_angle_between_planes(hand_plane["right"], self.tracker.detector_plane))

    def stop(self) -> None:
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

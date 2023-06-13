import random
import time
from PyQt5.QtCore import QThread

from src.Calculators import get_hand_plane
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
            self.tracker.get_detector_plane()

            left_hand_plane = get_hand_plane(self.tracker.hand_hist, 'left')
            if left_hand_plane:
                print("left angle : ",  calc_angle_between_planes(left_hand_plane, self.tracker.detector_plane))

            right_hand_plane = get_hand_plane(self.tracker.hand_hist, 'right')
            if right_hand_plane:
                print("right angle : ", calc_angle_between_planes(right_hand_plane, self.tracker.detector_plane))

    def stop(self) -> None:
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

import json

import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, QThread, Qt

from utils.HandTracker import HandTracker
from utils.constants import LINES_HAND, COR_VECT, SAG_VECT


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.tracker = HandTracker()

    def draw(self, frame, hands):
        frame = cv2.putText(frame, f"FPS: {self.tracker.fps.get()}:",
                            (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        if hands:
            for hand in hands:
                for i, landmark in enumerate(hand.landmarks):
                    frame = cv2.circle(frame, landmark, radius=1, color=(0, 0, 255), thickness=10)
                    frame = cv2.putText(frame, str(i), landmark + 10, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                                        cv2.LINE_AA)

                for line in LINES_HAND:
                    frame = cv2.line(frame, hand.landmarks[line[0]], hand.landmarks[line[1]], color=(0, 0, 255),
                                     thickness=2)

        return frame

    @staticmethod
    def to_view(lm, view):
        if view == "cor":
            return lm[:, :2]
        if view == "sag":
            return lm[:, ::2]
        if view == "ax":
            return lm[:, 1:]
        raise NotImplementedError()

    def measure_coronal(self, hand, frame):

        cor = dict()
        cor_lm = self.to_view(hand.world_landmarks, view="cor")

        for key in COR_VECT.keys():
            v1 = cor_lm[COR_VECT[key][0][0]] - cor_lm[COR_VECT[key][0][1]]
            v2 = cor_lm[COR_VECT[key][1][0]] - cor_lm[COR_VECT[key][1][1]]
            cor[key] = self.angle_between(v1, v2)

            if frame.any():
                landmark = (hand.landmarks[COR_VECT[key][0][0]] + hand.landmarks[COR_VECT[key][1][0]]) / 2
                landmark[0] -= 10
                frame = cv2.putText(frame, str(int(cor[key])), landmark.astype(int), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1, cv2.LINE_AA)

        return cor, frame

    def measure_sagittal(self, hand):

        sag = dict()
        sag_lm = self.to_view(hand.world_landmarks, view="sag")

        for key in SAG_VECT.keys():
            v1 = sag_lm[SAG_VECT[key][0][0]] - sag_lm[SAG_VECT[key][0][1]]
            v2 = sag_lm[SAG_VECT[key][1][0]] - sag_lm[SAG_VECT[key][1][1]]
            sag[key] = self.angle_between(v1, v2)

        return sag

    @staticmethod
    def angle_between(v1, v2):
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return np.rad2deg(rad)

    @staticmethod
    def print_pretty_dict(x):
        print(json.dumps(x, sort_keys=True, indent=4))

    def run(self):

        while self._run_flag:
            frame, hands, bag = self.tracker.next_frame()
            if frame.any():
                frame = self.draw(frame, hands)
                for hand in hands:
                    cor, frame = self.measure_coronal(hand, frame)

                self.change_pixmap_signal.emit(frame)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.tracker.exit()
        self.wait()

from typing import List

import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, QThread

from src.HandTracker import HandTracker
from src.utils.FPS import FPS
from src.utils.MediapipeUtils import HandRegion


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, tracker: HandTracker) -> None:
        super().__init__()
        self._run_flag = True
        self.tracker = tracker

    @staticmethod
    def draw(frame: np.ndarray, fps: FPS, hands: List[HandRegion] = None, keypoints: bool = False,
             depth: bool = False) -> np.ndarray:
        frame = fps.draw(frame, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if hands:
            for hand in hands:
                for i, xyz in enumerate(hand.xyz):
                    xy = hand.landmarks[i]
                    z = xyz.astype(int)[-1]
                    frame = cv2.circle(frame, xy, radius=1, color=(0, 0, 255), thickness=10)

                    if keypoints:
                        frame = cv2.putText(frame, str(i), xy + 10, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                                            cv2.LINE_AA)

                    if depth:
                        frame = cv2.putText(frame, f"[{z}]", xy + 20, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                                            cv2.LINE_AA)

        return frame

    def run(self) -> None:

        while self._run_flag:

            frame, hands = self.tracker.next_frame()
            depth = self.tracker.get_depth_frame()

            if frame.any():
                frame = self.draw(frame, self.tracker.fps, hands, depth=True)
                depth = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)
                frame = cv2.addWeighted(depth, 0.4, frame, 0.6, 0)
                self.change_pixmap_signal.emit(frame)

    def stop(self) -> None:
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.tracker.exit()
        self.wait()

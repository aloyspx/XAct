from typing import List

import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, QThread

from src.HandTracker import HandTracker
from src.Visualisers import draw


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, tracker: HandTracker) -> None:
        super().__init__()
        self._run_flag = True
        self.tracker = tracker

    def run(self) -> None:
        while self._run_flag:

            frame, hands = self.tracker.next_frame()
            depth = self.tracker.get_depth_frame()

            if frame.any():
                frame = draw(frame, self.tracker.fps, hands, depth=True)
                depth = cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)
                frame = cv2.addWeighted(depth, 0.4, frame, 0.6, 0)
                self.change_pixmap_signal.emit(frame)

    def stop(self) -> None:
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.tracker.exit()
        self.wait()

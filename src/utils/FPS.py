"""
@author: geaxx
"""
import time
import cv2
import numpy as np

from collections import deque
from typing import Tuple


class FPS:  # To measure the number of frame per second
    def __init__(self, average_of: int = 30) -> None:
        self.fps = None
        self.start = None
        self.timestamps = deque(maxlen=average_of)
        self.nbf = -1

    def update(self) -> None:
        self.timestamps.append(time.monotonic())
        if len(self.timestamps) == 1:
            self.start = self.timestamps[0]
            self.fps = 0
        else:
            self.fps = (len(self.timestamps) - 1) / (self.timestamps[-1] - self.timestamps[0])
        self.nbf += 1

    def get(self) -> float:
        return self.fps

    def get_global(self) -> float:
        return self.nbf / (self.timestamps[-1] - self.start)

    def nb_frames(self) -> int:
        return self.nbf + 1

    def draw(self, frame: np.ndarray, orig: Tuple[int, int] = (0, 0), font: int = cv2.FONT_HERSHEY_SIMPLEX,
             size: int = 2, color: Tuple[int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        return cv2.putText(frame, f"FPS={self.get():.2f}", orig, font, size, color, thickness)

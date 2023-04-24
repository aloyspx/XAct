import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, QThread

from src.Constants import LINES_HAND
from src.HandTracker import HandTracker
from src.Geometry import calc_angle_between_planes


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

    def run(self):
        count = 0
        while self._run_flag:

            frame, hands = self.tracker.next_frame()
            if frame.any():
                frame = self.draw(frame, hands)

                if len(self.tracker.hand_hist["left"]) == 10 and count == 10:
                    h_plane = self.tracker.get_hand_plane(display=False)["left"]
                    d_plane = self.tracker.detector_plane
                    from src.Visualisers import viz_matplotlib
                    viz_matplotlib(d_plane, [], [])
                    print(calc_angle_between_planes(h_plane, d_plane))
                    count = 0

                self.change_pixmap_signal.emit(frame)

                if hands:
                    count+=1

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.tracker.exit()
        self.wait()

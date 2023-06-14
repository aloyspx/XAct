from typing import List

import cv2
# import open3d as o3d
import numpy as np

from src.utils.FPS import FPS
from src.utils.MediapipeUtils import HandRegion


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

            # for line in LINES_HAND:
            #     frame = cv2.line(frame, hand.landmarks[line[0]], hand.landmarks[line[1]], color=(0, 0, 255),
            #                      thickness=2)

    return frame

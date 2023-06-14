from typing import List

import numpy as np
import pyransac3d as pyransac


def get_hand_plane(hand_coords) -> List[int]:
    # find the best fitting plane with ransac
    plane = pyransac.Plane()
    plane_equation, inlier_points = plane.fit(hand_coords, thresh=10, minPoints=hand_coords.shape[0],
                                              maxIteration=100)

    # For simplicity, make sure that the normal vector always has the same sense
    for i in range(len(plane_equation)):
        if plane_equation[i] < 0:
            plane_equation[i] = abs(plane_equation[i])

    return np.round(plane_equation, 1)

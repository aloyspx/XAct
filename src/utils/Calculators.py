from typing import List

import numpy as np
import pyransac3d as pyransac


def calc_angle_between_planes(P: List[float], Q: List[float]) -> float:
    assert len(P) == 4 and len(Q) == 4

    n1 = P[:-1]
    n2 = Q[:-1]
    ang = np.arccos(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-8))

    ang = np.rad2deg(ang)

    if ang > 90:
        return 180 - ang
    return ang


def calc_smallest_distance_between_points_and_surface(points: np.ndarray, plane: np.ndarray):
    normal = plane[:3]
    return np.array([np.abs(np.dot(point, normal) + plane[-1]) / np.linalg.norm(normal) for point in points])


def get_hand_plane(hand_coords) -> List[int]:
    # find the best fitting plane with ransac
    plane = pyransac.Plane()
    plane_equation, inlier_points = plane.fit(hand_coords, thresh=10, minPoints=hand_coords.shape[0],
                                              maxIteration=100)

    return plane_equation

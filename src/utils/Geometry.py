from typing import List

import numpy as np


def calc_angle_between_planes(P: List[float], Q: List[float]) -> float:
    assert len(P) == 4 and len(Q) == 4

    n1 = P[:-1]
    n2 = Q[:-1]
    ang = np.arccos(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-8))

    return np.rad2deg(ang)


def calc_smallest_distance_between_points_and_surface(points: np.ndarray, plane: np.ndarray):
    plane_normal = plane[:3]
    plane_point = np.array([0, 0, plane[3] / plane[2]])

    return np.abs(np.dot(points - plane_point, plane_normal)) / np.linalg.norm(plane_normal)

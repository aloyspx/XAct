from typing import List

import numpy as np


def calc_angle_between_planes(P: List[float], Q: List[float]) -> float:
    assert len(P) == 4 and len(Q) == 4

    n1 = np.round(P[:-1], 1)
    n2 = np.round(Q[:-1], 1)
    ang = np.arccos(np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2) + 1e-8))

    return np.rad2deg(ang)

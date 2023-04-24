import numpy as np


def calc_angle_between_planes(P, Q):
    assert len(P) == 4 and len(Q) == 4

    n1 = P[:-1]
    n2 = Q[:-1]
    cos_angle = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))

    return np.rad2deg(np.arctan(cos_angle))

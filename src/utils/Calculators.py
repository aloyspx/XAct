import math
from typing import List, Tuple

import numpy as np
import pyransac3d as pyransac
from numpy import ndarray


def calc_angle_between_lines(p: List[float], q: List[float]) -> int:
    m1, b1 = p
    m2, b2 = q
    # Compute the angle between the two lines in radians
    angle = np.arctan(abs((m2 - m1) / (1 + m1 * m2)))
    # Convert the angle to degrees
    return int(np.degrees(angle))


# Least Square Regression
def calc_best_fitting_line(points: ndarray) -> List[float]:
    # Convert the list of points to a numpy array
    points = np.array(points)
    # Extract the x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    # Compute the mean of the x and y coordinates
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    # Compute the slope of the line
    m = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    # Compute the y-intercept of the line
    b = y_mean - m * x_mean
    return [m, b]


def calc_projection_3d_points_to_2d_plane(points: ndarray, plane: List[float]) -> ndarray:
    a, b, c, d = plane
    # Define the normal vector of the plane
    normal = np.array([a, b, c])
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    # Define the origin of the plane
    origin = np.array([0, 0, d / c])
    # Define two orthogonal directions for the x and y axes on the plane
    e1 = np.array([1, 0, -a / c])
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(normal, e1)

    projected_points = []
    for point in points:
        # Compute the vector from the origin of the plane to the point
        v = point - origin
        # Compute the dot product of v and the normal vector
        s = np.dot(v, normal)
        # Compute the projection of the point onto the plane
        projection = point - s * normal
        # Compute the coordinates of the projection in the plane's coordinate system
        t1 = np.dot(projection - origin, e1)
        t2 = np.dot(projection - origin, e2)
        projected_points.append([t1, t2])

    return np.array(projected_points)


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


def calc_smallest_distance_between_two_points(p1: ndarray, p2: ndarray) -> float:
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


def get_hand_plane(hand_coords) -> List[int]:
    # find the best fitting plane with ransac
    plane = pyransac.Plane()
    plane_equation, inlier_points = plane.fit(hand_coords, thresh=10, minPoints=int(hand_coords.size * 0.8),
                                              maxIteration=100)

    return plane_equation

from typing import List

import cv2
# import open3d as o3d
import numpy as np
from matplotlib import pyplot as plt

from src.utils.FPS import FPS
from src.utils.MediapipeUtils import HandRegion


def viz_matplotlib(planes, coords=None, pts=None):
    plt.close('all')

    for azim in [0, 45, 90, 135, 180]:
        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim([-1000, 200])
        ax.view_init(elev=0, azim=azim)

        # Plot the points
        if coords.any():
            ax.scatter(coords[:, 0], coords[:, 1], -coords[:, 2], c='r', marker='o')
            # ax.scatter(coords[pts, 0], coords[pts, 1], -coords[pts, 2], c='b', marker='o')
            # outliers = list(set(list(np.arange(len(coords)))).difference(set(list(pts))))
            # ax.scatter(coords[outliers, 0], coords[outliers, 1], -coords[outliers, 2], c='r', marker='x')

        for plane in planes:
            a, b, c, d = plane
            xx, yy = np.meshgrid(range(-1000, 1000), range(-1000, 1000))
            zz = (-a * xx - b * yy - d) * 1. / c
            ax.plot_surface(xx, yy, -zz)

        # Show the plot
        plt.show()


def plot_lines_op3d(vis, points, line_set, spheres):
    # Update the LineSet object
    line_set.points = o3d.utility.Vector3dVector(points)
    vis.update_geometry(line_set)

    # Update the spheres
    for j, sphere in enumerate(spheres):
        sphere.translate(points[j] - sphere.get_center())
        vis.update_geometry(sphere)

    # Render and capture events
    vis.poll_events()
    vis.update_renderer()


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

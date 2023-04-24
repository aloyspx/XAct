import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt

from src.Constants import LINES_HAND


def viz_open3d(plane_eq, points):
    # Create a point cloud from the list of points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Create lines from the line indices
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(points)
    lines.lines = o3d.utility.Vector2iVector(LINES_HAND)

    # Create a mesh plane from the plane equation
    a, b, c, d = plane_eq
    plane = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=0.1)
    plane.compute_vertex_normals()
    R = o3d.geometry.get_rotation_matrix_from_xyz((np.arcsin(c), 0, 0))
    plane.rotate(R)
    plane.translate((0, 0, d))

    # Visualize the point cloud, lines and the plane
    o3d.visualization.draw_geometries([pcd, lines, plane])


def viz_matplotlib(plane_equation, coords, pts):
    plt.close('all')
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim([-1000, 0])

    # Plot the points
    if coords:
        ax.scatter(coords[pts, 0], coords[pts, 1], -coords[pts, 2], c='b', marker='o')
        outliers = list(set(list(np.arange(len(coords)))).difference(set(list(pts))))
        ax.scatter(coords[outliers, 0], coords[outliers, 1], -coords[outliers, 2], c='r', marker='x')

    # Plot the plane
    a, b, c, d = plane_equation
    xx, yy = np.meshgrid(range(0, 1152), range(0, 648))
    zz = (-a * xx - b * yy - d) * 1. / c
    ax.plot_surface(xx, yy, -zz)

    # Show the plot
    plt.show()

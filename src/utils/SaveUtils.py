import threading
import copy
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

from src.utils.Constants import PROXIMAL_LINKS


def save(detector_plane, hand_coords, protocol_name, save_dir="outputs"):
    def save_thread():
        output_folder = f"{save_dir}/{protocol_name}-{str(datetime.now()).replace(' ', '')}/"

        os.makedirs(output_folder, exist_ok=True)

        visualize_plane_hand_animation(detector_plane, hand_coords, output_file_name=f"{output_folder}/animation.mp4")
        np.savez(f"{output_folder}/data.npz", {"hand": np.array(hand_coords), "detector_plane": detector_plane})

    t = threading.Thread(target=save_thread)
    t.start()


def visualize_plane_hand_animation(plane, coords_list, output_file_name="animation"):
    # Plane equation coefficients
    a, b, c, d = plane

    # Create x,y grid
    xx, yy = np.meshgrid(range(-200, 200), range(-200, 200))

    # Calculate corresponding z values for the plane
    zz = (d - a * xx - b * yy) / c

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-(np.max(coords_list) + 100),
                -(np.max(coords_list) - 100))
    ax.view_init(elev=30, azim=0)  # Set the viewpoint

    # Plot the plane
    ax.plot_surface(xx, yy, zz, alpha=0.5)

    # Set up the scatter plot for the points
    scat = ax.scatter([], [], [])

    # Set up the line plots for the connections between points
    lines = [ax.plot([], [], [])[0] for _ in range(len(PROXIMAL_LINKS))]

    # Define the update function for the animation
    def update(frame):
        coords = coords_list[frame]
        points = copy.deepcopy(coords)
        points[:, 2] = -points[:, 2]

        # Update the scatter plot data
        scat._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

        # Update the line plot data
        for i, connection in enumerate(PROXIMAL_LINKS):
            x_values = [points[connection[0]][0], points[connection[1]][0]]
            y_values = [points[connection[0]][1], points[connection[1]][1]]
            z_values = [points[connection[0]][2], points[connection[1]][2]]
            lines[i].set_data(x_values, y_values)
            lines[i].set_3d_properties(z_values)

        return scat,

    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=range(len(coords_list)), blit=True)

    # Show the animation
    if ".mp4" not in output_file_name[-4:]:
        output_file_name += ".mp4"
    ani.save(output_file_name)

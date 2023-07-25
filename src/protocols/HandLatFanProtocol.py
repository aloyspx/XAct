import numpy as np
import pyransac3d
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem

from src.protocols.BaseProtocol import BaseProtocol
from src.utils.Calculators import get_hand_plane, calc_angle_between_planes, \
    calc_smallest_distance_between_points_and_surface, calc_projection_3d_points_to_2d_plane, calc_best_fitting_line, \
    calc_angle_between_lines, calc_smallest_distance_between_two_points
from src.utils.Constants import HAND_KEYPOINTS, FINGER_LINKS


class HandLatFanProtocol(BaseProtocol):
    def __init__(self, handedness: str, table_widget: QtWidgets.QTableWidget):
        super().__init__(f"Hand_Lateral_{handedness}", handedness, table_widget)

    def check_constraints(self):
        self.table_widget.clearContents()

        hand = self.dict_to_ndarray(self.parameters["hand"])
        self.hist.append(hand)
        detector_plane = self.parameters["detector_plane"]

        # 1. Check that the angle between the palm and the detector plane is more than 80 degrees
        hand_plane = get_hand_plane(hand[[1, 2, 5, 9, 13, 17]])
        angle = int(calc_angle_between_planes(detector_plane, hand_plane))
        is_angle = (70 < angle)

        # Display unmet constraints
        if not is_angle and True:
            self.table_widget.setItem(1, 0, QTableWidgetItem("Palm Angle"))
            self.table_widget.setItem(1, 1, QTableWidgetItem(f"{angle} deg"))

        # 2. Check that the pinky keypoints are less than 50mm from the table
        distances = calc_smallest_distance_between_points_and_surface(hand, detector_plane).astype(int)
        pinky_close = list(distances[17:] < self.parameters["hand_calibration"][self.handedness][17:])

        # Display unmet constraints
        i = 2
        for j, b in enumerate(pinky_close):
            if not b:
                self.table_widget.setItem(i, 0, QTableWidgetItem(HAND_KEYPOINTS[17 + j]))
                self.table_widget.setItem(i, 1, QTableWidgetItem(f"{distances[17 + j]} mm"))
                i += 1

        # 3. Check that the fingers are fanned
        hand_2d_proj = calc_projection_3d_points_to_2d_plane(hand, detector_plane)

        # get the links for individual fingers except index where we only want the metacarpal
        fingers = [FINGER_LINKS[0], FINGER_LINKS[1][:2]] + FINGER_LINKS[2:]

        finger_eqs = [calc_best_fitting_line(hand_2d_proj[finger]) for finger in fingers]

        # all fingers except the thumb
        finger_angles = [calc_angle_between_lines(finger_eqs[i], finger_eqs[i + 1]) for i in range(1, len(finger_eqs) - 1)]

        # add the thumb and middle finger
        finger_angles = [calc_angle_between_lines(finger_eqs[0], finger_eqs[3])] + finger_angles

        are_fingers = all([10 < finger_angles[i] < 30 for i in range(len(finger_angles))])

        # 4. Check that the thumb tip is pressed against the index tip
        thb_idx_dist = calc_smallest_distance_between_two_points(hand[4], hand[8])
        is_thb_idx = (thb_idx_dist < 20)

        correct_kpts = np.array(21 * [False])
        correct_kpts[[0, 1, 2, 5, 9, 13, 17]] = is_angle
        correct_kpts[[3, 6, 7, 10, 11, 12, 14, 15, 16, 18, 19, 20]] = are_fingers
        correct_kpts[[4, 8]] = is_thb_idx
        correct_kpts[18:] = all(pinky_close)

        return (is_angle and pinky_close and are_fingers and is_thb_idx), correct_kpts

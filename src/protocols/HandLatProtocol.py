from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem

from src.protocols.BaseProtocol import BaseProtocol
from src.utils.Calculators import get_hand_plane
from src.utils.Constants import HAND_KEYPOINTS
from src.utils.Geometry import calc_angle_between_planes, calc_smallest_distance_between_points_and_surface


class HandLatProtocol(BaseProtocol):
    def __init__(self, handedness: str, table_widget: QtWidgets.QTableWidget):
        super().__init__(f"Hand_Lateral_{handedness}", handedness, table_widget)

    def check_constraints(self):
        self.table_widget.clearContents()

        # 1. Camera tilt
        camera_tilt = int(self.parameters["camera_tilt"])
        is_tilt = (-65 < camera_tilt < -25)
        if not is_tilt:
            self.table_widget.setItem(0, 0, QTableWidgetItem("Camera tilt"))
            self.table_widget.setItem(0, 1, QTableWidgetItem(f"{camera_tilt} deg"))

        hand = self.dict_to_ndarray(self.parameters["hand"])
        detector_plane = self.parameters["detector_plane"]

        # 2. Check that the angle between the hand and the detector plane is less than 15 degrees
        hand_plane = get_hand_plane(hand[1:])
        angle = int(calc_angle_between_planes(detector_plane, hand_plane))
        is_angle = (80 < angle < 100)

        # Display unmet constraints
        if not is_angle and True:
            self.table_widget.setItem(1, 0, QTableWidgetItem("Hand Angle"))
            self.table_widget.setItem(1, 1, QTableWidgetItem(f"{angle} deg"))

        # 3. Check that the pinky keypoints are less than 50mm from the table
        distances = calc_smallest_distance_between_points_and_surface(hand, detector_plane).astype(int)
        pinky_close = list(distances[17:] < 50)

        # Display unmet constraints
        i = 2
        for j, b in enumerate(pinky_close):
            if not b:
                self.table_widget.setItem(i, 0, QTableWidgetItem(HAND_KEYPOINTS[17 + j]))
                self.table_widget.setItem(i, 1, QTableWidgetItem(f"{distances[17 + j]} mm"))
                i += 1

        return is_tilt and is_angle and all(pinky_close)

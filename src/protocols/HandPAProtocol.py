from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem

from src.protocols.BaseProtocol import BaseProtocol
from src.utils.Calculators import get_hand_plane
from src.utils.Constants import HAND_KEYPOINTS
from src.utils.Geometry import calc_angle_between_planes, calc_smallest_distance_between_points_and_surface


class HandPAProtocol(BaseProtocol):
    def __init__(self, handedness: str, table_widget: QtWidgets.QTableWidget):
        super().__init__(f"Hand_PosteriorAnterior_{handedness}", handedness, table_widget)

    def check_constraints(self):
        self.table_widget.clearContents()

        hand = self.dict_to_ndarray(self.parameters["hand"])
        self.hist.append(hand)

        detector_plane = self.parameters["detector_plane"]

        # 1. Check that the angle between the hand and the detector plane is less than 15 degrees
        hand_plane = get_hand_plane(hand[1:])
        angle = int(calc_angle_between_planes(detector_plane, hand_plane))
        is_angle = (angle > 165) or (angle < 15)

        # Display unmet constraints
        if not is_angle:
            self.table_widget.setItem(0, 0, QTableWidgetItem("Hand Angle"))
            self.table_widget.setItem(0, 1, QTableWidgetItem(f"{angle} deg"))

        # 2. Check that all finger keypoints are less than 50mm from the table and wrist is less than 75mm
        distances = calc_smallest_distance_between_points_and_surface(hand, detector_plane).astype(int)
        wrist_close = list(distances[:2] < 75)
        rest_close = list(distances[2:] < 50)
        all_close = wrist_close + rest_close

        # Display unmet constraints
        i = 1
        for j, b in enumerate(all_close):
            if not b:
                self.table_widget.setItem(i, 0, QTableWidgetItem(HAND_KEYPOINTS[j]))
                self.table_widget.setItem(i, 1, QTableWidgetItem(f"{distances[j]} mm"))
                i += 1

        return is_angle and all(all_close)

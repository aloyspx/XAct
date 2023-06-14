from typing import List

import numpy as np
from PyQt5 import QtWidgets

from src.utils.Constants import HAND_KEYPOINTS


class BaseProtocol:
    def __init__(self, protocol_name: str, handedness: str, table_widget: QtWidgets.QTableWidget):
        self.protocol_name = protocol_name
        self.handedness = handedness
        self.table_widget = table_widget
        self.parameters = self.create_default_parameters()
        print(f"Created constraint protocol: {self.protocol_name}")

    def create_default_parameters(self):

        hand = {key: None for key in HAND_KEYPOINTS}
        detector_plane = [0., 0., 0., 0.]

        return {
            "handedness": self.handedness,
            "hand": hand,
            "detector_plane": detector_plane
        }

    def set_detector_parameter(self, detector_plane: List[int]):
        assert len(detector_plane) == 4
        self.parameters["detector_plane"] = detector_plane

    def set_hand_parameter(self, hand_hist: np.ndarray):

        hist = hand_hist[self.handedness]

        if len(hist) == 0:
            print(f"No history present. This means the {self.handedness} hand was never detected.")
            return []

        # filter out points with depth not detected and calculate mean with remaining coordinate history
        masked_arr = np.ma.masked_equal(hist, 0)
        coords_3d = np.median(masked_arr, axis=0)

        for coord, coord_key in zip(coords_3d, self.parameters["hand"].keys()):
            self.parameters["hand"][coord_key] = coord

    def key_exists(self, keys: List[str]):
        d = self.parameters
        for key in keys:
            if key in d:
                d = d[key]
            else:
                return False
        return True

    # This is the function that must be implemented in all child classes
    def check_constraints(self):
        pass

    @staticmethod
    def dict_to_ndarray(d: dict) -> np.ndarray:
        return np.array(list(d.values()))

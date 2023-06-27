from typing import List, Dict, Any

import numpy as np
from PyQt5 import QtWidgets

from src.utils.Constants import HAND_KEYPOINTS


class BaseProtocol:
    def __init__(self, protocol_name: str, handedness: str, table_widget: QtWidgets.QTableWidget) -> None:
        self.protocol_name = protocol_name
        self.handedness = handedness
        self.table_widget = table_widget
        self.hist = []
        self.parameters = self.create_default_parameters()
        print(f"Created constraint protocol: {self.protocol_name}")

    @staticmethod
    def create_default_parameters() -> Dict[str, Any]:

        return {
            "hand": {key: None for key in HAND_KEYPOINTS},
            "detector_plane": [0., 0., 0., 0.],
            "hand_calibration": {"left": [75, 75] + [50 for _ in range(19)],
                                 "right": [75, 75] + [50 for _ in range(19)]},
            "camera_tilt": 0.
        }

    def set_detector_parameter(self, detector_plane: List[int]) -> bool:
        if len(detector_plane) != 4 and detector_plane == [0, 0, 0, 0]:
            return False
        else:
            self.parameters["detector_plane"] = detector_plane
            return True

    def set_hand_parameter(self, hand_hist: np.ndarray) -> bool:

        hist = hand_hist[self.handedness]

        if len(hist) == 0:
            print(f"No history present. This means the {self.handedness} hand was never detected.")
            self.parameters["hand"] = {key: None for key in HAND_KEYPOINTS}
            return False
        else:
            # filter out points with depth not detected and calculate mean with remaining coordinate history
            masked_arr = np.ma.masked_equal(hist, 0)
            coords_3d = np.median(masked_arr, axis=0)

            for coord, coord_key in zip(coords_3d, self.parameters["hand"].keys()):
                self.parameters["hand"][coord_key] = coord

            return True

    def set_hand_calibration_parameters(self, hand_calibrations):
        self.parameters["hand_calibration"] = hand_calibrations

    def set_camera_tilt(self, tilt: float) -> None:
        self.parameters["camera_tilt"] = tilt

    # This is the function that must be implemented in all child classes
    def check_constraints(self) -> bool:
        pass

    @staticmethod
    def dict_to_ndarray(d: dict) -> np.ndarray:
        return np.array(list(d.values()))

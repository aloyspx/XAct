from pprint import pprint
from typing import List

import numpy as np


class BaseProtocol:
    def __init__(self, protocol_name):
        self.protocol_name = protocol_name
        self.parameters = self.create_default_parameters()

    def get_parameters(self):
        return self.parameters

    @staticmethod
    def create_default_parameters():
        hand_keypoints = ['Wrist', 'Thumb_MCP', 'Thumb_IP', 'Thumb_Tip', 'Index_MCP', 'Index_PIP', 'Index_DIP',
                          'Index_Tip', 'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Middle_Tip', 'Ring_MCP', 'Ring_PIP',
                          'Ring_DIP', 'Ring_Tip', 'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP', 'Pinky_Tip']

        left_hand = {key: None for key in hand_keypoints}
        right_hand = {key: None for key in hand_keypoints}

        return {
            "left_hand": left_hand,
            "right_hand": right_hand
        }

    def key_exists(self, keys):
        d = self.parameters
        for key in keys:
            if key in d:
                d = d[key]
            else:
                return False
        return True

    def set_parameter(self, hand_hist: np.ndarray, key: str):

        hist = hand_hist[key]

        if len(hist) == 0:
            print(f"No history present. This means the {key} hand was never detected.")
            return []

        # filter out points with depth not detected and calculate mean with remaining coordinate history
        masked_arr = np.ma.masked_equal(hist, 0)
        coords_3d = np.mean(masked_arr, axis=0)

        assert len(coords_3d) == len(self.parameters.keys())

        for coord, coord_key in zip(coords_3d, self.parameters.keys()):
            self.parameters[key][coord_key] = coord


if __name__ == "__main__":
    protocol = BaseProtocol("Test")
    protocol.set_parameter()



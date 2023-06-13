from src.protocols.BaseProtocol import BaseProtocol


class HandLatProtocol(BaseProtocol):
    def __init__(self, handedness):
        super().__init__(f"Hand_Lateral_{handedness}")

        self.handedness = handedness

    def update_hand_parameters(self, hand_hist):
        self.set_hand_parameter(hand_hist, self.handedness)

    def update_detector_parameters(self, detector_plane):
        self.set_detector_parameter(detector_plane)

    def check_constraints(self):
        pass


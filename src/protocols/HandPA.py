from src.Calculators import get_hand_plane
from src.protocols.BaseProtocol import BaseProtocol


class HandPA(BaseProtocol):
    def __init__(self, handedness):
        self.handedness = handedness
        super().__init__(f"Hand_PosteriorAnterior_{handedness}")

    def update_parameters(self, hand_hist):
        self.set_parameter(hand_hist, self.handedness)

    def check_constraints(self):
        plane = get_hand_plane(self.parameters[self.handedness])

        return angle < 15.


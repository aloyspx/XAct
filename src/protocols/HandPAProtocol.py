from src.utils.Calculators import get_hand_plane
from src.utils.Geometry import calc_angle_between_planes, calc_smallest_distance_between_points_and_surface
from src.protocols.BaseProtocol import BaseProtocol


class HandPAProtocol(BaseProtocol):
    def __init__(self, handedness):
        super().__init__(f"Hand_PosteriorAnterior_{handedness}")

        self.handedness = handedness

    def update_hand_parameters(self, hand_hist):
        self.set_hand_parameter(hand_hist, self.handedness)

    def update_detector_parameters(self, detector_plane):
        self.set_detector_parameter(detector_plane)

    def check_constraints(self):
        # Two constraints with similar intent

        # 1. Check that the angle between the hand and the detector plane is less than 10 degrees
        hand_plane = get_hand_plane(self.parameters[self.handedness])
        is_angle = (calc_angle_between_planes(self.parameters["detector_plane"], hand_plane) < 10)

        # 2. Check that all finger keypoints are less than 40mm from the table and wrist is less than 60mm
        distances = calc_smallest_distance_between_points_and_surface(self.parameters[self.handedness],
                                                                      self.parameters["detector_plane"])
        is_close = all(distances[:2] < 60) and all(distances[2:] < 40)

        return is_angle and is_close


import time
from PyQt5.QtCore import QThread

from src.utils.Constants import PROTOCOLS
from src.HandTracker import HandTracker
from src.protocols.HandLatProtocol import HandLatProtocol
from src.protocols.HandObqProtocol import HandObqProtocol
from src.protocols.HandPAProtocol import HandPAProtocol


class ConstraintThread(QThread):

    def __init__(self, tracker: HandTracker) -> None:
        super().__init__()
        self._run_flag = True
        self.tracker = tracker
        self.constraint = None

    def set_protocol(self, constraint_name):
        idx = PROTOCOLS.index(constraint_name)

        if idx == 0:
            self.constraint = None
        elif idx == 1:
            self.constraint = HandPAProtocol(handedness="left")
        elif idx == 2:
            self.constraint = HandPAProtocol(handedness="right")
        elif idx == 3:
            self.constraint = HandObqProtocol(handedness="left")
        elif idx == 4:
            self.constraint = HandObqProtocol(handedness="right")
        elif idx == 5:
            self.constraint = HandLatProtocol(handedness="left")
        elif idx == 6:
            self.constraint = HandLatProtocol(handedness="right")
        else:
            raise NotImplementedError

    def run(self) -> None:
        while self._run_flag:
            time.sleep(2.5)

            if self.constraint:
                self.constraint.update_hand_parameters(self.tracker.hand_hist)
                self.constraint.update_detector_parameters(self.tracker.detector_plane)
                self.constraint.check_constraints()

    def stop(self) -> None:
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

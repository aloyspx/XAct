import asyncio
import time

from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal

from src.HandTracker import HandTracker
from src.protocols.HandLatProtocol import HandLatProtocol
from src.protocols.HandObqProtocol import HandObqProtocol
from src.protocols.HandPAProtocol import HandPAProtocol
from src.utils.SaveUtils import save


class ConstraintThread(QThread):
    change_light_color_signal = pyqtSignal(str)

    def __init__(self, tracker: HandTracker, table_widget: QtWidgets.QTableWidget) -> None:
        super().__init__()
        self._run_flag = True
        self.tracker = tracker
        self.table_widget = table_widget
        self.hand_calibrations = None
        self.constraint = None

    def set_protocol(self, constraint_idx):

        if self.constraint:
            save(self.tracker.detector_plane, self.constraint.hist, self.constraint.protocol_name)

        self.tracker.reset_hand_hist()

        if constraint_idx == 0:
            self.constraint = None
        elif constraint_idx == 1:
            self.constraint = HandPAProtocol("left", self.table_widget)
        elif constraint_idx == 2:
            self.constraint = HandPAProtocol("right", self.table_widget)
        elif constraint_idx == 3:
            self.constraint = HandObqProtocol("left", self.table_widget)
        elif constraint_idx == 4:
            self.constraint = HandObqProtocol("right", self.table_widget)
        elif constraint_idx == 5:
            self.constraint = HandLatProtocol("left", self.table_widget)
        elif constraint_idx == 6:
            self.constraint = HandLatProtocol("right", self.table_widget)
        else:
            raise NotImplementedError

        if constraint_idx != 0 and self.hand_calibrations:
            self.constraint.set_hand_calibration_parameters(self.hand_calibrations)

    def set_hand_calibration(self, hand_calibrations):

        if self.constraint is None:
            self.hand_calibrations = hand_calibrations
        else:
            self.hand_calibrations = hand_calibrations
            self.constraint.set_hand_calibration_parameters(hand_calibrations)

    def run(self) -> None:
        while self._run_flag:
            time.sleep(0.5)

            if self.constraint is None:
                self.change_light_color_signal.emit("blue")
                continue

            self.constraint.set_camera_tilt(self.tracker.get_camera_tilt())
            is_hand_detected = self.constraint.set_hand_parameter(self.tracker.hand_hist)
            is_table_detected = self.constraint.set_detector_parameter(self.tracker.detector_plane)

            if is_hand_detected and is_table_detected:
                is_correct, correct_kpts = self.constraint.check_constraints()
                self.tracker.correct_kpts = correct_kpts
                if is_correct:
                    self.change_light_color_signal.emit("green")
                else:
                    self.change_light_color_signal.emit("red")

    def stop(self) -> None:
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

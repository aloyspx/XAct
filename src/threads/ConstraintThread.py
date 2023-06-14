import time

from PyQt5 import QtWidgets
from PyQt5.QtCore import QThread, pyqtSignal

from src.HandTracker import HandTracker
from src.protocols.HandLatProtocol import HandLatProtocol
from src.protocols.HandObqProtocol import HandObqProtocol
from src.protocols.HandPAProtocol import HandPAProtocol


class ConstraintThread(QThread):
    change_light_color_signal = pyqtSignal(str)

    def __init__(self, tracker: HandTracker, table_widget: QtWidgets.QTableWidget) -> None:
        super().__init__()
        self._run_flag = True
        self.tracker = tracker
        self.table_widget = table_widget
        self.constraint = None

    def set_protocol(self, constraint_idx):

        self.tracker.reset_hand_hist()

        if constraint_idx == 0:
            self.constraint = None
        elif constraint_idx == 1:
            self.constraint = HandPAProtocol("left", self.table_widget)
        elif constraint_idx == 2:
            self.constraint = HandPAProtocol("right", self.table_widget)
        elif constraint_idx == 3:
            self.constraint = HandObqProtocol("left")
        elif constraint_idx == 4:
            self.constraint = HandObqProtocol("right")
        elif constraint_idx == 5:
            self.constraint = HandLatProtocol("left")
        elif constraint_idx == 6:
            self.constraint = HandLatProtocol("right")
        else:
            raise NotImplementedError

    def run(self) -> None:
        while self._run_flag:
            time.sleep(1)

            if not self.constraint:
                self.change_light_color_signal.emit("blue")
                continue

            self.constraint.set_hand_parameter(self.tracker.hand_hist)
            self.constraint.set_detector_parameter(self.tracker.detector_plane)

            if self.constraint.check_constraints():
                self.change_light_color_signal.emit("green")
            else:
                self.change_light_color_signal.emit("red")

    def stop(self) -> None:
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

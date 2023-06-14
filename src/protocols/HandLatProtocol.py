from PyQt5 import QtWidgets

from src.protocols.BaseProtocol import BaseProtocol


class HandLatProtocol(BaseProtocol):
    def __init__(self, handedness: str, table_widget: QtWidgets.QTableWidget):
        super().__init__(f"Hand_Lateral_{handedness}", handedness, table_widget)

    def check_constraints(self):
        pass

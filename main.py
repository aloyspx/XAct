import sys

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow

from src.HandTracker import HandTracker
from src.threads.ConstraintThread import ConstraintThread
from src.threads.VideoThread import VideoThread
from src.utils.Constants import PROTOCOLS


class App(QMainWindow):

    def __init__(self):
        # Main Window
        super().__init__()

        self.setObjectName("MainWindow")
        self.setFixedSize(1030, 525)
        self.central_widget = QtWidgets.QWidget(self)
        self.central_widget.setObjectName("central_widget")
        self.setCentralWidget(self.central_widget)

        # Video Viewer
        self.v_w, self.v_h = 800, 450
        self.viewer = QtWidgets.QLabel(self.central_widget)
        self.viewer.setGeometry(QtCore.QRect(10, 10, self.v_w, self.v_h))
        self.viewer.setAutoFillBackground(False)
        self.viewer.setFrameShape(QtWidgets.QFrame.Box)
        self.viewer.setObjectName("viewer")

        # Calibration Button
        self.calibrateButton = QtWidgets.QPushButton(self.central_widget)
        self.calibrateButton.setGeometry(QtCore.QRect(350, 470, 121, 31))
        self.calibrateButton.setObjectName("calibrateButton")
        self.calibrateButton.clicked.connect(self.on_calibrateButton_clicked)

        # LED Light Viewer
        self.led = QtWidgets.QLabel(self.central_widget)
        self.led.setGeometry(QtCore.QRect(820, 10, 25, 25))
        self.led.setObjectName("led_indicator")
        rgb_image = cv2.imread("artifacts/blue_led.png", cv2.IMREAD_UNCHANGED)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGBA)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGBA8888)
        p = convert_to_qt_format.scaled(25, 25, Qt.KeepAspectRatio)
        self.led.setPixmap(QtGui.QPixmap.fromImage(p))

        # Dropdown Protocol Picker
        self.protocolDropdown = QtWidgets.QComboBox(self.central_widget)
        self.protocolDropdown.setGeometry(QtCore.QRect(820, 40, 195, 45))
        self.protocolDropdown.setObjectName("protocolDropdown")
        self.protocolDropdown.addItems(PROTOCOLS)
        self.protocolDropdown.currentIndexChanged.connect(self.on_protocolDropdown_changed)

        # Table Widget Parameters
        self.tableWidget = QtWidgets.QTableWidget(self.central_widget)
        self.tableWidget.setObjectName("parameterTable")

        self.tableWidget.setRowCount(8)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setHorizontalHeaderLabels(["Constraint", "Value"])

        row_height = self.tableWidget.rowHeight(0)
        column_width = self.tableWidget.columnWidth(0)
        self.tableWidget.setGeometry(QtCore.QRect(820, 120, 2 * column_width, int(8.75 * row_height)))
        self.tableWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tableWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.tableWidget.verticalHeader().hide()

        # Menu bar
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1025, 22))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)

        # Status bar
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        # Exit
        self.actionExit = QtWidgets.QAction(self)
        self.actionExit.setObjectName("actionExit")

        # Final UI Setup
        self.retranslate_ui(self)
        QtCore.QMetaObject.connectSlotsByName(self)

        # Calculates information about positioning in the background
        self.tracker = HandTracker()

        # create the video capture thread
        self.video_thread = VideoThread(self.tracker)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

        # create thread to evaluate constraints in the background
        self.constraints_thread = ConstraintThread(self.tracker, self.tableWidget)
        self.constraints_thread.change_light_color_signal.connect(self.update_light)
        self.constraints_thread.start()

    def retranslate_ui(self, main_window) -> None:
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.viewer.setText(_translate("MainWindow", "Viewer"))
        self.calibrateButton.setText(_translate("MainWindow", "Calibrate"))

    def update_image(self, cv_img: np.ndarray) -> None:
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.viewer.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img: np.ndarray) -> QPixmap:
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(self.v_w, self.v_h, Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)

    def update_light(self, color: str) -> None:
        if color == "blue":
            fpath = "artifacts/blue_led.png"
        elif color == "red":
            fpath = "artifacts/red_led.png"
        elif color == "green":
            fpath = "artifacts/green_led.png"
        else:
            raise NotImplementedError

        rgb_image = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGBA)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGBA8888)
        p = convert_to_qt_format.scaled(25, 25, Qt.KeepAspectRatio)
        self.led.setPixmap(QtGui.QPixmap.fromImage(p))

    def on_calibrateButton_clicked(self) -> None:
        self.tracker.get_detector_plane()

    def on_protocolDropdown_changed(self, idx: int) -> None:
        self.constraints_thread.set_protocol(idx)

    def closeEvent(self, event) -> None:
        self.video_thread.stop()
        self.constraints_thread.stop()
        sys.exit(0)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = App()
    ui.show()
    sys.exit(app.exec_())

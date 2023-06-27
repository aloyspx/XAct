import sys

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from src.HandTracker import HandTracker
from src.threads.ConstraintThread import ConstraintThread
from src.threads.VideoThread import VideoThread
from src.utils.Calculators import calc_smallest_distance_between_points_and_surface
from src.utils.Constants import PROTOCOLS


class App(QMainWindow):

    def __init__(self):
        # Main Window
        super().__init__()

        self.pressed = False

        self.setObjectName("XAct")
        self.setFixedSize(1600, 800)
        self.central_widget = QtWidgets.QWidget(self)
        self.central_widget.setObjectName("central_widget")
        self.setCentralWidget(self.central_widget)

        # Video Viewer
        self.v_w, self.v_h = 1200, 675
        self.viewer = QtWidgets.QLabel(self.central_widget)
        self.viewer.setGeometry(QtCore.QRect(25, 25, self.v_w, self.v_h))
        self.viewer.setAutoFillBackground(False)
        self.viewer.setFrameShape(QtWidgets.QFrame.Box)
        self.viewer.setObjectName("viewer")

        # Calibration Button
        self.calibrateTableButton = QtWidgets.QPushButton(self.central_widget)
        self.calibrateTableButton.setGeometry(QtCore.QRect(385, 720, 240, 50))
        self.calibrateTableButton.setObjectName("calibrateTableButton")
        self.calibrateTableButton.clicked.connect(self.on_calibrateTableButton_clicked)

        self.calibrateHandButton = QtWidgets.QPushButton(self.central_widget)
        self.calibrateHandButton.setGeometry(QtCore.QRect(645, 720, 240, 50))
        self.calibrateHandButton.setObjectName("calibrateHandButton")
        self.calibrateHandButton.clicked.connect(self.on_calibrateHandButton_clicked)

        # LED Light Viewer
        self.led = QtWidgets.QLabel(self.central_widget)
        self.led.setGeometry(QtCore.QRect(1250, 25, 50, 50))
        self.led.setObjectName("led_indicator")
        rgb_image = cv2.imread("artifacts/blue_led.png", cv2.IMREAD_UNCHANGED)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGBA)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGBA8888)
        p = convert_to_qt_format.scaled(50, 50, Qt.KeepAspectRatio)
        self.led.setPixmap(QtGui.QPixmap.fromImage(p))

        # Dropdown Protocol Picker
        self.protocolDropdown = QtWidgets.QComboBox(self.central_widget)
        self.protocolDropdown.setGeometry(QtCore.QRect(1250, 90, 325, 45))
        self.protocolDropdown.setObjectName("protocolDropdown")
        self.protocolDropdown.addItems(PROTOCOLS)
        self.protocolDropdown.currentIndexChanged.connect(self.on_protocolDropdown_changed)

        # Table Widget Parameters
        self.tableWidget = QtWidgets.QTableWidget(self.central_widget)
        self.tableWidget.setObjectName("parameterTable")

        self.tableWidget.setRowCount(16)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setColumnWidth(0, 150)
        self.tableWidget.setColumnWidth(1, 175)
        self.tableWidget.setHorizontalHeaderLabels(["Constraint", "Value"])

        row_height = self.tableWidget.rowHeight(0)
        self.tableWidget.setGeometry(QtCore.QRect(1250, 150, 325, int(16.75 * row_height)))
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
        self.calibrateTableButton.setText(_translate("MainWindow", "Calibrate Table"))
        self.calibrateHandButton.setText(_translate("MainWindow", "Calibrate Hand"))

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
        p = convert_to_qt_format.scaled(50, 50, Qt.KeepAspectRatio)
        self.led.setPixmap(QtGui.QPixmap.fromImage(p))

    def on_calibrateTableButton_clicked(self) -> None:
        self.tracker.get_detector_plane()

    def on_calibrateHandButton_clicked(self) -> None:
        if not self.pressed:
            QMessageBox.about(self, "Warning", "Please lay hand (or hands) flat, palm down and in the center "
                                               "of the frame. Then press the button again.")
            self.pressed = True
        else:
            self.pressed = False

            hand_calibrations = {}
            for key in self.tracker.hand_hist.keys():
                if self.tracker.hand_hist[key]:
                    masked_arr = np.ma.masked_equal(self.tracker.hand_hist[key], 0)
                    coords_3d = np.median(masked_arr, axis=0)
                    dists = calc_smallest_distance_between_points_and_surface(coords_3d, self.tracker.detector_plane)
                    hand_calibrations[key] = np.maximum(np.round_(dists, decimals=-1), [75, 75] + [50 for _ in range(19)])
                else:
                    self.tracker.hand_hist[key] = [75, 75] + [50 for _ in range(19)]

            self.constraints_thread.set_hand_calibration(hand_calibrations)

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

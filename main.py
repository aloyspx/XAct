import sys

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap

from src.utils.Constants import PROTOCOLS
from src.HandTracker import HandTracker
from src.threads.VideoThread import VideoThread


class App:

    def __init__(self, main_window):
        main_window.setObjectName("MainWindow")
        main_window.setFixedSize(1025, 525)
        self.central_widget = QtWidgets.QWidget(main_window)
        self.central_widget.setObjectName("central_widget")

        self.v_w, self.v_h = 800, 450
        self.viewer = QtWidgets.QLabel(self.central_widget)
        self.viewer.setGeometry(QtCore.QRect(10, 10, self.v_w, self.v_h))
        self.viewer.setAutoFillBackground(False)
        self.viewer.setFrameShape(QtWidgets.QFrame.Box)
        self.viewer.setObjectName("viewer")

        self.calibrateButton = QtWidgets.QPushButton(self.central_widget)
        self.calibrateButton.setGeometry(QtCore.QRect(350, 470, 121, 31))
        self.calibrateButton.setObjectName("calibrateButton")
        self.calibrateButton.clicked.connect(self.on_calibrateButton_clicked)

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

        self.protocolDropdown = QtWidgets.QComboBox(self.central_widget)
        self.protocolDropdown.setGeometry(QtCore.QRect(820, 40, 195, 41))
        self.protocolDropdown.setObjectName("protocolDropdown")
        self.protocolDropdown.addItems(PROTOCOLS)
        self.protocolDropdown.currentIndexChanged.connect(self.on_protocolDropdown_changed)
        main_window.setCentralWidget(self.central_widget)

        self.menubar = QtWidgets.QMenuBar(main_window)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1025, 22))
        self.menubar.setObjectName("menubar")
        main_window.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(main_window)
        self.statusbar.setObjectName("statusbar")
        main_window.setStatusBar(self.statusbar)

        self.actionExit = QtWidgets.QAction(main_window)
        self.actionExit.setObjectName("actionExit")

        self.retranslateUi(main_window)
        QtCore.QMetaObject.connectSlotsByName(main_window)

        # Calculates information about positioning in the background
        self.tracker = HandTracker()

        # create the video capture thread
        self.video_thread = VideoThread(self.tracker)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.start()

    def retranslateUi(self, main_window) -> None:
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

    def on_calibrateButton_clicked(self):
        self.tracker.get_detector_plane()

    def on_protocolDropdown_changed(self, idx):
        print("Protocol changed to:", PROTOCOLS[idx])



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = App(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

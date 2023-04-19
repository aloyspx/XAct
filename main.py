import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

from utils.VideoThread import VideoThread


class App:

    def __init__(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(1025, 525)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.v_w, self.v_h = 800, 450
        self.viewer = QtWidgets.QLabel(self.centralwidget)
        self.viewer.setGeometry(QtCore.QRect(10, 10, self.v_w, self.v_h))
        self.viewer.setAutoFillBackground(False)
        self.viewer.setFrameShape(QtWidgets.QFrame.Box)
        self.viewer.setObjectName("viewer")

        self.calibrateButton = QtWidgets.QPushButton(self.centralwidget)
        self.calibrateButton.setGeometry(QtCore.QRect(210, 470, 121, 31))
        self.calibrateButton.setObjectName("calibrateButton")

        self.viewButton = QtWidgets.QPushButton(self.centralwidget)
        self.viewButton.setGeometry(QtCore.QRect(350, 470, 121, 31))
        self.viewButton.setObjectName("viewButton")

        self.historyButton = QtWidgets.QPushButton(self.centralwidget)
        self.historyButton.setGeometry(QtCore.QRect(490, 470, 121, 31))
        self.historyButton.setObjectName("historyButton")

        self.led = QtWidgets.QLabel(self.centralwidget)
        self.led.setGeometry(QtCore.QRect(820, 10, 25, 25))
        self.led.setObjectName("led_indicator")
        rgb_image = cv2.imread("artifacts/blue_led.png", cv2.IMREAD_UNCHANGED)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGRA2RGBA)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGBA8888)
        p = convert_to_Qt_format.scaled(25, 25, Qt.KeepAspectRatio)
        self.led.setPixmap(QtGui.QPixmap.fromImage(p))

        self.protocolDropdown = QtWidgets.QComboBox(self.centralwidget)
        self.protocolDropdown.setGeometry(QtCore.QRect(820, 40, 195, 41))
        self.protocolDropdown.setObjectName("protocolDropdown")
        self.protocolDropdown.addItems(["No Protocol", "Protocol1", "Protocol2", "Protocol3"])
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1025, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.viewer.setText(_translate("MainWindow", "Viewer"))
        self.calibrateButton.setText(_translate("MainWindow", "Calibrate"))
        self.viewButton.setText(_translate("MainWindow", "Change View"))
        self.historyButton.setText(_translate("MainWindow", "View History"))

    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.viewer.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.v_w, self.v_h, Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = App(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

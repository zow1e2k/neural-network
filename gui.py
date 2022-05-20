import os
import sys

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from network import Network


class Ui_MainWindow(object):
    network = None

    def create_ui(self, MainWindow):
        self.network = Network()
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 768)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("\n"
                                 "background-color: rgba(0, 0, 0, 0);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.learnNetwork = QtWidgets.QPushButton(self.centralwidget)
        self.learnNetwork.setGeometry(QtCore.QRect(10, 90, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.learnNetwork.setFont(font)
        self.learnNetwork.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.learnNetwork.setStyleSheet("border-color: rgb(255, 0, 0);\n"
                                        "background-color: rgba(155, 116, 60, 150);\n"
                                        "border-radius: 25px;\n"
                                        "color: rgb(255, 255, 255);")
        self.learnNetwork.setObjectName("learnNetwork")

        self.saveNetwork = QtWidgets.QPushButton(self.centralwidget)
        self.saveNetwork.setGeometry(QtCore.QRect(10, 10, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.saveNetwork.setFont(font)
        self.saveNetwork.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.saveNetwork.setStyleSheet("border-color: rgb(255, 0, 0);\n"
                                       "background-color: rgba(155, 116, 60, 150);\n"
                                       "border-radius: 25px;\n"
                                       "color: rgb(255, 255, 255);\n"
                                       "")
        self.saveNetwork.setObjectName("saveNetwork")

        self.loadNetwork = QtWidgets.QPushButton(self.centralwidget)
        self.loadNetwork.setGeometry(QtCore.QRect(10, 200, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.loadNetwork.setFont(font)
        self.loadNetwork.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.loadNetwork.setStyleSheet("border-color: rgb(255, 0, 0);\n"
                                       "background-color: rgba(155, 116, 60, 150);\n"
                                       "border-radius: 25px;\n"
                                       "color: rgb(255, 255, 255);\n"
                                       "")
        self.loadNetwork.setObjectName("loadNetwork")

        self.loadImage = QtWidgets.QPushButton(self.centralwidget)
        self.loadImage.setGeometry(QtCore.QRect(300, 390, 451, 32))
        self.loadImage.setStyleSheet("border-color: rgb(255, 0, 0);\n"
                                     "background-color: rgba(155, 116, 60, 150);\n"
                                     "border-radius: 25px;\n"
                                     "color: rgb(255, 255, 255);")
        self.loadImage.setObjectName("loadImage")
        self.graphicsView = QtWidgets.QLabel(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(190, 10, 641, 351))
        self.graphicsView.setStyleSheet("border: 2px solid grey;\n"
                                        "border-style: solid;\n"
                                        "border-color: rgb(255, 255, 255);\n"
                                        "background-color: rgb(255, 255, 255);\n"
                                        "")
        self.graphicsView.setObjectName("graphicsView")
        self.learnNetwork.raise_()
        self.loadNetwork.raise_()

        self.saveNetwork.raise_()
        self.graphicsView.raise_()
        self.loadImage.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 853, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslate_ui(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        self.add_func()

    def add_func(self):
        self.learnNetwork.clicked.connect(self.network.learn_model)
        self.loadNetwork.clicked.connect(self.network.load_model)
        self.saveNetwork.clicked.connect(self.network.save_model)
        self.loadImage.clicked.connect(self.get_file_name)

    def retranslate_ui(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle("Neural Network")
        self.learnNetwork.setText("Learn model")
        self.loadNetwork.setText("Load model")
        self.saveNetwork.setText("Save model")
        self.loadImage.setText("Upload image")

    def get_file_name(self):
        file_filter = 'Images (*.png *.jpg)'
        response = QFileDialog.getOpenFileName(
            caption='Select a data file',
            directory=os.getcwd(),
            filter=file_filter,
            initialFilter='Images (*.png *.jpg *.jpeg)'
        )
        print(response[0])
        self.show_image(response[0])
        response = self.network.analyze_image(response[0])

        error = QMessageBox()
        error.setWindowTitle("Result")
        error.setText(response)
        error.exec_()

    def show_image(self, path):
        self.graphicsView.setPixmap(QtGui.QPixmap(path))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.create_ui(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

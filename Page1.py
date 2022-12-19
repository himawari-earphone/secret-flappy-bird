import os
import time

import numpy as np
import cv2
import threading

from PyQt5.QtCore import pyqtSignal, QThread

from class_face.dete_face import face_detect
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog
from keras.models import Sequential, load_model

global label1, label2, label3, textend
textend = False
label1 = "Unknown"
label2 = "Unknown"
label3 = "Unknown"


class UpdateThread(QThread):
    # 实时显示追加线程（要继承QThread， 继承threading.Thread不行）
    signal = pyqtSignal(str)  # 信号

    def run(self):
        while 1:
            if textend:
                break
            self.signal.emit("<h1>&nbsp;姓名</h1> <p align='center'>&nbsp;</p> <h2 align='center'>{}</h2>"
                             "<p align='center'>&nbsp;""</p>"
                             "<h1>&nbsp;性别</h1><p align='center'>&nbsp;</p><h2 align='center'>{}</h2>"
                             "<p align='center'>&nbsp;</p>"
                             "<h1>&nbsp;情绪</h1><p align='center'>&nbsp;</p><h2 align='center'>{}</h2>"
                             .format(label1, label2, label3))  # 发射信号(实参类型要和定义信号的参数类型一致)
            time.sleep(0.01)


class Page1:
    def __init__(self, ui, mainWnd):
        self.ui = ui
        self.mainWnd = mainWnd
        self.Ori = True  # 第一次open不需要close
        self.model_gender = load_model('gender_classify_middle_hiar_man.h5')
        self.model_smile = load_model('smile_classify_middle_hiar_man.h5')
        self.label_dictG = {0: 'female', 1: 'male'}
        self.label_dictS = {0: 'no smile', 1: 'smile'}
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_recognizer.read('class_face/train.yml')
        self.update_thread = UpdateThread()
        self.update_thread.signal.connect(self.slot_text_browser)  # 连接槽函数
        self.update_thread.start()
        # 信号槽设置
        ui.cameraload.clicked.connect(self.Open_camera)
        ui.pictureload.clicked.connect(self.Open_picture)
        ui.videoload.clicked.connect(self.Open_video)
        ui.agreelogin.clicked.connect(self.nextpage)
        ui.agreelogin.setEnabled(False)

        # 创建一个关闭事件并设为未触发
        self.stopEvent = threading.Event()
        self.stopEvent.clear()

    def activate(self):
        self.ui.agreelogin.setEnabled(True)
        _translate = QtCore.QCoreApplication.translate
        self.ui.agreelogin.setText(_translate("MainWindow", "Passed"))

    def Close(self):
        # 关闭事件设为触发，关闭视频播放
        self.stopEvent.set()

    def Open_camera(self):
        if not self.Ori:
            self.Close()
        else:
            self.Ori = False
        self.cap = cv2.VideoCapture(0)
        # 创建视频显示线程
        th = threading.Thread(target=self.loadcamera)
        th.start()

    def loadcamera(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            # RGB转BGR
            if success:
                self.show(frame)
                cv2.waitKey(1)
            # 判断关闭事件是否已触发
            if self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.ui.LoginLabel.clear()
                break

    def Open_video(self):
        if not self.Ori:
            self.Close()
        else:
            self.Ori = False
        path = QFileDialog.getOpenFileNames(self.mainWnd, '选择视频', os.getcwd(), "All Files(*);;mp4 Files(*.mp4)")
        self.cap = cv2.VideoCapture(path[0][0])
        # 创建视频显示线程
        th = threading.Thread(target=self.loadvideo)
        th.start()

    def loadvideo(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            # RGB转BGR
            if success:
                self.show(frame)
                cv2.waitKey(1)
            # 判断关闭事件是否已触发
            if self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.ui.LoginLabel.clear()
                break

    def Open_picture(self):
        if not self.Ori:
            self.Close()
        else:
            self.Ori = False
        path = QFileDialog.getOpenFileNames(self.mainWnd, '选择图片', os.getcwd(),
                                            "All Files(*);;jpg Files(*.jpg);;png Files(*.png)")
        self.cap = path[0][0]
        print(self.cap)
        # 创建视频显示线程
        th = threading.Thread(target=self.loadpicture)
        th.start()

    def loadpicture(self):
        temp = np.fromfile(self.cap, dtype=np.uint8)  # 先用numpy把图片文件存入内存：raw_data，把图片数据看做是纯字节数据
        frame = cv2.imdecode(temp, 1)
        self.show(frame)
        while 1:
            # 判断关闭事件是否已触发
            cv2.waitKey(1)
            if self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.ui.LoginLabel.clear()
                break

    def cropAndPred(self, image, x, y, w, h):
        img = image[y:y + h, x:x + w:]
        cv2.imwrite('1.png', img)
        crpim = cv2.resize(img, (96, 96))

        im2Arr = np.asarray(crpim)
        x_test_nor = im2Arr.astype('float32') / 255.0
        x_test_nor = x_test_nor[np.newaxis, :]

        PredictionG = self.model_gender.predict(x_test_nor)
        PredictionG = np.int64(PredictionG > 0.5)
        PredictionS = self.model_smile.predict(x_test_nor)
        PredictionS = np.int64(PredictionS > 0.5)

        return PredictionG[0], PredictionS[0]

    def slot_text_browser(self, text):
        self.ui.textBrowser.clear()
        self.ui.textBrowser.append(text)

    def show(self, frame):
        global label1, label2, label3
        dirs = os.listdir('class_face/train_img_data')
        face, rect = face_detect(frame)
        if not face is None:
            label = self.face_recognizer.predict(face)
            if label[1] < 80:
                label1 = dirs[label[0]]
                self.activate()
            else:
                label1 = 'unknown'
            (x, y, w, h) = rect
            labelG, labelS = self.cropAndPred(frame, x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 100), 2)
            label2 = self.label_dictG[labelG[0]]
            label3 = self.label_dictS[labelS[0]]

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, [640, 480])
        img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.ui.LoginLabel.setPixmap(QPixmap.fromImage(img))

    def nextpage(self):
        self.Close()
        self.ui.stackedWidget.setCurrentIndex(2)

import os
import numpy as np
import cv2
import threading
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

# 此页弃用
class Page2:
    def __init__(self, ui, mainWnd):
        self.ui = ui
        self.mainWnd = mainWnd
        temp = np.fromfile('0aE5z2u.png', dtype=np.uint8)  # 先用numpy把图片文件存入内存：raw_data，把图片数据看做是纯字节数据
        frame = cv2.imdecode(temp, 1)
        frame = cv2.resize(frame, [640, 480])
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        self.ui.game1label.setPixmap(QPixmap.fromImage(img))

        # 信号槽设置
        ui.game1button.clicked.connect(self.nextpage)

    def nextpage(self):
        self.ui.stackedWidget.setCurrentIndex(2)

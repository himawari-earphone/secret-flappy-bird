import numpy as np
import cv2
import threading
from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap
from flappy_bird.flappy import Flappy
from flappy_bird.draw_utils import get_arm_angles, draw_pose, quit_game
import torch
from torchvision import transforms
from flappy_bird.utils.datasets import letterbox
from flappy_bird.utils.general import non_max_suppression_kpt
from flappy_bird.utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_one_box
from flappy_bird.smooth import KeypointSmoothing
import time


class Page3:
    def __init__(self, ui, mainWnd):
        self.ui = ui
        self.mainWnd = mainWnd
        self.Opening = False

        # 信号槽设置
        ui.exit1.clicked.connect(self.lastpage)
        ui.cap_control1.clicked.connect(self.Open_camera)

        # 创建一个关闭事件并设为未触发
        self.stopEvent = threading.Event()
        self.stopEvent.clear()

    def Close(self):
        # 关闭事件设为触发，关闭视频播放
        self.stopEvent.set()

    def Open_camera(self):
        if self.Opening:
            self.Close()
            _translate = QtCore.QCoreApplication.translate
            self.ui.cap_control1.setStyleSheet("border-image: url(:icon/button6.png);\n"
                                               "image: url(:/icon/play.png);\n"
                                               "")
            self.Opening = False
        else:
            _translate = QtCore.QCoreApplication.translate
            self.ui.cap_control1.setStyleSheet("border-image: url(:icon/button6.png);\n"
                                               "image: url(:/icon/pause.png);\n"
                                               "")
            self.Opening = True
            self.cap = cv2.VideoCapture(0)
            # 创建视频显示线程
            th = threading.Thread(target=self.loadcamera)
            th.start()

    def loadcamera(self):
        flappy = Flappy('icon')  # Flappy Bird
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
        model = weigths['model']
        _ = model.float().eval()
        if torch.cuda.is_available():
            model.half().to(device)
        frame_count = 0
        total_fps = 0

        # 分辨率
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))

        # 图片缩放
        vid_write_image = letterbox(self.cap.read()[1], (frame_width), stride=64, auto=True)[0]
        resize_height, resize_width = vid_write_image.shape[:2]

        # 保存结果视频
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("result_keypoint.mp4",
                              fourcc, 30,
                              (resize_width, resize_height))

        while self.cap.isOpened():
            success, frame = self.cap.read()
            # RGB转BGR
            if success:
                frame = cv2.flip(frame, 1)  # 镜像操作
                frame = cv2.GaussianBlur(frame, (0, 0), 5)

                orig_image = frame
                image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                image = letterbox(image, (frame_width), stride=64, auto=True)[0]
                image = transforms.ToTensor()(image)
                image = torch.tensor(np.array([image.numpy()]))
                image = image.to(device)
                image = image.half()

                start_time = time.time()
                with torch.no_grad():
                    output, _ = model(image)
                end_time = time.time()

                # 计算fps
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                # 非极大值抑制NMS
                output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'],
                                                 nkpt=model.yaml['nkpt'],
                                                 kpt_label=True)
                # 得到检测框、关键点数据
                output = output_to_keypoint(output)  # [batch_id, class_id, x, y, w, h, conf]
                nimg = image[0].permute(1, 2, 0) * 255
                nimg = nimg.cpu().numpy().astype(np.uint8)
                nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

                idx = 0
                for idx in range(output.shape[0]):
                    kpts = output[idx, 7:].T
                    plot_skeleton_kpts(nimg, kpts, 3)
                    # 检测框转换
                    x1 = output[idx, 2] - output[idx, 4] * 0.5
                    y1 = output[idx, 3] - output[idx, 5] * 0.5
                    x2 = output[idx, 2] + output[idx, 4] * 0.5
                    y2 = output[idx, 3] + output[idx, 5] * 0.5
                    bbox = [x1, y1, x2, y2]
                    break

                # 获取关键点数据
                pose_data = list()
                num_kpts = len(kpts) // 3  # 51//3=17!
                for kid in range(num_kpts):
                    x_coord, y_coord = kpts[3 * kid], kpts[3 * kid + 1]
                    pose_data.append([x_coord, y_coord])
                pose_data = np.array(pose_data)

                # 计算双臂与身体的夹角
                arm_angles = get_arm_angles(bbox, pose_data,
                                            KeypointSmoothing(frame_width, frame_height, "OneEuro"))

                # 根据双臂动作控制游戏画面更新，叠加到摄像头画面中做展示
                out_img, alive = flappy.update_flappy_pose(nimg, arm_angles, fps)

                out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
                out_img = out_img[24:512-24, :, :]
                out_img = cv2.resize(out_img, [900, 675])
                img = QImage(out_img.data, out_img.shape[1], out_img.shape[0], QImage.Format_RGB888)
                self.ui.DisplayLabel1.setPixmap(QPixmap.fromImage(img))
                if not alive:
                    time.sleep(3)
            # 判断关闭事件是否已触发
            if self.stopEvent.is_set():
                # 关闭事件置为未触发，清空显示label
                self.stopEvent.clear()
                self.ui.DisplayLabel1.clear()
                break

    def lastpage(self):
        if self.Opening:
            self.Close()
            _translate = QtCore.QCoreApplication.translate
            self.ui.cap_control1.setStyleSheet("border-image: url(:icon/button6.png);\n"
                                               "image: url(:/icon/pause.png);\n"
                                               "")
            self.Opening = False
        self.ui.stackedWidget.setCurrentIndex(0)

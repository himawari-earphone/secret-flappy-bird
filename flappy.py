from vector import vector
from random import randrange
import os
import cv2
import random
import numpy as np


class Flappy(object):
    def __init__(self, icon_path):
        self.score = 0
        self.speed = 5
        self.flapping = False
        self.bird = vector(0, 0)  # 小鸟的初始位置：画面中央
        self.wolves = []
        self.bird_img = cv2.imread(os.path.join(icon_path, 'bird.png'))
        self.bird_img = cv2.resize(self.bird_img, (64, 64))
        self.bird_mask = cv2.imread(os.path.join(icon_path, 'bird.png'), 0)
        self.bird_mask = cv2.resize(self.bird_mask, (64, 64))
        self.wolf_img = cv2.imread(os.path.join(icon_path, 'pig.png'))
        # self.wolf_img = cv2.resize(self.wolf_img, (64, 64))
        self.wolf_mask = cv2.imread(os.path.join(icon_path, 'pig.png'), 0)
        # self.wolf_mask = cv2.resize(self.wolf_mask, (64, 64))
        self.game_over = cv2.imread(os.path.join(icon_path, 'die.png'))
        self.game_over_mask = cv2.imread(os.path.join(icon_path, 'die.png'), 0)
        self.defeat = cv2.imread(os.path.join(icon_path, 'defeat.png'))
        self.defeat_mask = cv2.imread(os.path.join(icon_path, 'defeat.png'), 0)
        self.pause = False

    def tap(self):
        '''tap动作：控制小鸟上升'''
        up = vector(0, -40)
        self.bird.move(up)

    def inside(self, point, h, w):
        '''判断某个点是否在画面中'''
        return - w//2 < point.x < w // 2 and -h // 2 < point.y < h//2

    def draw_icon(self, icon, mask, img_data, x, y, size):
        '''绘制图标（小鸟、狼、gameover）'''
        h, w = img_data.shape[:2]
        tmp = img_data[np.clip(int(y + h//2) - size, 0, h):np.clip(int(y + h//2) + size, 0, h),
                        np.clip(int(x + w//2) - size, 0, w):np.clip(int(x + w//2) + size, 0, w)]

        tmp = cv2.copyTo(icon, mask, tmp)
        return img_data
    
    def rand_gen_wolves(self, img_data):
        '''在画面最右侧随机生成新的狼群'''

        h, w = img_data.shape[:2]
        
        # 计算最右侧的狼群距离画面右边界的距离
        if len(self.wolves) > 0:
            pre_wolf_x = self.wolves[-1].x
            pre_interval = w//2 - pre_wolf_x

        # 上下两侧狼群的空隙，要能容纳小鸟通过
        h_bird = self.bird_img.shape[0]
        h_wolf = self.wolf_img.shape[0]
        rand_interval = randrange(h_bird*3, h_bird*4)

        # 画面中没有狼群，或者最右侧狼群距离画面右边界超过一段距离，此时可以生成新的狼群
        if len(self.wolves) == 0 or pre_interval >= rand_interval:            
            max_num = (h - h_bird) // h_wolf   # 画面高度可容纳一列狼群的最大数量
            wolf_num = randrange(1, max_num)   # 随机产生狼群数量
            wolf_num_up = randrange(wolf_num)  # 安放在画面上方的狼群数量

            for ix in range(wolf_num_up):            # 添加画面上方的狼群
                y = -h//2 + ix * h_wolf
                wolf = vector(w//2 - 2, y)
                self.wolves.append(wolf)

            for ix in range(wolf_num - wolf_num_up):  # 添加画面下方的狼群
                y = h//2 - ix * h_wolf
                wolf = vector(w//2 - 2, y)
                self.wolves.append(wolf)

    def draw(self, alive, img_data):
        '''刷新游戏画面'''
        
        h, w = img_data.shape[:2]

        # 刷新狼群
        for wolf in self.wolves:
            img_data = self.draw_icon(self.wolf_img, self.wolf_mask, img_data, wolf.x, wolf.y, 32)

        if alive:  # 如果没挂，刷新小鸟
            img_data = self.draw_icon(self.bird_img, self.bird_mask, img_data, self.bird.x, self.bird.y, 32)
        else:     # 挂了，显示挂掉的图标，刷新游戏参数
            img_data = self.draw_icon(self.game_over, self.game_over_mask, img_data, self.bird.x, self.bird.y, 24)
            img_data = self.draw_icon(self.defeat, self.defeat_mask, img_data, 0, 0, 200)

        return img_data

    def move(self, img_data):
        '''更新所有物体的位置'''

        # 小鸟下降一段距离
        self.bird.y += self.speed // 2

        # 狼群左移一段距离
        for wolf in self.wolves:
            wolf.x -= self.speed

        # 画面最右侧生成新的狼群
        self.rand_gen_wolves(img_data)

        # 移除超出画面最左侧的狼群
        h, w = img_data.shape[:2]
        while len(self.wolves) > 0 and not self.inside(self.wolves[0], h, w):
            self.wolves.pop(0)

        # 判断小鸟是否触碰到画面上下边缘
        if not self.inside(self.bird, h, w):
            img = self.draw(False, img_data)
            return img, False

        # 判断小鸟是否触碰到狼群
        for wolf in self.wolves:
            if abs(wolf - self.bird) < 55:
                img = self.draw(False, img_data)
                return img, False

        # 刷新游戏画面
        img = self.draw(True, img_data)
        return img, True
    
    def update_flappy_keyboard(self, img_data, pressed_key):
        '''使用键盘控制游戏'''

        # 按下空格键，小鸟上升一段距离
        if pressed_key == ord(' '):
            self.tap()

        # 更新游戏状态
        frame, alive = self.move(img_data)
            
        # 显示当前存活时长
        self.score += 1
        cv2.putText(img_data, 'score: %d' % self.score, (5, 25), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
        
        # 如果挂掉了，更新游戏参数
        if not alive:
            self.bird = vector(0, 0)
            self.wolves = []
            self.score = 0

        return frame
    
    def is_arms_open(self, arm_angles, threshold):
        '''判断双臂是否展开（左右大臂、小臂与垂直方向的夹角都超过某个阈值）'''

        arms_open = False

        flag = True
        for angle in arm_angles:
            if angle < threshold:
                flag = False
                break
        if flag:
            arms_open = True
        return arms_open
    
    def is_arms_close(self, arm_angles, threshold):
        '''判断双臂是否落下（左右大臂、小臂与垂直方向的夹角都小于某个阈值）'''

        arms_close = False

        flag = True
        for angle in arm_angles:
            if angle > threshold:
                flag = False
                break
        if flag:
            arms_close = True
        return arms_close

    def update_flappy_pose(self, img_data, arm_angles, fps):
        '''体感控制游戏'''

        # 判断左右两臂的展开状态
        arms_open = self.is_arms_open(arm_angles, 45)
        arms_close = self.is_arms_close(arm_angles, 30)
        
        # 判断是否有扇动翅膀的动作（双臂由展开到落下）
        if self.flapping and arms_close:
            self.flapping = False
            self.tap()

        if arms_open:
            self.flapping = True

        # 更新游戏状态
        frame, alive = self.move(img_data)

        # 显示当前存活时长
        self.score += 1 / fps
        cv2.putText(img_data, 'score: %d' % int(self.score), (5, 23),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
        # # 显示游戏名
        # cv2.putText(img_data, 'Flappy Bird', (450, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 100, 255), 3)

        # 如果挂掉了，更新游戏参数
        if not alive:
            self.bird = vector(0, 0)
            self.wolves = []
            self.score = 0
            self.flapping = False
        return frame, alive


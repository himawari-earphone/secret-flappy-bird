import cv2
import numpy as np

# 需要画出的关节
link_pairs = [
    [[5, 6], [11, 12]],
    [[0, 5], [5, 7], [7, 9], [5, 11]],
    [[0, 6], [6, 8], [8, 10], [6, 12]],
]

# 关节对应的颜色
color_palette = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]


def get_cross_angle(point1_1, point1_2, point2_1, point2_2):
    '''计算两个向量的夹角'''
    if point1_1[0] < 0 or point1_2[0] < 0 or point2_1[0] < 0 or point2_2[0] < 0:
        return -1.0

    arr_0 = np.array([(point1_2[0] - point1_1[0]),
                      (point1_2[1] - point1_1[1])])
    arr_1 = np.array([(point2_2[0] - point2_1[0]),
                      (point2_2[1] - point2_1[1])])
    cos_val = (float(arr_0.dot(arr_1)) /
               (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))
    angle = np.arccos(cos_val) * (180 / np.pi)
    if not (angle >= 0 and angle <= 360):
        angle = -1.0
    return angle


def get_arm_angles(bbox, pose_data, keypoints_smooth):
    '''计算左右大臂、小臂与垂直方向的夹角'''
    pose_data = rematch_pose(bbox, pose_data, keypoints_smooth)

    # 右臂
    r_shoulder = 5
    r_elbow = 7
    r_wrist = 9
    angle1 = get_cross_angle(pose_data[r_shoulder], pose_data[r_elbow], pose_data[r_shoulder],
                             (pose_data[r_shoulder][0], pose_data[r_shoulder][1] + 10))
    angle2 = get_cross_angle(pose_data[r_elbow], pose_data[r_wrist], pose_data[r_elbow],
                             (pose_data[r_elbow][0], pose_data[r_elbow][1] + 10))

    # 左臂
    l_shoulder = 6
    l_elbow = 8
    l_wrist = 10
    angle3 = get_cross_angle(pose_data[l_shoulder], pose_data[l_elbow], pose_data[l_shoulder], (
        pose_data[l_shoulder][0], pose_data[l_shoulder][1] + 10))
    angle4 = get_cross_angle(pose_data[l_elbow], pose_data[l_wrist],
                             pose_data[l_elbow], (pose_data[l_elbow][0], pose_data[l_elbow][1] + 10))

    return [angle1, angle2, angle3, angle4]


def rematch_pose(bbox, pose_data, keypoints_smooth):
    '''根据检测框更新关键点在原始画面中的坐标，并作平滑处理'''
    x0, y0, x1, y1 = list(map(int, bbox))
    pose_data[:, 0] = pose_data[:, 0] * (x1 - x0) + x0
    pose_data[:, 1] = pose_data[:, 1] * (y1 - y0) + y0
    pose_data = keypoints_smooth.smooth_process(pose_data)
    return pose_data


def draw_pose(img, bbox, pose_data, score_thre):
    '''根据关键点画出关节'''
    for part, color in zip(link_pairs, color_palette):
        for pair in part:
            if pose_data[pair[0]][2] > score_thre and pose_data[pair[1]][2] > score_thre:
                cv2.line(img, (int(pose_data[pair[0]][0]), int(pose_data[pair[0]][1])), 
                (int(pose_data[pair[1]][0]), int(pose_data[pair[1]][1])), color, thickness=3)


def quit_game(img, pose_data):

    # r_elbow = 7
    # r_wrist = 9
    # l_elbow = 8
    # l_wrist = 10
    # angle_cross = get_cross_angle(pose_data[l_wrist], pose_data[l_elbow], pose_data[r_wrist], pose_data[r_elbow])
    # angle_l = get_cross_angle(pose_data[l_wrist], pose_data[l_elbow], [0, 0], [10, 0])
    # if angle_cross > 70 and angle_cross < 110 and angle_l > 30 and angle_l < 60:
    #     return True
    r_wrist = 9
    l_wrist = 10
    arr_0 = np.array([(pose_data[l_wrist][0] - pose_data[r_wrist][0]),
                      (pose_data[l_wrist][1] - pose_data[r_wrist][1])])
    dist = np.sqrt(arr_0.dot(arr_0))
    if dist < 60000:
        return img, True
    cv2.putText(img, 'Close your hands to exit the game', (5, 763),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
    return img, False
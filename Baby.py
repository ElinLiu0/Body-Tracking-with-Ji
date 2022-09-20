#Author: Elin.Liu
# Date: 2022-09-11 23:24:00
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-09-11 23:24:00

import time
import cv2
import mediapipe as mp
import sys

# 初始化MediaPipe绘图工具，以及样式
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# 初始化OpenCV窗口
window = cv2.namedWindow("Gi", cv2.WINDOW_FULLSCREEN)
# 读取坤坤视频
cap = cv2.VideoCapture('data.flv')
# 设置视频缓冲区
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
# 初始化FPS计时器和计数器
fps_start_time = 0
fps = 0

# 定义Processing处理函数


def processing(image):
    # 使用cv2.putText绘制FPS
    cv2.putText(image, "FPS: {:.2f}".format(
        fps), (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
    # 使用image.flags.writeable = False将图像标记为只读，以加快处理速度
    image.flags.writeable = False
    # 使用cv2.resize将图像缩放到适合的尺寸
    image = cv2.resize(image, (640, 480))
    # 使用cv2.cvtColor将图像转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 使用MediaPipe Pose检测关键点
    results = pose.process(image)

    # 解锁图像读写
    image.flags.writeable = True
    # 将图像转换回BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # 使用draw_landmarks()绘制关键点
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # 返回处理后的图像
    return image


# 初始化MediaPipe Pose类
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    # 当视频打开时
    while cap.isOpened():
        # 读取视频帧和状态
        success, image = cap.read()
        # 如果初始化失败，则推出进程
        if not success:
            print("")
            exit(1)
        # 初始化FPS结束点计时器
        fps_end_time = time.time()
        # 计算FPS
        fps = 1.0 / (fps_end_time - fps_start_time)
        # 重置FPS开始点计时器
        fps_start_time = fps_end_time
        # 创建线程处理图像
        image = processing(image)
        # 显示图像
        cv2.imshow('Gi', image)
        # 按下q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()

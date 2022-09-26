#Author: Elin.Liu
# Date: 2022-09-11 23:24:00
# Last Modified by:   Elin.Liu
# Last Modified time: 2022-09-11 23:24:00

import time
import cv2
import mediapipe as mp
import sys

# 初始化MediaPipe绘图工具，以及样式

origin_path = sys.argv[1]
compare_path = sys.argv[2]

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
# 初始化OpenCV窗口
window_origin = cv2.namedWindow("Origin", cv2.WINDOW_FULLSCREEN)
window_compare = cv2.namedWindow("Compare", cv2.WINDOW_FULLSCREEN)
# 读取坤坤视频
origin = cv2.VideoCapture(origin_path)
compare = cv2.VideoCapture(compare_path)

# 初始化FPS计时器和计数器
fps_start_time = 0
fps = 0

# 定义scoring函数，用于计算两个关键点之间的距离


def scoring(origin_image, compare_image):
    scores = []
    origin_points = pose.process(origin_image).pose_landmarks
    compare_points = pose.process(compare_image).pose_landmarks
    for i in range(33):
        try:
            origin_point = origin_points.landmark[i]
            compare_point = compare_points.landmark[i]
            origin_x = origin_point.x
            origin_y = origin_point.y
            compare_x = compare_point.x
            compare_y = compare_point.y
            score = (compare_x / origin_x + compare_y / origin_y) / 2
            scores.append(score)
        except:
            scores.append("Nan")
    return scores

# 定义Processing处理函数


def processing(image, scores):
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
    try:
        mark_points = results.pose_landmarks.landmark
        if len(scores) == len(mark_points):
            for i in range(len(scores)):
                score = scores[i]
                mark_point = mark_points[i]
                mark_x = mark_point.x
                mark_y = mark_point.y
                cv2.putText(image, "{:.2f}".format(
                    score), (int(mark_x * 640), int(mark_y * 480)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    except:
        cv2.putText(image, "Nan", (10, 85 + 1 * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # 返回处理后的图像
    return image


# 初始化MediaPipe Pose类
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    # 当视频打开时
    while origin.isOpened() and compare.isOpened():
        # 读取视频帧和状态
        success_origin, image_origin = origin.read()
        success_compare, image_compare = compare.read()
        # 如果初始化失败，则推出进程
        if not success_origin or not success_compare:
            print("")
            exit(1)
        # 初始化FPS结束点计时器
        fps_end_time = time.time()
        # 计算FPS
        fps = 1.0 / (fps_end_time - fps_start_time)
        # 重置FPS开始点计时器
        fps_start_time = fps_end_time
        # 创建线程处理图像
        image_origin = processing(image_origin, [1 for i in range(33)])
        image_compare = processing(
            image_compare, scores=scoring(image_origin, image_compare))
        # 显示图像
        cv2.imshow('Origin', image_origin)
        cv2.imshow('Compare', image_compare)
        # 按下q键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
origin.release()
compare.release()

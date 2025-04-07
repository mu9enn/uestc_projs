import cv2
import numpy as np
import os
import math
from camera import *
from pers_trans import *
from color_detect import *
import serial


import numpy as np

def inv_transform(point, matrix):
    """
    对单个点进行透视逆变换
    """
    x, y = point
    pt = np.array([x, y, 1.0], dtype=np.float32)
    matrix_inv = np.linalg.inv(matrix)
    dst = matrix_inv @ pt
    dst /= dst[2]

    return (int(dst[0]), int(dst[1]))

def compute_image_scale(pts, real_distance_cm=10.0):
    """
    计算图像的比例尺，首先找出距离最远的两个点作为 pt1 和 pt2。
    """
    max_dist = 0
    pt1, pt2 = None, None

    # 遍历所有点对，找到距离最远的两个点
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dist = np.linalg.norm(np.array(pts[j]) - np.array(pts[i]))
            if dist > max_dist:
                max_dist = dist
                pt1, pt2 = pts[i], pts[j]

    if pt1 is not None and pt2 is not None:
        image_scale = max_dist / real_distance_cm
        return image_scale
    else:
        raise ValueError("No points found")


def real_displace(image_point, image_scale, offset=(0, 0), xy=True, image=None):
    """
      image_point: 图像中目标点的坐标，格式 (x, y)
      image_scale: 图像比例尺（像素/cm）
      offset: 机械臂在图像上的坐标偏移（像素）
      xy: 如果为 True 返回 (x, y) 坐标，否则返回 (距离, 角度)
    """
    # 计算机械臂坐标系下的目标点坐标
    real_x = (image_point[0] - offset[0]) / image_scale
    real_y = (image_point[1] - offset[1]) / image_scale
    if xy:
        return (real_x, real_y)
    else:
        distance = math.sqrt(real_x ** 2 + real_y ** 2)
        angle_rad = math.atan2(real_y, real_x)
        angle_deg = math.degrees(angle_rad)
        return distance, angle_deg
    


def generate_command(start_xy, end_xy, strt_z=200, end_z=0, grip=1, speed=3):
    """
    生成控制字符串命令（共23位）
    start_xy, end_xy: 起点和终点坐标 (x, y)，单位已缩放为 mm
    """

    def format_coordinate(val, width, signed=False):
        """格式化单个坐标为指定宽度字符串，signed=True 表示前缀符号位"""
        if signed:
            sign = '1' if val < 0 else '0'
            val = abs(int(val))
            return sign + str(val).zfill(width - 1)
        else:
            return str(int(val)).zfill(width)
        
    strt_x, strt_y = start_xy
    end_x, end_y = end_xy

    start_str = (
        format_coordinate(strt_x, 3) +                   # 起点 x
        format_coordinate(strt_y, 4, signed=True) +      # 起点 y (带符号)
        format_coordinate(strt_z, 3) +                     # 起点 z
        str(grip)                                        # 夹爪状态（1夹）
    )
    end_str = (
        format_coordinate(end_x, 3) +                    # 终点 x
        format_coordinate(end_y, 4, signed=True) +       # 终点 y
        format_coordinate(end_z, 3) +                   # 终点 z
        '0'                                              # 不夹
    )
    final_str = start_str + end_str + str(speed)
    return final_str


def send_to_robot(data_str, port='COM3', baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        ser.write(data_str.encode('utf-8'))
        ser.close()
        print("发送给机械臂的数据:", data_str)
    except Exception as e:
        print("串口通信错误:", e)



def main():
    img = load_image()
    pts = select_points(img)
    matrix, transformed = compute_trans(img, pts)
    mask, detected = color_detect(transformed, color_name='black')
    contours, dart_centers, _ = contours_center(detected, iter=0, print_=True)
    contour_img = transformed.copy()
    cv2.drawContours(contour_img, contours, -1, 255, 2)
    print('contours number:', len(contours))
    # 显示结果图像
    cv2.imshow("Warped Perspective with contour", contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # -------------------------------------------------------------------------------
    farest_distant = 10.0
    offset = (20, 30)  # 机械手到图像原点的偏移
    scale = compute_image_scale(dart_centers, real_distance_cm=farest_distant)
    print("Image scale (px/cm):", scale)


    for center in dart_centers:

        start_xy = real_displace(center, scale, offset=offset, xy=True)
        target_dart_img = select_points(img, point_num=1, show_point=inv_transform(center, matrix))
        end_xy = real_displace(target_dart_img[0], scale, offset=offset, xy=True)
        command = generate_command(start_xy, end_xy, strt_z=30, end_z=30, grip=1, speed=3)
        # 坐标 ( 如z的30 ) 以毫米为单位
        print(command)


if __name__ == "__main__":
    main()

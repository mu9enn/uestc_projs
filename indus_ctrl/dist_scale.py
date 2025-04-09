import cv2
import numpy as np
import os
import math
from camera import *
from pers_trans import *
from color_detect import *
import serial

def inv_transform(point, matrix):
    # 对单个点进行透视逆变换
    x, y = point
    pt = np.array([x, y, 1.0], dtype=np.float32)
    matrix_inv = np.linalg.inv(matrix)
    dst = matrix_inv @ pt
    dst /= dst[2]
    return (int(dst[0]), int(dst[1]))

def compute_image_scale(pts, real_distance_cm=10.0):
    # 计算图像的比例尺
    max_dist = 0
    pt1, pt2 = None, None
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
    def format_coordinate(val, width, signed=False):
        if signed:
            sign = '1' if val < 0 else '0'
            val = abs(int(val))
            return sign + str(val).zfill(width - 1)
        else:
            return str(int(val)).zfill(width)

    strt_x, strt_y = start_xy
    end_x, end_y = end_xy
    start_str = (
        format_coordinate(strt_x, 3) +
        format_coordinate(strt_y, 4, signed=True) +
        format_coordinate(strt_z, 3) +
        str(grip)
    )
    end_str = (
        format_coordinate(end_x, 3) +
        format_coordinate(end_y, 4, signed=True) +
        format_coordinate(end_z, 3) +
        '0'
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
    cv2.imshow("Warped Perspective with contour", contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    farest_distant = 10.0
    offset = (20, 30)
    scale = compute_image_scale(dart_centers, real_distance_cm=farest_distant)
    print("Image scale (px/cm):", scale)

    for center in dart_centers:
        start_xy = real_displace(center, scale, offset=offset, xy=True)
        target_dart_img = select_points(transformed, point_num=1, show_point=center)
        end_xy = real_displace(target_dart_img[0], scale, offset=offset, xy=True)
        command = generate_command(start_xy, end_xy, strt_z=30, end_z=30, grip=1, speed=3)
        print(command)
        # send_to_robot(command)

if __name__ == "__main__":
    main()

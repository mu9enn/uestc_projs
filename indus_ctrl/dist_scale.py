import cv2
import numpy as np
import os
import math
from camera import *
from pers_trans import *
from color_detect import *


# def compute_image_scale(p1, p4, real_distance_cm=10.0):
#     """
#     计算图像的比例尺，同时在图像上可视化两点及连接线。
#     """
#     pixel_distance = np.linalg.norm(np.array(p4) - np.array(p1))
#     image_scale = pixel_distance / real_distance_cm
#     return image_scale

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
        return image_scale, pt1, pt2
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
        return real_x, real_y
    else:
        distance = math.sqrt(real_x ** 2 + real_y ** 2)
        angle_rad = math.atan2(real_y, real_x)
        angle_deg = math.degrees(angle_rad)
        return distance, angle_deg


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
    scale = compute_image_scale(dart_centers, real_distance_cm=farest_distant)
    print("Image scale (px/cm):", scale)

    offset = (20, 30)  # 机械手到图像原点的偏移
    # target_dart_img = dart_centers[0]
    target_dart_img = select_points(img, point_num=1)

    # 计算实际位移，返回实际坐标
    real_coord = real_displace(target_dart_img, scale, offset=offset, xy=True)
    print("Real coordinate:", real_coord)


if __name__ == "__main__":
    main()

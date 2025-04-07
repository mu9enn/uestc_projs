import cv2
import numpy as np
import os
from camera import *
from pers_trans import *


def color_detect(img, color_name, json_path="color_ranges.json"):
    """
    根据 color_name，从 json_path 读取对应的 HSV 范围，
    然后对 img 进行 inRange 操作，返回 mask。
    """

    # 红色提取
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if color_name == "black":
        lower_black = np.array([0, 0, 0])  # 低饱和度、低亮度
        upper_black = np.array([180, 255, 50])  # 亮度不超过 50（可根据需求调整）
        mask = cv2.inRange(hsv, lower_black, upper_black)

    else:

        if color_name == "red":
            lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
            lower_red2, upper_red2 = np.array([170, 50, 50]), np.array([180, 255, 255])

        elif color_name == "red+green":
            lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([77, 255, 255])
            lower_red2, upper_red2 = np.array([170, 50, 50]), np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(red_mask1, red_mask2)

    red_extracted = cv2.bitwise_and(img, img, mask=mask)

    detected = red_extracted.copy()

    return mask, detected


def contours_center(image, kernel_size=3, iter=3, min_area=50, max_area=10000, print_=False):
    def remove_inner_contours(contours, hierarchy):
        filtered_contours = []

        for i in range(len(contours)):
            # hierarchy[i][3] == -1 表示没有父轮廓，即它是最外层的轮廓
            if hierarchy[0][i][3] == -1:
                filtered_contours.append(contours[i])

        return filtered_contours

    if iter:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iter)

    # 转换为灰度图并二值化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    # 红色绿色检测时为(50,255), 黑色检测时为(0,255)

    # 轮廓检测：过滤被大轮廓包含的小轮廓 & 面积极小的轮廓

    contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = remove_inner_contours(contours, hierarchy)
    contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
    if print_:
        print([cv2.contourArea(contour) for contour in contours])

    dart_centers = []
    for i in range(len(contours)):
        points = contours[i][:, 0]
        center = points.mean(axis=0)
        dart_centers.append(tuple(center.astype(int)))  # integer tuple

    return contours, dart_centers, binary


if __name__ == "__main__":
    # save_dir = r"D:\code_proj\uestc_projs\indus_ctrl\test.jpg" # 图片保存路径
    # img = capture_image(save_dir, cam=0)
    img = load_image()
    pts = select_points(img)

    matrix, transformed = compute_trans(img, pts)
    # print("The calculated perspective transformation matrix：")
    # print(matrix)
    mask, detected = color_detect(transformed, color_name='black')
    contours, dart_centers, _ = contours_center(detected, iter=0, print_=True)

    contour_img = transformed.copy()
    cv2.drawContours(contour_img, contours, -1, 255, 2)

    cv2.imshow("Original Image", img)
    cv2.imshow("Color detected", detected)
    cv2.imshow("Warped Perspective with contour", contour_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

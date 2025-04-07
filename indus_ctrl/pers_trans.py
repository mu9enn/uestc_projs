import cv2
import numpy as np
import os
from camera import *


def select_points(img, point_num=4):
    """在图像上点击{point_num}个点，并返回这{point_num}个点的列表"""
    pts = []
    img_copy = img.copy()

    def click_event(event, x, y, flags, param):
        nonlocal pts, img_copy
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img_copy, str(len(pts)), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(f"Select {point_num} Points", img_copy)

    cv2.imshow(f"Select {point_num} Points", img_copy)
    cv2.setMouseCallback(f"Select {point_num} Points", click_event)
    if point_num == 4:
        print(
            f"Please click on {point_num} points on the image in sequence (top left, bottom left, bottom right, top right in order)")
    else:
        print(f"Please click on {point_num} points on the image.")
    # 循环直到点击{point_num}个点
    while True:
        cv2.imshow(f"Select {point_num} Points", img_copy)
        # 按ESC退出（可选）
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if len(pts) == point_num:
            break
    cv2.destroyWindow(f"Select {point_num} Points")
    return pts


def compute_trans(img, pts):
    # 将选取的{point_num}个点转换为浮点型数组
    pts1 = np.float32(pts)
    h, w = img.shape[:2]
    pts2 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (w, h))

    return matrix, result


if __name__ == "__main__":
    save_dir = r"D:\code_proj\uestc_projs\indus_ctrl"  # 图片保存路径
    img = capture_image(save_dir, cam=1)
    pts = select_points(img)
    matrix, result = compute_trans(img, pts)
    print("The calculated perspective transformation matrix：")
    print(matrix)

    cv2.imshow("Original Image", img)
    cv2.imshow("Warped Perspective", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

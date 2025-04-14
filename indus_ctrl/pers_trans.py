import numpy as np
from camera import *
import numpy as np

from camera import *


def select_points(img, point_num=4, show_point=None):
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

    if show_point:
        cv2.circle(img_copy, show_point, 5, (255, 0, 0), -1)
    cv2.imshow(f"Select {point_num} Points", img_copy)
    cv2.setMouseCallback(f"Select {point_num} Points", click_event)

    if point_num == 4:
        print(f"请依次点击图像上的 {point_num} 个点（顺序： 左上 左下 右下 右上）")
        # 让图像坐标系与机械臂坐标系对齐
    else:
        print(f"请点击图像上的 {point_num} 个点。")

    while True:
        cv2.imshow(f"Select {point_num} Points", img_copy)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if len(pts) == point_num:
            break
    cv2.destroyWindow(f"Select {point_num} Points")
    return pts

# def compute_trans(img, pts):
#     pts1 = np.float32(pts)
#     h, w = img.shape[:2]
#     pts2 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
#     matrix = cv2.getPerspectiveTransform(pts1, pts2)
#     result = cv2.warpPerspective(img, matrix, (w, h))
#
#     return matrix, result

def compute_trans(img, pts):
    pts = np.float32(pts)

    h, w = img.shape[:2]

    min_x = min(pts[:, 0])
    max_x = max(pts[:, 0])
    min_y = min(pts[:, 1])
    max_y = max(pts[:, 1])

    # pts1 = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
    # matrix = cv2.getPerspectiveTransform(pts, pts1)
    # result = cv2.warpPerspective(img, matrix, (w, h))
    # width = result.shape[1]

    # 目标坐标系的四个角，覆盖整个变换后的图像区域
    pts2 = np.float32([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])

    width_297 = max_x - min_x
    width_210 = max_y - min_y
    matrix = cv2.getPerspectiveTransform(pts, pts2)
    result = cv2.warpPerspective(img, matrix, (w, h))

    return matrix, result, width_297, width_210


def align_trans(img):
    rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    out_img = cv2.flip(rotated_img, 1)
    return out_img

if __name__ == "__main__":
    save_dir = r"D:\code_proj\uestc_projs\indus_ctrl"

    img = capture_image(save_dir, cam=1)
    pts = select_points(img)
    matrix, result = compute_trans(img, pts)
    print("The calculated perspective transformation matrix：")
    print(matrix)

    cv2.imshow("Original Image", img)
    cv2.imshow("Warped Perspective", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

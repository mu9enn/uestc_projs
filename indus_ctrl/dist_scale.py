import math
from color_detect import *
import math

from color_detect import *


def inv_transform(point, matrix):
    # 对单个点进行透视逆变换
    x, y = point
    pt = np.array([x, y, 1.0], dtype=np.float32)
    matrix_inv = np.linalg.inv(matrix)
    dst = matrix_inv @ pt
    dst /= dst[2]
    return (int(dst[0]), int(dst[1]))


def compute_image_scale(width, real_width=10.0):
    image_scale = width / real_width
    return image_scale


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


def main():
    img = load_image()
    pts = select_points(img)
    matrix, transformed = compute_trans(img, pts)

    width = transformed.shape[0]
    real_width = 297
    offset = (20, 30)
    scale = compute_image_scale(width, real_width)
    print("Image scale (px/cm):", scale)

    for idx in range(6):
        center = select_points(transformed, point_num=1, show_point=None)
        xy = real_displace(center[0], scale, offset=offset, xy=True)
        print("real_position (mm) :", xy)


if __name__ == "__main__":
    main()

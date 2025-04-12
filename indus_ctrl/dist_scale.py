import math

from color_detect import *


def compute_image_scale(width, real_width=10.0):
    image_scale = width / real_width
    return image_scale


def real_displace(image_point, image_scale, offset=(0, 0), xy=True, image=None):
    real_x = image_point[0] / image_scale - offset[0]
    real_y = image_point[1] / image_scale - offset[1]
    if xy:
        return (real_x, real_y)
    else:
        distance = math.sqrt(real_x ** 2 + real_y ** 2)
        angle_rad = math.atan2(real_y, real_x)
        angle_deg = math.degrees(angle_rad)
        return distance, angle_deg


def main():
    save_dir = r"D:\code_proj\uestc_projs\indus_ctrl"
    # img = capture_image(save_dir, cam=0)
    img = load_image(image_path=os.path.join(save_dir, "captured.jpg"))

    pts = select_points(img)
    matrix, transformed = compute_trans(img, pts)
    transformed = align_trans(transformed)

    width = transformed.shape[1]
    real_width = 297
    offset = (0, 0)  # mm单位
    scale = compute_image_scale(width, real_width)
    print("Image scale (px/cm):", scale)

    for idx in range(6):
        center = select_points(transformed, point_num=1, show_point=None)
        xy = real_displace(center[0], scale, offset=offset, xy=True)
        print("real_position (mm) :", xy)


if __name__ == "__main__":
    main()

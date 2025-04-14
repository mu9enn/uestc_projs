import math

from color_detect import *


def compute_image_scale(width, real_width=10.0):
    image_scale = width / real_width
    return image_scale


def real_displace(image_point, image_scale_297,image_scale_210, offset=(0, 0), xy=True, image=None):
    real_x = image_point[0] / image_scale_210 - offset[0]
    real_y = image_point[1] / image_scale_297 - offset[1]
    if xy:
        return (real_x, real_y)
    else:
        distance = math.sqrt(real_x ** 2 + real_y ** 2)
        angle_rad = math.atan2(real_y, real_x)
        angle_deg = math.degrees(angle_rad)
        return distance, angle_deg


def main():
    save_dir = r"D:\code_proj\uestc_projs\indus_ctrl"
    img = capture_image(save_dir, cam=0)
    pts = select_points(img)
    matrix, transformed, width_297, width_210 = compute_trans(img, pts)
    transformed = align_trans(transformed)
    mask, detected = color_detect(transformed, color_name='green')
    contours, dart_centers, _ = contours_center(detected, iter=0, print_=True)
    contour_img = transformed.copy()
    cv2.drawContours(contour_img, contours, -1, 255, 2)
    print('contours number:', len(contours))
    cv2.imshow("color detected", detected)
    cv2.waitKey(0)
    cv2.imshow("Warped Perspective with contour", contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    width = transformed.shape[1]
    real_width = 210
    offset = (101.5-200, 139.125)
    scale_210 = compute_image_scale(width_210, 210)
    scale_297 = compute_image_scale(width_297, 297)

    for center in dart_centers:
        xy = real_displace(center, scale_297, scale_210, offset=offset, xy=True)
        real_position = (float(xy[0]), float(xy[1]))
        print("real_position (mm) :", real_position)

def test():
    save_dir = r"D:\code_proj\uestc_projs\indus_ctrl"
    img = capture_image(save_dir, cam=0)

    pts = select_points(img)
    print(pts)  # np.array([(147, 282), (342, 261), (347, 114), (209, 131)])

    matrix, transformed, width_297, width_210 = compute_trans(img, pts)
    transformed = align_trans(transformed)

    offset = (0,0)  # mm单位, 参考点()
    scale_210 = compute_image_scale(width_210, 210)
    scale_297 = compute_image_scale(width_297, 297)
    # print("Image scale (px/cm):", scale)

    for idx in range(99):
        center = select_points(transformed, point_num=1, show_point=None)
        xy = real_displace(center[0], scale_297, scale_210, offset=offset, xy=True)
        print("real_position (mm) :", xy)


if __name__ == "__main__":
    # main()
    test()
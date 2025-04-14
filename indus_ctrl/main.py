import serial

from dist_scale import *


def blacken_right_25(image):
    h, w = image.shape[:2]
    cutoff = int(0.75 * w)
    image[:, cutoff:] = 0

    return image


def generate_command(start_xy, end_xy, strt_z=200, end_z=0, grip=1):
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
    final_str = start_str + end_str
    return final_str


def send_to_robot(data_str, port='COM3', baudrate=115200):
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        ser.write(data_str.encode('utf-8'))
        ser.close()
        print("发送给机械臂的数据:", data_str)
    except Exception as e:
        print("串口通信错误:", e)


def main():
    save_dir = r"D:\code_proj\uestc_projs\indus_ctrl"
    img = capture_image(save_dir, cam=0)
    # img = load_image()
    # pts = select_points(img)
    pts = np.array([(144, 286), (340, 264), (344, 117), (204, 131)])

    matrix, transformed, width_297,width_210 = compute_trans(img, pts)
    transformed = align_trans(transformed)

    transformed = blacken_right_25(transformed)

    mask, detected = color_detect(transformed, color_name='green')
    contours, dart_centers, _ = contours_center(detected, iter=0, print_=True)
    contour_img = transformed.copy()
    cv2.drawContours(contour_img, contours, -1, 255, 2)
    print('contours number:', len(contours))
    cv2.imshow("Warped Perspective with contour", contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # width = transformed.shape[0]
    speed = 2
    offset = (344.2012-274, 506.38498+86)  # mm单位
    scale_210 = compute_image_scale(width_210, 210)
    scale_297 = compute_image_scale(width_297, 297)

    command_se = ''

    for center in dart_centers:
        start_xy = real_displace(center, scale_297, scale_210, offset=offset, xy=True)
        print('start point:', start_xy)
        target_dart_img = select_points(transformed, point_num=1, show_point=center)
        end_xy = real_displace(target_dart_img[0], scale_297, scale_210, offset=offset, xy=True)
        print('end point:', end_xy)
        command = generate_command(start_xy, end_xy, strt_z=40, end_z=40, grip=1)
        command_se = command_se + command

    send_to_robot(command_se + str(speed))



if __name__ == "__main__":
    main()

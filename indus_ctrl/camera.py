import os

import cv2
import matplotlib.pyplot as plt


def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.show()


def capture_image(save_dir, cam=0, img_name="captured.jpg"):
    """打开摄像头，实时显示视频流，按q键保存一帧为1.jpg"""
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print("Unable to open camera")
        return None

    print("Press 'q' to capture the image and exit the camera")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read camera")
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 保存当前帧到指定路径
            img_path = os.path.join(save_dir, img_name)
            cv2.imwrite(img_path, frame)
            print(f"Image saved to: {img_path}")
            break

    cap.release()
    cv2.destroyAllWindows()
    img = cv2.imread(img_path)
    return img


def load_image(image_path=r"D:\code_proj\uestc_projs\indus_ctrl\test.jpg"):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return img


if __name__ == "__main__":
    save_dir = r"D:\code_proj\uestc_projs\indus_ctrl"  # 图片保存路径
    img = capture_image(save_dir, cam=1)
    show_image(img)

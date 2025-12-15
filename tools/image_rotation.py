import cv2
import numpy as np

def rotate_image(image, angle):
    # 获取图像尺寸
    height, width = image.shape[:2]

    # 计算旋转中心点
    center = (width // 2, height // 2)

    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算新图像的边界
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # 调整旋转矩阵
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # 执行旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    return rotated_image

def main():
    # 读取图像
    image_path = 'path/to/your/image.jpg'  # 请替换为实际的图像路径
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 设置旋转角度（以度为单位）
    angle = 45  # 可以根据需要修改角度

    # 旋转图像
    rotated = rotate_image(image, angle)

    # 显示原图和旋转后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Rotated Image', rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
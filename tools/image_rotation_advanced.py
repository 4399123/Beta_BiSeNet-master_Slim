#!/usr/bin/python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import argparse
import os

def rotate_image(image, angle, scale=1.0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, border_value=(0, 0, 0)):
    """
    旋转图像任意角度

    参数:
        image: 输入图像
        angle: 旋转角度（度）
        scale: 缩放因子
        interpolation: 插值方法
        border_mode: 边界填充模式
        border_value: 边界填充值

    返回:
        rotated_image: 旋转后的图像
    """
    # 获取图像尺寸
    height, width = image.shape[:2]

    # 计算旋转中心点
    center = (width // 2, height // 2)

    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    # 计算新图像的边界
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # 调整旋转矩阵
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # 执行旋转
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=border_value
    )

    return rotated_image

def parse_args():
    parser = argparse.ArgumentParser(description='图像旋转工具')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', '-o', type=str, help='输出图像路径')
    parser.add_argument('--angle', '-a', type=float, default=45.0, help='旋转角度（度）')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='缩放因子')
    parser.add_argument('--interpolation', type=int, default=cv2.INTER_LINEAR,
                        help='插值方法: 0=INTER_NEAREST, 1=INTER_LINEAR, 2=INTER_CUBIC')
    parser.add_argument('--border', type=int, default=cv2.BORDER_CONSTANT,
                        help='边界模式: 0=BORDER_CONSTANT, 1=BORDER_REPLICATE, 4=BORDER_REFLECT')
    parser.add_argument('--border-value', type=int, nargs=3, default=[0, 0, 0],
                        help='边界填充值 (B,G,R)')
    parser.add_argument('--display', '-d', action='store_true', help='显示图像')

    return parser.parse_args()

def main():
    args = parse_args()

    # 读取图像
    image = cv2.imread(args.input)

    if image is None:
        print(f"无法读取图像: {args.input}")
        return

    # 旋转图像
    rotated = rotate_image(
        image,
        args.angle,
        args.scale,
        args.interpolation,
        args.border,
        tuple(args.border_value)
    )

    # 保存图像
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(args.output, rotated)
        print(f"旋转后的图像已保存到: {args.output}")

    # 显示图像
    if args.display:
        cv2.imshow('Original Image', image)
        cv2.imshow('Rotated Image', rotated)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
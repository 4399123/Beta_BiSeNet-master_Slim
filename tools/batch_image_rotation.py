#!/usr/bin/python
# -*- encoding: utf-8 -*-

import cv2
import numpy as np
import argparse
import os
import glob
from tqdm import tqdm

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

def rotate_around_point(image, angle, point=None, scale=1.0):
    """
    围绕指定点旋转图像

    参数:
        image: 输入图像
        angle: 旋转角度（度）
        point: 旋转中心点 (x, y)，默认为图像中心
        scale: 缩放因子

    返回:
        rotated_image: 旋转后的图像
    """
    height, width = image.shape[:2]

    if point is None:
        # 默认围绕图像中心旋转
        point = (width // 2, height // 2)

    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(point, angle, scale)

    # 计算新图像的边界
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # 调整旋转矩阵
    rotation_matrix[0, 2] += (new_width / 2) - point[0]
    rotation_matrix[1, 2] += (new_height / 2) - point[1]

    # 执行旋转
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))

    return rotated_image

def parse_args():
    parser = argparse.ArgumentParser(description='批量图像旋转工具')
    parser.add_argument('--input-dir', '-i', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output-dir', '-o', type=str, required=True, help='输出图像目录')
    parser.add_argument('--angle', '-a', type=float, default=45.0, help='旋转角度（度）')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='缩放因子')
    parser.add_argument('--pattern', '-p', type=str, default='*.jpg,*.png,*.jpeg', help='文件匹配模式，用逗号分隔')
    parser.add_argument('--interpolation', type=int, default=cv2.INTER_LINEAR,
                        help='插值方法: 0=INTER_NEAREST, 1=INTER_LINEAR, 2=INTER_CUBIC')
    parser.add_argument('--border', type=int, default=cv2.BORDER_CONSTANT,
                        help='边界模式: 0=BORDER_CONSTANT, 1=BORDER_REPLICATE, 4=BORDER_REFLECT')
    parser.add_argument('--border-value', type=int, nargs=3, default=[0, 0, 0],
                        help='边界填充值 (B,G,R)')
    parser.add_argument('--recursive', '-r', action='store_true', help='递归处理子目录')

    return parser.parse_args()

def process_images(args):
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 获取所有匹配的文件
    patterns = args.pattern.split(',')
    image_files = []

    if args.recursive:
        for pattern in patterns:
            image_files.extend(glob.glob(os.path.join(args.input_dir, '**', pattern), recursive=True))
    else:
        for pattern in patterns:
            image_files.extend(glob.glob(os.path.join(args.input_dir, pattern)))

    if not image_files:
        print(f"在 {args.input_dir} 中未找到匹配的图像文件")
        return

    print(f"找到 {len(image_files)} 个图像文件")

    # 处理每个图像
    for image_path in tqdm(image_files, desc="处理图像"):
        # 读取图像
        image = cv2.imread(image_path)

        if image is None:
            print(f"无法读取图像: {image_path}")
            continue

        # 旋转图像
        rotated = rotate_image(
            image,
            args.angle,
            args.scale,
            args.interpolation,
            args.border,
            tuple(args.border_value)
        )

        # 确定输出路径
        rel_path = os.path.relpath(image_path, args.input_dir)
        output_path = os.path.join(args.output_dir, rel_path)

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存图像
        cv2.imwrite(output_path, rotated)

    print(f"所有图像处理完成，结果保存在: {args.output_dir}")

def main():
    args = parse_args()
    process_images(args)

if __name__ == "__main__":
    main()
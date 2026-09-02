#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '..')
import os
import os.path as osp
import cv2
import numpy as np
import argparse
import torch
from tqdm import tqdm

# 导入项目中的模型
from lib.models import model_factory
from configs import set_cfg_from_file

# 导入图像旋转函数
from image_rotation_advanced import rotate_image

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
            default='../configs/bisenetv1_blueface_efficientnet_b3.py',)
    parse.add_argument('--weight-path', type=str, default='../pt/best.pt',)
    parse.add_argument('--img-path', type=str, default='./imgs/test.jpg',)
    parse.add_argument('--angle', type=float, default=30.0, help='旋转角度（度）')
    return parse.parse_args()

def main():
    args = parse_args()
    cfg = set_cfg_from_file(args.config)

    # 加载模型
    net = model_factory[cfg.model_type](n_classes=cfg.n_cats, use_fp16=cfg.use_fp16)
    net.load_state_dict(torch.load(args.weight_path, map_location='cpu'))
    net.cuda()
    net.eval()

    # 读取图像
    img = cv2.imread(args.img_path)
    if img is None:
        print(f"无法读取图像: {args.img_path}")
        return

    # 显示原始图像
    cv2.imshow('Original Image', img)

    # 旋转图像
    rotated_img = rotate_image(img, args.angle)

    # 显示旋转后的图像
    cv2.imshow('Rotated Image', rotated_img)

    # 对原始图像进行分割
    orig_result = segment_image(net, img)
    cv2.imshow('Original Segmentation', orig_result)

    # 对旋转后的图像进行分割
    rotated_result = segment_image(net, rotated_img)
    cv2.imshow('Rotated Segmentation', rotated_result)

    # 等待用户按键
    print("按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def segment_image(net, img):
    # 预处理
    img = cv2.resize(img, (512, 512))
    img = img[:, :, ::-1]  # BGR to RGB
    img = img.astype(np.float32) / 255.0
    img = img - np.array([0.485, 0.456, 0.406])[None, None, :]
    img = img / np.array([0.229, 0.224, 0.225])[None, None, :]
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).cuda()

    # 推理
    with torch.no_grad():
        out = net(img)[0]

    # 后处理
    pred = out.argmax(dim=1).squeeze().cpu().numpy()

    # 创建可视化结果
    color_map = {
        0: [0, 0, 0],       # 背景 - 黑色
        1: [255, 0, 0],     # 类别1 - 红色
        2: [0, 255, 0],     # 类别2 - 绿色
        3: [0, 0, 255],     # 类别3 - 蓝色
        4: [255, 255, 0],   # 类别4 - 黄色
        5: [255, 0, 255],   # 类别5 - 洋红色
        6: [0, 255, 255],   # 类别6 - 青色
        7: [128, 0, 0],     # 类别7 - 深红色
        8: [0, 128, 0],     # 类别8 - 深绿色
    }

    result = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for label, color in color_map.items():
        result[pred == label] = color

    return result

if __name__ == "__main__":
    main()
# encoding=utf-8
import cv2
import numpy as np
from openvino import Core
import time

# 路径设置
model_path = r'./onnx/best-smi.onnx'
pic_path = r'./onnx/11.jpg'
w, h = 512, 512

# 调色板
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
palette[0] = [255, 255, 255]
palette[1] = [0, 255, 0]
palette[2] = [0, 0, 255]
palette[3] = [255, 0, 0]
palette[4] = [255, 255, 0]
palette[5] = [255, 0, 255]
palette[6] = [171, 130, 255]
palette[7] = [155, 211, 255]
palette[8] = [0, 255, 255]

# OpenVINO 模型
core = Core()
compiled_model = core.compile_model(model_path, 'CPU')
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# 图像预处理：OpenCV 读取 BGR，直接作为 pred 模型输入。
img = cv2.imread(pic_path, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f'无法读取图片: {pic_path}')
img = cv2.resize(img, (w, h))
imgbak = img.copy()
img = np.ascontiguousarray(np.transpose(img.astype(np.float32), (2, 0, 1))[None, ...])

# 模型推理
for i in range(100):
    start_time = time.time()
    out = compiled_model({input_layer: img})[output_layer].astype(int)
    print('耗时：{:.5f}s'.format(time.time() - start_time))





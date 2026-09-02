# encoding=utf-8
import onnx
import onnxruntime as ort
import numpy as np
import cv2

# 路径设置
onnx_path = r'./onnx/best-smi.onnx'
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

# ONNX 模型
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

# 图像预处理：OpenCV 读取 BGR，直接作为 ONNX pred 图的输入。
img = cv2.imread(pic_path, cv2.IMREAD_COLOR)
if img is None:
    raise FileNotFoundError(f'无法读取图片: {pic_path}')
img = cv2.resize(img, (w, h))
imgbak = img.copy()
img = np.ascontiguousarray(np.transpose(img.astype(np.float32), (2, 0, 1))[None, ...])

# 模型推理
out = session.run(None, input_feed={'input': img})
out = out[0].astype('int')
pred = palette[out].squeeze()

# 保存图像
n = 0
cv2.imwrite('./onnx/mask_{}.jpg'.format(n), pred)
img = cv2.addWeighted(imgbak, 0.3, pred, 0.7, 0)
cv2.imwrite('./onnx/out_{}.jpg'.format(n), img)

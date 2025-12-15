#encoding=gbk
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

#路径配置
onnx_path=r'./onnx/best-smi.onnx'
pic_path=r'./onnx/11.jpg'
w,h=512,512

#调色板配置
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
palette[0]=[255,255,255]
palette[1]=[0,255,0]
palette[2]=[0,0,255]
palette[3]=[255,0,0]
palette[4]=[255,255,0]
palette[5]=[255,0,255]
palette[6]=[171,130,255]
palette[7]=[155,211,255]
palette[8]=[0,255,255]

mean=(120,114,104)
std=(70,69,73)

#onnx模型载入
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])


#输入图像预处理
img=Image.open(pic_path).resize((w,h))
imgbak=img.copy()
img=np.array(img).astype(np.float32)  # 注意输入type一定要np.float32
img-=mean                             #减均值
img/=std                              #除方差
img=np.array([np.transpose(img,(2,0,1))])

#模型推理
out = session.run(None,input_feed = { 'input' : img })
# out=np.argmax(out[0],axis=1)
out=out[0].astype('uint8').squeeze()
print(out.shape)



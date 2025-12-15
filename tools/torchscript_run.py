#encoding=gbk
import torch
import numpy as np
from PIL import Image
import cv2

#路径配置
pt_path=r'./torchscript/tlib.pt'
pic_path=r'./torchscript/11.jpg'
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

#pt模型载入
model=torch.jit.load(pt_path)
model.eval()


#输入图像预处理
img=cv2.imread(pic_path)
imgbak=img.copy()
img=img[:,:,::-1]
img=img.astype(np.float32)
img-=mean                             #减均值
img/=std                              #除方差
img=np.array([np.transpose(img,(2,0,1))])
img=torch.from_numpy(img)


#模型推理
out = model(img)
out=out.squeeze().detach().cpu().numpy()
out=out.astype('uint8')
pred= palette[out]


#保存图像
n=0
cv2.imwrite('./torchscript/mask_{}.jpg'.format(n), pred)

img1=np.array(imgbak)
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img=cv2.addWeighted(img1,0.3,pred,0.7,0)
cv2.imwrite('./torchscript/out_{}.jpg'.format(n), img)



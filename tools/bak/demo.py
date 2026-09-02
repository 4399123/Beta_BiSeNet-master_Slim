
import sys
sys.path.insert(0, '..')
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file


# uncomment the following line if you want to reduce cpu usage, see issue #231
#  torch.set_num_threads(4)

torch.set_grad_enabled(False)
np.random.seed(42)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='../configs/bisenetv2_catdog.py',)
parse.add_argument('--weight-path', type=str, default='../pt/best.pt',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./imgs/283.jpg',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.46962251, 0.4464104, 0.40718787),  # coco, rgb
    std=(0.27469736, 0.27012361, 0.28515933),
)
im = cv2.imread(args.img_path)[:, :, ::-1]
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

# shape divisor
org_size = im.size()[2:]
new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

# inference
im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
out = net(im)[0]
out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
out = out.argmax(dim=1)

# visualize
out = out.squeeze().detach().cpu().numpy()
pred = palette[out]
cv2.imwrite('./res.jpg', pred)

img1 = cv2.imread(args.img_path)
img2=cv2.imread('./res.jpg')

img=cv2.addWeighted(img1,0.3,img2,0.7,0)
cv2.imwrite('./out.jpg', img)

cv2.imshow('result.png',img)
cv2.waitKey(0)


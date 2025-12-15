
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
from imutils import paths
from tqdm import tqdm
import os
from utils_zl import replace_batchnorm
from PIL  import  Image
from timm.utils import reparameterize_model
# np.set_printoptions(threshold=np.inf)

# uncomment the following line if you want to reduce cpu usage, see issue #231
#  torch.set_num_threads(4)

torch.set_grad_enabled(False)
np.random.seed(123)

# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='../configs/topformer_blueface_efficientnet_lite1.py',)
parse.add_argument('--weight-path', type=str, default='../pt/X/topformer_efficientnet_lite1.pt',)
parse.add_argument('--imgs_path', default=r'./imgs',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)


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
palette[9]=[107,25,88]
palette[10]=[11,60,132]

if not os.path.exists('./out'):
    os.makedirs('./out')


# define model
net = model_factory[cfg.model_type](n_classes=cfg.n_cats, aux_mode='eval',use_fp16=False)
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
net.eval()
net=reparameterize_model(net)
# net.cuda()

# prepare data
to_tensor = T.ToTensor(
    # mean=(0.4, 0.4, 0.4),  # coco, rgb
    # std=(0.2, 0.2, 0.2),
    mean=(0.46962251, 0.4464104, 0.40718787),  # coco, rgb
    std=(0.27469736, 0.27012361, 0.28515933),
)

images=list(paths.list_images(args.imgs_path))
n=0
for image in tqdm(images):
    _,basename=os.path.split(image)
    name=basename.split('.')[0]
    im = cv2.imread(image)
    img1=im.copy()
    im=im[:,:,::-1]
    # im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)

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
    out=out.astype('uint8')
    out1=out.copy()
    out[out>0]=1

    if(len(np.unique(out))==1): cv2.imwrite('./out/out_{}.jpg'.format(name), img1)

    contours, hierarchy = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for k in range(len(contours)):
        point = contours[k][0][0]
        x = point[0]
        y = point[1]
        id = int(out1[y][x])
        color = palette[id]
        line=np.array([contours[k]])
        cv2.drawContours(img1, line, -1, color=(int(color[0]), int(color[1]), int(color[2])), thickness=3)

    cv2.imwrite('./out/out_{}.jpg'.format(name), img1)
    n+=1




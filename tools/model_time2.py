
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
import time


# uncomment the following line if you want to reduce cpu usage, see issue #231
#  torch.set_num_threads(4)

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='../configs/bisenetv1_blueface_convnext_small.py',)
parse.add_argument('--weight-path', type=str, default='../pt/best.pt',)
parse.add_argument('--imgs_path', default='./ceshi/2048',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
palette[0]=[255,255,255]

# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
net.eval()
# net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.46962251, 0.4464104, 0.40718787),  # coco, rgb
    std=(0.27469736, 0.27012361, 0.28515933),
)

images=list(paths.list_images(args.imgs_path))
m=1
netalltime=0
start=time.time()
for image in tqdm(images):
    im = cv2.imread(image)[:, :, ::-1]
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()

    # shape divisor
    org_size = im.size()[2:]
    new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

    # inference
    im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
    st=time.time()
    out = net(im)[0]
    ed=time.time()
    netalltime+=ed-st
    out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
    out = out.argmax(dim=1)
    m+=1
end=time.time()
print("前处理+模型推理+后处理：{}ms".format((end-start)/m*1000))
print('模型推理：{}ms'.format(netalltime/m*1000))



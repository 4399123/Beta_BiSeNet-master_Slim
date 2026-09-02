
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
from PIL  import  Image
# np.set_printoptions(threshold=np.inf)

# uncomment the following line if you want to reduce cpu usage, see issue #231
#  torch.set_num_threads(4)

torch.set_grad_enabled(False)
np.random.seed(123)



# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='../configs/bisenetv1_blueface_fbnetv3_g.py',)
parse.add_argument('--weight-path', type=str, default='../pt/X22/v1_blueface_fbnetv3_g.pt',)
parse.add_argument('--imgs_path', default=r'./imgs',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)

# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='pred')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
net.eval()
# net.cuda()

mean = (0.46962251, 0.4464104, 0.40718787)  # coco, rgb
std = (0.27469736, 0.27012361, 0.28515933)

batch_size=3
outpath=r'./output_bs3'
if not os.path.exists(outpath):
    os.makedirs(outpath)

images=list(paths.list_images(args.imgs_path))
picnum=len(images)

epochs=picnum//batch_size

n=0

for epoch in tqdm(range(epochs)):
    names=[]
    img_bs = []
    for i in range(batch_size):
        imgpath=images[n]
        basename = os.path.basename(imgpath)
        name=basename.split('.')[0]
        names.append(name)
        imgbak=cv2.imread(imgpath)
        img=imgbak[:,:,::-1]
        img=img.astype(np.float32)
        img/=255.0
        img-=mean
        img/=std
        img = np.transpose(img, (2, 0, 1))
        img_bs.append(img)
        n+=1
    img_bs = np.array(img_bs)
    ts_img_bs=torch.from_numpy(img_bs)

    out = net(ts_img_bs)
    out=out.cpu().numpy()

    for j in range(batch_size):
        outp = out[j].astype('uint8')
        outpicpath = os.path.join(outpath, '{}.png'.format(names[j]))
        outp[outp > 0] = 255
        cv2.imwrite(outpicpath, outp)





# for image in tqdm(images):
#     _,basename=os.path.split(image)
#     labelid=basename.split('_')
#     im = cv2.imread(image)[:, :, ::-1]
#     img1=im.copy()
#     # im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
#     im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)
#
#     # shape divisor
#     org_size = im.size()[2:]
#     new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]
#
#     # inference
#     im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
#     out = net(im)[0]
#     out = F.interpolate(out, size=org_size, align_corners=False, mode='bilinear')
#     out = out.argmax(dim=1)
#
#     # visualize
#     out = out.detach().cpu().numpy()






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
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
torch.set_grad_enabled(False)
np.random.seed(123)

# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='../configs/bisenetv1_blueface_efficientnet_b3.py',)
parse.add_argument('--weight-path', type=str, default='../pt/U11/v1_blueface_efficientnet_b3.pt',)
parse.add_argument('--imgs_path', default='./imgs',)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)

grayimgpath=r'./out_gray/'
mergeimgparh=r'./out_test_merge/'
cutimgpath=r'./out_test_cut/'
if not os.path.exists(grayimgpath):
    os.makedirs(grayimgpath)
if not os.path.exists(mergeimgparh):
    os.makedirs(mergeimgparh)
if not os.path.exists(cutimgpath):
    os.makedirs(cutimgpath)


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

images=list(paths.list_images(args.imgs_path))
m=0
for image in tqdm(images):
    _,basename=os.path.split(image)
    im = cv2.imread(image)[:, :, ::-1]
    img1=cv2.imread(image)
    img2=cv2.imread(image)
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
    # im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)

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
    # bi_imgpath='./out_gray/{}'.format(basename)
    bi_imgpath=os.path.join(grayimgpath,'{}'.format(basename))
    cv2.imwrite(bi_imgpath, out)

    bi_img=cv2.imread(bi_imgpath,0)
    bi_img[bi_img>0]=255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bi_img)
    # print(stats)
    n=0
    for st in stats[1:]:
        x1,y1,w,h,s=st[0],st[1],st[2],st[3],st[4]
        if(s<10): continue
        if(w<5 or h<5):continue
        x2=st[0]+st[2]
        y2=st[1]+st[3]

        x1=max(0,x1-10)
        y1=max(0,y1-10)
        x2=min(512,x2+10)
        y2=min(512,y2+10)

        cv2.rectangle(img1,(x1,y1),(x2,y2),(0,255,0),3)
        # cv2.imwrite('./out_test_merge/{}.jpg'.format(m),img1)
        cv2.imwrite(os.path.join(mergeimgparh,'{}.jpg'.format(m)),img1)

        patch_img=img2[y1:y2,x1:x2]
        # cv2.imwrite('./out_test_cut/{}_{}.jpg'.format(m,n), patch_img)
        cv2.imwrite(os.path.join(cutimgpath,'msa_{}_{}.jpg'.format(m,n)), patch_img)
        n+=1

    m+=1


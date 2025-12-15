
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

import lib.data.transform_cv2 as T
from lib.data.sampler import RepeatedDistSampler
import math

from lib.data.cityscapes_cv2 import CityScapes
from lib.data.coco import CocoStuff
from lib.data.ade20k import ADE20k
from lib.data.customer_dataset import CustomerDataset
from lib.data.catdog_dataset import CatDogDataset
from lib.data.coco80_dataset import COCO80Dataset
from lib.data.crack_dataset import CrackDataset
from lib.data.blueface_dataset import BlueFaceDataset




def get_data_loader(cfg, mode='train'):
    if mode == 'train':
        train_model=0
        if(train_model==0):
            trans_func = T.TransformationTrain(cfg.scales, cfg.cropsize)
            batchsize = cfg.ims_per_gpu
            annpath = cfg.train_im_anns
            shuffle = True
            drop_last = True
        elif(train_model==1):
            trans_func = T.TransformationTrain2(cfg.scales, cfg.cropsize)
            batchsize = cfg.ims_per_gpu
            annpath = cfg.train_im_anns
            shuffle = True
            drop_last = True
        else:
            trans_func = T.TransformationTrain3()
            batchsize = cfg.ims_per_gpu
            annpath = cfg.train_im_anns
            shuffle = True
            drop_last = True

    elif mode == 'val':
        trans_func = T.TransformationVal()
        batchsize = cfg.eval_ims_per_gpu
        annpath = cfg.val_im_anns
        shuffle = False
        drop_last = False

    ds = eval(cfg.dataset)(cfg.im_root, annpath, trans_func=trans_func, mode=mode)
    img_nums=len(ds)

    if dist.is_initialized():
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            # assert not cfg.max_iter is None
            # n_train_imgs = cfg.ims_per_gpu * dist.get_world_size() * cfg.max_iter
            n_train_imgs=math.ceil(img_nums/(cfg.ims_per_gpu * dist.get_world_size()))* cfg.ims_per_gpu * dist.get_world_size()
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=shuffle)
        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dl = DataLoader(
            ds,
            batch_sampler=batchsampler,
            num_workers=8,
            pin_memory=True,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=0,
            pin_memory=False,
        )
    return dl

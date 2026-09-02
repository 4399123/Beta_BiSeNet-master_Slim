
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

import lib.data.transform_cv2 as T
from lib.data.sampler import RepeatedDistSampler
import math
import random
import numpy as np

from lib.data.cityscapes_cv2 import CityScapes
from lib.data.coco import CocoStuff
from lib.data.ade20k import ADE20k
from lib.data.customer_dataset import CustomerDataset
from lib.data.catdog_dataset import CatDogDataset
from lib.data.coco80_dataset import COCO80Dataset
from lib.data.crack_dataset import CrackDataset
from lib.data.blueface_dataset import BlueFaceDataset


def worker_init_fn(worker_id):
    """Give every (rank, worker) pair an independent numpy/random stream.

    PyTorch only auto-seeds torch and Python's `random` per worker, NOT numpy.
    Since all augmentations in transform_cv2.py use np.random, without this the
    numpy random sequence is identical across all workers and all DDP ranks,
    which collapses augmentation diversity (worse as world_size grows).
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    base_seed = torch.initial_seed() % (2 ** 31)
    seed = (base_seed + rank * 10007 + worker_id) % (2 ** 31)
    np.random.seed(seed)
    random.seed(seed)




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
            # Standard DDP: total samples per epoch == dataset size, sharded
            # across ranks. Each rank processes ~img_nums / world_size samples,
            # so adding GPUs actually speeds up training. Round up to a multiple
            # of (ims_per_gpu * world_size) so every rank gets the same number
            # of full batches.
            world_size = dist.get_world_size()
            n_train_imgs = math.ceil(
                img_nums / (cfg.ims_per_gpu * world_size)
            ) * cfg.ims_per_gpu * world_size
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
            worker_init_fn=worker_init_fn,
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

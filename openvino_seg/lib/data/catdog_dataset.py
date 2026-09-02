#!/usr/bin/python
# -*- encoding: utf-8 -*-


import lib.data.transform_cv2 as T
from lib.data.base_dataset import BaseDataset


class CatDogDataset(BaseDataset):

    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(CatDogDataset, self).__init__(dataroot, annpath, trans_func, mode)
        self.lb_ignore = 255

        self.to_tensor = T.ToTensor(
            mean=(0.46962251, 0.4464104,  0.40718787), # coco, rgb
            std=(0.27469736, 0.27012361, 0.28515933),
        )



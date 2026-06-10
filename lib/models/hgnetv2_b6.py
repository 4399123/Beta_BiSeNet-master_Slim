#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import timm
from timm.models import load_checkpoint

class HGNetV2_B6(nn.Module):
    def __init__(self):
        super(HGNetV2_B6, self).__init__()
        self.out_indices = [1, 2, 3]
        self.selected_feature_extractor = timm.create_model('hgnetv2_b6.ssld_stage2_ft_in1k', features_only=True, out_indices=self.out_indices, pretrained=False)
        try:
            load_checkpoint(self.selected_feature_extractor, '../lib/premodels/hgnetv2_b6.pth', remap=True)
        except FileNotFoundError:
            load_checkpoint(self.selected_feature_extractor, '../premodels/hgnetv2_b6.pth', remap=True)

    def forward(self, x):
        x=self.selected_feature_extractor(x)
        feat8 =x[0] # 1/8
        feat16 = x[1] # 1/16
        feat32 = x[2] # 1/32
        return feat8, feat16, feat32

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


if __name__ == "__main__":
    net = HGNetV2_B6()
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    print(out[0].size())
    print(out[1].size())
    print(out[2].size())
    net.get_params()

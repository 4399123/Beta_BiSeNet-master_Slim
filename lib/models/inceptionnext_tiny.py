#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import timm
from timm.models import load_checkpoint


class InceptionNeXt_Tiny(nn.Module):
    def __init__(self):
        super(InceptionNeXt_Tiny, self).__init__()
        self.out_indices = [1, 2, 3]
        self.selected_feature_extractor = timm.create_model(
            'inception_next_tiny.sail_in1k',
            features_only=True,
            out_indices=self.out_indices,
            pretrained=False,
        )
        try:
            load_checkpoint(self.selected_feature_extractor, '../lib/premodels/inceptionnext_tiny.pth', remap=True)
        except Exception:
            load_checkpoint(self.selected_feature_extractor, '../premodels/inceptionnext_tiny.pth', remap=True)

    def forward(self, x):
        x = self.selected_feature_extractor(x)
        feat8 = x[0]   # 1/8
        feat16 = x[1]  # 1/16
        feat32 = x[2]  # 1/32
        return feat8, feat16, feat32



if __name__ == "__main__":
    net = InceptionNeXt_Tiny()
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    print(out[0].size())
    print(out[1].size())
    print(out[2].size())

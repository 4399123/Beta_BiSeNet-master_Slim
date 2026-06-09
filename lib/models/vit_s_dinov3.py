#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import os
import timm
from timm.models import load_checkpoint
print(timm.__file__)  # 打印路径验证，会显示你工程里的 timm 路径

class Vit_S_DinoV3(nn.Module):
    """
    ViT-Small DINOv3 backbone，输出三个尺度的特征图。

    ViT 的 patch size=16，所有 transformer block 输出均为原图 1/16 分辨率，
    无法像 CNN 那样原生得到 1/8 和 1/32 的层级特征。
    本实现的策略：
      - feat8  (1/8)  : 取浅层 block 的 1/16 特征 → 双线性上采样 ×2
      - feat16 (1/16) : 取中层 block 的 1/16 特征，直接使用
      - feat32 (1/32) : 取深层 block 的 1/16 特征 → 步长2最大池化下采样

    out_indices 对应 timm ViT features_only 的三个抽头位置（0-indexed block编号），
    默认取 [2, 5, 11]，即浅/中/深三层，适配 vit_small 的12个 block。
    """

    @staticmethod
    def _remap_keys(state_dict, model):
        """
        权重文件 key 无 'model.' 前缀（原始 ViT 导出），
        但 features_only=True 时 timm 将模型包在 FeatureGetterNet.model 里，
        需要统一加上 'model.' 前缀才能匹配。
        """
        remapped = {}
        for k, v in state_dict.items():
            new_k = f'model.{k}' if not k.startswith('model.') else k
            remapped[new_k] = v
        return remapped

    def __init__(self, out_indices=(2, 5, 11)):
        super(Vit_S_DinoV3, self).__init__()
        self.out_indices = list(out_indices)
        self.selected_feature_extractor = timm.create_model(
            'vit_small_patch16_dinov3.lvd_1689m',
            features_only=True,
            out_indices=self.out_indices,
            pretrained=False,
        )

        # features_only 模型不支持 remap（按顺序对齐），必须用 filter_fn 做 key 重映射
        ckpt_paths = [
            '../lib/premodels/vit_s_dinov3.pth',
            '../premodels/vit_s_dinov3.pth',
        ]
        for path in ckpt_paths:
            if os.path.isfile(path):
                load_checkpoint(
                    self.selected_feature_extractor, path,
                    strict=False, remap=False,
                    filter_fn=self._remap_keys,
                )
                break

        embed_dim = self.selected_feature_extractor.model.embed_dim  # vit_small: 384

        # 将浅层 1/16 特征上采样 ×2，用轻量卷积对齐通道，得到 1/8 特征
        self.up_to_8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
        )

        # 将深层 1/16 特征下采样 ×2，得到 1/32 特征
        self.down_to_32 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim * 2),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = self.selected_feature_extractor(x)
        # feats[0]: 浅层 1/16,  feats[1]: 中层 1/16,  feats[2]: 深层 1/16
        feat_shallow, feat_mid, feat_deep = feats[0], feats[1], feats[2]

        feat8  = self.up_to_8(feat_shallow)   # 1/8,  通道 embed_dim//2
        feat16 = feat_mid                      # 1/16, 通道 embed_dim
        feat32 = self.down_to_32(feat_deep)    # 1/32, 通道 embed_dim*2

        return feat8, feat16, feat32


if __name__ == "__main__":
    net = Vit_S_DinoV3()
    net.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = net(x)
    # 期望输出（输入224×224）:
    #   feat8  : (2, 192, 28, 28)
    #   feat16 : (2, 384, 14, 14)
    #   feat32 : (2, 768,  7,  7)
    print('feat8  :', out[0].size())
    print('feat16 :', out[1].size())
    print('feat32 :', out[2].size())

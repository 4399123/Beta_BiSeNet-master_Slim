#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import os
import timm
from timm.models import load_checkpoint
print(timm.__file__)  # 打印路径验证，会显示你工程里的 timm 路径

class Vit_B_DinoV3(nn.Module):
    """
    ViT-Base DINOv3 backbone，支持三种输出模式，由 out_indices 长度自动判断。

    模型：vit_base_patch16_dinov3，patch size=16，embed_dim=768，共 12 个 transformer block。
    所有 block 输出均为原图 1/16 分辨率，无法像 CNN 那样原生得到多尺度层级特征。

    【单尺度模式】len(out_indices) == 1，例如 (11,)：
      - feat16 (1/16) : 指定 block 的原始 patch token，通道 768
      - up_to_8 / down_to_32 分支不构建，零多余计算
      返回：feat16  （单个 Tensor）

    【双尺度模式】len(out_indices) == 2，例如 (3, 11)：
      - feat_shallow (1/16) : 浅层 block 的原始 patch token，通道 768（空间细节丰富）
      - feat_deep    (1/16) : 深层 block 的原始 patch token，通道 768（语义最强）
      - up_to_8 / down_to_32 分支不构建，零多余计算
      返回：(feat_shallow, feat_deep)

    【三尺度模式】len(out_indices) == 3，默认 (3, 6, 11)：
      - feat8  (1/8)  : 浅层 block → 双线性上采样 ×2，通道 384
      - feat16 (1/16) : 中层 block → 直接使用，通道 768
      - feat32 (1/32) : 深层 block → 步长2卷积下采样，通道 1536
      返回：(feat8, feat16, feat32)
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

    def __init__(self, out_indices=(3, 6, 11)):
        super(Vit_B_DinoV3, self).__init__()
        self.out_indices = list(out_indices)
        n = len(self.out_indices)
        assert n in (1, 2, 3), "out_indices 长度须为 1（单尺度）、2（双尺度）或 3（三尺度）"
        self.single_scale = (n == 1)
        self.dual_scale   = (n == 2)
        self.triple_scale = (n == 3)

        self.selected_feature_extractor = timm.create_model(
            'vit_base_patch16_dinov3.lvd_1689m',
            features_only=True,
            out_indices=self.out_indices,
            pretrained=False,
        )

        # features_only 模型不支持 remap（按顺序对齐），必须用 filter_fn 做 key 重映射
        ckpt_paths = [
            '../lib/premodels/vit_b_dinov3.pth',
            '../premodels/vit_b_dinov3.pth',
        ]
        for path in ckpt_paths:
            if os.path.isfile(path):
                load_checkpoint(
                    self.selected_feature_extractor, path,
                    strict=False, remap=False,
                    filter_fn=self._remap_keys,
                )
                break

        embed_dim = self.selected_feature_extractor.model.embed_dim  # vit_base: 768

        if self.triple_scale:
            # 【三尺度模式】构建 up/down 分支
            # 将浅层 1/16 特征上采样 ×2，用轻量卷积压缩通道，得到 1/8 特征（384ch）
            self.up_to_8 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim // 2),
                nn.ReLU(inplace=True),
            )
            # 将深层 1/16 特征步长2卷积下采样，得到 1/32 特征（1536ch）
            self.down_to_32 = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim * 2),
                nn.ReLU(inplace=True),
            )
        # 【单尺度 / 双尺度模式】不构建任何额外分支，零多余参数

    def forward(self, x):
        feats = self.selected_feature_extractor(x)

        if self.single_scale:
            # 【单尺度模式】直接返回唯一抽头的 patch token（1/16, 768ch）
            return feats[0]

        if self.dual_scale:
            # 【双尺度模式】返回 (浅层 patch token, 深层 patch token)，均为 (1/16, 768ch)
            return feats[0], feats[1]

        # 【三尺度模式】
        # feats[0]: 浅层 1/16,  feats[1]: 中层 1/16,  feats[2]: 深层 1/16
        feat_shallow, feat_mid, feat_deep = feats[0], feats[1], feats[2]

        feat8  = self.up_to_8(feat_shallow)   # 1/8,  通道 embed_dim//2 (384)
        feat16 = feat_mid                      # 1/16, 通道 embed_dim    (768)
        feat32 = self.down_to_32(feat_deep)    # 1/32, 通道 embed_dim*2  (1536)

        return feat8, feat16, feat32


if __name__ == "__main__":
    net_3scale = Vit_B_DinoV3()                    # 三尺度（默认）
    net_2scale = Vit_B_DinoV3(out_indices=(3, 11)) # 双尺度（skip 用）
    net_1scale = Vit_B_DinoV3(out_indices=(11,))   # 单尺度

    configs = [
        (net_3scale, '三尺度', ['feat8', 'feat16', 'feat32']),
        (net_2scale, '双尺度', ['feat_shallow', 'feat_deep']),
        (net_1scale, '单尺度', ['feat16']),
    ]
    x = torch.randn(2, 3, 512, 512)
    for net, label, names in configs:
        net.eval()
        with torch.no_grad():
            out = net(x)
        print(f'── {label} ──')
        if isinstance(out, tuple):
            for name, t in zip(names, out):
                print(f'  {name}: {tuple(t.shape)}')
        else:
            print(f'  {names[0]}: {tuple(out.shape)}')
    # 期望输出（输入 512×512）:
    # ── 三尺度 ──
    #   feat8  : (2, 384,  64, 64)
    #   feat16 : (2, 768,  32, 32)
    #   feat32 : (2, 1536, 16, 16)
    # ── 双尺度 ──
    #   feat_shallow : (2, 768, 32, 32)
    #   feat_deep    : (2, 768, 32, 32)
    # ── 单尺度 ──
    #   feat16 : (2, 768, 32, 32)

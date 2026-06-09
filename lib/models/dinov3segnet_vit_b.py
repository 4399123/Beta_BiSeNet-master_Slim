#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
DinoV3SegNet_Vit_B — 单路 + 单层 skip 语义分割网络。

设计原则：
  - 主路径：ViT-Base block 11（最深语义层），经 SPPM 全局上下文增强。
  - Skip 路径：ViT-Base block 3（浅层，空间细节更丰富），轻量投影后在
    第一步上采样（1/16 → 1/8）前融入主路径，补充边缘细节。
  - 两个抽头都在同一分辨率（1/16），无额外分辨率变换，参数增量极小。
  - 无深度监督辅助头，结构简洁。

通道流（输入 H×W）：
  backbone (block 3)  → feat_shallow (B, 768, H/16, W/16)
  backbone (block 11) → feat_deep    (B, 768, H/16, W/16)

  feat_deep    → proj_deep (256) → SPPM → ctx      (B, 256, H/16, W/16)
  feat_shallow → proj_skip (128)         → skip     (B, 128, H/16, W/16)

  ctx (256) + skip (128) → skip_fuse conv → fused   (B, 256, H/16, W/16)
      │  ×2 上采样 + ConvBNReLU
      ▼  (B, 256, H/8, W/8)
      │  ×2 上采样 + ConvBNReLU
      ▼  (B, 128, H/4, W/4)
      │  ×4 上采样 + ConvBNReLU
      ▼  (B, 64,  H,   W)
      │  Dropout + Conv 1×1
      ▼  logits (B, n_classes, H, W)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.cuda.amp import autocast
from .vit_b_dinov3 import Vit_B_DinoV3


# ─────────────────────────────────────────────
# 基础模块
# ─────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """标准 Conv-BN-ReLU"""

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.bn   = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ─────────────────────────────────────────────
# TRT 静态池化
# ─────────────────────────────────────────────

class TRT_FixedAvgPool2d(nn.Module):
    """
    TensorRT 友好的静态平均池化。
    初始化时固定 kernel/stride，避免 adaptive_avg_pool 在 TRT 中动态解析失败。
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        if output_size in ((1, 1), 1):
            self.is_global = True
            self.pool = None
        else:
            self.is_global = False
            if isinstance(output_size, int):
                output_size = (output_size, output_size)
            if isinstance(input_size, int):
                input_size = (input_size, input_size)
            stride_h = input_size[0] // output_size[0]
            stride_w = input_size[1] // output_size[1]
            kernel_h = input_size[0] - (output_size[0] - 1) * stride_h
            kernel_w = input_size[1] - (output_size[1] - 1) * stride_w
            self.pool = nn.AvgPool2d(
                kernel_size=(kernel_h, kernel_w),
                stride=(stride_h, stride_w),
                padding=0,
            )

    def forward(self, x):
        return x.mean(dim=(2, 3), keepdim=True) if self.is_global else self.pool(x)


# ─────────────────────────────────────────────
# SPPM
# ─────────────────────────────────────────────

class SPPM_TRT(nn.Module):
    """
    Simple Pyramid Pooling Module（TRT 静态版）。
    在 1/16 特征图上聚合多尺度全局上下文，
    充分激活 DINOv3 block 11 的大感受野语义。

    流程：多尺度 FixedAvgPool → 1×1 降维 → 上采样 → Concat → 1×1 融合
    """

    def __init__(self, in_channels, out_channels,
                 pool_sizes=(1, 5, 9, 13), input_feat_shape=(32, 32)):
        super().__init__()
        hidden = in_channels // 4
        self.stages = nn.ModuleList([
            nn.Sequential(
                TRT_FixedAvgPool2d(input_size=input_feat_shape, output_size=s),
                nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            )
            for s in pool_sizes
        ])
        concat_channels = in_channels + len(pool_sizes) * hidden
        self.conv_out = ConvBNReLU(concat_channels, out_channels, ks=1, padding=0)

    def forward(self, x):
        h, w = x.shape[2:]
        priors = [x]
        for stage in self.stages:
            feat = stage(x)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            priors.append(feat)
        return self.conv_out(torch.cat(priors, dim=1))


# ─────────────────────────────────────────────
# 渐进式上采样解码头
# ─────────────────────────────────────────────

class ProgressiveUpsampleHead(nn.Module):
    """
    三步渐进上采样头：1/16 → 1/8 → 1/4 → 原图。

    分步上采样（×2, ×2, ×4）比一次 ×16 插值边缘质量更好，
    每步配合 ConvBNReLU 做特征精炼，控制通道数逐步收窄。

    通道变化：256 → 256(×2) → 128(×2) → 64(×4) → n_classes
    """

    def __init__(self, in_chan, n_classes):
        super().__init__()
        # 1/16 → 1/8
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(in_chan, 256, ks=3, padding=1),
        )
        # 1/8 → 1/4
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(256, 128, ks=3, padding=1),
        )
        # 1/4 → 原图
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            ConvBNReLU(128, 64, ks=3, padding=1),
        )
        self.dropout  = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(64, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.dropout(x)
        return self.conv_out(x)


# ─────────────────────────────────────────────
# 主网络
# ─────────────────────────────────────────────

class DinoV3SegNet_Vit_B(nn.Module):
    """
    DinoV3SegNet_Vit_B — 单路 + 单层 skip ViT-Base DINOv3 语义分割网络。

    backbone 双尺度模式 out_indices=(3, 11)：
      - block 3  → feat_shallow (768, 1/16)：空间细节较丰富，作为 skip
      - block 11 → feat_deep    (768, 1/16)：语义最强，作为主路径

    Args:
        n_classes (int): 分割类别数。
        aux_mode  (str): 'train' | 'eval' | 'pred'。
        use_fp16  (bool): 是否启用混合精度。
        img_size  (tuple): (H, W)，须为 32 的倍数，用于 SPPM 静态池化参数。
    """

    EMBED_DIM  = 768   # ViT-Base embed_dim
    INNER_DIM  = 256   # 主路径内部维度
    SKIP_DIM   = 128   # skip 投影维度

    def __init__(self, n_classes, aux_mode='train', use_fp16=False,
                 img_size=(512, 512)):
        super().__init__()
        self.use_fp16 = use_fp16
        self.aux_mode = aux_mode
        self.img_size = img_size
        D = self.INNER_DIM
        S = self.SKIP_DIM

        # 1. Backbone：双尺度模式，block 3（浅层 skip）+ block 11（深层主路径）
        #    两个抽头均在 1/16 分辨率，768ch，forward 返回 (feat_shallow, feat_deep)
        self.backbone = Vit_B_DinoV3(out_indices=(3, 11))

        # 2. 主路径投影：深层语义 768 → 256
        self.proj_deep = ConvBNReLU(self.EMBED_DIM, D, ks=1, padding=0)

        # 3. Skip 路径投影：浅层空间 768 → 128（轻量，避免喧宾夺主）
        self.proj_skip = ConvBNReLU(self.EMBED_DIM, S, ks=1, padding=0)

        # 4. Skip 融合：将主路径 (256ch) 与 skip (128ch) 在通道维拼接后降回 256
        #    均在 1/16 分辨率，无需插值
        self.skip_fuse = ConvBNReLU(D + S, D, ks=3, padding=1)

        # 5. SPPM：在 1/16 分辨率上聚合全局上下文（接在 proj_deep 之后、skip 融合之前）
        sppm_h = img_size[0] // 16
        sppm_w = img_size[1] // 16
        self.sppm = SPPM_TRT(
            in_channels=D, out_channels=D,
            pool_sizes=(1, 5, 9, 13),
            input_feat_shape=(sppm_h, sppm_w),
        )

        # 6. 渐进式上采样解码头（入口已是融合后的 256ch）
        self.head = ProgressiveUpsampleHead(D, n_classes)

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            # ── Backbone（双尺度：block 3 + block 11）──────────
            feat_shallow, feat_deep = self.backbone(x)
            # feat_shallow: (B, 768, H/16, W/16)  ← block 3,  空间细节
            # feat_deep   : (B, 768, H/16, W/16)  ← block 11, 语义最强

            # ── 主路径：深层语义投影 + 全局上下文增强 ──────────
            ctx  = self.proj_deep(feat_deep)   # (B, 256, H/16, W/16)
            ctx  = self.sppm(ctx)              # (B, 256, H/16, W/16)

            # ── Skip 路径：浅层空间投影 ───────────────────────
            skip = self.proj_skip(feat_shallow)  # (B, 128, H/16, W/16)

            # ── Skip 融合：concat → conv，均在 1/16，无需插值 ──
            fused = self.skip_fuse(torch.cat([ctx, skip], dim=1))  # (B, 256, H/16, W/16)

            # ── 渐进上采样 → logits ──────────────────────────
            logits = self.head(fused)   # (B, n_classes, H, W)

            if self.aux_mode == 'train':
                return logits,   # tuple，与训练框架接口一致

            elif self.aux_mode == 'eval':
                return logits,

            elif self.aux_mode == 'pred':
                return torch.argmax(logits, dim=1).float()

            else:
                raise NotImplementedError(f"Unknown aux_mode: {self.aux_mode}")


# ─────────────────────────────────────────────
# 快速验证
# ─────────────────────────────────────────────

if __name__ == "__main__":
    img_h, img_w = 512, 512
    n_classes = 19

    try:
        print(f"Initializing DinoV3SegNet_Vit_B (single skip)  {img_h}x{img_w} ...")
        net = DinoV3SegNet_Vit_B(
            n_classes=n_classes, aux_mode='train',
            img_size=(img_h, img_w),
        )
        net.train()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net.to(device)
        in_ten = torch.randn(2, 3, img_h, img_w, device=device)

        print("Running forward pass ...")
        out, = net(in_ten)

        print(f"\nInput : {tuple(in_ten.shape)}")
        print(f"Output: {tuple(out.shape)}")

        assert out.shape[2:] == (img_h, img_w), "Output shape mismatch!"
        print("\nSuccess: output shape matches input perfectly.")

        # 统计可训练参数量
        total = sum(p.numel() for p in net.parameters())
        trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Total params    : {total / 1e6:.1f} M")
        print(f"Trainable params: {trainable / 1e6:.1f} M")

    except Exception:
        import traceback
        traceback.print_exc()

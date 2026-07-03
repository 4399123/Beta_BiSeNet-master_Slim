"""
FastEfficientBiSeNet + InceptionNeXt-Tiny (Pro 版本)

针对原版的渐进式改造，重点：精度优先，保持 TensorRT 友好。
1) 加入 1/4 (stage0) 特征参与解码
2) Head 渐进上采样：1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/1，最后只做一次 4x bilinear
3) UAFM 升级为 SP-CA 版（双池化 channel attention + 通道维 mean/max spatial attention）
4) SPPM 替换为 DAPPM (DDRNet) 的 TRT 友好实现
5) Aux head 改接 backbone 原始 feat16/feat32 (而不是 projected/sppm 后)

不修改 fastefficientbisenet_inceptionnext_tiny.py。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.cuda.amp import autocast

import timm
from timm.models import load_checkpoint

# 复用已有的基础组件，避免重复定义
from .fastefficientbisenet_inceptionnext_tiny import (
    ConvBNReLU,
    TRT_FixedAvgPool2d,
    SegmentationHead,
)


# ---------------------------------------------------------------------------
# 1) 4-stage Backbone (本地包装，不修改 inceptionnext_tiny.py)
# ---------------------------------------------------------------------------
class _InceptionNeXt_Tiny_4Stage(nn.Module):
    """返回 stride 4/8/16/32 四级特征 (通道 96/192/384/768)。"""

    def __init__(self):
        super().__init__()
        self.out_indices = [0, 1, 2, 3]
        self.selected_feature_extractor = timm.create_model(
            'inception_next_tiny.sail_in1k',
            features_only=True,
            out_indices=self.out_indices,
            pretrained=False,
        )
        try:
            load_checkpoint(self.selected_feature_extractor,
                            '../lib/premodels/inceptionnext_tiny.pth', remap=True)
        except Exception:
            load_checkpoint(self.selected_feature_extractor,
                            '../premodels/inceptionnext_tiny.pth', remap=True)

    def forward(self, x):
        feats = self.selected_feature_extractor(x)
        feat4 = feats[0]   # 1/4 , 96
        feat8 = feats[1]   # 1/8 , 192
        feat16 = feats[2]  # 1/16, 384
        feat32 = feats[3]  # 1/32, 768
        return feat4, feat8, feat16, feat32


# ---------------------------------------------------------------------------
# 3) UAFM-SP-CA (PP-LiteSeg style)
# ---------------------------------------------------------------------------
class UAFM_SP_CA(nn.Module):
    """
    Unified Attention Fusion Module with Spatial + Channel attention.

    - Channel: 对 high 与 low 分别做 avg/max 全局池化, 拼接 4C -> MLP -> C, sigmoid
    - Spatial: 对 high 与 low 沿通道做 mean/max, 拼接 4 通道 -> 2 层 3x3 conv -> 1, sigmoid
    - 融合: out = high * (ch*sp) + low * (1 - ch*sp)  (PP-LiteSeg 风格门控)
    """

    def __init__(self, high_chan, low_chan, out_chan):
        super().__init__()
        self.conv_high = ConvBNReLU(high_chan, out_chan, ks=1, padding=0)
        self.conv_low = ConvBNReLU(low_chan, out_chan, ks=1, padding=0)

        # Channel attention
        self.ch_attn = nn.Sequential(
            nn.Conv2d(4 * out_chan, out_chan // 2, kernel_size=1, bias=False),
            BatchNorm2d(out_chan // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan // 2, out_chan, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Spatial attention
        self.sp_attn = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

        self.conv_out = ConvBNReLU(out_chan, out_chan, ks=3, padding=1)

    @staticmethod
    def _ch_stats(t):
        # (B, C, 1, 1) avg / max
        avg = F.adaptive_avg_pool2d(t, 1)
        mx = F.adaptive_max_pool2d(t, 1)
        return avg, mx

    @staticmethod
    def _sp_stats(t):
        # (B, 1, H, W) mean / max along channel
        mean = t.mean(dim=1, keepdim=True)
        mx, _ = t.max(dim=1, keepdim=True)
        return mean, mx

    def forward(self, x_high, x_low):
        high = self.conv_high(x_high)
        low = self.conv_low(x_low)

        high = F.interpolate(high, size=low.size()[2:],
                             mode='bilinear', align_corners=False)

        # Channel attention
        avg_h, max_h = self._ch_stats(high)
        avg_l, max_l = self._ch_stats(low)
        ch = self.ch_attn(torch.cat([avg_h, max_h, avg_l, max_l], dim=1))

        # Spatial attention
        mean_h, mxh = self._sp_stats(high)
        mean_l, mxl = self._sp_stats(low)
        sp = self.sp_attn(torch.cat([mean_h, mxh, mean_l, mxl], dim=1))

        atten = ch * sp  # broadcast: (B, C, H, W)
        out = high * atten + low * (1.0 - atten)
        return self.conv_out(out)


# ---------------------------------------------------------------------------
# 4) DAPPM (DDRNet) - TRT 友好实现
# ---------------------------------------------------------------------------
class DAPPM_TRT(nn.Module):
    """
    Deep Aggregation Pyramid Pooling Module，五分支级联：
        scale0: 1x1
        scale1: pool to (H/2, W/2) + 1x1
        scale2: pool to (H/4, W/4) + 1x1
        scale3: pool to (H/8, W/8) + 1x1
        scale4: global pool + 1x1
    级联: p_i = process_i(upsample(scale_i) + p_{i-1})
    输出: compression(concat([p0..p4])) + shortcut(x)
    """

    def __init__(self, in_channels, out_channels,
                 input_feat_shape=(20, 20), mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        H, W = input_feat_shape

        def _safe(t):
            return (max(1, t[0]), max(1, t[1]))

        s1 = _safe((H // 2, W // 2))
        s2 = _safe((H // 4, W // 4))
        s3 = _safe((H // 8, W // 8))

        def _fixed_pool(out_size):
            # 全局分支若走 x.mean(dim=(2,3)) 会导出为 ONNX ReduceMean，
            # TensorRT 解析时会丢失通道维，导致后续 BatchNorm 报
            # "shift weights has count C but 1 was expected"。
            # 这里改用满核 AvgPool2d，保持为标准 Pooling 节点，通道维正确。
            if out_size == 1 or out_size == (1, 1):
                return nn.AvgPool2d(kernel_size=(H, W))
            return TRT_FixedAvgPool2d(input_size=(H, W), output_size=out_size)

        def _branch_with_pool(out_size):
            return nn.Sequential(
                _fixed_pool(out_size),
                BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            )

        # scale0: 无下采样, BN+ReLU+1x1
        self.scale0 = nn.Sequential(
            BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
        )
        self.scale1 = _branch_with_pool(s1)
        self.scale2 = _branch_with_pool(s2)
        self.scale3 = _branch_with_pool(s3)
        self.scale4 = _branch_with_pool(1)  # global

        def _make_process():
            return nn.Sequential(
                BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels,
                          kernel_size=3, padding=1, bias=False),
            )

        self.process1 = _make_process()
        self.process2 = _make_process()
        self.process3 = _make_process()
        self.process4 = _make_process()

        self.compression = nn.Sequential(
            BatchNorm2d(mid_channels * 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels * 5, out_channels, kernel_size=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        H, W = x.shape[2:]

        x0 = self.scale0(x)

        def _up(t):
            return F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)

        x1 = _up(self.scale1(x))
        x2 = _up(self.scale2(x))
        x3 = _up(self.scale3(x))
        x4 = _up(self.scale4(x))

        p1 = self.process1(x1 + x0)
        p2 = self.process2(x2 + p1)
        p3 = self.process3(x3 + p2)
        p4 = self.process4(x4 + p3)

        out = self.compression(torch.cat([x0, p1, p2, p3, p4], dim=1)) + self.shortcut(x)
        return out


# ---------------------------------------------------------------------------
# 主模型：精度向 Pro 版
# ---------------------------------------------------------------------------
class FastEfficientBiSeNet_InceptionNeXt_Tiny_Pro(nn.Module):
    def __init__(self, n_classes, aux_mode='train', use_fp16=False,
                 img_size=(512, 512)):
        super().__init__()
        self.use_fp16 = use_fp16
        self.aux_mode = aux_mode
        self.img_size = img_size

        # 1) 4-stage backbone
        self.backbone = _InceptionNeXt_Tiny_4Stage()

        # Backbone 通道
        self.c2_chan = 96    # 1/4
        self.c3_chan = 192   # 1/8
        self.c4_chan = 384   # 1/16
        self.c5_chan = 768   # 1/32

        # 解码主通道
        self.dec_chan = 192

        # 投影到统一通道
        self.proj_c5 = ConvBNReLU(self.c5_chan, self.dec_chan, ks=1, padding=0)
        self.proj_c4 = ConvBNReLU(self.c4_chan, self.dec_chan, ks=1, padding=0)
        self.proj_c3 = ConvBNReLU(self.c3_chan, self.dec_chan, ks=1, padding=0)
        self.proj_c2 = ConvBNReLU(self.c2_chan, self.dec_chan, ks=1, padding=0)

        # 4) DAPPM 替换 SPPM
        sppm_feat_h = img_size[0] // 32
        sppm_feat_w = img_size[1] // 32
        self.dappm = DAPPM_TRT(
            in_channels=self.dec_chan,
            out_channels=self.dec_chan,
            input_feat_shape=(sppm_feat_h, sppm_feat_w),
        )

        # 2) + 3) 渐进上采样 + UAFM-SP-CA: 1/16 -> 1/8 -> 1/4
        self.fuse_c4 = UAFM_SP_CA(self.dec_chan, self.dec_chan, self.dec_chan)  # 与 c4 融合 -> 1/16
        self.fuse_c3 = UAFM_SP_CA(self.dec_chan, self.dec_chan, self.dec_chan)  # 与 c3 融合 -> 1/8
        self.fuse_c2 = UAFM_SP_CA(self.dec_chan, self.dec_chan, self.dec_chan)  # 与 c2 融合 -> 1/4

        # 输出头：在 1/4 上做最后一次 4x bilinear，比一次 8x 上采样更精细
        self.head = SegmentationHead(self.dec_chan, n_classes, scale_factor=4)

        # 5) Aux head 直接接 backbone 原始特征
        if self.aux_mode == 'train':
            self.aux_head_c4 = SegmentationHead(self.c4_chan, n_classes, scale_factor=16)
            self.aux_head_c5 = SegmentationHead(self.c5_chan, n_classes, scale_factor=32)

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            H, W = x.size()[2:]

            # Encoder (4 stages)
            feat4, feat8, feat16, feat32 = self.backbone(x)

            # Projections to dec_chan
            c5 = self.proj_c5(feat32)
            c4 = self.proj_c4(feat16)
            c3 = self.proj_c3(feat8)
            c2 = self.proj_c2(feat4)

            # Multi-scale context
            c5_ctx = self.dappm(c5)

            # Progressive top-down fusion
            f16 = self.fuse_c4(c5_ctx, c4)  # 1/16
            f8  = self.fuse_c3(f16, c3)     # 1/8
            f4  = self.fuse_c2(f8, c2)      # 1/4

            # Final head: 1/4 -> 1/1
            logits = self.head(f4)

            if self.aux_mode == 'train':
                # 直接接 backbone 原始特征作为更强的中间监督
                aux_out1 = self.aux_head_c4(feat16)
                aux_out2 = self.aux_head_c5(feat32)
                return logits, aux_out1, aux_out2

            elif self.aux_mode == 'eval':
                return logits,

            elif self.aux_mode == 'pred':
                pred = torch.argmax(logits, dim=1)
                return pred.float()  # trt11,不能使用 float(),否则会报错
            else:
                raise NotImplementedError


if __name__ == "__main__":
    # 关键：保证 img_height, img_width 是 32 的倍数
    img_height, img_width = 640, 640
    n_classes = 19

    try:
        print(f"Initializing PRO model with image size: {img_height}x{img_width}...")
        net = FastEfficientBiSeNet_InceptionNeXt_Tiny_Pro(
            n_classes=n_classes, aux_mode='train',
            img_size=(img_height, img_width),
        )
        net.train()

        if torch.cuda.is_available():
            net.cuda()
            in_ten = torch.randn(2, 3, img_height, img_width).cuda()
        else:
            in_ten = torch.randn(2, 3, img_height, img_width)

        print("Running Forward Pass...")
        out, aux1, aux2 = net(in_ten)

        print("\nResults:")
        print(f"Input:  {in_ten.shape}")
        print(f"Output: {out.shape}")
        print(f"Aux1:   {aux1.shape}")
        print(f"Aux2:   {aux2.shape}")

        assert out.shape[2:] == (img_height, img_width), "Output shape mismatch!"
        print("\nSuccess: Output shape matches Input shape perfectly.")

        # 参数量 & FLOPs 粗略统计
        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"\nTrainable params: {n_params/1e6:.2f}M")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError: {e}")

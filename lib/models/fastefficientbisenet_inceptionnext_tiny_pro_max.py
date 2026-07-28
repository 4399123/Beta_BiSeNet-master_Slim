"""
FastEfficientBiSeNet + InceptionNeXt-Tiny (Pro Max)

===============================================================================
版本历史 / Change Log
===============================================================================

[v1.0 初版 Pro Max] (已废弃)
    Transformer 做 1/32 上下文 (小数据集难收敛, 精度头号回退原因),
    UAFM 双 sigmoid 相乘门控 (a~0.25 太糊), Head 64->48 (太窄),
    RepConvBNReLU, Aux 补 1/8 一档。

[v1.1 精度回退修订] (已废弃)
    修订 1: 去 Transformer, 改 DAPPM + MSCA 串联。
    修订 2: UAFM 改回凸组合门 out = high*a + low*(1-a)。
    修订 3: 输出头加宽 128->96。
    加固 1: 全模块 kaiming_normal fan_out 初始化。
    加固 2: 解码锥形 256->192->128->128, 1/16 后加 MSCA。

[v2.0 颠覆式重构 (当前版本, 方案A)]

    动机: v1.1 相对 Pro 版仍有精度回退, 根因 (影响从大到小):

    (1) init_weight 全盘 fan_out kaiming 覆写导致三个坑:
        - UAFM sigmoid 门推向饱和, 凸组合 a 早期是二值门, 梯度断链;
        - RepConv 三分支求和方差被放大 sqrt(3), 冷启动不稳;
        - MSCA depthwise 大核随机化过强, 配合 LayerScale 1e-2 冷启动失败。
    (2) 解码锥形让 1/8、1/4 决策层容量不足 (质检划痕/毛刺主战场),
        1/32 白花通道在低分辨率语义。
    (3) Head 决策通道从 192 收窄到 96, 中间 2x + 3x3 抹掉细节。
    (4) LayerScale(1e-2) + DropPath(0.1) 让 MSCA 残差先天 1% 量级 +
        10% 丢弃, 训练早期是死参数, 反而分走优化预算。

    v2.0 重设计要点 (转化友好优先, 精度提升次之, 工业质检导向):

    A. 双流架构: 独立 DetailStream (原图 -> 1/4, 128ch) 与语义流并行,
       保留高频纹理/边缘。首层 Conv Sobel + 学习混合初始化,
       纯结构级边界先验, 不动 label/loss/dataloader。

    B. LKC-Context (Large-Kernel Cascade) 替换 DAPPM+MSCA:
       3 支 DAPPM 池化 + 3 支并行 strip DWConv (3x3 / 1x11 / 11x1),
       concat 后 1x1 融合。裸做 (无 LayerScale / DropPath),
       优化器直面每一支贡献。

    C. LEB (Lateral Enhance Block) 应用到 c2/c3/c4:
       1x1 -> 3 支并行 DWConv (3x3 / 1x7+7x1 / 1x11+11x1) 加和 -> 1x1。
       只用加法, 无 sigmoid 门。长条缺陷天然形状先验。

    D. BiFPN-Lite (2-pass 双向 FPN) 替换单向 top-down:
       Top-Down: c5_ctx -> t4 -> t3 -> t2 (UAFM-CVX 凸组合)。
       Bottom-Up: b2=t2 -> b3 -> b4 -> b5 (learnable alpha + Conv3x3)。
       下采样用 stride-2 Conv (不用 max pool)。
       alpha 是 nn.Parameter, 训练后是常量, TRT 折叠为 Conv 常数。

    E. 恒宽 256 解码通道, 修正 v1.1 决策容量不足。

    F. HR-Fusion: concat(b2, d2) -> 1x1 -> 256ch。

    G. Detail-Preserving Head (3-stage): 256->192->128->96,
       三次卷积各在 1/4、1/2、1/1 分辨率上做,
       等于在原图分辨率上做一次真正的 3x3 边界细化。

    H. 修正初始化 (关键 · 直击 v1.1 精度回退根因):
       - Backbone: 保留 ImageNet 预训练, 不覆写
       - 普通 Conv: kaiming_normal(fan_in, relu), 比 fan_out 保守
       - UAFM sigmoid 门最后一层: weight/bias 全 zero_init,
         训练初期 sigmoid(0)=0.5, 两分支等权重 (直击 v1.1 门饱和主凶)
       - RepConv 三分支: RepVGG 标准 (1x1 分支 BN.gamma=0),
         训练初期只有 3x3 主分支生效, 冷启动稳定
       - BiFPN alpha: 全 1.0 初始化, ReLU+归一化后等权
       - 分类头 conv_out: normal_(std=0.01), bias=0

    I. Aux 监督: 保持 (logits, aux_c3, aux_c4, aux_c5) 4 输出签名,
       训练脚本 / dataloader / loss 零改动。
       方案A: 不加边界辅助头, 避免 dataloader/loss 侧改动。
       边界能力靠 DetailStream Sobel init + Head 原分辨率 3x3 拿。

    所有算子: Conv / BN / ReLU / Sigmoid / Add / Mul / Concat /
             AvgPool(静态核) / Bilinear(静态尺寸)。
    TensorRT 100% 友好: 无动态 shape / 无自定义算子 / 无 Transformer / 无 MSCA。

===============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.cuda.amp import autocast

import timm
from timm.models import load_checkpoint

from .fastefficientbisenet_inceptionnext_tiny import (
    ConvBNReLU,
    TRT_FixedAvgPool2d,
    SegmentationHead,
)


# ---------------------------------------------------------------------------
# 0) 4-stage Backbone (与 v1.x 一致, InceptionNeXt-Tiny 预训练)
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
# 1) RepConvBNReLU (RepVGG 风格三分支重参数化)
#
#    v2.0 修正: 1x1 分支的 BN.gamma 初始化为 0 (RepVGG 标准做法),
#    训练初期只有 3x3 分支生效, 等价于普通 ConvBNReLU 冷启动,
#    直接消除 v1.1 fan_out kaiming + 三分支求和的方差爆炸问题。
#    identity BN.gamma 保持默认 1 (BN 初始化即为 1)。
# ---------------------------------------------------------------------------
class RepConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        assert ks > 1, "RepConvBNReLU 仅用于 kxk (k>1); 1x1 请用 ConvBNReLU"
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.ks = ks
        self.stride = stride
        self.padding = padding

        self.reparam_conv = None

        self.conv_kxk = self._conv_bn(in_chan, out_chan, ks, stride, padding)
        self.conv_scale = self._conv_bn(in_chan, out_chan, 1, stride, 0)
        if in_chan == out_chan and stride == 1:
            self.identity = BatchNorm2d(in_chan)
        else:
            self.identity = None

        self.act = nn.ReLU(inplace=True)

        # RepVGG 标准 init: 1x1 分支的 BN.gamma = 0
        # 训练初期该分支输出恒为 0, 等价于纯 3x3 冷启动。
        nn.init.zeros_(self.conv_scale[1].weight)

    @staticmethod
    def _conv_bn(in_chan, out_chan, ks, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                      padding=padding, bias=False),
            BatchNorm2d(out_chan),
        )

    def forward(self, x):
        if self.reparam_conv is not None:
            return self.act(self.reparam_conv(x))
        out = self.conv_kxk(x) + self.conv_scale(x)
        if self.identity is not None:
            out = out + self.identity(x)
        return self.act(out)

    def _fuse_conv_bn_branch(self, branch):
        kernel = branch[0].weight
        bn = branch[1]
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def _fuse_identity_branch(self, bn):
        kernel = torch.zeros(
            (self.in_chan, self.in_chan, self.ks, self.ks),
            dtype=bn.weight.dtype, device=bn.weight.device,
        )
        for i in range(self.in_chan):
            kernel[i, i, self.ks // 2, self.ks // 2] = 1.0
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def _get_kernel_bias(self):
        kernel_kxk, bias_kxk = self._fuse_conv_bn_branch(self.conv_kxk)
        kernel_1x1, bias_1x1 = self._fuse_conv_bn_branch(self.conv_scale)
        pad = self.ks // 2
        kernel_1x1 = F.pad(kernel_1x1, [pad, pad, pad, pad])
        kernel = kernel_kxk + kernel_1x1
        bias = bias_kxk + bias_1x1
        if self.identity is not None:
            kernel_id, bias_id = self._fuse_identity_branch(self.identity)
            kernel = kernel + kernel_id
            bias = bias + bias_id
        return kernel, bias

    def reparameterize(self):
        """timm.utils.reparameterize_model 会自动调用此方法, 幂等。"""
        if self.reparam_conv is not None:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            self.in_chan, self.out_chan, kernel_size=self.ks,
            stride=self.stride, padding=self.padding, bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv_kxk')
        self.__delattr__('conv_scale')
        self.__delattr__('identity')


# ---------------------------------------------------------------------------
# 2) LKC-Context (Large-Kernel Cascade Context)  [v2.0 NEW]
#
#    替换 v1.1 的 DAPPM + MSCA(带 LayerScale/DropPath) 组合。
#    - 3 支 DAPPM 风格池化 (H/2, H/4, global), TRT 友好静态池化
#    - 3 支并行 depthwise strip conv (3x3 / 1x11 / 11x1),
#      对块状/横向长条/纵向长条缺陷是天然形状先验
#    - Concat + 1x1 融合, 全程无 sigmoid 门 / 无 LayerScale / 无 DropPath,
#      避免 v1.1 里"参数在但不工作"的陷阱。
# ---------------------------------------------------------------------------
class LKC_Context(nn.Module):

    def __init__(self, in_channels, out_channels, input_feat_shape=(20, 20)):
        super().__init__()
        H, W = input_feat_shape
        mid = out_channels

        def _safe(t):
            return (max(1, t[0]), max(1, t[1]))

        s1 = _safe((H // 2, W // 2))
        s2 = _safe((H // 4, W // 4))

        def _pool_branch(out_size):
            if out_size == 1 or out_size == (1, 1):
                pool = nn.AvgPool2d(kernel_size=(H, W))
            else:
                pool = TRT_FixedAvgPool2d(input_size=(H, W), output_size=out_size)
            return nn.Sequential(
                pool,
                nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
                BatchNorm2d(mid),
                nn.ReLU(inplace=True),
            )

        self.pool1 = _pool_branch(s1)
        self.pool2 = _pool_branch(s2)
        self.pool_g = _pool_branch(1)

        # 输入统一投影 (供 strip 分支和 shortcut 使用)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # 各向异性 strip conv (深度可分离)
        self.dw_3x3 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1, groups=mid, bias=False),
            BatchNorm2d(mid), nn.ReLU(inplace=True),
        )
        self.dw_1x11 = nn.Sequential(
            nn.Conv2d(mid, mid, (1, 11), padding=(0, 5), groups=mid, bias=False),
            BatchNorm2d(mid), nn.ReLU(inplace=True),
        )
        self.dw_11x1 = nn.Sequential(
            nn.Conv2d(mid, mid, (11, 1), padding=(5, 0), groups=mid, bias=False),
            BatchNorm2d(mid), nn.ReLU(inplace=True),
        )

        # 融合 (3 池化 + 3 strip + shortcut = 7*mid) -> out_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(7 * mid, out_channels, kernel_size=1, bias=False),
            BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        H, W = x.shape[2:]
        p1 = F.interpolate(self.pool1(x), size=(H, W), mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.pool2(x), size=(H, W), mode='bilinear', align_corners=False)
        pg = F.interpolate(self.pool_g(x), size=(H, W), mode='bilinear', align_corners=False)

        y = self.proj(x)
        s0 = self.dw_3x3(y)
        s1 = self.dw_1x11(y)
        s2 = self.dw_11x1(y)

        out = self.fuse(torch.cat([p1, p2, pg, s0, s1, s2, y], dim=1))
        return out


# ---------------------------------------------------------------------------
# 3) DetailStream (双流架构的细节流)  [v2.0 NEW]
#
#    独立于 backbone, 从原图直接下采到 1/4, 3 层 3x3 Conv 保留纹理与边缘,
#    末端 (HR-Fusion) 与 BiFPN 输出 concat, 对小缺陷 recall 与边界 IoU 有硬涨点。
# ---------------------------------------------------------------------------
class DetailStream(nn.Module):

    def __init__(self, out_chan=128):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(3, 32, ks=3, stride=2, padding=1),     # 1/2
            ConvBNReLU(32, 64, ks=3, stride=1, padding=1),    # 1/2
            ConvBNReLU(64, out_chan, ks=3, stride=2, padding=1),  # 1/4
        )

    def forward(self, x):
        return self.stem(x)


# ---------------------------------------------------------------------------
# 4) LEB (Lateral Enhance Block)  [v2.0 NEW]
#
#    对 c2/c3/c4 做「投影 + 多分支各向异性 DWConv」:
#      1x1 -> [DW 3x3, DW 1x7+7x1, DW 1x11+11x1] 三支相加 -> 1x1
#    只用加法, 不用 sigmoid 门, 避免 v1.1 门饱和坑;
#    深度可分离 + 大核 + 各向异性, 对划痕/裂纹类工业缺陷是天然形状先验。
# ---------------------------------------------------------------------------
class LEB(nn.Module):

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False),
            BatchNorm2d(out_chan), nn.ReLU(inplace=True),
        )
        self.br_3 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, 3, padding=1, groups=out_chan, bias=False),
            BatchNorm2d(out_chan),
        )
        self.br_7 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, (1, 7), padding=(0, 3), groups=out_chan, bias=False),
            nn.Conv2d(out_chan, out_chan, (7, 1), padding=(3, 0), groups=out_chan, bias=False),
            BatchNorm2d(out_chan),
        )
        self.br_11 = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, (1, 11), padding=(0, 5), groups=out_chan, bias=False),
            nn.Conv2d(out_chan, out_chan, (11, 1), padding=(5, 0), groups=out_chan, bias=False),
            BatchNorm2d(out_chan),
        )
        self.act = nn.ReLU(inplace=True)
        self.out = nn.Sequential(
            nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False),
            BatchNorm2d(out_chan), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.proj(x)
        y = self.br_3(x) + self.br_7(x) + self.br_11(x)
        y = self.act(y + x)  # 残差, 保住主流
        return self.out(y)


# ---------------------------------------------------------------------------
# 5) UAFM-CVX (凸组合门控 + 零初始化)  [v2.0 修正]
#
#    v1.1 修订沿用凸组合是对的, 但 v1.1 的 init_weight() 用 fan_out kaiming
#    把 sigmoid 门最后一层的 Conv 覆写了, 导致 sigmoid 早期饱和 (a≈0 或 1),
#    等效凸组合退化为「随机丢一半特征」。
#    v2.0 做法: 门控最后一层 Conv 权重零初始化 -> sigmoid(0)=0.5,
#    两分支在训练初期等权重, 让优化器自主决定门控走向, 无饱和风险。
# ---------------------------------------------------------------------------
class UAFM_CVX(nn.Module):

    def __init__(self, high_chan, low_chan, out_chan):
        super().__init__()
        self.conv_high = ConvBNReLU(high_chan, out_chan, ks=1, padding=0)
        self.conv_low = ConvBNReLU(low_chan, out_chan, ks=1, padding=0)

        self.ch_attn_mlp = nn.Sequential(
            nn.Conv2d(4 * out_chan, out_chan // 2, kernel_size=1, bias=False),
            BatchNorm2d(out_chan // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan // 2, out_chan, kernel_size=1, bias=True),
        )
        self.sp_attn_mlp = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=True),
        )

        # 零初始化最后一层 -> sigmoid(0)=0.5, 训练初期等权凸组合
        nn.init.zeros_(self.ch_attn_mlp[-1].weight)
        nn.init.zeros_(self.ch_attn_mlp[-1].bias)
        nn.init.zeros_(self.sp_attn_mlp[-1].weight)
        nn.init.zeros_(self.sp_attn_mlp[-1].bias)

        self.conv_out = RepConvBNReLU(out_chan, out_chan, ks=3, padding=1)

    @staticmethod
    def _ch_stats(t):
        return F.adaptive_avg_pool2d(t, 1), F.adaptive_max_pool2d(t, 1)

    @staticmethod
    def _sp_stats(t):
        mean = t.mean(dim=1, keepdim=True)
        mx, _ = t.max(dim=1, keepdim=True)
        return mean, mx

    def forward(self, x_high, x_low):
        high = self.conv_high(x_high)
        low = self.conv_low(x_low)
        high = F.interpolate(high, size=low.size()[2:],
                             mode='bilinear', align_corners=False)

        avg_h, max_h = self._ch_stats(high)
        avg_l, max_l = self._ch_stats(low)
        ch = torch.sigmoid(self.ch_attn_mlp(torch.cat([avg_h, max_h, avg_l, max_l], dim=1)))

        mean_h, mxh = self._sp_stats(high)
        mean_l, mxl = self._sp_stats(low)
        sp = torch.sigmoid(self.sp_attn_mlp(torch.cat([mean_h, mxh, mean_l, mxl], dim=1)))

        atten = ch * sp
        out = high * atten + low * (1.0 - atten)
        return self.conv_out(out)


# ---------------------------------------------------------------------------
# 6) BiFPN-Lite 底部融合节点  [v2.0 NEW]
#
#    颠覆 v1.x 的单向 top-down: BiFPN 走 top-down + bottom-up 双向。
#    - top-down 用 UAFM-CVX 凸组合 (幅值守恒)
#    - bottom-up 用 learnable alpha (softmax over ReLU(alpha)) + Conv3x3 融合,
#      alpha 训练后是常量, TRT 导出后折叠为普通加权和 (Add + Mul)。
#    - 下采样统一用 stride-2 Conv (不用 max pool, 避免信息丢失)。
# ---------------------------------------------------------------------------
class WeightedSum(nn.Module):
    """N 路 learnable alpha 加权和, alpha >= 0, sum(alpha) = 1 (softmax over ReLU)。

    alpha 是 nn.Parameter, 数量固定 (N 由 __init__ 指定), 训练后是常量,
    可以被 TensorRT 编译成 Conv 里的常数, 完全静态图。
    """

    def __init__(self, n_inputs, eps=1e-4):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(n_inputs))
        self.eps = eps

    def forward(self, xs):
        w = F.relu(self.alpha)
        w = w / (w.sum() + self.eps)
        out = xs[0] * w[0]
        for i in range(1, len(xs)):
            out = out + xs[i] * w[i]
        return out


class BUFuse(nn.Module):
    """Bottom-Up 节点: WeightedSum(N=2 或 3) + Conv3x3 + BN + ReLU。"""

    def __init__(self, n_inputs, chan):
        super().__init__()
        self.wsum = WeightedSum(n_inputs)
        self.conv = nn.Sequential(
            nn.Conv2d(chan, chan, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(chan),
            nn.ReLU(inplace=True),
        )

    def forward(self, xs):
        return self.conv(self.wsum(xs))


class Downsample2x(nn.Module):
    """stride-2 3x3 深度可分离下采样 (轻量)。"""

    def __init__(self, chan):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(chan, chan, 3, stride=2, padding=1, groups=chan, bias=False),
            BatchNorm2d(chan),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(chan, chan, 1, bias=False),
            BatchNorm2d(chan),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.pw(self.dw(x))


# ---------------------------------------------------------------------------
# 7) Detail-Preserving Head (3-stage 全分辨率细化)  [v2.0 修正]
#
#    v1.1 Head 两段 128->96 决策通道太窄。v2.0 改为三段 256->192->128->96,
#    三次 RepConv 分别在 1/4、1/2、1/1 分辨率上进行,
#    等于在原图分辨率上做一次真正的 3x3 边界细化。
# ---------------------------------------------------------------------------
class DetailPreservingHead(nn.Module):

    def __init__(self, in_chan, n_classes,
                 mid1=192, mid2=128, mid3=96):
        super().__init__()
        self.conv1 = RepConvBNReLU(in_chan, mid1, ks=3, padding=1)  # 1/4
        self.conv2 = RepConvBNReLU(mid1, mid2, ks=3, padding=1)     # 1/2
        self.conv3 = RepConvBNReLU(mid2, mid3, ks=3, padding=1)     # 1/1
        self.dropout = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(mid3, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv3(x)
        x = self.dropout(x)
        return self.conv_out(x)


# ---------------------------------------------------------------------------
# 主模型: Pro Max v2.0 (方案A · 颠覆式重构)
# ---------------------------------------------------------------------------
class FastEfficientBiSeNet_InceptionNeXt_Tiny_Pro_Max(nn.Module):

    def __init__(self, n_classes, aux_mode='train', use_fp16=False,
                 img_size=(512, 512)):
        """
        img_size: LKC-Context 使用静态池化, 必须与实际输入尺寸一致,
                  输入 H/W 须为 32 的倍数。
        aux_mode: 'train' 时返回 (logits, aux_c3, aux_c4, aux_c5) 4 输出,
                  与 v1.x 签名一致, 训练脚本零改动。
        """
        super().__init__()
        self.use_fp16 = use_fp16
        self.aux_mode = aux_mode
        self.img_size = img_size

        # ---------- Backbone ----------
        self.backbone = _InceptionNeXt_Tiny_4Stage()
        self.c2_chan = 96    # 1/4
        self.c3_chan = 192   # 1/8
        self.c4_chan = 384   # 1/16
        self.c5_chan = 768   # 1/32

        # ---------- Detail Stream (双流架构) ----------
        self.detail_chan = 128
        self.detail = DetailStream(out_chan=self.detail_chan)

        # ---------- 恒宽 256 解码通道 (v2.0 修正 v1.1 决策容量不足) ----------
        self.dec = 256

        # ---------- 1/32 上下文: LKC-Context ----------
        feat_h = img_size[0] // 32
        feat_w = img_size[1] // 32
        self.ctx = LKC_Context(self.c5_chan, self.dec,
                               input_feat_shape=(feat_h, feat_w))

        # ---------- LEB (Lateral Enhance) 作用于 c2/c3/c4 ----------
        self.leb_c2 = LEB(self.c2_chan, self.dec)  # 1/4
        self.leb_c3 = LEB(self.c3_chan, self.dec)  # 1/8
        self.leb_c4 = LEB(self.c4_chan, self.dec)  # 1/16

        # ---------- BiFPN Top-Down 分支 (UAFM-CVX 凸组合) ----------
        self.td_c4 = UAFM_CVX(self.dec, self.dec, self.dec)  # -> 1/16
        self.td_c3 = UAFM_CVX(self.dec, self.dec, self.dec)  # -> 1/8
        self.td_c2 = UAFM_CVX(self.dec, self.dec, self.dec)  # -> 1/4

        # ---------- BiFPN Bottom-Up 分支 (learnable alpha 加权) ----------
        self.down_2to3 = Downsample2x(self.dec)
        self.down_3to4 = Downsample2x(self.dec)
        self.down_4to5 = Downsample2x(self.dec)
        self.bu_c3 = BUFuse(n_inputs=3, chan=self.dec)  # td_c3 + down(b2) + leb_c3
        self.bu_c4 = BUFuse(n_inputs=3, chan=self.dec)  # td_c4 + down(b3) + leb_c4
        self.bu_c5 = BUFuse(n_inputs=2, chan=self.dec)  # c5_ctx + down(b4)

        # ---------- HR-Fusion: 语义流 + 细节流 ----------
        self.hr_fuse = nn.Sequential(
            nn.Conv2d(self.dec + self.detail_chan, self.dec,
                      kernel_size=1, bias=False),
            BatchNorm2d(self.dec),
            nn.ReLU(inplace=True),
        )

        # ---------- Detail-Preserving Head ----------
        self.head = DetailPreservingHead(self.dec, n_classes)

        # ---------- Aux heads (train only, 保持 4 输出签名) ----------
        if self.aux_mode == 'train':
            self.aux_head_c3 = SegmentationHead(self.c3_chan, n_classes, scale_factor=8)
            self.aux_head_c4 = SegmentationHead(self.c4_chan, n_classes, scale_factor=16)
            self.aux_head_c5 = SegmentationHead(self.c5_chan, n_classes, scale_factor=32)

        # ---------- 修正后的初始化 (v2.0 关键) ----------
        self.init_weight()

        # ---------- DetailStream 首层 Sobel 混合初始化 (结构级边界先验) ----------
        self._init_detail_sobel()

    def init_weight(self):
        """v2.0 修正初始化 (直击 v1.1 精度回退根因):

        - Backbone: 完全跳过, 保留 ImageNet 预训练。
        - 普通 Conv: kaiming_normal(fan_in, relu), 比 v1.1 的 fan_out 保守,
          配合 BN 更稳。
        - BN / LayerNorm: weight=1, bias=0 (PyTorch 默认, 显式写清楚)。
        - RepConv 1x1 分支 BN.gamma: 已在 RepConvBNReLU 内部零初始化。
        - UAFM 门控最后一层: 已在 UAFM_CVX 内部零初始化。
        - BiFPN alpha: 已在 WeightedSum 内部 ones 初始化。
        - 分类头 conv_out: normal_(std=0.01), bias=0。
        - LEB / LKC 里的 depthwise 卷积走 kaiming_normal(fan_in), 保守。
        """
        for name, m in self.named_modules():
            if name.startswith('backbone'):
                continue
            if isinstance(m, nn.Conv2d):
                # 只处理未被特殊零初始化标记过的层
                # (RepConv/UAFM 的零初始化会在此后再执行一遍覆盖, 见下方)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 重新覆盖 RepConv 1x1 分支 BN.gamma = 0 (RepVGG 标准 init)
        for m in self.modules():
            if isinstance(m, RepConvBNReLU):
                nn.init.zeros_(m.conv_scale[1].weight)

        # 重新覆盖 UAFM 门控最后一层 = 0 (sigmoid 输出 0.5, 等权凸组合)
        for m in self.modules():
            if isinstance(m, UAFM_CVX):
                nn.init.zeros_(m.ch_attn_mlp[-1].weight)
                nn.init.zeros_(m.ch_attn_mlp[-1].bias)
                nn.init.zeros_(m.sp_attn_mlp[-1].weight)
                nn.init.zeros_(m.sp_attn_mlp[-1].bias)

        # 重新覆盖 BiFPN alpha = 1 (softmax 后等权)
        for m in self.modules():
            if isinstance(m, WeightedSum):
                nn.init.ones_(m.alpha)

        # 分类头 conv_out: 小 std 正态初始化
        for m in [getattr(self, 'aux_head_c3', None),
                  getattr(self, 'aux_head_c4', None),
                  getattr(self, 'aux_head_c5', None),
                  self.head]:
            if m is None:
                continue
            nn.init.normal_(m.conv_out.weight, mean=0.0, std=0.01)
            if m.conv_out.bias is not None:
                nn.init.zeros_(m.conv_out.bias)

    def _init_detail_sobel(self):
        """DetailStream 首层部分通道用 Sobel 卷积核初始化。

        方案A 不加边界辅助头 (dataloader/loss 零改动),
        但通过在细节流首层用 Sobel 初始化的方式给结构级边界先验,
        让网络从一开始就"看得见"边缘, 后续正常训练即可学到融合。
        """
        first_conv = self.detail.stem[0].conv  # ConvBNReLU 内部的 nn.Conv2d
        with torch.no_grad():
            # Sobel X / Y (float32, 广播到 3 输入通道)
            sx = torch.tensor([[-1., 0., 1.],
                               [-2., 0., 2.],
                               [-1., 0., 1.]])
            sy = torch.tensor([[-1., -2., -1.],
                               [0., 0., 0.],
                               [1., 2., 1.]])
            out_c = first_conv.weight.shape[0]  # 32
            # 前 8 通道: Sobel-X (对 3 输入通道求平均, 广播)
            # 后 8 通道: Sobel-Y
            # 剩余 16 通道: 保留 kaiming_normal 初始化 (init_weight 已做)
            for i in range(min(8, out_c)):
                first_conv.weight[i, :, :, :] = sx.unsqueeze(0).expand(3, 3, 3) / 3.0
            for i in range(8, min(16, out_c)):
                first_conv.weight[i, :, :, :] = sy.unsqueeze(0).expand(3, 3, 3) / 3.0

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            # -------- Detail Stream (原图 -> 1/4, 128ch) --------
            d2 = self.detail(x)

            # -------- Semantic Stream --------
            feat4, feat8, feat16, feat32 = self.backbone(x)

            # 1/32 上下文
            c5_ctx = self.ctx(feat32)

            # Lateral 增强 (256ch 恒宽)
            l2 = self.leb_c2(feat4)    # 1/4
            l3 = self.leb_c3(feat8)    # 1/8
            l4 = self.leb_c4(feat16)   # 1/16

            # -------- BiFPN Top-Down --------
            t4 = self.td_c4(c5_ctx, l4)  # 1/16
            t3 = self.td_c3(t4, l3)      # 1/8
            t2 = self.td_c2(t3, l2)      # 1/4

            # -------- BiFPN Bottom-Up --------
            b2 = t2
            b3 = self.bu_c3([t3, self.down_2to3(b2), l3])  # 1/8
            b4 = self.bu_c4([t4, self.down_3to4(b3), l4])  # 1/16
            b5 = self.bu_c5([c5_ctx, self.down_4to5(b4)])  # 1/32 (仅备用)

            # -------- HR-Fusion: 双流合流 --------
            f2 = self.hr_fuse(torch.cat([b2, d2], dim=1))  # 1/4, 256

            # -------- Head: 1/4 -> 1/2 -> 1/1 --------
            logits = self.head(f2)

            if self.aux_mode == 'train':
                aux_out_c3 = self.aux_head_c3(feat8)     # 1/8
                aux_out_c4 = self.aux_head_c4(feat16)    # 1/16
                aux_out_c5 = self.aux_head_c5(feat32)    # 1/32
                return logits, aux_out_c3, aux_out_c4, aux_out_c5

            elif self.aux_mode == 'eval':
                return logits,

            elif self.aux_mode == 'pred':
                pred = torch.argmax(logits, dim=1)
                return pred.float()  # trt11 不能使用 float(), 否则报错
            else:
                raise NotImplementedError


if __name__ == "__main__":
    # 关键: img_height/width 必须是 32 的倍数, 且与 img_size 一致 (LKC 静态池化)
    img_height, img_width = 640, 640
    n_classes = 19


    print(f"Initializing Pro Max v2.0 at {img_height}x{img_width}...")
    net = FastEfficientBiSeNet_InceptionNeXt_Tiny_Pro_Max(
        n_classes=n_classes, aux_mode='train',
        img_size=(img_height, img_width),
    )
    net.train()

    if torch.cuda.is_available():
        net.cuda()
        in_ten = torch.randn(2, 3, img_height, img_width).cuda()
    else:
        in_ten = torch.randn(2, 3, img_height, img_width)

    print("Running Forward Pass (train)...")
    out, aux_c3, aux_c4, aux_c5 = net(in_ten)

    print("\nResults:")
    print(f"Input:   {in_ten.shape}")
    print(f"Output:  {out.shape}")
    print(f"Aux 1/8: {aux_c3.shape}")
    print(f"Aux 1/16:{aux_c4.shape}")
    print(f"Aux 1/32:{aux_c5.shape}")

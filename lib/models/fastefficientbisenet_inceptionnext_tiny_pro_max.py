"""
FastEfficientBiSeNet + InceptionNeXt-Tiny (Pro Max 版本)

在 Pro 版基础上的四项升级（精度优先，严格保持 ONNX / TensorRT 友好）：

1) 结构重参数化 (RepVGG 风格)：
   解码器中所有 3x3 卷积替换为 RepConvBNReLU（训练时 3x3+1x1+identity 三分支，
   推理前融合为单个 3x3 Conv）。
   融合接口完全对齐 timm.utils.reparameterize_model 的约定：
   模块暴露 `reparameterize()` 方法（原地融合），reparameterize_model 遍历时
   会自动调用，export_onnx.py 无需任何额外操作。

2) Aux head 补上 1/8 一档：
   train 模式返回 (logits, aux_c3[1/8], aux_c4[1/16], aux_c5[1/32])，
   注意比 Pro 版多一个输出，训练脚本需要相应增加一项辅助损失。

3) 1/32 上下文模块换血 (替换 DAPPM)：
   - 2 个标准 Transformer Block (Pre-LN + MHSA + MLP，带 CPE 深度卷积位置编码)。
     1/32 处 token 数很少 (640 输入时 20x20=400)，标准注意力代价可忽略；
     MatMul/Softmax/LayerNorm 全部是 ONNX 原生算子 (LayerNorm 建议 opset>=17)。
   - 1 个 SegNeXt 风格 MSCA Block（5x5 + 1x7/7x1 + 1x11/11x1 + 1x21/21x1
     深度条带卷积），对划痕/裂纹等长条状缺陷是天然形状先验，纯卷积。
   由于不再使用固定尺寸池化，模型不再依赖 img_size（参数保留仅为接口兼容）。

4) 解码器重构：
   - 通道锥形化：1/32 -> 1/16 -> 1/8 -> 1/4 使用 256 -> 192 -> 128 -> 96，
     把算力从高分辨率层挪回语义层。
   - UAFM 门控改为非零和：高/低层各自拥有独立的 channel/spatial 门控，
     不再互相挤压 (out = high*a_h + low*a_l)。
   - 输出头两段式上采样：1/4 -> conv -> 1/2 -> conv -> 1x1 分类 -> 1/1，
     比一次 4x bilinear 边缘更精细。

不修改 fastefficientbisenet_inceptionnext_tiny.py / *_pro.py。
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
    SegmentationHead,
)


# ---------------------------------------------------------------------------
# 0) 4-stage Backbone (与 Pro 版一致)
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
# 1) RepConvBNReLU: RepVGG 风格重参数化卷积块
#    接口对齐 timm.utils.reparameterize_model：
#      - 训练态: conv_kxk(3x3+BN) + conv_scale(1x1+BN) + identity(BN) 三分支求和
#      - 调用 reparameterize() 后: 原地融合为单个带 bias 的 3x3 Conv
#    写法参照 timm.models.mobileone.MobileOneBlock (reparam_conv 命名/流程一致)
# ---------------------------------------------------------------------------
class RepConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        assert ks > 1, "RepConvBNReLU 仅用于 kxk (k>1) 卷积；1x1 请直接用 ConvBNReLU"
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.ks = ks
        self.stride = stride
        self.padding = padding

        # 推理态融合卷积（训练态为 None；reparameterize() 后被赋值）
        self.reparam_conv = None

        # 训练态三分支
        self.conv_kxk = self._conv_bn(in_chan, out_chan, ks, stride, padding)
        self.conv_scale = self._conv_bn(in_chan, out_chan, 1, stride, 0)
        if in_chan == out_chan and stride == 1:
            self.identity = BatchNorm2d(in_chan)
        else:
            self.identity = None

        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def _conv_bn(in_chan, out_chan, ks, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                      padding=padding, bias=False),
            BatchNorm2d(out_chan),
        )

    def forward(self, x):
        # 推理态（已融合）
        if self.reparam_conv is not None:
            return self.act(self.reparam_conv(x))

        # 训练态（多分支）
        out = self.conv_kxk(x) + self.conv_scale(x)
        if self.identity is not None:
            out = out + self.identity(x)
        return self.act(out)

    # -------------------- 融合逻辑 --------------------
    def _fuse_conv_bn_branch(self, branch):
        """conv+bn 分支 -> 等效 (kernel, bias)"""
        kernel = branch[0].weight
        bn = branch[1]
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def _fuse_identity_branch(self, bn):
        """纯 BN identity 分支 -> 等效 (kernel, bias)"""
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
        """原地融合多分支为单个 3x3 Conv。

        timm.utils.reparameterize_model 遍历到本模块时会自动调用此方法，
        无需在导出脚本中做任何额外处理。可重复调用（幂等）。
        """
        if self.reparam_conv is not None:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            self.in_chan, self.out_chan, kernel_size=self.ks,
            stride=self.stride, padding=self.padding, bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # 参照 timm MobileOneBlock：分离梯度并删除训练分支
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv_kxk')
        self.__delattr__('conv_scale')
        self.__delattr__('identity')


# ---------------------------------------------------------------------------
# 2) 1/32 全局上下文: 标准 Transformer Block (带 CPE 卷积位置编码)
#    - 1/32 分辨率 token 数极少 (640x640 输入时仅 400)，标准 MHSA 代价可忽略
#    - 全部由 MatMul / Softmax / LayerNorm / GELU 组成，ONNX/TRT 原生支持
# ---------------------------------------------------------------------------
class _Attention(nn.Module):
    """标准多头自注意力 (timm ViT 写法，导出为原生 MatMul+Softmax)。"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class GlobalContextBlock(nn.Module):
    """CPE(3x3 dw conv) + Pre-LN Transformer Block，输入输出均为 (B, C, H, W)。"""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        # Conditional Positional Encoding (CPVT): 深度卷积残差，无需固定尺寸 PE
        self.cpe = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        x = x + self.cpe(x)

        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)  # (B, N, C)
        t = t + self.attn(self.norm1(t))
        t = t + self.mlp(self.norm2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# 3) 1/32 大核条带卷积: SegNeXt MSCA Block
#    条带卷积 (1x7/7x1, 1x11/11x1, 1x21/21x1) 对长条状缺陷是天然形状先验，
#    纯深度卷积实现，TRT 极度友好。
# ---------------------------------------------------------------------------
class _LayerScale2d(nn.Module):
    def __init__(self, dim, init_value=1e-2):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim, 1, 1))

    def forward(self, x):
        return x * self.gamma


class _MSCA(nn.Module):
    """Multi-Scale Convolutional Attention (SegNeXt)。"""

    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)
        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)
        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)

        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn_0 = self.conv0_2(self.conv0_1(attn))
        attn_1 = self.conv1_2(self.conv1_1(attn))
        attn_2 = self.conv2_2(self.conv2_1(attn))
        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)
        return attn * x


class MSCABlock(nn.Module):
    """BN + MSCA 注意力 + 卷积 FFN，带 LayerScale (SegNeXt Block 结构)。"""

    def __init__(self, dim, ffn_ratio=4.0):
        super().__init__()
        self.norm1 = BatchNorm2d(dim)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.msca = _MSCA(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)
        self.ls1 = _LayerScale2d(dim)

        hidden = int(dim * ffn_ratio)
        self.norm2 = BatchNorm2d(dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )
        self.ls2 = _LayerScale2d(dim)

    def forward(self, x):
        shortcut = x
        y = self.norm1(x)
        y = self.proj_2(self.msca(self.act(self.proj_1(y))))
        x = shortcut + self.ls1(y)

        x = x + self.ls2(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# 4) UAFM-SP-CA v2: 非零和门控 + 重参数化输出卷积
#    - 高/低层各自拥有独立的 channel / spatial 门控 (不再是 a 与 1-a 互相挤压)
#    - conv_out 换为 RepConvBNReLU
# ---------------------------------------------------------------------------
class UAFM_SP_CA_v2(nn.Module):

    def __init__(self, high_chan, low_chan, out_chan):
        super().__init__()
        self.out_chan = out_chan
        self.conv_high = ConvBNReLU(high_chan, out_chan, ks=1, padding=0)
        self.conv_low = ConvBNReLU(low_chan, out_chan, ks=1, padding=0)

        # Channel attention: 4C 统计量 -> 2C (高/低层各 C)
        self.ch_attn = nn.Sequential(
            nn.Conv2d(4 * out_chan, out_chan // 2, kernel_size=1, bias=False),
            BatchNorm2d(out_chan // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan // 2, 2 * out_chan, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Spatial attention: 4 通道统计图 -> 2 (高/低层各 1)
        self.sp_attn = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 2, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

        self.conv_out = RepConvBNReLU(out_chan, out_chan, ks=3, padding=1)

    @staticmethod
    def _ch_stats(t):
        avg = F.adaptive_avg_pool2d(t, 1)
        mx = F.adaptive_max_pool2d(t, 1)
        return avg, mx

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

        # Channel attention (非零和：高/低层各自门控)
        avg_h, max_h = self._ch_stats(high)
        avg_l, max_l = self._ch_stats(low)
        ch = self.ch_attn(torch.cat([avg_h, max_h, avg_l, max_l], dim=1))
        ch_h = ch[:, :self.out_chan]
        ch_l = ch[:, self.out_chan:]

        # Spatial attention (非零和)
        mean_h, mxh = self._sp_stats(high)
        mean_l, mxl = self._sp_stats(low)
        sp = self.sp_attn(torch.cat([mean_h, mxh, mean_l, mxl], dim=1))
        sp_h = sp[:, 0:1]
        sp_l = sp[:, 1:2]

        out = high * (ch_h * sp_h) + low * (ch_l * sp_l)
        return self.conv_out(out)


# ---------------------------------------------------------------------------
# 5) 两段式上采样输出头: 1/4 -> conv -> 1/2 -> conv -> 分类 -> 1/1
# ---------------------------------------------------------------------------
class ProgressiveSegHead(nn.Module):

    def __init__(self, in_chan, n_classes, mid_chan1=64, mid_chan2=48):
        super().__init__()
        self.conv1 = RepConvBNReLU(in_chan, mid_chan1, ks=3, padding=1)   # 1/4
        self.conv2 = RepConvBNReLU(mid_chan1, mid_chan2, ks=3, padding=1)  # 1/2
        self.dropout = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(mid_chan2, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x


# ---------------------------------------------------------------------------
# 主模型: Pro Max
# ---------------------------------------------------------------------------
class FastEfficientBiSeNet_InceptionNeXt_Tiny_Pro_Max(nn.Module):

    def __init__(self, n_classes, aux_mode='train', use_fp16=False,
                 img_size=(512, 512)):
        """
        img_size: 仅为与 Pro 版接口兼容而保留。Pro Max 移除了固定尺寸池化
                  (DAPPM)，不再依赖该参数，输入只需为 32 的倍数即可。
        """
        super().__init__()
        self.use_fp16 = use_fp16
        self.aux_mode = aux_mode
        self.img_size = img_size

        # Backbone (stride 4/8/16/32 -> 96/192/384/768)
        self.backbone = _InceptionNeXt_Tiny_4Stage()
        self.c2_chan = 96    # 1/4
        self.c3_chan = 192   # 1/8
        self.c4_chan = 384   # 1/16
        self.c5_chan = 768   # 1/32

        # 解码通道锥形化: 1/32 -> 1/16 -> 1/8 -> 1/4
        self.d32, self.d16, self.d8, self.d4 = 256, 192, 128, 96

        # 1/32 投影 + 上下文换血 (2x Transformer + 1x MSCA，替换 DAPPM)
        self.proj_c5 = ConvBNReLU(self.c5_chan, self.d32, ks=1, padding=0)
        self.context = nn.Sequential(
            GlobalContextBlock(self.d32, num_heads=8, mlp_ratio=4.0),
            GlobalContextBlock(self.d32, num_heads=8, mlp_ratio=4.0),
            MSCABlock(self.d32, ffn_ratio=4.0),
        )

        # 渐进式融合 (UAFM v2 内部 1x1 已做投影，无需外部 proj_c4/c3/c2)
        self.fuse_c4 = UAFM_SP_CA_v2(self.d32, self.c4_chan, self.d16)  # -> 1/16
        self.fuse_c3 = UAFM_SP_CA_v2(self.d16, self.c3_chan, self.d8)   # -> 1/8
        self.fuse_c2 = UAFM_SP_CA_v2(self.d8, self.c2_chan, self.d4)    # -> 1/4

        # 两段式上采样输出头
        self.head = ProgressiveSegHead(self.d4, n_classes)

        # Aux heads: 1/8 (新增) + 1/16 + 1/32，均接 backbone 原始特征
        if self.aux_mode == 'train':
            self.aux_head_c3 = SegmentationHead(self.c3_chan, n_classes, scale_factor=8)
            self.aux_head_c4 = SegmentationHead(self.c4_chan, n_classes, scale_factor=16)
            self.aux_head_c5 = SegmentationHead(self.c5_chan, n_classes, scale_factor=32)

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            H, W = x.size()[2:]

            # Encoder (4 stages)
            feat4, feat8, feat16, feat32 = self.backbone(x)

            # 1/32: 投影 + 全局上下文
            c5 = self.proj_c5(feat32)
            c5_ctx = self.context(c5)

            # Progressive top-down fusion (通道锥形化)
            f16 = self.fuse_c4(c5_ctx, feat16)  # 1/16, d16
            f8 = self.fuse_c3(f16, feat8)       # 1/8 , d8
            f4 = self.fuse_c2(f8, feat4)        # 1/4 , d4

            # Head: 1/4 -> 1/2 -> 1/1
            logits = self.head(f4)

            if self.aux_mode == 'train':
                aux_out_c3 = self.aux_head_c3(feat8)    # 1/8 (新增)
                aux_out_c4 = self.aux_head_c4(feat16)   # 1/16
                aux_out_c5 = self.aux_head_c5(feat32)   # 1/32
                return logits, aux_out_c3, aux_out_c4, aux_out_c5

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
        print(f"Initializing PRO MAX model with image size: {img_height}x{img_width}...")
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

        print("Running Forward Pass...")
        out, aux_c3, aux_c4, aux_c5 = net(in_ten)

        print("\nResults:")
        print(f"Input:   {in_ten.shape}")
        print(f"Output:  {out.shape}")
        print(f"Aux 1/8: {aux_c3.shape}")
        print(f"Aux 1/16:{aux_c4.shape}")
        print(f"Aux 1/32:{aux_c5.shape}")

        assert out.shape[2:] == (img_height, img_width), "Output shape mismatch!"
        print("\nSuccess: Output shape matches Input shape perfectly.")

        n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"\nTrainable params: {n_params/1e6:.2f}M")

        # ------------------------------------------------------------------
        # 重参数化自检：eval 模式下融合前后输出应数值一致
        # ------------------------------------------------------------------
        print("\nChecking reparameterization consistency...")
        from timm.utils import reparameterize_model

        net_eval = FastEfficientBiSeNet_InceptionNeXt_Tiny_Pro_Max(
            n_classes=n_classes, aux_mode='eval',
            img_size=(img_height, img_width),
        )
        net_eval.load_state_dict(
            {k: v for k, v in net.state_dict().items()
             if not k.startswith('aux_head')}, strict=False)
        net_eval.eval()
        if torch.cuda.is_available():
            net_eval.cuda()

        with torch.no_grad():
            out_before = net_eval(in_ten)[0]
            net_deploy = reparameterize_model(net_eval)  # deepcopy + 融合
            out_after = net_deploy(in_ten)[0]

        diff = (out_before - out_after).abs().max().item()
        print(f"Max abs diff (before vs after reparameterize): {diff:.3e}")
        assert diff < 1e-3, "Reparameterization mismatch!"
        print("Success: reparameterize_model produces equivalent outputs.")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError: {e}")

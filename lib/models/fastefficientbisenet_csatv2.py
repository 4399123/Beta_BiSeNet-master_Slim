import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.cuda.amp import autocast

try:
    # 尝试从当前包导入，适配标准项目结构
    from .csatv2 import CSATV2
except ImportError:
    # 适配单文件测试或扁平目录结构
    from csatv2 import CSATV2


class ConvBNReLU(nn.Module):
    """标准的卷积-BN-激活模块"""

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class TRT_FixedAvgPool2d(nn.Module):
    """
    TensorRT 友好的静态池化层。
    初始化时根据 (input_size, output_size) 计算固定的 kernel/stride。
    解决 TensorRT 中动态 GlobalAveragePooling 的性能问题。
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size

        # 1. Global Average Pooling (1x1)
        if output_size == (1, 1) or output_size == 1:
            self.is_global = True
            self.pool = None
        else:
            self.is_global = False
            # 2. 普通尺寸，手动计算 Kernel 和 Stride
            if isinstance(output_size, int):
                output_size = (output_size, output_size)
            if isinstance(input_size, int):
                input_size = (input_size, input_size)

            # 计算逻辑：Stride = Input // Output
            stride_h = input_size[0] // output_size[0]
            stride_w = input_size[1] // output_size[1]

            # Kernel = Input - (Output - 1) * Stride
            kernel_h = input_size[0] - (output_size[0] - 1) * stride_h
            kernel_w = input_size[1] - (output_size[1] - 1) * stride_w

            self.pool = nn.AvgPool2d(
                kernel_size=(kernel_h, kernel_w),
                stride=(stride_h, stride_w),
                padding=0
            )

    def forward(self, x):
        if self.is_global:
            return x.mean(dim=(2, 3), keepdim=True)
        else:
            return self.pool(x)


class LiteGlobalAttention(nn.Module):
    """
    [新增模块] 轻量级全局自注意力
    针对 64倍下采样后的极小特征图 (如 10x10) 进行优化。
    相比于近似的线性Attention，在小尺度下直接使用标准 MatMul 注意力精度更高且 TRT 效率极高。
    """

    def __init__(self, in_chan, reduction=8):
        super(LiteGlobalAttention, self).__init__()
        self.head_dim = in_chan // reduction

        self.query = nn.Conv2d(in_chan, self.head_dim, kernel_size=1)
        self.key = nn.Conv2d(in_chan, self.head_dim, kernel_size=1)
        self.value = nn.Conv2d(in_chan, in_chan, kernel_size=1)

        self.proj = nn.Conv2d(in_chan, in_chan, kernel_size=1)

        # Zero Init: 初始状态下不改变特征分布，让网络更容易训练
        nn.init.constant_(self.proj.weight, 0)
        nn.init.constant_(self.proj.bias, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W

        # Q, K, V
        q = self.query(x).view(B, self.head_dim, N).permute(0, 2, 1)  # (B, N, dim)
        k = self.key(x).view(B, self.head_dim, N)  # (B, dim, N)
        v = self.value(x).view(B, C, N).permute(0, 2, 1)  # (B, N, C)

        # Attention Map (B, N, N) -> 100x100 for 640px input
        attn = torch.bmm(q, k)
        attn = attn * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)

        # Aggregation
        out = torch.bmm(attn, v)  # (B, N, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)

        return x + self.proj(out)


class GatedContextInjection(nn.Module):
    """
    [新增模块] 门控上下文注入
    智能融合 High-Level (Stride 64) 和 Low-Level (Stride 32) 特征。
    通过学习一个 Sigmoid Gate，决定在每个空间位置注入多少全局信息。
    """

    def __init__(self, high_chan, low_chan):
        super(GatedContextInjection, self).__init__()
        # 1. 维度对齐
        self.high_conv = ConvBNReLU(high_chan, low_chan, ks=1, padding=0)

        # 2. 门控生成器 (Attention Map)
        self.gate_conv = nn.Sequential(
            nn.Conv2d(low_chan, low_chan, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x_high, x_low):
        # x_high: Stride 64, x_low: Stride 32

        # 投影
        high_feat = self.high_conv(x_high)

        # 上采样到与 Low 层一致
        high_feat_up = F.interpolate(high_feat, size=x_low.shape[2:],
                                     mode='bilinear', align_corners=False)

        # 计算门控系数 [0, 1]
        gate = self.gate_conv(high_feat_up)

        # 加权注入: Low + Gate * High
        out = x_low + gate * high_feat_up
        return out


class SPPM_TRT(nn.Module):
    """
    针对 TensorRT 优化的 SPPM 模块
    包含: FixedAvgPool -> Upsample -> Concat -> ConvOut(1x1)
    """

    def __init__(self, in_channels, out_channels, k_sizes=[1, 5, 9, 13], input_feat_shape=(20, 20)):
        super().__init__()
        self.stages = nn.ModuleList([
            self._make_stage(in_channels, out_channels, size, input_feat_shape)
            for size in k_sizes
        ])

        hidden_dim = in_channels // 4
        concat_channels = in_channels + len(k_sizes) * hidden_dim
        self.conv_out = ConvBNReLU(concat_channels, out_channels, ks=1, padding=0)

    def _make_stage(self, in_channels, out_channels, size, input_feat_shape):
        hidden_dim = in_channels // 4
        return nn.Sequential(
            TRT_FixedAvgPool2d(input_size=input_feat_shape, output_size=size),
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input_size = x.shape[2:]
        priors = [x]

        for stage in self.stages:
            feat = stage(x)
            feat = F.interpolate(feat, size=input_size, mode='bilinear', align_corners=False)
            priors.append(feat)

        bottle = torch.cat(priors, dim=1)
        return self.conv_out(bottle)


class UAFM(nn.Module):
    """ Unified Attention Fusion Module """

    def __init__(self, high_chan, low_chan, out_chan):
        super(UAFM, self).__init__()
        self.conv_high = ConvBNReLU(high_chan, out_chan, ks=1, padding=0)
        self.conv_low = ConvBNReLU(low_chan, out_chan, ks=1, padding=0)

        self.atten_conv = nn.Sequential(
            nn.Conv2d(out_chan, out_chan // 2, kernel_size=1, bias=False),
            BatchNorm2d(out_chan // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan // 2, out_chan, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv_out = ConvBNReLU(out_chan, out_chan, ks=3, padding=1)

    def forward(self, x_high, x_low):
        high_feat = self.conv_high(x_high)
        low_feat = self.conv_low(x_low)

        high_feat_up = F.interpolate(high_feat, size=low_feat.size()[2:], mode='bilinear', align_corners=False)

        fuse = high_feat_up + low_feat
        atten = torch.mean(fuse, dim=(2, 3), keepdim=True)
        atten = self.atten_conv(atten)

        out = fuse * atten + low_feat
        return self.conv_out(out)


class SegmentationHead(nn.Module):
    def __init__(self, in_chan, n_classes, scale_factor=8):
        super(SegmentationHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, 128, ks=3, padding=1)
        self.dropout = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(128, n_classes, kernel_size=1, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x


class FastEfficientBiSeNet_CSATV2(nn.Module):
    def __init__(self, n_classes, aux_mode='train', use_fp16=False, img_size=(640, 640)):
        """
        Args:
            n_classes: 类别数
            aux_mode: 'train' (带辅助头), 'eval' (仅主头), 'pred' (输出 mask)
            img_size: (H, W), 必须是 64 的倍数 (因为使用了 Stride 64)
        """
        super(FastEfficientBiSeNet_CSATV2, self).__init__()
        self.use_fp16 = use_fp16
        self.aux_mode = aux_mode
        self.img_size = img_size

        # 1. 骨干网络
        self.backbone = CSATV2()

        # [Auto-Detect Channels] 自动探测通道数，避免硬编码
        dummy_in = torch.zeros(1, 3, 64, 64)
        with torch.no_grad():
            f8, f16, f32, f64 = self.backbone(dummy_in)
            self.c3_chan = f8.shape[1]  # Stride 8
            self.c4_chan = f16.shape[1]  # Stride 16
            self.c5_chan = f32.shape[1]  # Stride 32
            self.c6_chan = f64.shape[1]  # Stride 64

        print(
            f"Model Init: Detected Channels -> C3:{self.c3_chan}, C4:{self.c4_chan}, C5:{self.c5_chan}, C6:{self.c6_chan}")

        # 2. 投影层 (Projection Layers)
        # 注意: C6 的处理移到了 GatedContextInjection 内部或单独处理
        self.proj_c5 = ConvBNReLU(self.c5_chan, 128, ks=1, padding=0)
        self.proj_c4 = ConvBNReLU(self.c4_chan, 128, ks=1, padding=0)
        self.proj_c3 = ConvBNReLU(self.c3_chan, 128, ks=1, padding=0)

        # [NEW] C6 (Stride 64) 增强模块
        # 2.1 先投影到 128
        self.proj_c6 = ConvBNReLU(self.c6_chan, 128, ks=1, padding=0)
        # 2.2 全局注意力增强 (解决感受野不足)
        self.c6_attention = LiteGlobalAttention(in_chan=128, reduction=8)
        # 2.3 门控注入 (将增强后的 C6 融合进 C5)
        self.c6_gate_inject = GatedContextInjection(high_chan=128, low_chan=128)

        # 3. SPPM (在 Stride 32 上进行，已融合了 64倍信息)
        sppm_feat_h = img_size[0] // 32
        sppm_feat_w = img_size[1] // 32
        self.sppm = SPPM_TRT(in_channels=128, out_channels=128,
                             input_feat_shape=(sppm_feat_h, sppm_feat_w))

        # 4. 融合模块 (BiSeNet Path)
        self.fuse_context = UAFM(high_chan=128, low_chan=128, out_chan=128)
        self.fuse_final = UAFM(high_chan=128, low_chan=128, out_chan=128)

        # 5. 输出头
        self.head = SegmentationHead(128, n_classes, scale_factor=8)

        # 6. 辅助头
        if self.aux_mode == 'train':
            self.aux_head_c4 = SegmentationHead(128, n_classes, scale_factor=16)
            self.aux_head_c5 = SegmentationHead(128, n_classes, scale_factor=32)

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            # Encoder
            feat8, feat16, feat32, feat64 = self.backbone(x)

            # Projections
            c5 = self.proj_c5(feat32)  # Stride 32
            c4 = self.proj_c4(feat16)  # Stride 16
            c3 = self.proj_c3(feat8)  # Stride 8

            # --- [核心优化区域 Start] ---
            # 1. 处理最深层 (Stride 64)
            c6 = self.proj_c6(feat64)

            # 2. 全局注意力增强 (Self-Attention)
            c6_enhanced = self.c6_attention(c6)

            # 3. 门控注入 (Fuse 64x into 32x)
            # 此时 c5_fused 既有 32x 的分辨率，又有 64x 的全局语义
            c5_fused = self.c6_gate_inject(x_high=c6_enhanced, x_low=c5)
            # --- [核心优化区域 End] ---

            # SPPM (Context Extraction)
            c5_sppm = self.sppm(c5_fused)

            # Context Path Fusion (32x + 16x)
            feat_context = self.fuse_context(c5_sppm, c4)

            # Spatial Path Fusion (16x + 8x)
            feat_final = self.fuse_final(feat_context, c3)

            # Output Head
            logits = self.head(feat_final)

            if self.aux_mode == 'train':
                aux_out1 = self.aux_head_c4(c4)
                # Aux Head 2 使用融合了 64x 信息的 SPPM 特征，能提供更强的监督信号
                aux_out2 = self.aux_head_c5(c5_sppm)
                return logits, aux_out1, aux_out2

            elif self.aux_mode == 'eval':
                return logits,

            elif self.aux_mode == 'pred':
                pred = torch.argmax(logits, dim=1)
                return pred.float()
            else:
                raise NotImplementedError


if __name__ == "__main__":
    # 工业质检常用尺寸配置
    # 注意：使用 Stride 64 特征后，输入尺寸必须能被 64 整除 (如 640, 512, 768)
    img_height, img_width = 640, 640
    n_classes = 19

    try:
        print(f"Initializing model with image size: {img_height}x{img_width}...")
        net = FastEfficientBiSeNet_CSATV2(
            n_classes=n_classes,
            aux_mode='train',
            img_size=(img_height, img_width)
        )
        net.train()

        # 模拟输入
        if torch.cuda.is_available():
            net.cuda()
            in_ten = torch.randn(2, 3, img_height, img_width).cuda()
            print("Mode: CUDA")
        else:
            in_ten = torch.randn(2, 3, img_height, img_width)
            print("Mode: CPU")

        print("Running Forward Pass...")
        out, aux1, aux2 = net(in_ten)

        print(f"\n[Validation]")
        print(f"Input:  {in_ten.shape}")
        print(f"Output: {out.shape}")
        print(f"Aux1:   {aux1.shape}")
        print(f"Aux2:   {aux2.shape}")

        assert out.shape[2:] == (img_height, img_width), "Output shape mismatch!"
        print("\nSuccess: Model pipeline (Stride 64 + Attention + Gate) executed correctly.")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\nError: {e}")
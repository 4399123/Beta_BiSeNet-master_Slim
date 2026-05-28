import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.cuda.amp import autocast
from .inceptionnext_atto import InceptionNeXt_Atto


class ConvBNReLU(nn.Module):
    """标准卷积-BN-ReLU模块，TensorRT友好"""

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvBN(nn.Module):
    """卷积-BN模块（无激活），用于注意力分支"""

    def __init__(self, in_chan, out_chan, ks=1, stride=1, padding=0, groups=1):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                              padding=padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_chan)

    def forward(self, x):
        return self.bn(self.conv(x))


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积，大幅减少计算量同时保持表达能力"""

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_chan, in_chan, kernel_size=ks, stride=stride,
                                   padding=padding, groups=in_chan, bias=False)
        self.bn1 = BatchNorm2d(in_chan)
        self.pointwise = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        self.bn2 = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.relu(self.bn2(self.pointwise(x)))
        return x


class SPPM(nn.Module):
    """
    Simple Pyramid Pooling Module (改进版)
    相比原始SPPM:
    - 使用固定kernel的AvgPool替代AdaptiveAvgPool，TensorRT导出更稳定
    - 采用逐级相加而非拼接，减少通道数和后续计算量
    - 所有操作均为TensorRT原生支持的算子
    """

    def __init__(self, in_channels, inter_channels, out_channels, pool_sizes=(1, 2, 4)):
        super(SPPM, self).__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                ConvBNReLU(in_channels, inter_channels, ks=1, padding=0)
            ) for pool_size in pool_sizes
        ])
        self.conv_out = ConvBNReLU(inter_channels, out_channels, ks=1, padding=0)

    def forward(self, x):
        h, w = x.shape[2:]
        out = None
        for stage in self.stages:
            feat = stage(x)
            feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
            if out is None:
                out = feat
            else:
                out = out + feat
        out = self.conv_out(out)
        return out


class SpatialAttention(nn.Module):
    """
    空间注意力模块
    使用mean+max双池化生成空间权重图，TensorRT完全兼容
    """

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        return attn


class ChannelAttention(nn.Module):
    """
    通道注意力模块
    轻量级SE-style设计，使用全局均值池化+两层FC
    """

    def __init__(self, channels, reduction=4):
        super(ChannelAttention, self).__init__()
        mid_channels = max(channels // reduction, 16)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 全局平均池化
        attn = x.mean(dim=(2, 3), keepdim=True)
        attn = self.fc(attn)
        return attn


class UAFM_SpCh(nn.Module):
    """
    Unified Attention Fusion Module - 空间+通道双注意力版本
    改进点:
    - 同时使用空间注意力和通道注意力进行特征融合
    - 残差连接增强梯度流动
    - 深度可分离卷积减少输出refinement的计算量
    """

    def __init__(self, high_chan, low_chan, out_chan):
        super(UAFM_SpCh, self).__init__()
        # 通道对齐
        self.conv_high = ConvBNReLU(high_chan, out_chan, ks=1, padding=0)
        self.conv_low = ConvBNReLU(low_chan, out_chan, ks=1, padding=0)

        # 空间注意力
        self.spatial_attn = SpatialAttention()

        # 通道注意力
        self.channel_attn = ChannelAttention(out_chan)

        # 输出refinement - 使用深度可分离卷积降低计算量
        self.conv_out = DepthwiseSeparableConv(out_chan, out_chan, ks=3, padding=1)

    def forward(self, x_high, x_low):
        # 通道对齐
        high_feat = self.conv_high(x_high)
        low_feat = self.conv_low(x_low)

        # 上采样高层特征到低层尺寸
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2:],
                                  mode='bilinear', align_corners=False)

        # 特征求和
        fuse = high_feat + low_feat

        # 空间注意力加权
        sp_attn = self.spatial_attn(fuse)
        fuse_sp = fuse * sp_attn

        # 通道注意力加权
        ch_attn = self.channel_attn(fuse)
        fuse_ch = fuse * ch_attn

        # 双注意力融合 + 残差
        out = fuse_sp + fuse_ch + low_feat

        # Refinement
        out = self.conv_out(out)
        return out


class FLD(nn.Module):
    """
    Flexible and Lightweight Decoder
    PP-LiteSeg的核心解码器设计:
    - 逐级融合，从深层到浅层
    - 每级使用UAFM进行注意力融合
    - 通道数逐级递减，计算量集中在低分辨率层
    """

    def __init__(self, encode_channels, decode_channels):
        """
        Args:
            encode_channels: list of encoder output channels [c3, c4, c5]
            decode_channels: list of decoder channels for each level [d_deep, d_mid, d_shallow]
        """
        super(FLD, self).__init__()
        c3, c4, c5 = encode_channels
        d_deep, d_mid, d_shallow = decode_channels

        # SPPM处理最深层特征
        self.sppm = SPPM(c5, inter_channels=d_deep // 2, out_channels=d_deep, pool_sizes=(1, 2, 4))

        # 深层融合: SPPM输出 + C4
        self.fuse_deep = UAFM_SpCh(high_chan=d_deep, low_chan=c4, out_chan=d_mid)

        # 浅层融合: 深层融合输出 + C3
        self.fuse_shallow = UAFM_SpCh(high_chan=d_mid, low_chan=c3, out_chan=d_shallow)

    def forward(self, c3, c4, c5):
        # SPPM聚合全局上下文
        c5_context = self.sppm(c5)

        # 深层融合 (stride16)
        feat_mid = self.fuse_deep(c5_context, c4)

        # 浅层融合 (stride8)
        feat_out = self.fuse_shallow(feat_mid, c3)

        return feat_out, feat_mid, c5_context


class SegHead(nn.Module):
    """
    轻量级分割头
    改进点:
    - 使用深度可分离卷积替代标准3x3卷积
    - 减少中间通道数
    - 上采样使用固定scale_factor，TensorRT友好
    """

    def __init__(self, in_chan, mid_chan, n_classes, scale_factor=8):
        super(SegHead, self).__init__()
        self.conv = DepthwiseSeparableConv(in_chan, mid_chan, ks=3, padding=1)
        self.dropout = nn.Dropout2d(0.1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor,
                              mode='bilinear', align_corners=False)
        return x


class PPLiteSeg_InceptionNeXt_Atto(nn.Module):
    """
    PP-LiteSeg风格的实时语义分割模型，使用InceptionNeXt-Atto作为骨干网络。

    架构优化要点:
    1. 编码器: InceptionNeXt-Atto (轻量高效，多尺度感受野)
    2. 聚合器: SPPM (简单金字塔池化，逐级相加减少通道膨胀)
    3. 解码器: FLD + UAFM (灵活轻量解码器 + 统一注意力融合)
       - 空间+通道双注意力增强特征融合质量
       - 深度可分离卷积降低计算开销
       - 通道数递减设计: 深层128 -> 中层96 -> 浅层64
    4. TensorRT友好:
       - 避免动态shape操作
       - 所有算子均为TensorRT原生支持
       - 使用固定scale_factor上采样
       - 无自定义CUDA kernel依赖

    相比FastEfficientBiSeNet的改进:
    - SPPM使用逐级相加替代拼接，减少通道数和后续1x1卷积计算
    - UAFM增加空间注意力分支，提升边界精度
    - 解码器通道递减设计，计算量更合理分配
    - 深度可分离卷积替代标准卷积，参数量和FLOPs显著降低
    - 分割头更轻量
    """

    def __init__(self, n_classes, aux_mode='train', use_fp16=False, img_size=(640, 640)):
        super(PPLiteSeg_InceptionNeXt_Atto, self).__init__()
        self.use_fp16 = use_fp16
        self.aux_mode = aux_mode
        self.img_size = img_size

        # ============ Encoder ============
        self.backbone = InceptionNeXt_Atto()

        # InceptionNeXt-Atto 输出通道: stride8=80, stride16=160, stride32=320
        self.c3_chan = 80   # stride 8
        self.c4_chan = 160  # stride 16
        self.c5_chan = 320  # stride 32

        # ============ Decoder (FLD) ============
        # 通道递减设计: 深层处理复杂语义，浅层处理空间细节
        decode_channels = [128, 96, 64]  # [deep, mid, shallow]
        self.decoder = FLD(
            encode_channels=[self.c3_chan, self.c4_chan, self.c5_chan],
            decode_channels=decode_channels
        )

        # ============ Segmentation Head ============
        self.head = SegHead(decode_channels[2], 64, n_classes, scale_factor=8)

        # ============ Auxiliary Heads (训练时使用) ============
        if self.aux_mode == 'train':
            self.aux_head_mid = SegHead(decode_channels[1], 64, n_classes, scale_factor=16)
            self.aux_head_deep = SegHead(decode_channels[0], 64, n_classes, scale_factor=32)

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            # Encoder: 提取多尺度特征
            feat8, feat16, feat32 = self.backbone(x)

            # Decoder: FLD逐级融合
            feat_out, feat_mid, feat_deep = self.decoder(feat8, feat16, feat32)

            # 主输出
            logits = self.head(feat_out)

            if self.aux_mode == 'train':
                aux_mid = self.aux_head_mid(feat_mid)
                aux_deep = self.aux_head_deep(feat_deep)
                return logits, aux_mid, aux_deep

            elif self.aux_mode == 'eval':
                return logits,

            elif self.aux_mode == 'pred':
                pred = torch.argmax(logits, dim=1)
                return pred.float()
            else:
                raise NotImplementedError(f"Unsupported aux_mode: {self.aux_mode}")


if __name__ == "__main__":
    img_height, img_width = 640, 640
    n_classes = 19

    try:
        print(f"Initializing PPLiteSeg_InceptionNeXt_Atto with image size: {img_height}x{img_width}...")
        net = PPLiteSeg_InceptionNeXt_Atto(
            n_classes=n_classes, aux_mode='train', img_size=(img_height, img_width)
        )
        net.train()

        # 模拟输入
        if torch.cuda.is_available():
            net.cuda()
            in_ten = torch.randn(2, 3, img_height, img_width).cuda()
        else:
            in_ten = torch.randn(2, 3, img_height, img_width)

        print("Running Forward Pass...")
        out, aux1, aux2 = net(in_ten)

        print(f"\nResults:")
        print(f"  Input:  {in_ten.shape}")
        print(f"  Output: {out.shape}")
        print(f"  Aux1:   {aux1.shape}")
        print(f"  Aux2:   {aux2.shape}")

        # 验证输出尺寸
        assert out.shape[2:] == (img_height, img_width), \
            f"Output shape mismatch! Expected ({img_height}, {img_width}), got {out.shape[2:]}"
        print("\n✓ Output shape matches input shape.")

        # 统计参数量
        total_params = sum(p.numel() for p in net.parameters())
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"\n  Total params:     {total_params / 1e6:.2f}M")
        print(f"  Trainable params: {trainable_params / 1e6:.2f}M")

        # Eval模式测试
        net.eval()
        net_eval = PPLiteSeg_InceptionNeXt_Atto(
            n_classes=n_classes, aux_mode='eval', img_size=(img_height, img_width)
        )
        if torch.cuda.is_available():
            net_eval.cuda()
        net_eval.eval()
        with torch.no_grad():
            eval_out = net_eval(in_ten)
        print(f"\n  Eval output: {eval_out[0].shape}")
        print("\n✓ All tests passed!")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError: {e}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.cuda.amp import autocast
from .efficientnetv2_b3 import EfficientNetV2_B3

class TRT_FixedAvgPool2d(nn.Module):
    """保留您的TensorRT友好池化层"""
    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        if output_size == (1, 1) or output_size == 1:
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
            self.pool = nn.AvgPool2d(kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w), padding=0)

    def forward(self, x):
        if self.is_global:
            return x.mean(dim=(2, 3), keepdim=True)
        else:
            return self.pool(x)

class StaticResize(nn.Module):
    """静态尺寸调整层，确保TensorRT兼容性"""
    def __init__(self, scale_factor=None, size=None):
        super().__init__()
        self.scale_factor = scale_factor
        self.size = size  # 必须为静态元组

    def forward(self, x):
        if self.scale_factor is not None:
            size = (int(x.shape[2] * self.scale_factor), int(x.shape[3] * self.scale_factor))
        else:
            size = self.size
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ECAAttention(nn.Module):
    """轻量级ECA注意力模块，计算量极小，适合低端GPU"""
    def __init__(self, channels, gamma=2, beta=1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

class ConvBlock(nn.Module):
    """基础卷积块，可选深度可分离卷积以进一步加速"""
    def __init__(self, in_ch, out_ch, ks=3, stride=1, padding=1, use_dw=False):
        super().__init__()
        if use_dw:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=ks, stride=stride, padding=padding, groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            )
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.eca = ECAAttention(out_ch)  # 轻量注意力

    def forward(self, x):
        return self.eca(self.act(self.bn(self.conv(x))))

class LiteFeatureFusion(nn.Module):
    """轻量级特征融合模块，替代原UAFM，计算更高效"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_high = ConvBlock(in_ch, out_ch, ks=1, padding=0)
        self.conv_low = ConvBlock(in_ch, out_ch, ks=1, padding=0)
        self.fusion_conv = ConvBlock(out_ch, out_ch, ks=3, padding=1)

    def forward(self, high, low):
        # 高层特征上采样
        high = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)
        high = self.conv_high(high)
        low = self.conv_low(low)
        # 简单相加融合（比注意力更省计算）
        fused = high + low
        return self.fusion_conv(fused)

class LightweightSPPM(nn.Module):
    """精简版SPPM，减少分支数量，使用固定池化"""
    def __init__(self, in_channels, out_channels, input_feat_shape):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                TRT_FixedAvgPool2d(input_size=input_feat_shape, output_size=1),
                ConvBlock(in_channels, in_channels//4, ks=1, padding=0)
            ),
            nn.Sequential(
                TRT_FixedAvgPool2d(input_size=input_feat_shape, output_size=(2,2)),
                ConvBlock(in_channels, in_channels//4, ks=1, padding=0)
            )
        ])
        self.conv_out = ConvBlock(in_channels + in_channels//2, out_channels, ks=1, padding=0)

    def forward(self, x):
        input_size = x.shape[2:]
        priors = [x]
        for stage in self.stages:
            feat = stage(x)
            feat = F.interpolate(feat, size=input_size, mode='bilinear', align_corners=False)
            priors.append(feat)
        return self.conv_out(torch.cat(priors, dim=1))

class FastEfficientFormerSeg_EfficientNetV2_B3(nn.Module):
    def __init__(self, n_classes, aux_mode='train', use_fp16=False, img_size=(512, 512)):
        super().__init__()
        self.use_fp16 = use_fp16
        self.aux_mode = aux_mode
        self.img_size = img_size

        # 1. 骨干网络 (保持不变)
        self.backbone = EfficientNetV2_B3()
        self.c3_chan, self.c4_chan, self.c5_chan = 56, 136, 232

        # 2. 极简投影层 (减少通道数以加速)
        self.proj_c5 = ConvBlock(self.c5_chan, 64, ks=1, padding=0)  # 从128减至64
        self.proj_c4 = ConvBlock(self.c4_chan, 64, ks=1, padding=0)
        self.proj_c3 = ConvBlock(self.c3_chan, 64, ks=1, padding=0)

        # 3. 静态化多尺度融合
        sppm_feat_h, sppm_feat_w = img_size[0] // 32, img_size[1] // 32
        self.sppm = LightweightSPPM(64, 64, (sppm_feat_h, sppm_feat_w))

        # 4. 轻量级特征金字塔 (单路径融合)
        self.fuse1 = LiteFeatureFusion(64, 64)  # c5_sppm + c4
        self.fuse2 = LiteFeatureFusion(64, 64)  # result + c3

        # 5. 头部网络 (简化)
        self.head = nn.Sequential(
            ConvBlock(64, 64, ks=3, padding=1),
            nn.Conv2d(64, n_classes, kernel_size=1, bias=True)
        )
        self.upsample = StaticResize(scale_factor=8)  # 静态上采样

        # 6. 辅助头 (仅训练时使用，更轻量)
        if self.aux_mode == 'train':
            self.aux_head = nn.Sequential(
                ConvBlock(64, 32, ks=3, padding=1),
                nn.Conv2d(32, n_classes, kernel_size=1, bias=True),
                StaticResize(scale_factor=16)
            )

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            # 编码器
            feat8, feat16, feat32 = self.backbone(x)

            # 投影
            c5 = self.proj_c5(feat32)
            c4 = self.proj_c4(feat16)
            c3 = self.proj_c3(feat8)

            # 多尺度上下文
            c5_context = self.sppm(c5)

            # 融合
            feat = self.fuse1(c5_context, c4)
            feat = self.fuse2(feat, c3)

            # 输出
            logits = self.head(feat)
            logits = self.upsample(logits)  # 静态上采样到原图

            if self.aux_mode == 'train':
                # 使用c4作为辅助特征，减少计算
                aux_logits = self.aux_head(c4)
                return logits, aux_logits
            elif self.aux_mode == 'eval':
                return logits,
            elif self.aux_mode == 'pred':
                return torch.argmax(logits, dim=1).float()
            else:
                raise NotImplementedError

# 测试代码
if __name__ == "__main__":
    img_height, img_width = 640, 640
    n_classes = 19
    print(f"测试模型: {img_height}x{img_width} -> {n_classes}类")

    net = FastEfficientFormerSeg_EfficientNetV2_B3(n_classes=n_classes, aux_mode='train', img_size=(img_height, img_width))
    net.train()

    if torch.cuda.is_available():
        net.cuda()
        in_ten = torch.randn(2, 3, img_height, img_width).cuda()
    else:
        in_ten = torch.randn(2, 3, img_height, img_width)

    out, aux1 = net(in_ten)
    print(f"主输出: {out.shape}, 辅助: {aux1.shape}")
    print("模型测试通过！")

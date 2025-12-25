import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.cuda.amp import autocast
from .shvit_s3 import ShVit_S3



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
    def __init__(self, input_size, output_size):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        if isinstance(input_size, int):
            input_size = (input_size, input_size)

        self.output_size = output_size
        self.is_global = output_size == (1, 1)

        if self.is_global:
            self.pool = None
        else:
            # 【修正点】：增加 max(1, ...) 保护，防止 stride 为 0
            stride_h = max(1, input_size[0] // output_size[0])
            stride_w = max(1, input_size[1] // output_size[1])

            # 【修正点】：如果输入比输出还小，kernel 直接取输入尺寸（退化为全局池化）
            kernel_h = input_size[0] - (output_size[0] - 1) * stride_h
            kernel_w = input_size[1] - (output_size[1] - 1) * stride_w

            # 再次保护 kernel 必须大于 0
            kernel_h = max(1, kernel_h)
            kernel_w = max(1, kernel_w)

            self.pool = nn.AvgPool2d(
                kernel_size=(kernel_h, kernel_w),
                stride=(stride_h, stride_w),
                padding=0
            )

    def forward(self, x):
        if self.is_global or self.pool is None:
            return x.mean(dim=(2, 3), keepdim=True)
        return self.pool(x)


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

        # 计算拼接后的通道数：原始通道 + 4个分支的通道
        hidden_dim = in_channels // 4
        concat_channels = in_channels + len(k_sizes) * hidden_dim

        # 融合卷积：将 256 通道降维回 128 通道
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
            # 上采样回 input_size (feature map size)
            feat = F.interpolate(feat, size=input_size, mode='bilinear', align_corners=False)
            priors.append(feat)

        # 拼接
        bottle = torch.cat(priors, dim=1)
        # 融合降维
        out = self.conv_out(bottle)
        return out


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

        # 上采样高层特征
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
        # 保留 Head 内部的上采样，既然保证输入能被整除，这里的 scale_factor 是安全的
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x


class FastEfficientBiSeNet_SHViT_S3(nn.Module):
    def __init__(self, n_classes, aux_mode='train', use_fp16=False, img_size=(512, 512)):
        super(FastEfficientBiSeNet_SHViT_S3, self).__init__()
        self.use_fp16 = use_fp16
        self.aux_mode = aux_mode
        self.img_size = img_size

        # 1. 骨干网络
        self.backbone = ShVit_S3()

        # 【关键修改 1】：确认 SHViT 的通道数与下采样倍率
        # SHViT-S1 典型的 Stage 输出通道 (请根据你具体的 shvit_s1.py 确认)
        # 假设输出为: 1/8(C3), 1/16(C4), 1/32(C5), 1/64(C6)
        # 这里我们选取最后三层进行融合，或者包含 C6
        self.c_chans = [192, 352, 448] # 对应 1/16, 1/32, 1/64

        # 投影层：统一转换到 128 通道
        self.proj_c6 = ConvBNReLU(448, 128, ks=1, padding=0) # 1/64
        self.proj_c5 = ConvBNReLU(352, 128, ks=1, padding=0) # 1/32
        self.proj_c4 = ConvBNReLU(192, 128, ks=1, padding=0) # 1/16

        # 【关键修改 2】：针对 1/64 的 SPPM
        # 因为 SHViT 最大下采样是 64，SPPM 应该放在最深层
        sppm_feat_h = img_size[0] // 64
        sppm_feat_w = img_size[1] // 64
        self.sppm = SPPM_TRT(in_channels=128, out_channels=128,
                             input_feat_shape=(sppm_feat_h, sppm_feat_w))

        # 3. 融合模块 (UAFM 逐级融合)
        self.fuse_c6_c5 = UAFM(high_chan=128, low_chan=128, out_chan=128)
        self.fuse_c5_c4 = UAFM(high_chan=128, low_chan=128, out_chan=128)

        # 【关键修改 3】：输出头
        # 最后的融合特征在 1/16 尺度，所以还原原图需要 scale_factor=16
        self.head = SegmentationHead(128, n_classes, scale_factor=16)

        # 4. 辅助头 (对应调整缩放倍率)
        if self.aux_mode == 'train':
            self.aux_head_c5 = SegmentationHead(128, n_classes, scale_factor=32)
            self.aux_head_c6 = SegmentationHead(128, n_classes, scale_factor=64)

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            # 获取 Backbone 输出 (假设返回四层)
            # feat8(1/8), feat16(1/16), feat32(1/32), feat64(1/64)
            feat16, feat32, feat64 = self.backbone(x)

            # 投影
            c6 = self.proj_c6(feat64) # 1/64
            c5 = self.proj_c5(feat32) # 1/32
            c4 = self.proj_c4(feat16) # 1/16

            # 语义增强 (SPPM 处理最深层)
            c6_sppm = self.sppm(c6)

            # 逐级融合
            feat_fuse_1 = self.fuse_c6_c5(c6_sppm, c5) # (1/64, 1/32) -> 1/32
            feat_final = self.fuse_c5_c4(feat_fuse_1, c4) # (1/32, 1/16) -> 1/16

            # 主输出
            logits = self.head(feat_final)

            if self.aux_mode == 'train':
                aux_out1 = self.aux_head_c5(c5)
                aux_out2 = self.aux_head_c6(c6_sppm)
                return logits, aux_out1, aux_out2
            elif self.aux_mode == 'eval':
                return logits,
            elif self.aux_mode == 'pred':
                return torch.argmax(logits, dim=1).float()
            else:
                raise NotImplementedError


if __name__ == "__main__":
    # 模拟配置
    # 关键：保证 img_height, img_width 是 32 的倍数
    img_height, img_width = 640, 640
    n_classes = 19

    try:
        print(f"Initializing model with image size: {img_height}x{img_width}...")
        net = FastEfficientBiSeNet_SHViT_S3(n_classes=n_classes, aux_mode='train', img_size=(img_height, img_width))
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
        print(f"Input:  {in_ten.shape}")
        print(f"Output: {out.shape}")
        print(f"Aux1:   {aux1.shape}")
        print(f"Aux2:   {aux2.shape}")

        # 简单验证
        assert out.shape[2:] == (img_height, img_width), "Output shape mismatch!"
        print("\nSuccess: Output shape matches Input shape perfectly without external resize.")

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"\nError: {e}")
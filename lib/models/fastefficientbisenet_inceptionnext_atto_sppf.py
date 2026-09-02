import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.cuda.amp import autocast
from .inceptionnext_atto import InceptionNeXt_Atto

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


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (YOLOv5 风格)。
    用一个 k=5 的 MaxPool 串联 3 次，等效于 5/9/13 的多尺度感受野，
    但计算量更小；且全程 stride=1 + same padding，尺寸不变，
    不需要 AdaptiveAvgPool / interpolate，对 TensorRT 非常友好。

    结构: Conv1x1(降维) -> [x, mp(x), mp(mp(x)), mp(mp(mp(x)))] -> Concat -> Conv1x1(融合)
    """

    def __init__(self, in_channels, out_channels, k=5):
        super().__init__()
        hidden_dim = in_channels // 2
        self.cv1 = ConvBNReLU(in_channels, hidden_dim, ks=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = ConvBNReLU(hidden_dim * 4, out_channels, ks=1, padding=0)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


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


class FastEfficientBiSeNet_InceptionNeXt_Atto_SPPF(nn.Module):
    def __init__(self, n_classes, aux_mode='train', use_fp16=False, img_size=(512, 512)):
        """
        img_size: 用于计算 SPPM 静态池化参数，务必与实际输入一致。
        """
        super(FastEfficientBiSeNet_InceptionNeXt_Atto_SPPF, self).__init__()
        self.use_fp16 = use_fp16
        self.aux_mode = aux_mode
        self.img_size = img_size

        # 1. 骨干网络
        self.backbone = InceptionNeXt_Atto()

        # 通道定义 (需根据实际 Backbone 输出调整)
        self.c3_chan = 80  # Stride 8
        self.c4_chan = 160  # Stride 16
        self.c5_chan = 320  # Stride 32

        # 投影层
        self.proj_c5 = ConvBNReLU(self.c5_chan, 128, ks=1, padding=0)
        self.proj_c4 = ConvBNReLU(self.c4_chan, 128, ks=1, padding=0)
        self.proj_c3 = ConvBNReLU(self.c3_chan, 128, ks=1, padding=0)

        # 2. SPPF (尺寸无关，无需静态池化参数)
        self.sppf = SPPF(in_channels=128, out_channels=128, k=5)

        # 3. 融合模块
        self.fuse_context = UAFM(high_chan=128, low_chan=128, out_chan=128)
        self.fuse_final = UAFM(high_chan=128, low_chan=128, out_chan=128)

        # 4. 输出头 (Scale factor = 8, 还原回原图)
        self.head = SegmentationHead(128, n_classes, scale_factor=8)

        # 5. 辅助头
        if self.aux_mode == 'train':
            self.aux_head_c4 = SegmentationHead(128, n_classes, scale_factor=16)
            self.aux_head_c5 = SegmentationHead(128, n_classes, scale_factor=32)

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            # 获取输入尺寸，仅用于校验或备用，不再用于动态 Resize
            H, W = x.size()[2:]

            # Encoder
            feat8, feat16, feat32 = self.backbone(x)

            # Projections
            c5 = self.proj_c5(feat32)
            c4 = self.proj_c4(feat16)
            c3 = self.proj_c3(feat8)

            # SPPF
            c5_sppf = self.sppf(c5)

            # Context Fusion
            feat_context = self.fuse_context(c5_sppf, c4)

            # Spatial Fusion
            feat_final = self.fuse_final(feat_context, c3)

            # Output Head
            # 直接信任 Head 内部的 scale_factor=8 能还原回 (H, W)
            # 只要 H, W 是 32 的倍数，这里一定是对齐的
            logits = self.head(feat_final)

            if self.aux_mode == 'train':
                # 同理，信任 scale_factor 16 和 32
                aux_out1 = self.aux_head_c4(c4)
                aux_out2 = self.aux_head_c5(c5_sppf)
                return logits, aux_out1, aux_out2

            elif self.aux_mode == 'eval':
                return logits,

            elif self.aux_mode == 'pred':
                pred = torch.argmax(logits, dim=1)
                return  pred.float()   #trt11,不能使用float(),否则会报错
                # return pred
            else:
                raise NotImplementedError


if __name__ == "__main__":
    # 模拟配置
    # 关键：保证 img_height, img_width 是 32 的倍数
    img_height, img_width = 640, 640
    n_classes = 19

    try:
        print(f"Initializing model with image size: {img_height}x{img_width}...")
        net = FastEfficientBiSeNet_InceptionNeXt_Atto_SPPF(n_classes=n_classes, aux_mode='train', img_size=(img_height, img_width))
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
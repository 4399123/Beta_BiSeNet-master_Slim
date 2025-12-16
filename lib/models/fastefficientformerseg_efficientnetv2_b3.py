import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.cuda.amp import autocast
from .efficientnetv2_b3 import EfficientNetV2_B3


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
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size

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


class DAPPM_TRT(nn.Module):
    """
    针对 TensorRT 优化的 DAPPM 模块（受 DDRNet 启发）
    包含多尺度池化、深度可分离卷积、Upsample -> Concat -> ConvOut
    这比原 SPPM 提供更丰富的上下文聚合，提升精度，而计算开销控制在小分辨率上。
    """

    def __init__(self, in_channels, out_channels, input_feat_shape=(20, 20)):
        super().__init__()
        hidden_dim = in_channels // 4  # 分支通道，参考 DDRNet 设置

        # 本地分支（无池化）
        self.scale0 = ConvBNReLU(in_channels, hidden_dim, ks=1, padding=0)

        # 多尺度分支参数：(kernel, stride, padding)，固定值以兼容 TRT
        scale_params = [(5, 2, 2), (9, 4, 4), (17, 8, 8)]

        self.scales = nn.ModuleList()
        for k, s, p in scale_params:
            module = nn.Sequential(
                nn.AvgPool2d(kernel_size=k, stride=s, padding=p),
                ConvBNReLU(in_channels, hidden_dim, ks=1, padding=0),
                ConvBNReLU(hidden_dim, hidden_dim, ks=3, padding=1, groups=hidden_dim),  # 深度可分离卷积，提升特征深度
                ConvBNReLU(hidden_dim, hidden_dim, ks=1, padding=0)
            )
            self.scales.append(module)

        # 全局分支
        self.scale_global = nn.Sequential(
            TRT_FixedAvgPool2d(input_size=input_feat_shape, output_size=1),
            ConvBNReLU(in_channels, hidden_dim, ks=1, padding=0),
            ConvBNReLU(hidden_dim, hidden_dim, ks=1, padding=0),  # 简化全局处理
            ConvBNReLU(hidden_dim, hidden_dim, ks=1, padding=0)
        )

        # 拼接通道：5 个分支 (local + 3 scales + global)
        concat_channels = hidden_dim * 5

        # 融合卷积
        self.conv_out = ConvBNReLU(concat_channels, out_channels, ks=1, padding=0)

        # 快捷连接
        self.shortcut = ConvBNReLU(in_channels, out_channels, ks=1, padding=0)

    def forward(self, x):
        input_size = x.shape[2:]

        feats = [self.scale0(x)]

        for module in self.scales:
            feat = module(x)
            feat = F.interpolate(feat, size=input_size, mode='bilinear', align_corners=False)
            feats.append(feat)

        feat_g = self.scale_global(x)
        feat_g = F.interpolate(feat_g, size=input_size, mode='bilinear', align_corners=False)
        feats.append(feat_g)

        # 拼接并融合
        cat = torch.cat(feats, dim=1)
        out = self.conv_out(cat) + self.shortcut(x)
        return out


class AttentionFusionModule(nn.Module):
    """ Attention Fusion Module (inspired by SPCONet)
    整合局部和全局注意力机制，用于融合高低层特征，提升空间-上下文整合。
    已移除AdaptiveAvgPool2d，使用mean操作兼容TRT。
    """

    def __init__(self, in_chan, out_chan):
        super(AttentionFusionModule, self).__init__()
        self.conv_fuse = ConvBNReLU(in_chan * 2, out_chan, ks=1, padding=0)  # 初始融合

        # 局部注意力 (channel-wise，使用mean替换AdaptiveAvgPool2d)
        self.local_atten = nn.Sequential(
            nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 全局注意力 (spatial-wise, lightweight)
        self.global_atten = nn.Sequential(
            nn.Conv2d(out_chan, 1, kernel_size=1, bias=False),  # 压缩通道
            nn.Sigmoid()
        )

        self.conv_out = ConvBNReLU(out_chan, out_chan, ks=3, padding=1)

    def forward(self, x_high, x_low):
        # 上采样高层面并拼接
        high_up = F.interpolate(x_high, size=x_low.shape[2:], mode='bilinear', align_corners=False)
        fuse = torch.cat([high_up, x_low], dim=1)
        fuse = self.conv_fuse(fuse)

        # 应用局部注意力 (使用mean作为全局平均池化)
        local_input = fuse.mean(dim=(2, 3), keepdim=True)  # TRT友好替换
        local_a = self.local_atten(local_input)

        # 全局注意力
        global_a = self.global_atten(fuse)

        atten = local_a * global_a  # 组合注意力

        out = fuse * atten + x_low  # 残差连接
        return self.conv_out(out)


class MultiLevelInteract(nn.Module):
    """ Multi-level Interactive Fusion (inspired by MIFNet)
    多级交互融合，用于逐步耦合多尺度特征。
    """

    def __init__(self, chan, levels=2):
        super(MultiLevelInteract, self).__init__()
        self.levels = nn.ModuleList([
            ConvBNReLU(chan, chan, ks=3, padding=1, groups=chan)  # 深度可分离卷积 for each level
            for _ in range(levels)
        ])

    def forward(self, feats):
        # feats: list of features from different levels
        out = feats[0]
        for i in range(1, len(feats)):
            feat_up = F.interpolate(feats[i], size=out.shape[2:], mode='bilinear', align_corners=False)
            out = self.levels[i-1](out + feat_up)  # 逐级交互
        return out


class SpatialEnhanceBranch(nn.Module):
    """ Lightweight Spatial Enhance Branch (inspired by ISANet SFAB)
    轻量空间特征增强分支，用于捕获缺失空间信息，提升实时精度。
    """

    def __init__(self, in_chan, out_chan):
        super(SpatialEnhanceBranch, self).__init__()
        self.dw_conv = ConvBNReLU(in_chan, out_chan, ks=3, padding=1, groups=in_chan)  # 深度可分离
        self.pw_conv = ConvBNReLU(out_chan, out_chan, ks=1, padding=0)
        self.shortcut = ConvBNReLU(in_chan, out_chan, ks=1, padding=0)

    def forward(self, x):
        out = self.dw_conv(x)
        out = self.pw_conv(out)
        return out + self.shortcut(x)


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


class FastEfficientFormerSeg_EfficientNetV2_B3(nn.Module):
    def __init__(self, n_classes, aux_mode='train', use_fp16=False, img_size=(512, 512)):
        """
        img_size: 用于计算 DAPPM 静态池化参数，务必与实际输入一致。
        """
        super(FastEfficientFormerSeg_EfficientNetV2_B3, self).__init__()
        self.use_fp16 = use_fp16
        self.aux_mode = aux_mode
        self.img_size = img_size

        # 预定义归一化参数，避免JIT tracing警告
        self.register_buffer('mean', torch.tensor([120.0, 114.0, 104.0]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([70.0, 69.0, 73.0]).view(1, 3, 1, 1))

        # 1. 骨干网络
        self.backbone = EfficientNetV2_B3()

        # 通道定义 (需根据实际 Backbone 输出调整)
        self.c3_chan = 56  # Stride 8
        self.c4_chan = 136  # Stride 16
        self.c5_chan = 232  # Stride 32

        # 投影层
        self.proj_c5 = ConvBNReLU(self.c5_chan, 128, ks=1, padding=0)
        self.proj_c4 = ConvBNReLU(self.c4_chan, 128, ks=1, padding=0)
        self.proj_c3 = ConvBNReLU(self.c3_chan, 128, ks=1, padding=0)

        # 2. DAPPM (上下文聚合)
        dappm_feat_h = img_size[0] // 32
        dappm_feat_w = img_size[1] // 32
        self.dappm = DAPPM_TRT(in_channels=128, out_channels=128,
                               input_feat_shape=(dappm_feat_h, dappm_feat_w))

        # 3. 融合模块 (使用更新后的 AFM，无AdaptiveAvgPool2d)
        self.fuse_context = AttentionFusionModule(in_chan=128, out_chan=128)
        self.fuse_final = AttentionFusionModule(in_chan=128, out_chan=128)

        # 4. 多级交互融合
        self.multi_interact = MultiLevelInteract(chan=128, levels=2)

        # 5. 空间增强分支 (新增，受ISANet启发)
        self.spatial_enhance = SpatialEnhanceBranch(128, 128)

        # 6. 输出头
        self.head = SegmentationHead(128, n_classes, scale_factor=8)

        # 7. 辅助头
        if self.aux_mode == 'train':
            self.aux_head_c4 = SegmentationHead(128, n_classes, scale_factor=16)
            self.aux_head_c5 = SegmentationHead(128, n_classes, scale_factor=32)

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            H, W = x.size()[2:]

            # if self.aux_mode == 'pred':
            #     x = x.float()
            #     x = torch.flip(x, dims=[1])
            #     x = (x - self.mean) / self.std

            # Encoder
            feat8, feat16, feat32 = self.backbone(x)

            # Projections
            c5 = self.proj_c5(feat32)
            c4 = self.proj_c4(feat16)
            c3 = self.proj_c3(feat8)

            # DAPPM
            c5_dappm = self.dappm(c5)

            # Context Fusion
            feat_context = self.fuse_context(c5_dappm, c4)

            # Spatial Fusion
            feat_final = self.fuse_final(feat_context, c3)

            # 空间增强 (新增)
            feat_final = self.spatial_enhance(feat_final)

            # 多级交互
            multi_feats = [feat_final, feat_context, c5_dappm]
            feat_final = self.multi_interact(multi_feats)

            # Output Head
            logits = self.head(feat_final)

            if self.aux_mode == 'train':
                aux_out1 = self.aux_head_c4(c4)
                aux_out2 = self.aux_head_c5(c5_dappm)
                return logits, aux_out1, aux_out2

            elif self.aux_mode == 'eval':
                return logits,

            elif self.aux_mode == 'pred':
                feat_out = torch.argmax(logits, dim=1).to(dtype=torch.float32)
                # feat_out = feat_out[None, :, :, :]
                return feat_out
            else:
                raise NotImplementedError


if __name__ == "__main__":
    # 模拟配置
    # 关键：保证 img_height, img_width 是 32 的倍数
    img_height, img_width = 640, 640
    n_classes = 19

    try:
        print(f"Initializing model with image size: {img_height}x{img_width}...")
        net = FastEfficientFormerSeg_EfficientNetV2_B3(n_classes=n_classes, aux_mode='train', img_size=(img_height, img_width))
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
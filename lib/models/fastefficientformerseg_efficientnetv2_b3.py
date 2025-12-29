import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficientnetv2_b3 import EfficientNetV2_B3


# ==========================================
# 1. 基础组件 (SHViT 风格 & 动态 Shape 优化)
# ==========================================

class Conv2dNorm(nn.Sequential):
    """
    基础卷积单元：Conv + BN
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            groups: int = 1,
            bn_weight_init: int = 1,
            **kwargs,
    ):
        super().__init__()
        self.add_module('c', nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False, **kwargs))
        self.add_module('bn', nn.BatchNorm2d(out_channels))

        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        """融合 BN 到 Conv"""
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5

        m = nn.Conv2d(
            in_channels=c.in_channels,
            out_channels=c.out_channels,
            kernel_size=c.kernel_size,
            stride=c.stride,
            padding=c.padding,
            dilation=c.dilation,
            groups=c.groups,
            device=c.weight.device,
            dtype=c.weight.dtype,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class PolyRepConv(nn.Module):
    """
    多分支重参数化卷积 (Poly-Rep-Conv)
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(PolyRepConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups

        # 1. 3x3 分支
        self.branch_3x3 = Conv2dNorm(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups)

        # 2. 1x1 分支
        self.branch_1x1 = Conv2dNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)

        # 3. Identity 分支
        self.use_identity = (stride == 1 and in_channels == out_channels)
        if self.use_identity:
            self.branch_identity = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        out = self.branch_3x3(x) + self.branch_1x1(x)
        if self.use_identity:
            out += self.branch_identity(x)
        return out

    @torch.no_grad()
    def fuse(self) -> nn.Conv2d:
        # Fuse 3x3
        fused_3x3 = self.branch_3x3.fuse()
        kernel_final = fused_3x3.weight
        bias_final = fused_3x3.bias

        # Fuse 1x1 & Pad
        fused_1x1 = self.branch_1x1.fuse()
        kernel_1x1_pad = F.pad(fused_1x1.weight, [1, 1, 1, 1])
        kernel_final += kernel_1x1_pad
        bias_final += fused_1x1.bias

        # Fuse Identity
        if self.use_identity:
            id_conv_wrapper = Conv2dNorm(self.in_channels, self.out_channels, 1, 1, 0, groups=self.groups)
            id_conv = id_conv_wrapper.c
            nn.init.dirac_(id_conv.weight.data)

            id_bn = id_conv_wrapper.bn
            id_bn.weight.data.copy_(self.branch_identity.weight.data)
            id_bn.bias.data.copy_(self.branch_identity.bias.data)
            id_bn.running_mean.data.copy_(self.branch_identity.running_mean.data)
            id_bn.running_var.data.copy_(self.branch_identity.running_var.data)

            fused_id = id_conv_wrapper.fuse()
            kernel_id_pad = F.pad(fused_id.weight, [1, 1, 1, 1])

            kernel_final += kernel_id_pad
            bias_final += fused_id.bias

        m = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.groups,
            bias=True
        )
        m.weight.data.copy_(kernel_final)
        m.bias.data.copy_(bias_final)
        return m


class StripPoolingDynamic(nn.Module):
    """
    全动态 Strip Pooling。
    不使用 AvgPool 或 AdaptivePool，而是使用 torch.mean (ReduceMean)。
    这在 TensorRT 中对应 ReduceMean 算子，完全支持 Dynamic Shapes，无需预设尺寸。
    """

    def __init__(self, in_channels, out_channels):
        super(StripPoolingDynamic, self).__init__()

        # 1x1 Conv 降维
        self.conv1_1 = nn.Sequential(
            Conv2dNorm(in_channels, out_channels, 1),
            nn.SiLU(inplace=True)
        )

        # 处理池化后的特征
        self.conv_h = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.conv_w = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.conv1_1(x)

        # 1. 水平 Strip: 沿着 W 维度求平均 -> (B, C, H, 1)
        # 替代 AvgPool2d(kernel_size=(1, W))
        x_h = x.mean(dim=3, keepdim=True)

        # 2. 垂直 Strip: 沿着 H 维度求平均 -> (B, C, 1, W)
        # 替代 AvgPool2d(kernel_size=(H, 1))
        x_w = x.mean(dim=2, keepdim=True)

        # 3. 卷积提取特征
        x_h = self.conv_h(x_h)
        x_w = self.conv_w(x_w)

        # 4. 扩展回原尺寸
        # 利用 bilinear 插值，这比 expand/broadcast 在 TRT 中通常更安全，
        # 且避免了奇怪的 view 操作。
        target_size = x.shape[2:]
        x_h = F.interpolate(x_h, size=target_size, mode='bilinear', align_corners=False)
        x_w = F.interpolate(x_w, size=target_size, mode='bilinear', align_corners=False)

        # 5. 融合
        return self.act(self.bn(x + x_h + x_w))


# ==========================================
# 2. 核心网络 (无需 img_size)
# ==========================================

class FastEfficientFormerSeg_EfficientNetV2_B3(nn.Module):
    def __init__(self, n_classes, aux_mode='train',use_fp16=False, *args, **kwargs):
        """
        移除了 img_size 参数。现在支持任意尺寸输入。
        """
        super(FastEfficientFormerSeg_EfficientNetV2_B3, self).__init__()
        self.aux_mode = aux_mode

        # 1. Backbone
        self.backbone = EfficientNetV2_B3()

        # Backbone 通道 (已修正为 56, 136, 232)
        self.c3_chan = 56  # Stride 8
        self.c4_chan = 136  # Stride 16
        self.c5_chan = 232  # Stride 32

        self.embed_dim = 128

        # 2. 投影层
        self.proj_c5 = nn.Sequential(
            Conv2dNorm(self.c5_chan, self.embed_dim, 1),
            nn.SiLU(inplace=True)
        )
        self.proj_c4 = nn.Sequential(
            Conv2dNorm(self.c4_chan, self.embed_dim, 1),
            nn.SiLU(inplace=True)
        )
        self.proj_c3 = nn.Sequential(
            Conv2dNorm(self.c3_chan, self.embed_dim, 1),
            nn.SiLU(inplace=True)
        )

        # 3. Context Aggregation (Dynamic Strip Pooling)
        # 不再需要传入 feat_size
        self.context_agg = StripPoolingDynamic(
            in_channels=self.embed_dim,
            out_channels=self.embed_dim
        )

        # 4. Fusion Layer (PolyRepConv)
        self.fuse_block = nn.Sequential(
            PolyRepConv(self.embed_dim * 3, self.embed_dim, stride=1),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1)
        )

        # 5. Head
        self.head = nn.Conv2d(self.embed_dim, n_classes, kernel_size=1)

        # 6. Aux Head
        if self.aux_mode == 'train':
            self.aux_head_c4 = nn.Sequential(
                Conv2dNorm(self.embed_dim, 64, 3, 1, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, n_classes, 1)
            )

    def forward(self, x):
        input_shape = x.shape[2:]

        # Encoder
        c3, c4, c5 = self.backbone(x)

        # Projections
        p5 = self.proj_c5(c5)
        p4 = self.proj_c4(c4)
        p3 = self.proj_c3(c3)

        # Context (Dynamic Pooling)
        p5 = self.context_agg(p5)

        # Upsampling (统一到 p3 的 1/8 尺寸)
        # 动态获取当前 p3 的尺寸
        target_size = p3.shape[2:]
        p5_up = F.interpolate(p5, size=target_size, mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)

        # Concatenate
        feat_cat = torch.cat([p3, p4_up, p5_up], dim=1)

        # Fusion
        feat_fused = self.fuse_block(feat_cat)

        # Head
        logits = self.head(feat_fused)

        # Final Upsample (8x)
        logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)

        if self.aux_mode == 'train':
            aux_out = self.aux_head_c4(p4)
            aux_out = F.interpolate(aux_out, size=input_shape, mode='bilinear', align_corners=False)
            return logits, aux_out

        elif self.aux_mode == 'eval':
            return logits,

        elif self.aux_mode == 'pred':
            return torch.argmax(logits, dim=1).float()

        return logits

    @torch.no_grad()
    def fuse(self):
        """
        递归融合入口
        """

        def fuse_children(net):

            for child_name, child in net.named_children():
                if hasattr(child, 'fuse'):
                    fused = child.fuse()
                    setattr(net, child_name, fused)
                    fuse_children(fused)
                else:
                    fuse_children(child)

        print("Starting model fusion (RepConv -> Conv2d)...")
        fuse_children(self)
        print("Fusion complete.")


if __name__ == "__main__":
    # 配置
    CLASSES = 4

    # 1. 初始化 (不再需要 img_size)
    print(f"Initializing model (Dynamic Input Size)...")
    model = FastEfficientFormerSeg_EfficientNetV2_B3(n_classes=CLASSES, aux_mode='train')
    model.eval()

    # 2. 测试不同尺寸输入 (验证动态性)
    print("\n--- Testing Dynamic Shapes ---")

    # 尺寸 A: 640x640
    x1 = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        y1, _ = model(x1)
    print(f"Input: 640x640 -> Output: {y1.shape}")

    # 尺寸 B: 512x1024 (长宽不等)
    x2 = torch.randn(1, 3, 512, 1024)
    with torch.no_grad():
        y2, _ = model(x2)
    print(f"Input: 512x1024 -> Output: {y2.shape}")

    # 3. 融合测试
    print("\n--- Fusing Model ---")
    model.fuse()

    # 4. 融合后再次测试
    with torch.no_grad():
        y1_fused, _ = model(x1)

    diff = (y1 - y1_fused).abs().max()
    print(f"Fusion Difference: {diff.item():.8f}")

    if diff < 1e-4:
        print("✅ Success: Model is dynamic and fusion-ready.")
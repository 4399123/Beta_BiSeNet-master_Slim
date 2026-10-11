import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d
from torch.cuda.amp import autocast

try:
    from .qwen3_5_0_8b import Qwen3_5_0_8b
except ImportError:
    from qwen3_5_0_8b import Qwen3_5_0_8b


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


class LayerNorm2d(nn.Module):
    """对每个空间位置的通道维执行 LayerNorm，适配 ViT 的特征分布。"""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ViTProjection(nn.Module):
    """LayerNorm2d + 1x1 projection。"""

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.norm = LayerNorm2d(in_chan)
        self.proj = nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.proj(self.norm(x))))


class SPPF(nn.Module):
    """YOLO 风格快速空间金字塔池化：连续 3 次 5x5 MaxPool 聚合多尺度上下文。"""

    def __init__(self, in_chan, out_chan, pool_ks=5):
        super().__init__()
        hidden = max(out_chan // 2, 32)
        self.reduce = ConvBNReLU(in_chan, hidden, ks=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=pool_ks, stride=1, padding=pool_ks // 2)
        self.fuse = ConvBNReLU(hidden * 4, out_chan, ks=1, padding=0)

    def forward(self, x):
        x0 = self.reduce(x)
        x1 = self.pool(x0)
        x2 = self.pool(x1)
        x3 = self.pool(x2)
        return self.fuse(torch.cat((x0, x1, x2, x3), dim=1))


class GatedFusion(nn.Module):
    """通过逐通道门控在语义特征和浅层/空间特征之间自适应选择。"""

    def __init__(self, high_chan, low_chan, out_chan):
        super().__init__()
        self.high_proj = ConvBNReLU(high_chan, out_chan, ks=1, padding=0)
        self.low_proj = ConvBNReLU(low_chan, out_chan, ks=1, padding=0)
        self.gate = nn.Sequential(
            nn.Conv2d(out_chan * 2, out_chan, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = ConvBNReLU(out_chan, out_chan, ks=3, padding=1)

    def forward(self, high, low):
        if high.shape[2:] != low.shape[2:]:
            high = F.interpolate(high, size=low.shape[2:], mode='bilinear', align_corners=False)
        high = self.high_proj(high)
        low = self.low_proj(low)
        gate = self.gate(torch.cat((high, low), dim=1))
        return self.refine(gate * low + (1.0 - gate) * high)


class SpatialBranch(nn.Module):
    """轻量 CNN 空间分支，提供 Qwen stride-16 token 中缺失的真实 1/4、1/8 细节。"""

    def __init__(self, c2=48, c4=96, c8=160):
        super().__init__()
        self.stage2 = ConvBNReLU(3, c2, ks=3, stride=2, padding=1)
        self.stage4 = nn.Sequential(
            ConvBNReLU(c2, c4, ks=3, stride=2, padding=1),
            ConvBNReLU(c4, c4, ks=3, padding=1, groups=c4),
        )
        self.stage8 = nn.Sequential(
            ConvBNReLU(c4, c8, ks=3, stride=2, padding=1),
            ConvBNReLU(c8, c8, ks=3, padding=1, groups=c8),
        )

    def forward(self, x):
        x = self.stage2(x)
        feat4 = self.stage4(x)
        feat8 = self.stage8(feat4)
        return feat4, feat8


class QwenDenseDecoder(nn.Module):
    """Qwen 多层垂直融合 + SPPF + CNN 空间先验的渐进式解码器。"""

    def __init__(self, in_chan=768, embedding_dim=256, n_classes=19, dropout_ratio=0.1):
        super().__init__()
        self.proj_b3 = ViTProjection(in_chan, embedding_dim)
        self.proj_b7 = ViTProjection(in_chan, embedding_dim)
        self.proj_b11 = ViTProjection(in_chan, embedding_dim)

        self.sppf = SPPF(embedding_dim, embedding_dim)
        self.fuse_b7 = GatedFusion(embedding_dim, embedding_dim, embedding_dim)
        self.fuse_b3 = GatedFusion(embedding_dim, embedding_dim, embedding_dim)

        self.spatial = SpatialBranch(c2=48, c4=96, c8=160)
        self.up_to_8 = ConvBNReLU(embedding_dim, 160, ks=3, padding=1)
        self.fuse_s8 = GatedFusion(160, 160, 160)
        self.up_to_4 = ConvBNReLU(160, 96, ks=3, padding=1)
        self.fuse_s4 = GatedFusion(96, 96, 96)

        # 先在 1/4 尺度细化并压缩通道，再 ×4 到原图，最后执行 1×1 分类。
        self.final_refine = ConvBNReLU(96, 32, ks=3, padding=1)
        self.dropout = nn.Dropout2d(dropout_ratio)
        self.classifier = nn.Conv2d(32, n_classes, kernel_size=1, bias=True)

    def forward(self, image, feats, out_size=None):
        block3, block7, block11 = feats
        p3 = self.proj_b3(block3)
        p7 = self.proj_b7(block7)
        p11 = self.sppf(self.proj_b11(block11))

        semantic = self.fuse_b7(p11, p7)
        semantic = self.fuse_b3(semantic, p3)

        spatial4, spatial8 = self.spatial(image)
        x = F.interpolate(semantic, size=spatial8.shape[2:], mode='bilinear', align_corners=False)
        x = self.fuse_s8(self.up_to_8(x), spatial8)
        x = F.interpolate(x, size=spatial4.shape[2:], mode='bilinear', align_corners=False)
        x = self.fuse_s4(self.up_to_4(x), spatial4)
        x = self.final_refine(x)

        target_size = out_size if out_size is not None else (x.shape[2] * 4, x.shape[3] * 4)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return self.classifier(self.dropout(x))


class SegmentationHead(nn.Module):
    """用于训练阶段对中间层 (feat16 / feat32) 进行辅助监督学习的辅助分类头"""
    def __init__(self, in_chan, n_classes, scale_factor=16):
        super(SegmentationHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, 128, ks=3, padding=1)
        self.dropout = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(128, n_classes, kernel_size=1, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x, out_size=None):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        if out_size is not None:
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        elif self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        return x


class SegFormer_Qwen3_5_0_8b(nn.Module):
    """
    Qwen3.5 视觉骨干 + SPPF/门控垂直融合 + 高分辨率空间分支解码器。

    Qwen block 3/7/11 提供同尺度 1/16 语义特征；CNN 空间分支提供真实
    1/8、1/4 细节。最终特征在 1/4 尺度细化后直接上采样 4 倍，并通过
    1x1 卷积输出类别 logits。
    """
    def __init__(
            self,
            n_classes=19,
            aux_mode='train',
            use_fp16=False,
            embedding_dim=256,
            img_size=None,
            ckpt_path=None
    ):
        super(SegFormer_Qwen3_5_0_8b, self).__init__()
        self.use_fp16 = use_fp16
        self.aux_mode = aux_mode
        self.img_size = img_size

        # 1. 骨干网络：直接输出 block 3/7/11 的原始 1/16 特征。
        self.backbone = Qwen3_5_0_8b(ckpt_path=ckpt_path)
        self.in_channels = self.backbone.out_channels  # (768, 768, 768)

        # 2. SPPF + 门控垂直融合 + 轻量 CNN 空间分支解码器。
        self.decode_head = QwenDenseDecoder(
            in_chan=self.in_channels[0],
            embedding_dim=embedding_dim,
            n_classes=n_classes,
            dropout_ratio=0.1
        )

        # 3. 辅助头：分别监督 block 7 与 block 11 的原始 1/16 特征。
        if self.aux_mode == 'train':
            self.aux_head_c4 = SegmentationHead(self.in_channels[1], n_classes, scale_factor=16)
            self.aux_head_c5 = SegmentationHead(self.in_channels[2], n_classes, scale_factor=16)

        # 4. 导出 ONNX (pred 模式) 内嵌预处理 Buffer
        # 注意：输入为 0~255 原始像素，0.5 归一化对应均值和方差均为 127.5
        self.register_buffer('pixel_mean', torch.tensor([127.5, 127.5, 127.5]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std',  torch.tensor([127.5, 127.5, 127.5]).view(1, 3, 1, 1))

    def forward(self, x):
        with autocast(enabled=self.use_fp16):
            H, W = x.size()[2:]

            # pred 模式：支持直接输入原始 0~255 的 BGR 图像 (用于 ONNX / OpenVINO / TensorRT 导出部署)
            if self.aux_mode == 'pred':
                x = x.float()
                x = torch.flip(x, dims=[1])  # BGR 翻转为 RGB
                x = (x - self.pixel_mean) / self.pixel_std

            # Qwen 原始同尺度特征：block 3、7、11，均为 1/16。
            block3, block7, block11 = self.backbone(x)

            # 多层语义特征与原图高分辨率空间特征渐进融合。
            logits = self.decode_head(x, [block3, block7, block11], out_size=(H, W))

            if self.aux_mode == 'train':
                aux_out1 = self.aux_head_c4(block7, out_size=(H, W))
                aux_out2 = self.aux_head_c5(block11, out_size=(H, W))
                return logits, aux_out1, aux_out2

            elif self.aux_mode == 'eval':
                return logits,

            elif self.aux_mode == 'pred':
                pred = torch.argmax(logits, dim=1)
                pred = pred[:, None, :, :]
                return pred.float()
            else:
                raise NotImplementedError

    def get_params(self):
        """
        优化器参数分组接口：
        将主干与 Head 区分开，支持设置不同的学习率（如 Head 用 10x 学习率），
        并支持权重衰减与无衰减参数分离。
        """
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if hasattr(child, 'get_params'):
                child_wd_params, child_nowd_params = child.get_params()
            else:
                child_wd_params, child_nowd_params = [], []
                for p_name, param in child.named_parameters():
                    if param.requires_grad:
                        if ('weight' in p_name) and (param.dim() > 1):
                            child_wd_params.append(param)
                        else:
                            child_nowd_params.append(param)

            if 'head' in name or 'decode' in name or 'aux' in name:
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params

        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


# 兼容别名
FastSegFormer_Qwen3_5_0_8b = SegFormer_Qwen3_5_0_8b


if __name__ == "__main__":
    # 测试配置
    img_height, img_width = 512, 512
    n_classes = 19

    print("=" * 80)
    print("测试 SegFormer_Qwen3_5_0_8b 模型")
    print("=" * 80)

    # 1. 训练模式测试
    print(f"\n[1] 初始化训练模式 (输入尺寸: {img_height}x{img_width}, 类别数: {n_classes})...")
    net = SegFormer_Qwen3_5_0_8b(n_classes=n_classes, aux_mode='train', img_size=(img_height, img_width))
    net.train()

    in_ten = torch.randn(2, 3, img_height, img_width)
    out, aux1, aux2 = net(in_ten)

    print(f"主输出尺寸 (Logits):  {out.shape}")
    print(f"辅助头 1 尺寸 (Aux1):  {aux1.shape}")
    print(f"辅助头 2 尺寸 (Aux2):  {aux2.shape}")
    assert out.shape == (2, n_classes, img_height, img_width), "主输出尺寸不匹配！"

    # 2. 推理/部署模式测试 (输入 0~255 原始 uint8/float BGR 图像)
    print("\n[2] 测试 Pred / ONNX 导出模式 (自动预处理 + BGR 翻转 + ArgMax)...")
    net.aux_mode = 'pred'
    net.eval()
    raw_bgr = torch.randint(0, 256, (1, 3, img_height, img_width)).float()
    pred = net(raw_bgr)
    print(f"Pred 模式输出尺寸 (类别掩码): {pred.shape}")
    assert pred.shape == (1, 1, img_height, img_width), "Pred 输出尺寸不匹配！"

    # 3. 参数分组检查
    wd, nowd, lr_wd, lr_nowd = net.get_params()
    print(f"\n[3] 优化器参数分组:")
    print(f"Backbone Weight Decay:    {len(wd)}")
    print(f"Backbone No Weight Decay: {len(nowd)}")
    print(f"Head Weight Decay:        {len(lr_wd)}")
    print(f"Head No Weight Decay:     {len(lr_nowd)}")

    print("\n恭喜，模型所有测试均完全通过！")

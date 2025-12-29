import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 0. 模拟环境 (如果你在本地运行，请保留原来的 import)
# ==========================================
try:
    from efficientnetv2_b3 import EfficientNetV2_B3
except ImportError:
    print("⚠️ 未找到 efficientnetv2_b3，使用 DummyBackbone 进行测试...")


    class EfficientNetV2_B3(nn.Module):
        def forward(self, x):
            # 模拟 B3 的输出通道: 56, 136, 232; 下采样倍率: 8, 16, 32
            B, _, H, W = x.shape
            c3 = torch.randn(B, 56, H // 8, W // 8, device=x.device)
            c4 = torch.randn(B, 136, H // 16, W // 16, device=x.device)
            c5 = torch.randn(B, 232, H // 32, W // 32, device=x.device)
            return c3, c4, c5


# ==========================================
# 1. 你的原始组件 (保持不变)
# ==========================================
class Conv2dNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bn_weight_init=1,
                 **kwargs):
        super().__init__()
        self.add_module('c',
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False,
                                  **kwargs))
        self.add_module('bn', nn.BatchNorm2d(out_channels))
        nn.init.constant_(self.bn.weight, bn_weight_init)
        nn.init.constant_(self.bn.bias, 0)


class PolyRepConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(PolyRepConv, self).__init__()
        self.branch_3x3 = Conv2dNorm(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups)
        self.branch_1x1 = Conv2dNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, groups=groups)
        self.use_identity = (stride == 1 and in_channels == out_channels)
        if self.use_identity:
            self.branch_identity = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        out = self.branch_3x3(x) + self.branch_1x1(x)
        if self.use_identity:
            out += self.branch_identity(x)
        return out


class StripPoolingDynamic(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StripPoolingDynamic, self).__init__()
        self.conv1_1 = nn.Sequential(Conv2dNorm(in_channels, out_channels, 1), nn.SiLU(inplace=True))
        self.conv_h = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.conv_w = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x = self.conv1_1(x)
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True)
        x_h = self.conv_h(x_h)
        x_w = self.conv_w(x_w)
        target_size = x.shape[2:]
        x_h = F.interpolate(x_h, size=target_size, mode='bilinear', align_corners=False)
        x_w = F.interpolate(x_w, size=target_size, mode='bilinear', align_corners=False)
        return self.act(self.bn(x + x_h + x_w))


# ==========================================
# 2. 修改后的模型类 (支持分步执行)
# ==========================================
class ProfilingModel(nn.Module):
    def __init__(self, n_classes, aux_mode='eval'):
        super().__init__()
        self.aux_mode = aux_mode
        self.backbone = EfficientNetV2_B3()

        # 参数配置
        self.c3_chan, self.c4_chan, self.c5_chan = 56, 136, 232
        self.embed_dim = 128

        # 投影层
        self.proj_c5 = nn.Sequential(Conv2dNorm(self.c5_chan, self.embed_dim, 1), nn.SiLU(inplace=True))
        self.proj_c4 = nn.Sequential(Conv2dNorm(self.c4_chan, self.embed_dim, 1), nn.SiLU(inplace=True))
        self.proj_c3 = nn.Sequential(Conv2dNorm(self.c3_chan, self.embed_dim, 1), nn.SiLU(inplace=True))

        self.context_agg = StripPoolingDynamic(self.embed_dim, self.embed_dim)

        self.fuse_block = nn.Sequential(
            PolyRepConv(self.embed_dim * 3, self.embed_dim, stride=1),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.head = nn.Conv2d(self.embed_dim, n_classes, kernel_size=1)

    # --- 关键修改: 将 Forward 拆分为两部分 ---

    def extract_features(self, x):
        """仅运行 Backbone"""
        return self.backbone(x)

    def decode_head(self, features, input_shape):
        """运行 Neck, Decoder, Head 和 Upsample"""
        c3, c4, c5 = features

        # Projections
        p5 = self.proj_c5(c5)
        p4 = self.proj_c4(c4)
        p3 = self.proj_c3(c3)

        # Context
        p5 = self.context_agg(p5)

        # Upsampling & Fusion
        target_size = p3.shape[2:]
        p5_up = F.interpolate(p5, size=target_size, mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)

        feat_cat = torch.cat([p3, p4_up, p5_up], dim=1)
        feat_fused = self.fuse_block(feat_cat)

        # Head
        logits = self.head(feat_fused)

        # Final Upsample
        logits = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=False)

        return logits


# ==========================================
# 3. 性能分析器
# ==========================================
def run_profiling():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("⚠️ 警告: 正在使用 CPU 测试，时间可能不准确且无法反映生产环境性能。建议使用 GPU。")

    print(f"🚀 初始化模型 (Device: {device})...")
    model = ProfilingModel(n_classes=4).to(device)
    model.eval()

    # 输入数据 (640x640)
    B, C, H, W = 1, 3, 640, 640
    input_data = torch.randn(B, C, H, W).to(device)

    # 参数
    warmup_steps = 20
    test_steps = 100

    records_backbone = []
    records_decoder = []

    print(f"🔥 开始预热 ({warmup_steps} 次)...")
    with torch.no_grad():
        for _ in range(warmup_steps):
            feats = model.extract_features(input_data)
            _ = model.decode_head(feats, (H, W))

    print(f"⏱️ 开始正式测试 ({test_steps} 次)...")

    # 定义 CUDA 事件
    start_event = torch.cuda.Event(enable_timing=True)
    mid_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(test_steps):
        with torch.no_grad():
            torch.cuda.synchronize()
            start_event.record()

            # 1. 特征提取 (Backbone)
            features = model.extract_features(input_data)

            mid_event.record()

            # 2. 后续操作 (Decoder + Head + Upsample)
            _ = model.decode_head(features, (H, W))

            end_event.record()
            torch.cuda.synchronize()  # 等待 GPU 完成

            # 记录时间 (ms)
            records_backbone.append(start_event.elapsed_time(mid_event))
            records_decoder.append(mid_event.elapsed_time(end_event))

    # 计算统计数据
    avg_backbone = np.mean(records_backbone)
    avg_decoder = np.mean(records_decoder)
    total_time = avg_backbone + avg_decoder

    fps = 1000 / total_time

    print("\n" + "=" * 50)
    print(f"📊 性能分析报告 (Input: {H}x{W})")
    print("=" * 50)
    print(f"总平均耗时: \t{total_time:.4f} ms")
    print(f"预估 FPS: \t{fps:.1f}")
    print("-" * 50)

    # 打印 Backbone 数据
    ratio_backbone = (avg_backbone / total_time) * 100
    print(f"🔹 特征提取 (Backbone):")
    print(f"   耗时: {avg_backbone:.4f} ms")
    print(f"   占比: {ratio_backbone:.2f}%")

    # 打印 Decoder 数据
    ratio_decoder = (avg_decoder / total_time) * 100
    print(f"🔸 后续操作 (Decoder/Head):")
    print(f"   耗时: {avg_decoder:.4f} ms")
    print(f"   占比: {ratio_decoder:.2f}%")

    print("-" * 50)
    if ratio_decoder > 40:
        print("💡 建议: 后续操作占比很高 (>40%)。")
        print("   可以考虑优化 StripPooling (使用 TensorRT 插件) 或减少 Projection 的通道数。")
    else:
        print("💡 状态: 计算瓶颈主要在 Backbone，这是正常的。")
    print("=" * 50)


if __name__ == "__main__":
    run_profiling()
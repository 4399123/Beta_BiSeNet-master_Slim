import torch
import time
import torch.nn as nn
from torch.cuda.amp import autocast
import sys
import os

# 设置路径
sys.path.insert(0, '.')

try:
    from lib.models.fastefficientbisenet_efficientnetv2_b3 import FastEfficientBiSeNet_EfficientNetV2_B3
    print("成功导入 SHViT_S3 模型")
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

# 创建测试输入
def encoder_decoder_benchmark(batch_size=1, img_size=(512, 512), num_iterations=100, use_fp16=False):
    """测试编码器和解码器的耗时比"""

    # 创建模型
    model = FastEfficientBiSeNet_EfficientNetV2_B3(n_classes=19, aux_mode='train', use_fp16=use_fp16, img_size=img_size)
    model.eval()

    if torch.cuda.is_available():
        model.cuda()
        device = torch.device('cuda')
        print(f"使用 GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("使用 CPU")

    # 创建测试数据
    x = torch.randn(batch_size, 3, img_size[0], img_size[1]).to(device)
    if use_fp16:
        x = x.half()

    print(f"\n测试配置:")
    print(f"批大小: {batch_size}")
    print(f"图像尺寸: {img_size}")
    print(f"迭代次数: {num_iterations}")
    print(f"使用 FP16: {use_fp16}")
    print(f"设备: {device}")

    # Warm-up
    print("\n预热中...")
    with torch.no_grad():
        for _ in range(10):
            if use_fp16:
                with autocast(enabled=True):
                    model(x)
            else:
                model(x)

    # 测量编码器(特征提取)时间: 只运行 Backbone
    print("测量编码器耗时...")
    encoder_times = []

    with torch.no_grad():
        for i in range(num_iterations):
            start_time = time.time()

            # 只运行编码器(Backbone特征提取)
            with autocast(enabled=use_fp16):
                feat16, feat32, feat64 = model.backbone(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            encoder_times.append((end_time - start_time) * 1000)  # 转换为毫秒

    # 测量解码器(后续操作)时间: 投影层 + SPPM + 融合模块 + 分割头
    print("测量解码器耗时...")
    decoder_times = []

    with torch.no_grad():
        for i in range(num_iterations):
            # 先获取特征（不计算时间）
            with autocast(enabled=use_fp16):
                feat8, feat16, feat32 = model.backbone(x)

            start_time = time.time()

            # 只运行解码器部分
            with autocast(enabled=use_fp16):
                # 投影层
                c5 = model.proj_c5(feat32)
                c4 = model.proj_c4(feat16)
                c3 = model.proj_c3(feat8)

                # SPPM
                c5_sppm = model.sppm(c5)

                # 融合模块
                feat_context = model.fuse_context(c5_sppm, c4)
                feat_final = model.fuse_final(feat_context, c3)

                # 分割头
                logits = model.head(feat_final)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            decoder_times.append((end_time - start_time) * 1000)  # 转换为毫秒

    # 测量完整模型时间(编码器+解码器)
    print("测量完整模型耗时...")
    full_times = []

    with torch.no_grad():
        for i in range(num_iterations):
            start_time = time.time()

            with autocast(enabled=use_fp16):
                output = model(x)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            full_times.append((end_time - start_time) * 1000)  # 转换为毫秒

    # 验证：编码器 + 解码器 ≈ 完整模型
    expected_full_times = [enc + dec for enc, dec in zip(encoder_times, decoder_times)]
    verification_diff = sum(abs(full - expected) for full, expected in zip(full_times, expected_full_times)) / len(full_times)
    print(f"\n验证: 编码器+解码器 vs 完整模型差异: {verification_diff:.4f} ms")

    # 统计分析
    def analyze_times(times, name):
        avg = sum(times) / len(times)
        max_t = max(times)
        min_t = min(times)
        std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5

        print(f"\n{name}统计:")
        print(f"  平均时间: {avg:.2f} ms")
        print(f"  最大时间: {max_t:.2f} ms")
        print(f"  最小时间: {min_t:.2f} ms")
        print(f"  标准差: {std:.2f} ms")

        return avg

    # 分析各项耗时
    enc_avg = analyze_times(encoder_times, "编码器")
    dec_avg = analyze_times(decoder_times, "解码器")
    full_avg = analyze_times(full_times, "完整模型")
    # 正确计算比例：编码器和解码器构成100%的完整模型
    enc_ratio = (enc_avg / full_avg) * 100
    dec_ratio = (dec_avg / full_avg) * 100

    print(f"\n📊 耗时比例分析:")
    print(f"编码器(特征提取)占比: {enc_ratio:.1f}% ({enc_avg:.2f} ms)")
    print(f"解码器(后续处理)占比: {dec_ratio:.1f}% ({dec_avg:.2f} ms)")
    print(f"总占比: {enc_ratio + dec_ratio:.1f}% = 100%")
    print(f"总耗时: {full_avg:.2f} ms")

    # FPS 计算
    fps_encoder = 1000 / enc_avg if enc_avg > 0 else float('inf')
    fps_decoder = 1000 / dec_avg if dec_avg > 0 else float('inf')
    fps_full = 1000 / full_avg

    print(f"\n🎯 性能指标:")
    print(f"编码器 FPS: {fps_encoder:.1f}")
    print(f"解码器 FPS: {fps_decoder:.1f}")
    print(f"完整模型 FPS: {fps_full:.1f}")

    return {
        'encoder_avg_ms': enc_avg,
        'decoder_avg_ms': dec_avg,
        'full_avg_ms': full_avg,
        'encoder_ratio': enc_ratio,
        'decoder_ratio': dec_ratio,
        'fps_full': fps_full
    }

if __name__ == "__main__":
    print("BiSeNet 编码器解码器耗时比测试")
    print("=" * 50)

    # 测试不同配置
    configs = [
        {"batch_size": 1, "img_size": (512, 512), "use_fp16": False},
        {"batch_size": 1, "img_size": (512, 512), "use_fp16": True},
        {"batch_size": 4, "img_size": (512, 512), "use_fp16": False},
        {"batch_size": 4, "img_size": (512, 512), "use_fp16": True},
    ]

    results = []

    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"配置 {i+1}/{len(configs)}")
        result = encoder_decoder_benchmark(**config, num_iterations=50)
        results.append((config, result))

    # 总结对比
    print(f"\n{'='*60}")
    print("📈 配置对比总结")
    print(f"{'配置':<25} {'编码器占比':<12} {'解码器占比':<12} {'总FPS':<10}")

    for (config, result) in results:
        config_desc = f"BS{config['batch_size']}_{config['img_size'][0]}x{config['img_size'][1]}"
        if config['use_fp16']:
            config_desc += "_FP16"
        else:
            config_desc += "_FP32"

        # 直接使用模型返回的比例数据
        print(f"{config_desc:<25} {result['encoder_ratio']:<11.1f}% {result['decoder_ratio']:<11.1f}% {result['fps_full']:<9.1f}")
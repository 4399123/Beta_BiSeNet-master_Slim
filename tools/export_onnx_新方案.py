import argparse
import os
import os.path as osp
import sys

sys.path.insert(0, '.')

import torch
import onnx
from lib.models import model_factory
from configs import set_cfg_from_file
from utils_zl import replace_batchnorm
from timm.utils import reparameterize_model

# ----------------------------------------------------------------
# 1. 引入新工具 (onnxslim + polygraphy)
# ----------------------------------------------------------------
try:
    from onnxslim import slim

    print("[Info] onnxslim module loaded.")
except ImportError:
    print("Error: onnxslim not found. Please install: pip install onnxslim")
    sys.exit(1)

try:
    from polygraphy.backend.onnx import fold_constants

    print("[Info] polygraphy module loaded.")
except ImportError:
    print("Error: polygraphy not found. Please install: pip install polygraphy[onnx]")
    sys.exit(1)

torch.set_grad_enabled(False)


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
                       default='../configs/fastefficientbisenet_blueface_efficientnetv2_b3.py', )
    parse.add_argument('--weight-path', dest='weight_pth', type=str,
                       default='../pt/fastefficientbisenet_efficientnetv2_b3.pt')
    parse.add_argument('--outpath', dest='out_pth', type=str,
                       default='./onnx/best.onnx')
    parse.add_argument('--ousmitpath', dest='outsmi_pth', type=str,
                       default='./onnx/best-smi.onnx')
    parse.add_argument('--aux-mode', dest='aux_mode', type=str,
                       default='pred')
    args = parse.parse_args()

    # ----------------------------------------------------------------
    # 2. 模型加载与 PyTorch 侧预处理
    # ----------------------------------------------------------------
    cfg = set_cfg_from_file(args.config)
    if cfg.use_sync_bn: cfg.use_sync_bn = False

    print(f"Loading model: {cfg.model_type} from {args.weight_pth}")
    net = model_factory[cfg.model_type](cfg.n_cats, aux_mode=args.aux_mode, use_fp16=False)
    net.load_state_dict(torch.load(args.weight_pth, map_location='cpu'), strict=False)
    net.eval()

    # 保持原有的重参数化逻辑，这对推理加速至关重要
    # replace_batchnorm(net)
    print("Applying reparameterize_model (folding BN/Conv)...")
    reparameterize_model(net)

    dummy_input = torch.randn(1, 3, 512, 512)
    input_name = 'input'
    output_name = 'output'

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.out_pth), exist_ok=True)
    if os.path.dirname(args.outsmi_pth):
        os.makedirs(os.path.dirname(args.outsmi_pth), exist_ok=True)

    # ----------------------------------------------------------------
    # 3. 导出原始 ONNX
    # ----------------------------------------------------------------
    print(f'Step 1: Exporting raw ONNX to {args.out_pth}...')
    torch.onnx.export(net, dummy_input, args.out_pth,
                      input_names=[input_name],
                      output_names=[output_name],
                      verbose=False, opset_version=16,
                      do_constant_folding=True,  # PyTorch 自带折叠
                      dynamic_axes={
                          input_name: {0: 'batch_size'},
                          output_name: {0: 'batch_size'}
                      }
                      )

    # ----------------------------------------------------------------
    # 4. 现代化优化流水线: onnxslim -> polygraphy
    # ----------------------------------------------------------------
    print('Step 2: Optimizing with onnxslim (Architecture Simplification)...')
    try:
        # 替代了原来的 onnxoptimizer + simplify
        # model_check=True 会自动验证模型有效性
        slimmed_model = slim(args.out_pth, model_check=True)
        print(' -> onnxslim optimization complete.')
    except Exception as e:
        print(f"Error during onnxslim: {e}")
        sys.exit(1)

    print('Step 3: Sanitizing with Polygraphy (TensorRT Compatibility)...')
    try:
        # 使用 Polygraphy 进行最终的常量折叠和清理
        final_model = fold_constants(slimmed_model)

        # 保存最终模型
        onnx.save(final_model, args.outsmi_pth)
        print(f' -> Optimization successful! Saved to: {args.outsmi_pth}')

        # 简单对比大小
        raw_size = os.path.getsize(args.out_pth) / 1024 / 1024
        final_size = os.path.getsize(args.outsmi_pth) / 1024 / 1024
        print(f" -> Size: {raw_size:.2f} MB -> {final_size:.2f} MB")

    except Exception as e:
        print(f"Error during Polygraphy: {e}")
        # 兜底策略：如果 Polygraphy 失败，保存 onnxslim 的结果
        onnx.save(slimmed_model, args.outsmi_pth)
        print(f" -> Saved onnxslim result only (Polygraphy failed).")


if __name__ == "__main__":
    main()
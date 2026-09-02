import argparse
import os
from pathlib import Path
import sys
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, ".")

import numpy as np
import onnx
import torch
from torch.utils.data import DataLoader, Dataset

from onnx_segmentation_utils import parse_hw, preprocess_image, read_annotation_pairs


def _patch_ppq_environment():
    """Apply runtime patches to PPQ for high ONNX opset (>=14) and Resize op compatibility."""
    import torch.nn.functional as F
    from ppq.api import register_operation_handler
    from ppq.core import TargetPlatform
    import ppq.IR.base.opdef as opdef

    # 1. Patch Resize_forward handler in PPQ Torch Executor
    def patched_resize_forward(op, values, ctx=None, **kwargs):
        value = values[0]
        scale_factor = values[2].cpu() if (len(values) > 2 and values[2] is not None) else None
        size = values[-1].cpu().tolist() if (len(values) == 4 and values[-1] is not None) else None
        mode = op.attributes.get("mode", "nearest")
        if mode == "cubic":
            mode = "bicubic"
        linear_mode_map = {1: "linear", 2: "bilinear", 3: "trilinear"}
        coordinate_transformation_mode = op.attributes.get("coordinate_transformation_mode", "half_pixel")
        align_corners = op.attributes.get("align_corners", 0) == 1
        if coordinate_transformation_mode == "align_corners":
            align_corners = True
        elif coordinate_transformation_mode in ("half_pixel", "pytorch_half_pixel", "asymmetric"):
            align_corners = False

        if size is None or len(size) == 0:
            size = None
            if scale_factor is not None:
                if scale_factor.numel() == 1:
                    scale_factor = scale_factor.item()
                else:
                    scale_factor = scale_factor.tolist()
                    if len(scale_factor) == 4:
                        scale_factor = scale_factor[2:]
        else:
            scale_factor = None
            if len(size) == 4:
                size = size[2:]

        if size is not None:
            size = [int(s) for s in size]
        mode_str = linear_mode_map.get(len(size or [1, 1]), mode) if mode == "linear" else mode
        return F.interpolate(
            value,
            size=size,
            scale_factor=scale_factor,
            mode=mode_str,
            align_corners=align_corners if mode_str in ("linear", "bilinear", "bicubic", "trilinear") else None,
        )

    for plat in TargetPlatform:
        register_operation_handler(patched_resize_forward, "Resize", plat)

    # 2. Patch Resize socket definition
    def patched_resize_socket(op):
        in_plat = [TargetPlatform.UNSPECIFIED, TargetPlatform.SOI, TargetPlatform.SOI, TargetPlatform.SOI]
        return opdef.OpSocket(
            op=op,
            in_plat=in_plat[: op.num_of_input],
            links=[opdef.VLink(in_idx=0, out_idx=0)],
        )

    opdef.DEFAULT_SOCKET_TABLE["Resize"] = patched_resize_socket

    # 3. Relax strict opset version check
    def relaxed_check_opset(op, min_version_supported, max_version_supported, strict_check=True):
        return

    opdef.CHECK_OPSET = relaxed_check_opset


class BlueFaceDataset(Dataset):
    """PyTorch Dataset yielding preprocessed BlueFace images for calibration."""

    def __init__(self, pairs, input_size, max_samples=None):
        if max_samples is not None:
            pairs = pairs[:max_samples]
        if not pairs:
            raise ValueError("No calibration samples were selected")
        self.pairs = pairs
        self.input_size = input_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        image_path, _ = self.pairs[idx]
        image_np = preprocess_image(image_path, self.input_size)  # [1, 3, H, W]
        return torch.from_numpy(image_np.squeeze(0))


def find_sensitive_nodes(
    onnx_path: Path,
    preserve_decoder: bool = True,
    preserve_late_backbone: bool = True,
) -> List[str]:
    """Find numerically sensitive segmentation nodes to preserve in FP32."""
    model = onnx.load(str(onnx_path))
    sensitive_nodes = []
    for node in model.graph.node:
        name_lower = node.name.lower()
        is_stem_or_head = "stem_0" in name_lower or "/head/" in name_lower
        is_decoder = any(
            prefix in name_lower
            for prefix in ("/proj_c", "/sppf/", "/fuse_context/", "/fuse_final/")
        )
        is_late_backbone = "/stages_2/" in name_lower or "/stages_3/" in name_lower
        if (
            is_stem_or_head
            or (preserve_decoder and is_decoder)
            or (preserve_late_backbone and is_late_backbone)
        ) and node.op_type in ("Conv", "Gemm", "MatMul"):
            sensitive_nodes.append(node.name)
    return sensitive_nodes


def quantize_with_ppq(args, pairs, input_size, max_samples):
    """Execute high-precision PTQ pipeline using PPQ (CLE + Bias Correction + Percentile + AdaRound)."""
    _patch_ppq_environment()

    from ppq import TargetPlatform
    from ppq.api import QuantizationSettingFactory, export_ppq_graph, quantize_onnx_model

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA is not available, falling back to CPU for PPQ.")
        device = "cpu"

    dataset = BlueFaceDataset(pairs, input_size, max_samples)
    batch_size = max(1, args.batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    calib_steps = min(512, max(8, len(dataloader)))

    print(f"\n[PPQ Quantization Pipeline]")
    print(f"  Device:             {device}")
    print(f"  Calibration data:   {len(dataset)} samples (running {calib_steps} batches, batch_size={batch_size})")
    print(f"  Target Platform:    {args.target_platform}")
    print(f"  Activation Calib:   {args.calibration_method}")
    print(f"  Equalization (CLE): {args.equalization}")
    print(f"  Bias Correction:    {args.bias_correction}")
    print(f"  AdaRound (Opt):     {args.adaround} (steps={args.adaround_steps}, lr={args.adaround_lr})")

    # Read original ONNX opset version
    orig_model = onnx.load(str(args.input_path))
    orig_opset = 16
    for imp in orig_model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            orig_opset = max(orig_opset, imp.version)

    # Configure PPQ Quantization Setting
    setting = QuantizationSettingFactory.default_setting()
    setting.equalization = args.equalization
    setting.bias_correct = args.bias_correction
    setting.blockwise_reconstruction = args.adaround
    if args.adaround:
        setting.blockwise_reconstruction_setting.steps = args.adaround_steps
        setting.blockwise_reconstruction_setting.lr = args.adaround_lr
        setting.blockwise_reconstruction_setting.collecting_device = device

    # Map activation calibration algorithm
    calib_algo_map = {
        "percentile": "percentile",
        "mse": "mse",
        "entropy": "kl",
        "kl": "kl",
        "minmax": "minmax",
    }
    setting.quantize_activation_setting.calib_algorithm = calib_algo_map.get(args.calibration_method, "percentile")

    # Handle sensitive nodes (keep in FP32). An explicit allow-list takes
    # precedence and is useful for accuracy-first mixed-precision searches.
    exclude_nodes = set()
    quantize_patterns = [p.strip() for p in args.quantize_node_patterns.split(",") if p.strip()]
    if quantize_patterns:
        matched = []
        for node in orig_model.graph.node:
            if node.op_type not in ("Conv", "MatMul"):
                continue
            if any(pattern.lower() in node.name.lower() for pattern in quantize_patterns):
                matched.append(node.name)
            else:
                exclude_nodes.add(node.name)
        if not matched:
            raise ValueError(
                f"--quantize-node-patterns did not match any Conv/MatMul nodes: {quantize_patterns}"
            )
        print(f"  Explicit INT8 node allow-list ({len(matched)} nodes):")
        for node_name in matched:
            print(f"    - {node_name}")
    elif args.preserve_sensitive:
        detected = find_sensitive_nodes(
            args.input_path, args.preserve_decoder, args.preserve_late_backbone
        )
        exclude_nodes.update(detected)
    if args.exclude_nodes:
        exclude_nodes.update(n.strip() for n in args.exclude_nodes.split(",") if n.strip())

    if exclude_nodes:
        print(f"  Preserving sensitive nodes in FP32 ({len(exclude_nodes)} nodes):")
        for node_name in sorted(exclude_nodes):
            print(f"    - {node_name}")
            setting.dispatching_table.append(operation=node_name, platform=TargetPlatform.FP32)

    platform_map = {
        "onnxruntime": TargetPlatform.ONNXRUNTIME,
        "trt_int8": TargetPlatform.TRT_INT8,
        "openvino": TargetPlatform.OPENVINO_INT8,
    }
    target_platform = platform_map.get(args.target_platform, TargetPlatform.ONNXRUNTIME)

    if args.quantize_conv_only:
        # Keep numerically fragile non-convolution operators in floating point
        # for both PPQ and ORT. This is especially important for InstanceNorm,
        # attention gates and Resize in this segmentation model.
        for node in orig_model.graph.node:
            if node.op_type not in ("Conv", "MatMul"):
                setting.dispatching_table.append(operation=node.name, platform=TargetPlatform.FP32)

    def collate_fn(batch):
        return batch.to(device)

    height, width = input_size
    quantized_graph = quantize_onnx_model(
        onnx_import_file=str(args.input_path),
        calib_dataloader=dataloader,
        calib_steps=calib_steps,
        input_shape=[1, 3, height, width],
        platform=target_platform,
        setting=setting,
        collate_fn=collate_fn,
        device=device,
        verbose=1,
    )

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_export = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")

    export_ppq_graph(
        graph=quantized_graph,
        platform=target_platform,
        graph_save_to=str(temp_export),
    )

    # Post-process exported ONNX: ensure opset version matches modern ONNX ops (e.g. HardSwish)
    actual_exported = temp_export if temp_export.is_file() else temp_export.with_suffix(".onnx")
    if not actual_exported.is_file():
        # Fallback in case ppq modified filename
        for candidate in output_path.parent.glob(f"{output_path.stem}*"):
            if "tmp" in candidate.name and candidate.is_file():
                actual_exported = candidate
                break

    exported_model = onnx.load(str(actual_exported))
    has_default = False
    for imp in exported_model.opset_import:
        if imp.domain in ("", "ai.onnx"):
            imp.domain = ""
            imp.version = max(imp.version, orig_opset)
            has_default = True
    if not has_default:
        exported_model.opset_import.append(onnx.helper.make_opsetid("", orig_opset))

    try:
        onnx.checker.check_model(exported_model)
    except Exception as e:
        print(f"[Warning] ONNX model checker note: {e}")

    onnx.save(exported_model, str(output_path))
    if actual_exported.is_file() and actual_exported != output_path:
        actual_exported.unlink()

    # Sync copy to both tools/onnx and onnx for seamless IDE and script compatibility
    try:
        alt_output = Path("onnx") / output_path.name if "tools" in str(output_path) else Path("tools/onnx") / output_path.name
        alt_output.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(output_path), str(alt_output))
    except Exception:
        pass

    if output_path.is_file():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n[Success] INT8 ONNX model successfully saved to:")
        print(f"  Path: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
    else:
        raise RuntimeError(f"Failed to save INT8 ONNX model to {output_path}")

    # Verify model with ONNX Runtime
    verify_onnx_model(output_path, pairs, input_size)


def verify_onnx_model(output_path: Path, pairs, input_size):
    """Verify that exported INT8 model can be loaded by ONNX Runtime and run inference successfully."""
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print(f"[Verification] Validating model with ONNX Runtime...")
    print(f"  Target file: {output_path}")

    if not output_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {output_path}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size:   {size_mb:.2f} MB")

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    print(f"  Input meta:  name='{input_meta.name}', shape={input_meta.shape}, type={input_meta.type}")
    print(f"  Output meta: name='{output_meta.name}', shape={output_meta.shape}, type={output_meta.type}")

    sample_img_path = pairs[0][0]
    sample_img = preprocess_image(sample_img_path, input_size)
    print(f"  Running inference on sample: {Path(sample_img_path).name} ...")
    pred = session.run(None, {input_meta.name: sample_img})[0]
    unique_classes, counts = np.unique(pred, return_counts=True)
    print(f"  Inference Result: SUCCESS!")
    print(f"    - Output shape: {pred.shape}")
    print(f"    - Predicted classes: {unique_classes.tolist()}")
    print(f"    - Class pixel counts: {dict(zip(unique_classes.tolist(), counts.tolist()))}")
    print("=" * 60)


def quantize_with_ort(args, pairs, input_size, max_samples):
    """Fallback standard ONNX Runtime static quantization."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class ORTCalibrationReader(CalibrationDataReader):
        def __init__(self, pairs, input_name, input_size, max_samples):
            if max_samples is not None:
                pairs = pairs[:max_samples]
            self.pairs = pairs
            self.input_name = input_name
            self.input_size = input_size
            self.index = 0

        def get_next(self):
            if self.index >= len(self.pairs):
                return None
            image_path, _ = self.pairs[self.index]
            self.index += 1
            if self.index == 1 or self.index % 50 == 0 or self.index == len(self.pairs):
                print(f"  Calibrating {self.index}/{len(self.pairs)}: {image_path.name}")
            return {self.input_name: preprocess_image(image_path, self.input_size)}

        def rewind(self):
            self.index = 0

    reader = ORTCalibrationReader(pairs, args.input_name, input_size, max_samples)
    method_map = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
        "mse": CalibrationMethod.Distribution,
    }
    method = method_map.get(args.calibration_method, CalibrationMethod.Percentile)

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Quantizing with ORT {args.input_path} ({len(reader.pairs)} samples)...")

    nodes_to_exclude = []
    nodes_to_quantize = None
    source_model = onnx.load(str(args.input_path))
    quantize_patterns = [p.strip() for p in args.quantize_node_patterns.split(",") if p.strip()]
    if quantize_patterns:
        nodes_to_quantize = [
            node.name
            for node in source_model.graph.node
            if node.op_type in ("Conv", "MatMul")
            and any(pattern.lower() in node.name.lower() for pattern in quantize_patterns)
        ]
        if not nodes_to_quantize:
            raise ValueError(
                f"--quantize-node-patterns did not match any Conv/MatMul nodes: {quantize_patterns}"
            )
        print(f"  Explicit INT8 node allow-list ({len(nodes_to_quantize)} nodes)")
    elif args.preserve_sensitive:
        nodes_to_exclude.extend(
            find_sensitive_nodes(
                args.input_path, args.preserve_decoder, args.preserve_late_backbone
            )
        )
    if args.exclude_nodes:
        nodes_to_exclude.extend(n.strip() for n in args.exclude_nodes.split(",") if n.strip())

    # The network contains InstanceNorm, attention gates and several elementwise
    # products.  Quantizing those activations with a single tensor range causes
    # severe saturation on small segmentation classes.  By default ORT PTQ is
    # restricted to convolution nodes (the high-impact INT8 kernels); all other
    # operators remain FP32 while still using static QDQ calibration for Conv.
    op_types_to_quantize = None
    if args.quantize_conv_only and nodes_to_quantize is None:
        # ORT can select operator types directly. This avoids creating QDQ
        # ranges for every intermediate tensor (and is much faster than
        # excluding hundreds of individual nodes).
        op_types_to_quantize = ["Conv", "MatMul"]

    quantize_static(
        model_input=args.input_path,
        model_output=output_path,
        calibration_data_reader=reader,
        quant_format=(QuantFormat.QOperator if args.quant_format == "qoperator" else QuantFormat.QDQ),
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=method,
        nodes_to_quantize=nodes_to_quantize,
        nodes_to_exclude=nodes_to_exclude if nodes_to_exclude else None,
        op_types_to_quantize=op_types_to_quantize,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    if output_path.is_file():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n[Success] INT8 ONNX model successfully saved to:")
        print(f"  Path: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
    else:
        raise RuntimeError(f"Failed to save INT8 ONNX model to {output_path}")

    # Verify model with ONNX Runtime
    verify_onnx_model(output_path, pairs, input_size)


def main():
    parser = argparse.ArgumentParser(description="High-Precision Static INT8 Quantization for Segmentation ONNX Models")
    default_input = "./onnx/best-smi.onnx" if Path("./onnx/best-smi.onnx").is_file() else "./onnx/best.onnx"
    parser.add_argument("--input-path", default=default_input, help="Source FP32 ONNX model")
    parser.add_argument("--output-path", default="./onnx/best_int8.onnx", help="Output static INT8 ONNX model")
    parser.add_argument(
        "--dataset-root",
        default=r"D:\F\ABlueFaceProj\20240618\Seg\BlueFaceDataX2",
        help="BlueFaceDataX2 root directory",
    )
    parser.add_argument("--annotations", default="train.txt", help="Calibration file relative to --dataset-root")
    parser.add_argument("--input-name", default="input", help="ONNX input tensor name")
    parser.add_argument("--input-size", default="512,512", help="Fixed ONNX input size as H,W")
    parser.add_argument("--max-samples", type=int, default=300, help="Maximum calibration images; use 0 for all")
    parser.add_argument("--batch-size", type=int, default=4, help="Calibration DataLoader batch size")
    parser.add_argument("--engine", choices=("ppq", "ort"), default="ppq", help="Quantization engine (ppq or ort)")
    parser.add_argument(
        "--quantize-conv-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantize only Conv/MatMul and keep nonlinear/normalization/resize ops in FP32",
    )
    parser.add_argument(
        "--quantize-node-patterns",
        default=(
            "/proj_c3/conv/Conv,/proj_c4/conv/Conv,/proj_c5/conv/Conv,"
            "/sppf/cv1/conv/Conv,/sppf/cv2/conv/Conv,"
            "/fuse_context/conv_high/conv/Conv,/fuse_context/conv_low/conv/Conv,"
            "/fuse_context/conv_out/fused_conv/Conv,"
            "/fuse_final/conv_high/conv/Conv,/fuse_final/conv_low/conv/Conv,"
            "/fuse_final/conv_out/fused_conv/Conv,/head/conv/fused_conv/Conv"
        ),
        help="Comma-separated node-name substrings for the validated CPU QOperator profile",
    )
    parser.add_argument(
        "--quant-format",
        choices=("qoperator", "qdq"),
        default="qoperator",
        help="ORT output representation; QOperator is usually faster on CPU, QDQ is more portable",
    )
    parser.add_argument(
        "--calibration-method",
        choices=("percentile", "mse", "entropy", "minmax"),
        default="percentile",
        help="Activation calibration algorithm",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--target-platform", choices=("onnxruntime", "trt_int8", "openvino"), default="onnxruntime")

    # High-precision PTQ options
    parser.add_argument("--adaround", action=argparse.BooleanOptionalAction, default=False, help="Enable AdaRound blockwise reconstruction optimization")
    parser.add_argument("--adaround-steps", type=int, default=200, help="Number of AdaRound optimization steps")
    parser.add_argument("--adaround-lr", type=float, default=1e-3, help="Learning rate for AdaRound")
    parser.add_argument("--equalization", action=argparse.BooleanOptionalAction, default=False, help="Enable Cross-Layer Equalization (CLE)")
    parser.add_argument("--bias-correction", action=argparse.BooleanOptionalAction, default=False, help="Enable Bias Correction")
    parser.add_argument("--preserve-sensitive", "--exclude-head", dest="preserve_sensitive", action=argparse.BooleanOptionalAction, default=True, help="Preserve sensitive nodes (Head, Stem) in FP32")
    parser.add_argument(
        "--preserve-decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep projection, SPPF, fusion attention and segmentation head nodes in FP32",
    )
    parser.add_argument(
        "--preserve-late-backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep backbone stages 2/3 in FP32; stage 0/1 remain statically quantized",
    )
    parser.add_argument("--exclude-nodes", default="", help="Comma-separated custom node names to force in FP32")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.is_file():
        # Resolve paths consistently from either the project root or tools/.
        candidates = [SCRIPT_DIR / input_path, SCRIPT_DIR / "onnx" / input_path.name,
                      Path.cwd() / input_path, Path.cwd() / "onnx" / input_path.name]
        input_path = next((candidate for candidate in candidates if candidate.is_file()), input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"FP32 ONNX model not found: {input_path}")
    args.input_path = input_path

    input_size = parse_hw(args.input_size)
    pairs = read_annotation_pairs(args.dataset_root, args.annotations)
    max_samples = None if args.max_samples == 0 else args.max_samples

    if args.engine == "ppq":
        quantize_with_ppq(args, pairs, input_size, max_samples)
    else:
        quantize_with_ort(args, pairs, input_size, max_samples)


if __name__ == "__main__":
    main()


# encoding: utf-8
import argparse
import os
import os.path as osp
import shutil
import sys
from pathlib import Path



import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.transformers import float16

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def resolve_path(rel_or_abs):
    p = Path(rel_or_abs)
    if p.is_file():
        return str(p)
    candidates = [
        SCRIPT_DIR / rel_or_abs,
        PROJECT_ROOT / rel_or_abs,
        SCRIPT_DIR / "onnx" / Path(rel_or_abs).name,
        PROJECT_ROOT / "onnx" / Path(rel_or_abs).name,
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return str(p)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert FP32 ONNX model to FP16 with optional FP32 IO")
    parser.add_argument(
        "--input-path",
        dest="input_pth",
        type=str,
        default="./onnx/best-smi.onnx",
        help="Path to the input FP32 ONNX model",
    )
    parser.add_argument(
        "--output-path",
        dest="output_pth",
        type=str,
        default="./onnx/best_fp16.onnx",
        help="Path to save the output FP16 ONNX model",
    )
    parser.add_argument(
        "--keep-io-types",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep graph input/output types as FP32 (recommended, transparent to downstream apps)",
    )
    return parser.parse_args()


def refresh_tensor_type_metadata(model):
    """基于实际节点连接重新推导全部中间张量的类型和形状。"""

    def iter_graphs(graph):
        yield graph
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    yield from iter_graphs(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for subgraph in attr.graphs:
                        yield from iter_graphs(subgraph)

    removed = 0
    for graph in iter_graphs(model.graph):
        removed += len(graph.value_info)
        del graph.value_info[:]

    print(f"  [fix] Cleaned {removed} intermediate ValueInfo entries")
    try:
        return onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as e:
        print(f"  [warn] Shape inference note: {e}")
        return model


def ensure_topological_order(model):
    """确保插入的 input Cast 节点排在最前面，满足 ONNX 拓扑排序规范。"""
    input_names = {inp.name for inp in model.graph.input}
    input_casts = []
    other_nodes = []
    for node in model.graph.node:
        if node.op_type == "Cast" and any(inp in input_names for inp in node.input):
            input_casts.append(node)
        else:
            other_nodes.append(node)

    if input_casts:
        del model.graph.node[:]
        model.graph.node.extend(input_casts + other_nodes)
        print(f"  [fix] Topologically sorted {len(input_casts)} input Cast node(s)")
    return model


def convert_to_fp16(input_path, output_path, keep_io_types=True):
    input_path = resolve_path(input_path)
    if not osp.isfile(input_path):
        raise FileNotFoundError(f"Input model not found: {input_path}")

    print(f"\n============================================================")
    print(f"  FP32 -> FP16 ONNX Conversion")
    print(f"============================================================")
    print(f"  Input model  : {input_path} ({os.path.getsize(input_path)/(1024*1024):.2f} MB)")
    print(f"  Output model : {output_path}")
    print(f"  Keep IO FP32 : {keep_io_types} (Input accepts standard Float32: {keep_io_types})")

    model = onnx.load(input_path)

    print("\n[1/4] Converting internal weights and operators to Float16...")
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_types,
        disable_shape_infer=False,
    )

    print("[2/4] Ensuring graph topological order and refreshing metadata...")
    model_fp16 = ensure_topological_order(model_fp16)
    model_fp16 = refresh_tensor_type_metadata(model_fp16)

    print("[3/4] Validating model with onnx.checker...")
    try:
        onnx.checker.check_model(model_fp16)
        print("  [OK] onnx.checker passed successfully!")
    except Exception as e:
        print(f"  [WARN] onnx.checker warning: {e}")

    print("[4/4] Validating with ONNX Runtime CPU execution...")
    try:
        sess = ort.InferenceSession(model_fp16.SerializeToString(), providers=["CPUExecutionProvider"])
        inp_meta = sess.get_inputs()[0]
        out_meta = sess.get_outputs()[0]
        print(f"  [OK] ONNX Runtime load passed!")
        print(f"    - Input  name: '{inp_meta.name}', shape: {inp_meta.shape}, type: {inp_meta.type}")
        print(f"    - Output name: '{out_meta.name}', shape: {out_meta.shape}, type: {out_meta.type}")

        # Run test inference with float32 input
        test_dtype = np.float32 if inp_meta.type == "tensor(float)" else np.float16
        dummy = np.random.randn(1, 3, 512, 512).astype(test_dtype)
        res = sess.run(None, {inp_meta.name: dummy})[0]
        print(f"    - Test inference successful! Output shape: {res.shape}, dtype: {res.dtype}")
    except Exception as e:
        raise RuntimeError(f"ONNX Runtime validation failed: {e}") from e

    # Save model
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, str(output_path))
    print(f"\n[Saved] FP16 model saved to: {output_path} ({os.path.getsize(output_path)/(1024*1024):.2f} MB)")

    # Dual-sync to ./onnx and ./tools/onnx
    sync_targets = []
    if "tools" in output_path.parts:
        root_onnx = PROJECT_ROOT / "onnx" / output_path.name
        sync_targets.append(root_onnx)
    else:
        tools_onnx = SCRIPT_DIR / "onnx" / output_path.name
        sync_targets.append(tools_onnx)

    for target in sync_targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(output_path), str(target))
            print(f"[Sync]  Auto-synced to: {target}")
        except Exception as e:
            print(f"[Sync]  Warning: could not sync to {target}: {e}")

    print("\n[Done] Conversion Complete!")
    return model_fp16


if __name__ == "__main__":
    args = parse_args()
    convert_to_fp16(args.input_pth, args.output_pth, keep_io_types=args.keep_io_types)

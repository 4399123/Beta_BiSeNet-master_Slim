import argparse
import os.path as osp
import sys
sys.path.insert(0, '.')

from onnxconverter_common import float16
import onnx
from onnx import TensorProto, helper
import os

parse = argparse.ArgumentParser()
parse.add_argument('--input-path', dest='input_pth', type=str,
                   default='./onnx/best-smi.onnx',
                   help='Path to the input FP32 ONNX model')
parse.add_argument('--output-path', dest='output_pth', type=str,
                   default='./onnx/best_fp16.onnx',
                   help='Path to save the output FP16 ONNX model')
args = parse.parse_args()


def remove_bad_cast_chain_and_add_clean_cast(model):
    """
    处理两种情况：
    A. onnxconverter_common 把 argmax→cast(float32) 转成了畸形双 Cast 链 → 全部移除，追加干净 Cast
    B. 图输出本身就是 int64（argmax 直连输出，没有 Cast）→ 直接追加 Cast(to=float32)
    C. 图输出是 float16（keep_io_types 没保住）→ 追加 Cast(to=float32)
    """
    graph = model.graph

    # 建立 tensor_name -> 生产节点 的反向索引
    producer = {}
    for node in graph.node:
        for out_name in node.output:
            producer[out_name] = node

    def collect_cast_chain(tensor_name):
        """从图输出向上追溯，收集连续 Cast 节点链。
        返回 (chain_nodes, upstream_non_cast_tensor_name)"""
        chain = []
        current = tensor_name
        visited = set()
        while current in producer:
            if current in visited:
                break  # 防止节点名==tensor名导致的死循环
            visited.add(current)
            node = producer[current]
            if node.op_type != "Cast":
                break
            chain.append(node)
            current = node.input[0]
        return chain, current

    node_list = list(graph.node)
    nodes_to_remove = set()
    new_nodes_to_add = []

    for out in graph.output:
        out_name = out.name
        current_elem_type = out.type.tensor_type.elem_type
        # elem_type: 1=float32, 7=int64, 10=float16

        cast_chain, upstream_tensor = collect_cast_chain(out_name)

        needs_fix = (
            len(cast_chain) > 0              # 有 Cast 链（可能是畸形链）
            or current_elem_type == TensorProto.INT64    # 输出是 int64
            or current_elem_type == TensorProto.FLOAT16  # 输出是 float16
        )

        if not needs_fix:
            print(f"  [ok] 图输出 '{out_name}' 类型已是 float32，无需修复")
            continue

        # 移除旧 Cast 链
        if cast_chain:
            print(f"  [fix] 图输出 '{out_name}': 移除 {len(cast_chain)} 个旧 Cast 节点")
            for n in cast_chain:
                attrs = {a.name: a.i for a in n.attribute}
                print(f"        移除 node='{n.name}'  to={attrs.get('to')}")
                nodes_to_remove.add(id(n))
        else:
            # 没有 Cast 链，upstream_tensor 就是图输出本身的生产者输出
            upstream_tensor = out_name
            # 此时需要把图输出名让给新节点，用临时名承接上游
            tmp_name = out_name + "__int_tmp"
            # 找到生产 out_name 的节点，把它的输出改名
            if out_name in producer:
                prod_node = producer[out_name]
                for i, o in enumerate(prod_node.output):
                    if o == out_name:
                        prod_node.output[i] = tmp_name
                        print(f"  [fix] 将节点 '{prod_node.name}' 的输出 '{out_name}' 改名为 '{tmp_name}'")
                upstream_tensor = tmp_name

        # 追加干净的 Cast(to=float32)
        clean_cast = helper.make_node(
            "Cast",
            inputs=[upstream_tensor],
            outputs=[out_name],
            name=f"/clean_cast_{out_name}",
            to=TensorProto.FLOAT
        )
        print(f"  [fix] 追加 Cast(to=float32): '{upstream_tensor}' → '{out_name}'")
        new_nodes_to_add.append(clean_cast)

        # 更新图输出类型声明为 float32
        out.type.tensor_type.elem_type = TensorProto.FLOAT

    # 重建节点列表
    filtered_nodes = [n for n in node_list if id(n) not in nodes_to_remove]
    filtered_nodes.extend(new_nodes_to_add)
    del graph.node[:]
    graph.node.extend(filtered_nodes)

    return model


def convert_to_fp16(input_path, output_path):
    """
    Convert FP32 ONNX model to FP16.
    内部权重 & 计算用 float16，输入/输出保持 float32（ONNX Runtime 兼容）。
    """
    print(f"Loading FP32 model from: {input_path}")
    model = onnx.load(input_path)

    print("Converting to FP16...")
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=False
    )

    print("Fixing Cast node chain...")
    model_fp16 = remove_bad_cast_chain_and_add_clean_cast(model_fp16)

    # 验证
    print("Validating with onnx.checker...")
    try:
        onnx.checker.check_model(model_fp16)
        print("  ✓ onnx.checker passed")
    except Exception as e:
        print(f"  ✗ onnx.checker warning: {e}")

    # 保存
    print(f"Saving to: {output_path}")
    output_dir = osp.dirname(output_path)
    if output_dir and not osp.exists(output_dir):
        os.makedirs(output_dir)
    onnx.save(model_fp16, output_path)

    print("\n✓ Done!")
    print(f"  Input  : {input_path} (FP32)")
    print(f"  Output : {output_path} (FP16 weights, FP32 IO)")
    print("\nModel IO:")
    for inp in model_fp16.graph.input:
        t = inp.type.tensor_type.elem_type
        print(f"  Input : '{inp.name}'  dtype={t}  (1=float32, 10=float16)")
    for out in model_fp16.graph.output:
        t = out.type.tensor_type.elem_type
        print(f"  Output: '{out.name}'  dtype={t}  (1=float32, 10=float16)")

    return model_fp16


if __name__ == '__main__':
    if not osp.exists(args.input_pth):
        raise FileNotFoundError(f"Input model not found: {args.input_pth}")
    convert_to_fp16(args.input_pth, args.output_pth)

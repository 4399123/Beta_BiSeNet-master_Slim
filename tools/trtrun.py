# -*- coding: utf-8 -*-
import os
import glob

import tensorrt as trt
import numpy as np
from PIL import Image
import cv2
import pycuda.driver as cuda

# ============================================================
# 路径配置
# ============================================================
engine_path = r'./trt/best-smi.engine'
# 支持单张图片路径 或 图片文件夹路径
input_path  = r'./imgs'    # 文件夹 或 单张图片
output_dir  = r'./trt'     # 结果保存目录
w, h = 512, 512

# ============================================================
# 调色板配置（与 onnxrun.py 保持一致）
# ============================================================
palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)
palette[0] = [255, 255, 255]
palette[1] = [0,   255,   0]
palette[2] = [0,     0, 255]
palette[3] = [255,   0,   0]
palette[4] = [255, 255,   0]
palette[5] = [255,   0, 255]
palette[6] = [171, 130, 255]
palette[7] = [155, 211, 255]
palette[8] = [0,   255, 255]

mean = np.array([120, 114, 104], dtype=np.float32)
std  = np.array([ 70,  69,  73], dtype=np.float32)

# TRT dtype -> numpy dtype 映射
TRT_DTYPE_MAP = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF:  np.float16,
    trt.DataType.INT8:  np.int8,
    trt.DataType.INT32: np.int32,
    trt.DataType.BOOL:  np.bool_,
}
if hasattr(trt.DataType, 'INT64'):
    TRT_DTYPE_MAP[trt.DataType.INT64] = np.int64

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# ============================================================
# 工具函数
# ============================================================
def load_engine(path: str) -> trt.ICudaEngine:
    with open(path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f'无法加载 engine 文件：{path}')
    return engine


def preprocess(img_path: str, w: int, h: int):
    img_pil = Image.open(img_path).convert('RGB').resize((w, h))
    img_bak = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img = np.array(img_pil, dtype=np.float32)
    img -= mean
    img /= std
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))[np.newaxis])  # (1,3,h,w)
    return img, img_bak


def infer_one(context: trt.IExecutionContext,
              engine: trt.ICudaEngine,
              input_data: np.ndarray,
              stream: cuda.Stream) -> np.ndarray:
    """执行单次推理，stream 由外部传入复用。"""
    input_data = np.ascontiguousarray(input_data, dtype=np.float32)
    use_new_api = hasattr(context, 'set_tensor_address')

    if use_new_api:
        input_name  = engine.get_tensor_name(0)
        output_name = engine.get_tensor_name(1)
        context.set_input_shape(input_name, input_data.shape)
        out_shape = tuple(context.get_tensor_shape(output_name))
        trt_dtype = engine.get_tensor_dtype(output_name)
    else:
        input_name  = engine.get_binding_name(0)
        output_name = engine.get_binding_name(1)
        context.set_binding_shape(0, input_data.shape)
        out_shape = tuple(context.get_binding_shape(1))
        trt_dtype = engine.get_binding_dtype(1)

    out_dtype = TRT_DTYPE_MAP.get(trt_dtype, np.float32)

    if any(d < 0 for d in out_shape):
        raise RuntimeError(
            f'输出 shape 含负数 {out_shape}，engine 可能不支持输入尺寸 {input_data.shape}'
        )

    output_data = np.empty(out_shape, dtype=out_dtype)
    d_input     = cuda.mem_alloc(input_data.nbytes)
    d_output    = cuda.mem_alloc(output_data.nbytes)

    cuda.memcpy_htod_async(d_input, input_data, stream)

    if use_new_api:
        context.set_tensor_address(input_name,  int(d_input))
        context.set_tensor_address(output_name, int(d_output))
        context.execute_async_v3(stream_handle=stream.handle)
    else:
        context.execute_async_v2(
            bindings=[int(d_input), int(d_output)],
            stream_handle=stream.handle
        )

    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    stream.synchronize()

    # 显式释放显存，防止推理循环中显存泄漏
    d_input.free()
    d_output.free()

    return output_data


def postprocess_and_save(out: np.ndarray, img_bak: np.ndarray, save_prefix: str):
    os.makedirs(os.path.dirname(os.path.abspath(save_prefix)), exist_ok=True)

    # out 为 float32 类别图（pred 模式），直接取整映射颜色
    idx_map = out.squeeze().astype(np.int32)   # (H, W)
    pred    = palette[idx_map]                 # (H, W, 3)

    mask_path    = f'{save_prefix}_mask.jpg'
    blended_path = f'{save_prefix}_out.jpg'

    cv2.imwrite(mask_path, pred)
    blended = cv2.addWeighted(img_bak, 0.3, pred, 0.7, 0)
    cv2.imwrite(blended_path, blended)

    print(f'  已保存 → {mask_path}  /  {blended_path}')


# ============================================================
# 收集待推理图片列表
# ============================================================
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

if os.path.isfile(input_path):
    img_list = [input_path]
elif os.path.isdir(input_path):
    img_list = sorted([
        p for p in glob.glob(os.path.join(input_path, '*'))
        if os.path.splitext(p)[1].lower() in IMG_EXTS
    ])
    if not img_list:
        raise FileNotFoundError(f'在 {input_path} 下未找到图片文件')
else:
    raise FileNotFoundError(f'input_path 不存在：{input_path}')

print(f'共找到 {len(img_list)} 张图片，开始推理...\n')

# ============================================================
# 手动管理 CUDA 上下文，避免 pycuda / TRT 析构顺序冲突
# ============================================================
cuda.init()
cuda_device  = cuda.Device(0)
cuda_ctx     = cuda_device.make_context()   # 手动创建，由我们控制生命周期

try:
    engine  = load_engine(engine_path)
    context = engine.create_execution_context()
    stream  = cuda.Stream()

    os.makedirs(output_dir, exist_ok=True)

    for idx, img_path in enumerate(img_list):
        print(f'[{idx+1}/{len(img_list)}] 处理：{img_path}')
        inp, img_bak = preprocess(img_path, w, h)
        out          = infer_one(context, engine, inp, stream)
        base_name    = os.path.splitext(os.path.basename(img_path))[0]
        save_prefix  = os.path.join(output_dir, base_name)
        postprocess_and_save(out, img_bak, save_prefix)

    print('\n全部推理完成。')

finally:
    # 按顺序显式释放：context → engine → stream，最后弹出 CUDA 上下文
    del context
    del engine
    del stream
    cuda_ctx.pop()        # 弹出上下文，TRT 资源已全部释放，不会再触发 invalid device context

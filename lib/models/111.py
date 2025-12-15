#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import os
from typing import Any, Callable, Dict, Optional, Union, Tuple
import logging

# region: 模型加载工具函数 (基本保持不变, 原始实现已相当不错)
# -------------------------------------------------------------------
try:
    import safetensors.torch

    _has_safetensors = True
except ImportError:
    _has_safetensors = False

_logger = logging.getLogger(__name__)


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def load_state_dict(
        checkpoint_path: str,
        use_ema: bool = True,
        device: Union[str, torch.device] = 'cpu',
) -> Dict[str, Any]:
    if not os.path.isfile(checkpoint_path):
        _logger.error(f"No checkpoint found at '{checkpoint_path}'")
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'")

    if str(checkpoint_path).endswith(".safetensors"):
        if not _has_safetensors:
            raise ImportError("`pip install safetensors` is required to load .safetensors files.")
        return safetensors.torch.load_file(checkpoint_path, device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict_key = ''
    if isinstance(checkpoint, dict):
        for key in ('state_dict_ema', 'model_ema', 'state_dict', 'model'):
            if key in checkpoint:
                state_dict_key = key
                break

    state_dict = checkpoint.get(state_dict_key, checkpoint) if state_dict_key else checkpoint
    state_dict = clean_state_dict(state_dict)
    _logger.info(f"Loaded '{state_dict_key or 'state_dict'}' from checkpoint '{checkpoint_path}'")
    return state_dict


def remap_state_dict(
        state_dict: Dict[str, Any],
        model: nn.Module,
        allow_reshape: bool = True
):
    """
    Remaps checkpoint by iterating over state dicts in order.
    Warning: This is a brittle method and should be used with caution.
    """
    out_dict = {}
    model_state_dict = model.state_dict()

    if len(model_state_dict) != len(state_dict):
        _logger.warning(
            f"State dict size mismatch: model has {len(model_state_dict)} tensors, checkpoint has {len(state_dict)}. Remap may fail.")

    for (ka, va), (kb, vb) in zip(model_state_dict.items(), state_dict.items()):
        if va.numel() != vb.numel():
            raise ValueError(
                f"Tensor size mismatch for {ka}: model shape {va.shape} vs checkpoint shape {vb.shape}. Remap failed.")

        if va.shape != vb.shape:
            if allow_reshape:
                _logger.warning(f"Reshaping tensor {kb} from {vb.shape} to {va.shape} for key {ka}.")
                vb = vb.reshape(va.shape)
            else:
                raise ValueError(f"Tensor shape mismatch for {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed.")
        out_dict[ka] = vb
    return out_dict


def load_checkpoint(
        model: nn.Module,
        checkpoint_path: str,
        use_ema: bool = True,
        device: Union[str, torch.device] = 'cpu',
        strict: bool = True,
        remap: bool = True,
        filter_fn: Optional[Callable] = None,
):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
            return
        else:
            raise NotImplementedError('Model does not support loading numpy checkpoints.')

    state_dict = load_state_dict(checkpoint_path, use_ema, device=device)

    if remap:
        state_dict = remap_state_dict(state_dict, model)
    elif filter_fn:
        state_dict = filter_fn(state_dict, model)

    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


# -------------------------------------------------------------------
# endregion


class Classifier(nn.Module):
    """
    A simple classifier head with global average pooling and two fully-connected layers.
    Args:
        in_ch (int): Number of input channels from the backbone.
        num_classes (int): Number of output classes.
        embedding_dim (int): The dimension of the feature embedding.
    """

    def __init__(self, in_ch: int, num_classes: int, embedding_dim: int):
        super().__init__()
        # <--- 优化点 2: 使用高效、标准的 AdaptiveAvgPool2d
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_ch, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            A tuple containing the feature embedding and the final logits.
        """
        x = self.pool(x)
        x = torch.flatten(x, 1)
        feature = F.relu(self.fc1(x))
        out = self.fc2(feature)
        return feature, out


class Net(nn.Module):
    """
    A network composed of a timm backbone and a custom classifier head.
    Args:
        model_name (str): The name of the model to load from timm.
        num_classes (int): The number of classes for the final classifier.
        embedding_dim (int): The dimension of the intermediate feature embedding.
        pretrained_path (str, optional): Path to the pretrained backbone weights.
    """

    def __init__(self, model_name: str, num_classes: int, embedding_dim: int = 512,
                 pretrained_path: Optional[str] = None):
        super().__init__()
        # 创建骨干网络，仅提取特征
        self.backbone = timm.create_model(
            model_name,
            features_only=True,
            out_indices=[-1],
            pretrained=False  # 先设置为False，我们手动加载
        )

        # <--- 优化点 1: 动态、安全地获取输入通道数
        # 从backbone的feature_info中获取最后一层输出的通道数
        fc_in_ch = self.backbone.feature_info[-1]['num_chs']

        # 手动加载预训练权重 (如果提供了路径)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from: {pretrained_path}")
            load_checkpoint(self.backbone, pretrained_path)
        else:
            _logger.warning(
                f"Pretrained path '{pretrained_path}' not found or not provided. Using randomly initialized backbone.")

        self.classifier = Classifier(fc_in_ch, num_classes, embedding_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        <--- 优化点 3: 移除 'mode' 参数，forward只负责前向传播
        """
        # backbone返回的是一个列表，因为out_indices=[-1]，所以我们取第一个元素
        features_map = self.backbone(x)[0]
        # 传递给分类头
        feature_embedding, logits = self.classifier(features_map)
        return feature_embedding, logits


if __name__ == "__main__":
    # --- 配置 ---
    MODEL_NAME = 'convnext_pico.d1_in1k'
    NUM_CLASSES = 9
    EMBEDDING_DIM = 128

    # 构建预训练权重路径
    # 这种方式比 try-except 更清晰
    filename = MODEL_NAME.split('.')[0]
    possible_paths = [
        f'./premodels/{filename}.pth',
        f'./utils/premodels/{filename}.pth',
    ]
    PRETRAINED_PATH = next((path for path in possible_paths if os.path.exists(path)), None)

    # --- 模型实例化 ---
    net = Net(
        model_name=MODEL_NAME,
        num_class=NUM_CLASSES,
        embeddingdim=EMBEDDING_DIM,
        pretrained_path=PRETRAINED_PATH
    )

    # --- 推理演示 ---
    net.eval()  # 切换到评估模式，这对于BN和Dropout层很重要

    # 创建一个随机输入张量
    input_tensor = torch.randn(3, 3, 224, 224)

    with torch.no_grad():  # 在推理时使用 no_grad 可以节省显存并加速
        # <--- 优化点 3 和 4 的实践: 在模型外部进行后处理
        # 1. 获取模型原始输出 (logits)
        _, logits = net(input_tensor)

        # 2. 计算概率
        probabilities = torch.softmax(logits, dim=1)

        # 3. 获取最高分的类别和分数
        scores, indices = torch.max(probabilities, dim=1)

    print("Predicted Indices:", indices)
    print("Confidence Scores:", scores)

    # 验证输出的类型和形状
    print("\n--- Verification ---")
    print("Indices shape:", indices.shape)  # torch.Size([3])
    print("Indices dtype:", indices.dtype)  # torch.int64
    print("Scores shape:", scores.shape)  # torch.Size([3])
    print("Scores dtype:", scores.dtype)  # torch.float32
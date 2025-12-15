#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import timm
import os
from typing import Any, Callable, Dict, Optional, Union
import logging
try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

_logger = logging.getLogger(__name__)

def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
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
    if checkpoint_path and os.path.isfile(checkpoint_path):
        # Check if safetensors or not and load weights accordingly
        if str(checkpoint_path).endswith(".safetensors"):
            assert _has_safetensors, "`pip install safetensors` to use .safetensors"
            checkpoint = safetensors.torch.load_file(checkpoint_path, device=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)

        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        state_dict = clean_state_dict(checkpoint[state_dict_key] if state_dict_key else checkpoint)
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()
def remap_state_dict(
        state_dict: Dict[str, Any],
        model: torch.nn.Module,
        allow_reshape: bool = True
):
    """ remap checkpoint by iterating over state dicts in order (ignoring original keys).
    This assumes models (and originating state dict) were created with params registered in same order.
    """
    out_dict = {}
    for (ka, va), (kb, vb) in zip(model.state_dict().items(), state_dict.items()):
        assert va.numel() == vb.numel(), f'Tensor size mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed.'
        if va.shape != vb.shape:
            if allow_reshape:
                vb = vb.reshape(va.shape)
            else:
                assert False,  f'Tensor shape mismatch {ka}: {va.shape} vs {kb}: {vb.shape}. Remap failed.'
        out_dict[ka] = vb
    return out_dict

def load_checkpoint(
        model: torch.nn.Module,
        checkpoint_path: str,
        use_ema: bool = True,
        device: Union[str, torch.device] = 'cpu',
        strict: bool = True,
        remap: bool = True,
        filter_fn: Optional[Callable] = None,
):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return

    state_dict = load_state_dict(checkpoint_path, use_ema, device=device)
    if remap:
        state_dict = remap_state_dict(state_dict, model)
    elif filter_fn:
        state_dict = filter_fn(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys

class EfficientNet_Lite3(nn.Module):
    def __init__(self):
        super(EfficientNet_Lite3, self).__init__()
        self.out_indices = [2, 3,4]
        self.selected_feature_extractor = timm.create_model('tf_efficientnet_lite3.in1k', features_only=True, out_indices=self.out_indices,pretrained=False)

        load_checkpoint(self.selected_feature_extractor, '../lib/premodels/efficientnet_lite3.pth')
        # load_checkpoint(self.selected_feature_extractor, '../premodels/efficientnet_lite3.pth')

    def forward(self, x):
        x=self.selected_feature_extractor(x)
        feat8 =x[0] # 1/8
        feat16 = x[1] # 1/16
        feat32 = x[2] # 1/32
        return feat8, feat16, feat32

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


if __name__ == "__main__":
    net = EfficientNet_Lite3()
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    print(out[0].size())
    print(out[1].size())
    print(out[2].size())
    net.get_params()

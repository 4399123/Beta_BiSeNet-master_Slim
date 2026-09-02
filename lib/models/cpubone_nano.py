#!/usr/bin/python
# -*- encoding: utf-8 -*-

import inspect
import os
import re

import torch
import torch.nn as nn
import timm
from timm.models import load_checkpoint as _timm_load_checkpoint

# 新版 timm 的 load_checkpoint 多了 weights_only 参数且默认 True，
# 遇到含非白名单 pickle 对象(如 argparse.Namespace)的权重会直接报错；老版 timm 没有该参数。
# 这里按签名动态决定是否传入，保证新旧版本都能用。
_SUPPORTS_WEIGHTS_ONLY = 'weights_only' in inspect.signature(_timm_load_checkpoint).parameters


def _is_safetensors_file(path):
    """lcnetv2_small.pth 实际是 safetensors 格式(扩展名却是 .pth)，
    而 timm 只按扩展名判断格式，因此这里按文件头自行探测。"""
    if str(path).endswith('.safetensors'):
        return True
    try:
        with open(path, 'rb') as f:
            head = f.read(9)
    except OSError:
        return False
    # safetensors 布局: 8 字节小端 header 长度 + JSON(以 '{' 开头)
    return len(head) == 9 and head[8:9] == b'{'


def _flatten_seq_keys(state_dict):
    """features_only 包装会把第一层 Sequential 展平(stem.0 -> stem_0, stages.1 -> stages_1)，
    原始 backbone 权重需要做同样的键名转换。"""
    return {re.sub(r'^([A-Za-z_]\w*)\.(\d+)\.', r'\1_\2.', k): v for k, v in state_dict.items()}


def load_checkpoint(model, checkpoint_path, strict=False, remap=True, **kwargs):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    if _is_safetensors_file(checkpoint_path):
        import safetensors.torch
        state_dict = _flatten_seq_keys(safetensors.torch.load_file(checkpoint_path, device='cpu'))
        return model.load_state_dict(state_dict, strict=strict)

    if _SUPPORTS_WEIGHTS_ONLY:
        kwargs.setdefault('weights_only', False)
    return _timm_load_checkpoint(model, checkpoint_path, strict=strict, remap=remap, **kwargs)


class CPUBone_Nano(nn.Module):
    def __init__(self):
        super(CPUBone_Nano, self).__init__()
        self.out_indices = [1, 2,3]
        self.selected_feature_extractor = timm.create_model('cpubone_nano.r224_in1k', features_only=True, out_indices=self.out_indices,pretrained=False)
        try:
            load_checkpoint(self.selected_feature_extractor, '../lib/premodels/cpubone_nano.safetensors')
        except FileNotFoundError:
            load_checkpoint(self.selected_feature_extractor, '../premodels/cpubone_nano.safetensors')

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
    net = CPUBone_Nano()
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    print(out[0].size())
    print(out[1].size())
    print(out[2].size())
    net.get_params()

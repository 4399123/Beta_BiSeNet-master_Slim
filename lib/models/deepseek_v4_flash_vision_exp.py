#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import torch
import torch.nn as nn
import timm


class DeepSeek_V4_Flash_Vision_Exp(nn.Module):
    def __init__(self):
        super(DeepSeek_V4_Flash_Vision_Exp, self).__init__()
        self.out_indices = [1, 2, 3]  # 抽取 1/8, 1/16, 1/32 特征

        ckpt_path = '../lib/premodels/deepseek_v4_flash_vision_exp.safetensors'
        if not os.path.isfile(ckpt_path):
            ckpt_path = '../premodels/deepseek_v4_flash_vision_exp.safetensors'

        # 1. 创建完整模型并载入预训练权重（保证官方权重的 key 100% 精确匹配）
        self.model = timm.create_model(
            'deepseek_vit_412m.deepseek_v4_flash_vision_exp',
            pretrained=False,
            checkpoint_path=ckpt_path if os.path.isfile(ckpt_path) else None
        )

        # 2. 载入权重后，立即剪枝：剔除多余 stage 并移除分类头 fc
        self.model.prune_intermediate_layers(indices=self.out_indices, prune_head=True)

        # 3. （可选进阶）因为分割完全不需要 head 中的 pre_logits / norm 等辅助层，可以直接置空：
        self.model.head = nn.Identity()

    def forward(self, x):
        # 通过 forward_intermediates 提取特征
        feats = self.model.forward_intermediates(
            x,
            indices=self.out_indices,
            output_fmt='NCHW',
            intermediates_only=True
        )
        return feats[0], feats[1], feats[2]

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
    net = DeepSeek_V4_Flash_Vision_Exp()
    x = torch.randn(2, 3, 224, 224)
    out = net(x)
    print(out[0].size())
    print(out[1].size())
    print(out[2].size())
    net.get_params()

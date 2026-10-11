#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import torch
import torch.nn as nn
import timm


# ================= 全局配置：预训练权重文件名 =================
PRETRAINED_WEIGHTS_NAME = 'qwen3_5_0_8b.safetensors'
# ==========================================================


class ConvBNAct(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, bias=False, act=nn.GELU):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.act = act() if act is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Qwen3_5_0_8b(nn.Module):
    """返回 Qwen3 ViT block 3/7/11 的原始 1/16 NCHW 特征。"""

    def __init__(
            self,
            out_indices=(3, 7, 11),
            use_checkpoint=True,
            ckpt_path=None
    ):
        super(Qwen3_5_0_8b, self).__init__()
        assert len(out_indices) == 3, "out_indices 必须为 3 个阶段的索引 (浅层, 中层, 深层)"
        self.out_indices = list(out_indices)
        self.embed_dim = 768
        self.out_channels = (self.embed_dim,) * len(self.out_indices)

        if ckpt_path is None:
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            lib_dir = os.path.dirname(cur_dir)
            root_dir = os.path.dirname(lib_dir)
            candidates = [
                os.path.join(lib_dir, 'premodels', PRETRAINED_WEIGHTS_NAME),
                os.path.join(root_dir, 'premodels', PRETRAINED_WEIGHTS_NAME),
                os.path.join(root_dir, 'lib', 'premodels', PRETRAINED_WEIGHTS_NAME),
                os.path.join('.', 'lib', 'premodels', PRETRAINED_WEIGHTS_NAME),
                os.path.join('..', 'lib', 'premodels', PRETRAINED_WEIGHTS_NAME),
                os.path.join('..', 'premodels', PRETRAINED_WEIGHTS_NAME),
            ]
            for cand in candidates:
                if os.path.isfile(cand):
                    ckpt_path = os.path.abspath(cand)
                    break

        self.loaded_pretrained = False
        if ckpt_path and os.path.isfile(ckpt_path):
            print(f"正在加载预训练权重: {ckpt_path}")
            self.model = timm.create_model(
                'qwen3_vit_88m.qwen3_5_0_8b',
                pretrained=False,
                checkpoint_path=ckpt_path
            )
            self.loaded_pretrained = True
            print("预训练权重加载完成！")
        else:
            print(f"警告: 未找到预训练权重文件 (ckpt_path={ckpt_path})，使用随机初始化！")
            self.model = timm.create_model(
                'qwen3_vit_88m.qwen3_5_0_8b',
                pretrained=False
            )

        self.model.prune_intermediate_layers(indices=self.out_indices, prune_head=True)
        self.model.head = nn.Identity()

        if use_checkpoint and hasattr(self.model, 'set_grad_checkpointing'):
            self.model.set_grad_checkpointing(True)

    def forward(self, x):
        feats = self.model.forward_intermediates(
            x,
            indices=self.out_indices,
            output_fmt='NCHW',
            intermediates_only=True
        )
        return tuple(feats)

    def get_params(self):
        wd_params, nowd_params = [], []
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                wd_params.append(module.weight)
                if module.bias is not None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)):
                nowd_params += list(module.parameters())

        recorded_params = set(wd_params + nowd_params)
        for param in self.parameters():
            if param not in recorded_params:
                nowd_params.append(param)

        return wd_params, nowd_params


if __name__ == "__main__":
    net = Qwen3_5_0_8b()
    net.eval()

    # 验证预训练权重是否加载成功：
    # 打印 Block 0 注意力层的部分权重数值及其统计信息
    sample_weight = net.model.encoder.blocks[0].attn.qkv.weight[0, :5].detach().cpu()
    weight_norm = net.model.encoder.blocks[0].attn.qkv.weight.norm().item()
    print("\n--- 预训练权重加载验证 ---")
    print(f"Block 0 QKV Weight[:5]: {sample_weight.tolist()}")
    print(f"Block 0 QKV Weight L2 Norm: {weight_norm:.4f}")
    if net.loaded_pretrained:
        print("[验证结果] 预训练权重已成功加载进入模型！")
    else:
        print("[验证结果] 未加载预训练权重（为随机初始化）！")
    print("---------------------------\n")

    # 输入尺寸推荐为 16 的倍数
    x = torch.randn(2, 3, 224, 224)
    out8, out16, out32 = net(x)
    print("feat8  size:", out8.size())
    print("feat16 size:", out16.size())
    print("feat32 size:", out32.size())

    wd, nowd = net.get_params()
    print(f"Parameters: {len(wd)} weight decay, {len(nowd)} no weight decay")



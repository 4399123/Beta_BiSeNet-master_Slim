import argparse
import os.path as osp
import sys
sys.path.insert(0, '.')

import torch

from lib.models import model_factory
from configs import set_cfg_from_file
from timm.utils import reparameterize_model

torch.set_grad_enabled(False)

parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str,
        default='../configs/segformer_blueface_b0.py',)
parse.add_argument('--weight-path', dest='weight_pth', type=str,
        default='../pt/segformer_b0.pt')
parse.add_argument('--outpath', dest='out_pth', type=str,
        default='./torchscript/best.pt')
parse.add_argument('--aux-mode', dest='aux_mode', type=str,
        default='pred')
args = parse.parse_args()

cfg = set_cfg_from_file(args.config)
if cfg.use_sync_bn: cfg.use_sync_bn = False

net = model_factory[cfg.model_type](cfg.n_cats, aux_mode=args.aux_mode, use_fp16=False)
net.load_state_dict(torch.load(args.weight_pth, map_location='cpu'), strict=False)
net.eval()
reparameterize_model(net)

# 使用 JIT 将模型转换为 TorchScript 格式
dummy_input = torch.randn(1, 3, 512, 512)
traced_script_module = torch.jit.trace(net, dummy_input)

# 保存 TorchScript 模型
traced_script_module.save(args.out_pth)

print(f'TorchScript model saved to {args.out_pth}')
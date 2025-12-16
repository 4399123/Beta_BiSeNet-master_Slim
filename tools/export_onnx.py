import argparse
import os.path as osp
import sys
sys.path.insert(0, '.')

import torch

from lib.models import model_factory
from configs import set_cfg_from_file
import onnx
from onnxsim import simplify
import onnxoptimizer
from utils_zl import replace_batchnorm
from timm.utils import reparameterize_model
torch.set_grad_enabled(False)

parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str,
                   default='../configs/fastefficientformerseg_blueface_efficientnetv2_b3.py', )
parse.add_argument('--weight-path', dest='weight_pth', type=str,
                   default='../pt/fastefficientformerseg_efficientnetv2_b3.pt')
parse.add_argument('--outpath', dest='out_pth', type=str,
                   default='./onnx/best.onnx')
parse.add_argument('--ousmitpath', dest='outsmi_pth', type=str,
                   default='./onnx/best-smi.onnx')
parse.add_argument('--aux-mode', dest='aux_mode', type=str,
                   default='pred')
args = parse.parse_args()


cfg = set_cfg_from_file(args.config)
if cfg.use_sync_bn: cfg.use_sync_bn = False

net = model_factory[cfg.model_type](cfg.n_cats, aux_mode=args.aux_mode,use_fp16=False)
net.load_state_dict(torch.load(args.weight_pth, map_location='cpu'), strict=False)
net.eval()
# replace_batchnorm(net)
reparameterize_model(net)

#  dummy_input = torch.randn(1, 3, *cfg.crop_size)
dummy_input = torch.randn(1, 3, 960, 960)
# input_name = ['input']
# output_name = ['eval',]
input_name = 'input'
output_name = 'output'

torch.onnx.export(net, dummy_input, args.out_pth,
    input_names=[input_name],
    output_names=[output_name],
    verbose=False, opset_version=16,
    dynamic_axes={
                input_name: {0: 'batch_size'},
                output_name: {0: 'batch_size'}}
)


print('step 1 ok')
model = onnx.load(args.out_pth)

newmodel=onnxoptimizer.optimize(model)

model_simp, check = simplify(newmodel)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp,args.outsmi_pth)
print('step 2 ok')

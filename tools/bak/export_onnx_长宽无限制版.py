import argparse
import os.path as osp
import sys
sys.path.insert(0, '.')

import torch

from lib.models import model_factory
from configs import set_cfg_from_file
import onnx
from onnxsim import simplify

torch.set_grad_enabled(False)


parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str,
        default='../configs/../configs/bisenetv1_blueface_fbnetv3_g.py',)
parse.add_argument('--weight-path', dest='weight_pth', type=str,
        default='../pt/X22/v1_blueface_fbnetv3_g.pt')
parse.add_argument('--outpath', dest='out_pth', type=str,
        default='./onnx/fbnetv3_g.onnx')
parse.add_argument('--ousmitpath', dest='outsmi_pth', type=str,
        default='./onnx/fbnetv3_g-smi.onnx')
parse.add_argument('--aux-mode', dest='aux_mode', type=str,
        default='pred')
args = parse.parse_args()


cfg = set_cfg_from_file(args.config)
if cfg.use_sync_bn: cfg.use_sync_bn = False

net = model_factory[cfg.model_type](cfg.n_cats, aux_mode=args.aux_mode)
net.load_state_dict(torch.load(args.weight_pth, map_location='cpu'), strict=False)
net.eval()


#  dummy_input = torch.randn(1, 3, *cfg.crop_size)
dummy_input = torch.randn(1, 3, 512, 512)
# input_name = ['input']
# output_name = ['eval',]
input_name = 'input'
output_name = 'output'

torch.onnx.export(net, dummy_input, args.out_pth,
    input_names=[input_name],
    output_names=[output_name],
    verbose=True, opset_version=11,
    dynamic_axes={
                input_name: {0: 'batch_size', 2: 'input_height', 3: 'input_width'},
                output_name: {0: 'batch_size', 2: 'output_height', 3: 'output_width'}}
)

# torch.onnx.export(net, dummy_input, args.out_pth,
#     input_names=[input_name],
#     output_names=[output_name],
#     verbose=True, opset_version=11,
#     dynamic_axes={
#                 input_name: {0: 'batch_size'},
#                 output_name: {0: 'batch_size'}}
# )


print('step 1 ok')
model = onnx.load(args.out_pth)

model_simp, check = simplify(model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp,args.outsmi_pth)
print('step 2 ok')

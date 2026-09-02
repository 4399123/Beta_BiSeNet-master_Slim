import torch


def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse()
            setattr(net, child_name, fused)
            replace_batchnorm(fused)
        # elif isinstance(child, torch.nn.BatchNorm2d):
        #     setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)
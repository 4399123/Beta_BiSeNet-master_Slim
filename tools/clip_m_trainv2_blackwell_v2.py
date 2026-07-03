#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys

sys.path.insert(0, '..')
import os
import os.path as osp
import logging
import time
import json
import argparse
import numpy as np
from tabulate import tabulate
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.cuda.amp as amp

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.data import get_data_loader
from evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss, FocalLoss, PolyFocalLoss, IoULoss, DiceLoss,LogCoshDiceOhemLovaszLoss
from lib.ohem_ce_loss import DiceWithOhemCELoss, DiceWithFocalLoss, DiceBCELoss, OhemWithIoULoss, OhemWithFocalLoss, \
    GDiceWithOhemCELoss, LogCoshDiceLossWithOhemCELoss, FocalTverskyWithOhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg, print_log_msgs,print_log_msgs_segformer
from tqdm import tqdm
from timm.optim import AdamW, AdamP, RAdam, Lookahead, AdaBelief, Lion, Lamb
from timm.scheduler import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter
from timm.layers.norm_act import convert_sync_batchnorm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

## fix all random seeds (per-rank offset so each process draws a different
## augmentation stream; dist isn't initialized yet here, so read the rank from
## the environment variable set by torchrun / launch).
_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
torch.manual_seed(42 + _rank)
torch.cuda.manual_seed(42 + _rank)
np.random.seed(42 + _rank)
random.seed(42 + _rank)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')
printlabels = ['background', 'QPZZ', 'MDBD', 'MNYW', 'WW', 'LMPS', 'BMQQ', 'LMHH', 'KTAK']


def parse_args():
    parse = argparse.ArgumentParser()
    # parse.add_argument('--config', dest='config', type=str,
    #         default='../configs/bisenetv1_blueface_caformer_s36.py',)
    parse.add_argument('--config', dest='config', type=str,
                       default='../configs/fastefficientbisenet_blueface_inceptionnext_tiny_pro_max.py', )
    parse.add_argument('--finetune-from', type=str, default=None, )
    parse.add_argument("--local_rank", type=int)
    parse.add_argument('--scale-epochs-with-world-size', dest='scale_epochs',
                       type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'y', 't'),
                       default=True,
                       help='Whether to multiply max_epochs by world_size so the total '
                            'number of optimizer steps matches single-GPU training '
                            '(step-aligned comparison). Default: True. '
                            'Pass --scale-epochs-with-world-size False to disable.')
    return parse.parse_args()


args = parse_args()
print('Loading configuration:{}'.format(args.config))
cfg = set_cfg_from_file(args.config)


def set_model(lb_ignore=255):
    logger = logging.getLogger()
    net = model_factory[cfg.model_type](n_classes=cfg.n_cats, use_fp16=cfg.use_fp16, aux_mode='train')
    if not args.finetune_from is None:
        logger.info(f'load pretrained weights from {args.finetune_from}')
        msg = net.load_state_dict(torch.load(args.finetune_from,
                                             map_location='cpu'), strict=False)
        logger.info('\tmissing keys: ' + json.dumps(msg.missing_keys))
        logger.info('\tunexpected keys: ' + json.dumps(msg.unexpected_keys))
    # if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    if cfg.use_sync_bn: net = convert_sync_batchnorm(net)
    net.cuda()
    net.train()

    # loss_opt 说明:
    # 0: OhemCELoss
    # 1: DiceWithOhemCELoss
    # 2: DiceWithFocalLoss
    # 3: OhemWithIoULoss
    # 4: OhemWithFocalLoss
    # 5: DiceBCELoss
    # 6: GDiceWithOhemCELoss
    # 7: LogCoshDiceLossWithOhemCELoss
    # 8: LogCoshDiceOhemLovaszLoss
    # 9: FocalTverskyWithOhemCELoss (推荐工业质检: alpha=0.3 beta=0.7 偏重漏检惩罚)
    loss_opt = 8

    criteria_pre = 0
    criteria_aux = []   # num_aux_heads=0 时保持空列表，避免后续 zip 报错
    if (loss_opt == 0):
        criteria_pre = OhemCELoss(0.7, lb_ignore)
        criteria_aux = [OhemCELoss(0.7, lb_ignore) for _ in range(cfg.num_aux_heads)]
    elif (loss_opt == 1):
        criteria_pre = DiceWithOhemCELoss()
        criteria_aux = [DiceWithOhemCELoss() for _ in range(cfg.num_aux_heads)]
    elif (loss_opt == 2):
        criteria_pre = DiceWithFocalLoss()
        criteria_aux = [DiceWithFocalLoss() for _ in range(cfg.num_aux_heads)]
    elif (loss_opt == 3):
        criteria_pre = OhemWithIoULoss()
        criteria_aux = [OhemWithIoULoss() for _ in range(cfg.num_aux_heads)]
    elif (loss_opt == 4):
        criteria_pre = OhemWithFocalLoss()
        criteria_aux = [OhemWithFocalLoss() for _ in range(cfg.num_aux_heads)]
    elif (loss_opt == 5):
        criteria_pre = DiceBCELoss()
        criteria_aux = [DiceBCELoss() for _ in range(cfg.num_aux_heads)]
    elif (loss_opt == 6):
        criteria_pre = GDiceWithOhemCELoss()
        criteria_aux = [GDiceWithOhemCELoss() for _ in range(cfg.num_aux_heads)]
    elif (loss_opt == 7):
        criteria_pre = LogCoshDiceLossWithOhemCELoss()
        criteria_aux = [LogCoshDiceLossWithOhemCELoss() for _ in range(cfg.num_aux_heads)]
    elif (loss_opt == 8):
        criteria_pre = LogCoshDiceOhemLovaszLoss()
        criteria_aux = [LogCoshDiceOhemLovaszLoss() for _ in range(cfg.num_aux_heads)]
    elif (loss_opt == 9):
        criteria_pre = FocalTverskyWithOhemCELoss(ignore_index=lb_ignore, alpha=0.3, beta=0.7,
                                                  gamma=1.33, tversky_weight=1.0, ohem_weight=1.0)
        criteria_aux = [FocalTverskyWithOhemCELoss(ignore_index=lb_ignore, alpha=0.3, beta=0.7,
                                                   gamma=1.33, tversky_weight=1.0, ohem_weight=1.0)
                        for _ in range(cfg.num_aux_heads)]
    else:
        print('no such loss !!!')

    return net, criteria_pre, criteria_aux


# Optimizer choice. Lifted to module level so the LR-scaling helper can pick
# the right scaling rule (linear for SGD, sqrt for adaptive optimizers).
# 0: AdaBelief, 1: AdamP, 2: AdamW, 3: SGD, 4: Lion, 5: Lamb
OPTIM_OPT = 2


def get_lr_scale():
    """Compute LR scaling factor for the current effective batch size.

    Effective batch size = ims_per_gpu * world_size. We scale relative to
    `cfg.base_bs` if defined, otherwise relative to `cfg.ims_per_gpu` (i.e.
    the original lr_start is assumed to be tuned for single-GPU training).

    SGD uses linear scaling; adaptive optimizers (AdamP/AdamW/AdaBelief/
    Lion/Lamb) use sqrt scaling, which is the empirically safer choice and
    avoids blowing up the LR when scaling to many GPUs.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    base_bs = cfg.get('base_bs', cfg.ims_per_gpu)
    effective_bs = cfg.ims_per_gpu * world_size
    ratio = effective_bs / max(base_bs, 1)
    if OPTIM_OPT == 3:  # SGD: linear scaling
        return ratio
    return ratio ** 0.5  # adaptive optimizers: sqrt scaling


def set_optimizer(model, lr):
    if OPTIM_OPT == 0:
        optim = AdaBelief(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    elif OPTIM_OPT == 1:
        optim = AdamP(model.parameters(), lr=lr, weight_decay=cfg.weight_decay, nesterov=True)
    elif OPTIM_OPT == 2:
        optim = AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    elif OPTIM_OPT == 3:
        optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif OPTIM_OPT == 4:
        optim = Lion(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    elif OPTIM_OPT == 5:
        optim = Lamb(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f'Unknown OPTIM_OPT={OPTIM_OPT}')
    # base_optim = RAdam(model.parameters(), lr=lr, weight_decay=5e-4)
    # optim=Lookahead(base_optimizer=base_optim,k=5,alpha=0.5)

    return optim


def set_model_dist(net):
    # [Modified by Gemini] Fix for torchrun: check env var first
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = int(args.local_rank) if args.local_rank is not None else 0

    net = nn.parallel.DistributedDataParallel(
        net,
        device_ids=[local_rank, ],
        find_unused_parameters=True,
        output_device=local_rank
    )
    return net


def set_meters():
    # time_meter = TimeMeter(cfg.max_iter)
    time_meter = TimeMeter(cfg.max_epochs)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
                       for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def train(writer):
    logger = logging.getLogger()

    ## dataset
    dl = get_data_loader(cfg, mode='train')

    ## model
    net, criteria_pre, criteria_aux = set_model(dl.dataset.lb_ignore)

    ## optimizer with LR/warmup/lr_min scaled by effective batch size
    lr_scale = get_lr_scale()
    scaled_lr = cfg.lr_start * lr_scale
    scaled_lr_min = 5e-6 * lr_scale
    scaled_warmup_lr = 1e-4 * lr_scale
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    logger.info(
        'LR scaling: world_size={}, ims_per_gpu={}, base_bs={}, '
        'scale={:.4f}, lr_start: {:.2e} -> {:.2e}'.format(
            world_size, cfg.ims_per_gpu, cfg.get('base_bs', cfg.ims_per_gpu),
            lr_scale, cfg.lr_start, scaled_lr,
        )
    )
    optim = set_optimizer(net, scaled_lr)

    ## mixed precision training
    scaler = amp.GradScaler()

    ## ddp training
    net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    # Optionally scale the epoch count with world_size so that the total number
    # of optimizer steps matches single-GPU training (step-aligned comparison).
    max_epochs = cfg.max_epochs
    if args.scale_epochs and dist.is_initialized():
        max_epochs = cfg.max_epochs * dist.get_world_size()
        logger.info(
            'scale-epochs enabled: max_epochs {} -> {} (x world_size={})'.format(
                cfg.max_epochs, max_epochs, dist.get_world_size()))
    else:
        logger.info(
            'scale-epochs disabled: max_epochs={} (world_size={})'.format(
                max_epochs, dist.get_world_size() if dist.is_initialized() else 1))
    # keep ETA estimation consistent with the actual epoch count
    time_meter.max_iter = max_epochs

    lr_schdr = CosineLRScheduler(optimizer=optim,
                                 t_initial=max_epochs,
                                 lr_min=scaled_lr_min,
                                 # warmup_t=0.05 * cfg.max_epochs,
                                 warmup_t=2,
                                 warmup_lr_init=scaled_warmup_lr)

    miou = 0.0
    mprecision = 0.0
    mrecall = 0.0
    gap = int(len(dl) / 10)
    if (gap == 0): gap = 2
    ## train loop
    for epoch in range(max_epochs):
        if hasattr(dl, 'batch_sampler') and hasattr(dl.batch_sampler, 'sampler'):
            sampler = dl.batch_sampler.sampler
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

        lr_schdr.step(epoch)
        lr = optim.param_groups[0]['lr']
        writer.add_scalar('lr', lr, epoch)

        for it, (im, lb) in enumerate(dl):
            im = im.cuda()
            lb = lb.cuda()

            lb = torch.squeeze(lb, 1)

            optim.zero_grad()
            aux_weight = 0.5 * (1 - epoch / max_epochs)
            with amp.autocast(enabled=cfg.use_fp16):
                logits, *logits_aux = net(im)
                loss_pre = criteria_pre(logits, lb)
                if cfg.num_aux_heads > 0:
                    loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
                    loss = loss_pre + aux_weight * sum(loss_aux)
                else:
                    loss_aux = []
                    loss = loss_pre
            scaler.scale(loss).backward()
            # clip
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5, norm_type=2)

            # nan/inf 防线: 一旦 loss 异常, 跳过本次参数更新, 避免污染整个模型权重。
            # scaler.step 内部对 inf 梯度会自动跳过, 但 nan loss 仍需显式拦截。
            if not torch.isfinite(loss):
                logger.warning(
                    'non-finite loss detected at epoch {}, iter {}, skip this step'.format(epoch, it))
                scaler.update()
                optim.zero_grad(set_to_none=True)
                continue

            scaler.step(optim)
            scaler.update()
            torch.cuda.synchronize()

            # time_meter.update()
            loss_meter.update(loss.item())
            loss_pre_meter.update(loss_pre.item())
            if cfg.num_aux_heads > 0:
                _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

            ## print training log message
            # if (it + 1) % 200 == 0:
            if (it + 1) % gap == 0:
                # print_log_msg(epoch,cfg.max_epochs,it, len(dl), lr, time_meter, loss_meter,loss_pre_meter, loss_aux_meters)
                if cfg.num_aux_heads > 0:
                    print_log_msgs(epoch, max_epochs, it, len(dl), lr, loss_meter, loss_pre_meter, loss_aux_meters,
                               writer)
                else:
                    print_log_msgs_segformer(epoch, max_epochs, it, len(dl), lr, loss_meter,writer)
        interv, ets = time_meter.get()
        logger.info('ets:{},interv:{:.2f}s'.format(ets, interv))
        time_meter.update()

        torch.cuda.empty_cache()
        iou_heads, iou_content, f1_heads, f1_content, precision_heads, precision_content, recall_heads, recall_content = eval_model(
            cfg, net.module, printlabels)

        logger.info('\neval results of miou metric:')
        logger.info('\n' + tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))

        logger.info('\neval results of mprecision metric:')
        logger.info('\n' + tabulate(precision_content, headers=precision_heads, tablefmt='orgtbl'))

        logger.info('\neval results of mrecall metric:')
        logger.info('\n' + tabulate(recall_content, headers=recall_heads, tablefmt='orgtbl'))

        writer.add_scalar('miou', float(iou_content[-2][-1]), epoch)
        writer.add_scalar('mprecision', float(precision_content[-1][-1]), epoch)
        writer.add_scalar('mrecall', float(recall_content[-1][-1]), epoch)
        if (miou < float(iou_content[-2][-1])):
            miou = float(iou_content[-2][-1])
            mprecision = float(precision_content[-1][-1])
            mrecall = float(recall_content[-1][-1])
            if dist.get_rank() == 0: torch.save(net.module.state_dict(), '../pt/best.pt')
            logger.info("miou:{},mprecision:{},mrecall:{},save model!!!".format(miou, mprecision, mrecall))
        logger.info("best miou:{},mprecision:{},mrecall:{}".format(miou, mprecision, mrecall))

    return


def main(writer):
    # [Modified by Gemini] Fix for torchrun: check env var first
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = int(args.local_rank) if args.local_rank is not None else 0

    torch.cuda.set_device(local_rank)

    # [Modified by Gemini] Use nccl for GPU training instead of gloo
    dist.init_process_group(backend='nccl')

    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-train', cfg.respth)
    train(writer)


if __name__ == "__main__":
    writer = SummaryWriter()
    main(writer)
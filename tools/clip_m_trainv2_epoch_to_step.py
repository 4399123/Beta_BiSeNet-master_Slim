#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""
训练脚本 - Epoch-to-Step 自动转换版本
===========================================
本脚本实现了方案 A：Epoch-to-Step 自动转换，核心改进：
1. 配置文件仍使用 max_epochs（语义清晰："训练 300 轮"）
2. 训练开始前自动计算 total_steps = max_epochs × len(dataloader)
3. 学习率调度改为按 total_steps（精确控制）
4. 训练循环保留 epoch 结构（用于验证触发和数据洗牌）
5. 默认禁用 epoch 缩放（实现多卡线性加速）

关键优势：
- 保证数据遍历次数恒定（应对数据集大小变化）
- 实现多卡线性加速（4卡 ≈ 4× 加速）
- 无需修改配置文件格式
- 与 MMSegmentation 实践对齐
"""

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
    GDiceWithOhemCELoss, LogCoshDiceLossWithOhemCELoss, FocalTverskyWithOhemCELoss, ImprovedFocalTverskyWithOhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg, print_log_msgs,print_log_msgs_segformer
from tqdm import tqdm
from timm.optim import AdamW, AdamP, RAdam, Lookahead, AdaBelief, Lion, Lamb
from timm.scheduler import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter
from timm.layers.norm_act import convert_sync_batchnorm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

## fix all random seeds (per-rank offset so each process draws a different
## augmentation stream; dist isn't initialized yet here, so read the rank from
## the environment variable set by torchrun / launch).
_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', 0)))
torch.manual_seed(42 + _rank)
torch.cuda.manual_seed(42 + _rank)
np.random.seed(42 + _rank)
random.seed(42 + _rank)
#  torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')
printlabels = ['background', 'QPZZ', 'MDBD', 'MNYW', 'WW', 'LMPS', 'BMQQ', 'LMHH', 'KTAK']


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
                       default='../configs/fastefficientbisenet_blueface_inceptionnext_atto.py', )
    parse.add_argument('--finetune-from', type=str, default=None, )
    parse.add_argument("--local_rank", type=int)
    parse.add_argument('--scale-epochs-with-world-size', dest='scale_epochs',
                       type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'y', 't'),
                       default=False)

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
    # 1: LogCoshDiceOhemLovaszLoss
    # 2: FocalTverskyWithOhemCELoss (原始版本: 按全图像素数归一化)
    # 3: ImprovedFocalTverskyWithOhemCELoss (修复版: 按类别像素数归一化，解决小类别量级失衡问题)
    loss_opt = 3

    criteria_pre = 0
    criteria_aux = []   # num_aux_heads=0 时保持空列表，避免后续 zip 报错
    if (loss_opt == 0):
        criteria_pre = OhemCELoss(0.7, lb_ignore)
        criteria_aux = [OhemCELoss(0.7, lb_ignore) for _ in range(cfg.num_aux_heads)]
    elif (loss_opt == 1):
        criteria_pre = LogCoshDiceOhemLovaszLoss()
        criteria_aux = [LogCoshDiceOhemLovaszLoss() for _ in range(cfg.num_aux_heads)]
    elif (loss_opt == 2):
        criteria_pre = FocalTverskyWithOhemCELoss(ignore_index=lb_ignore, alpha=0.3, beta=0.7,
                                                  gamma=1.33, tversky_weight=1.0, ohem_weight=1.0)
        criteria_aux = [FocalTverskyWithOhemCELoss(ignore_index=lb_ignore, alpha=0.3, beta=0.7,
                                                   gamma=1.33, tversky_weight=1.0, ohem_weight=1.0)
                        for _ in range(cfg.num_aux_heads)]
    elif (loss_opt == 3):
        # 修复版 FocalTversky: 按类别像素数归一化，解决小类别量级失衡
        criteria_pre = ImprovedFocalTverskyWithOhemCELoss(
            ignore_index=lb_ignore, alpha=0.3, beta=0.7,
            gamma=1.33, tversky_weight=1.0, ohem_weight=1.0)
        criteria_aux = [ImprovedFocalTverskyWithOhemCELoss(
            ignore_index=lb_ignore, alpha=0.3, beta=0.7,
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

    ## 【改动2】计算总步数（关键改动）
    steps_per_epoch = len(dl)
    max_epochs = cfg.max_epochs

    # 如果启用了 epoch 缩放（不推荐），计算调整后的 epoch 数
    if args.scale_epochs and dist.is_initialized():
        max_epochs = cfg.max_epochs * dist.get_world_size()
        logger.info(
            'scale-epochs enabled (not recommended): max_epochs {} -> {} (x world_size={})'.format(
                cfg.max_epochs, max_epochs, dist.get_world_size()))
    else:
        logger.info(
            'scale-epochs disabled (recommended): max_epochs={} (world_size={})'.format(
                max_epochs, dist.get_world_size() if dist.is_initialized() else 1))

    total_steps = max_epochs * steps_per_epoch
    logger.info('='*80)
    logger.info(f'Training plan: {max_epochs} epochs × {steps_per_epoch} steps/epoch = {total_steps} total steps')
    logger.info('='*80)

    ## model
    net, criteria_pre, criteria_aux = set_model(dl.dataset.lb_ignore)

    ## optimizer with LR scaling
    lr_scale = get_lr_scale()
    scaled_lr = cfg.lr_start * lr_scale
    scaled_lr_min = max(1e-6, cfg.lr_start * lr_scale * 0.01)  # 不低于初始 LR 的 1%
    scaled_warmup_lr = 1e-4 * lr_scale
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    logger.info(
        'LR scaling: world_size={}, ims_per_gpu={}, base_bs={}, '
        'scale={:.4f}, lr_start: {:.2e} -> {:.2e}, lr_min: {:.2e}'.format(
            world_size, cfg.ims_per_gpu, cfg.get('base_bs', cfg.ims_per_gpu),
            lr_scale, cfg.lr_start, scaled_lr, scaled_lr_min
        )
    )
    optim = set_optimizer(net, scaled_lr)

    ## 【改动3】学习率调度改为基于总步数（关键改动）
    warmup_steps = max(5, int(0.05 * total_steps))  # warmup 至少 5 步或总数的 5%
    lr_schdr = CosineLRScheduler(
        optimizer=optim,
        t_initial=total_steps,  # 改为总步数
        lr_min=scaled_lr_min,
        warmup_t=warmup_steps,
        warmup_lr_init=scaled_warmup_lr
    )
    logger.info(f'Learning rate schedule: CosineLR with {warmup_steps} warmup steps')

    ## mixed precision training
    scaler = amp.GradScaler()

    ## ddp training
    net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()
    time_meter.max_iter = max_epochs  # 保持 ETA 估计的准确性

    miou = 0.0
    mprecision = 0.0
    mrecall = 0.0
    gap = int(len(dl) / 10)
    if gap == 0: gap = 2

    ## 【改动4】训练循环：保留 epoch 结构 + 添加全局 step 计数
    global_step = 0

    for epoch in range(max_epochs):
        # 设置 epoch（用于数据洗牌的随机种子）
        if hasattr(dl, 'batch_sampler') and hasattr(dl.batch_sampler, 'sampler'):
            sampler = dl.batch_sampler.sampler
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)

        for it, (im, lb) in enumerate(dl):
            im = im.cuda()
            lb = lb.cuda()
            lb = torch.squeeze(lb, 1)

            optim.zero_grad()

            # 辅助头权重仍可按 epoch 衰减（保留原逻辑）
            aux_weight = 0.5 * (1 - epoch / max_epochs)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=cfg.use_fp16):
                logits, *logits_aux = net(im)
                loss_pre = criteria_pre(logits, lb)
                if cfg.num_aux_heads > 0:
                    loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
                    loss = loss_pre + aux_weight * sum(loss_aux)
                else:
                    loss_aux = []
                    loss = loss_pre

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5, norm_type=2)

            # nan/inf 防线
            if not torch.isfinite(loss):
                logger.warning(
                    'non-finite loss detected at epoch {}, iter {}, global_step {}, skip this step'.format(
                        epoch, it, global_step))
                scaler.update()
                optim.zero_grad(set_to_none=True)
                global_step += 1
                continue

            scaler.step(optim)
            scaler.update()

            # 【改动5】学习率调度按全局步数（关键改动）
            lr_schdr.step(global_step + 1)
            lr = optim.param_groups[0]['lr']

            torch.cuda.synchronize()

            # 损失记录
            loss_meter.update(loss.item())
            loss_pre_meter.update(loss_pre.item())
            if cfg.num_aux_heads > 0:
                _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

            ## 打印训练日志（增强版）
            if (it + 1) % gap == 0:
                if cfg.num_aux_heads > 0:
                    print_log_msgs(epoch, max_epochs, it, len(dl), lr, loss_meter, loss_pre_meter, loss_aux_meters, writer)
                else:
                    print_log_msgs_segformer(epoch, max_epochs, it, len(dl), lr, loss_meter, writer)

                # 【改动6】额外记录全局步数进度（可选）
                logger.info(f'[Global] Step {global_step}/{total_steps} ({100*global_step/total_steps:.1f}%)')

            # TensorBoard 记录（按全局步数）
            if (it + 1) % 10 == 0:  # 每 10 步记录一次，避免过于频繁
                writer.add_scalar('lr_step', lr, global_step)
                writer.add_scalar('loss_step', loss.item(), global_step)

            global_step += 1

        # Epoch 结束：验证
        interv, ets = time_meter.get()
        logger.info('ets:{}, interv:{:.2f}s'.format(ets, interv))
        time_meter.update()

        torch.cuda.empty_cache()
        iou_heads, iou_content, f1_heads, f1_content, precision_heads, precision_content, recall_heads, recall_content = eval_model(
            cfg, net.module, printlabels)
        net.train()  # 重要：验证后恢复训练模式

        # 三个指标合并成一个表显示（按类别名对齐，不改动原始结果）
        p_map = {r[0]: r[-1] for r in precision_content}
        r_map = {r[0]: r[-1] for r in recall_content}
        merged = []
        for row in iou_content:
            name, ratio, iou = row[0], row[1], row[-1]
            if name == 'mious':
                merged.append(['mean', '-', iou,
                               p_map.get('mprecision', '-'), r_map.get('mrecall', '-')])
            else:
                merged.append([name, ratio, iou, p_map.get(name, '-'), r_map.get(name, '-')])
        logger.info('\neval results:\n' + tabulate(
            merged, headers=['class', 'ratio', 'iou', 'precision', 'recall'], tablefmt='orgtbl'))

        # TensorBoard 记录（按 epoch）
        writer.add_scalar('miou_epoch', float(iou_content[-2][-1]), epoch)
        writer.add_scalar('mprecision_epoch', float(precision_content[-1][-1]), epoch)
        writer.add_scalar('mrecall_epoch', float(recall_content[-1][-1]), epoch)

        if miou < float(iou_content[-2][-1]):
            miou = float(iou_content[-2][-1])
            mprecision = float(precision_content[-1][-1])
            mrecall = float(recall_content[-1][-1])
            if dist.get_rank() == 0:
                torch.save(net.module.state_dict(), '../pt/best.pt')
            logger.info("miou:{}, mprecision:{}, mrecall:{}, save model!!!".format(miou, mprecision, mrecall))
        logger.info("best miou:{}, mprecision:{}, mrecall:{}".format(miou, mprecision, mrecall))

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

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

# 导入 SWA 相关工具
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

from lib.models import model_factory
from configs import set_cfg_from_file
from lib.data import get_data_loader
from evaluate import eval_model
from lib.ohem_ce_loss import OhemCELoss, FocalLoss, PolyFocalLoss, IoULoss, DiceLoss, LogCoshDiceOhemLovaszLoss
from lib.ohem_ce_loss import DiceWithOhemCELoss, DiceWithFocalLoss, DiceBCELoss, OhemWithIoULoss, OhemWithFocalLoss, \
    GDiceWithOhemCELoss, LogCoshDiceLossWithOhemCELoss, FocalTverskyWithOhemCELoss,OptimizedCombinedLoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg, print_log_msgs
from tqdm import tqdm
from timm.optim import AdamW, AdamP, RAdam, Lookahead, AdaBelief, Lion, Lamb
from timm.scheduler import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter
from timm.layers.norm_act import convert_sync_batchnorm

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

## fix all random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)
#  torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')
printlabels = ['background', 'QPZZ', 'MDBD', 'MNYW', 'WW', 'LMPS', 'BMQQ', 'LMHH', 'KTAK']


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--config', dest='config', type=str,
                       default='../configs/fastefficientformerseg_blueface_efficientnetv2_b3.py', )
    parse.add_argument('--finetune-from', type=str, default=None, )
    parse.add_argument("--local_rank", type=int)
    # 增加 SWA 相关的命令行参数
    parse.add_argument('--use-swa', action='store_true', default=True, help='whether to use SWA')
    parse.add_argument('--swa-start', type=float, default=0.75, help='start swa at this fraction of total epochs')
    parse.add_argument('--swa-lr', type=float, default=0.01, help='swa learning rate')
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
    if cfg.use_sync_bn: net = convert_sync_batchnorm(net)
    net.cuda()
    net.train()

    loss_opt = 9
    criteria_pre = 0
    criteria_aux = 0
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
        criteria_pre = OptimizedCombinedLoss()
        criteria_aux = [OptimizedCombinedLoss() for _ in range(cfg.num_aux_heads)]
    else:
        print('no such loss !!!')

    return net, criteria_pre, criteria_aux


def set_optimizer(model):
    optim_opt = 2
    optim = 0
    if (optim_opt == 0):
        optim = AdaBelief(model.parameters(), lr=cfg.lr_start, weight_decay=cfg.weight_decay)
    elif (optim_opt == 1):
        optim = AdamP(model.parameters(), lr=cfg.lr_start, weight_decay=cfg.weight_decay, nesterov=True)
    elif (optim_opt == 2):
        optim = AdamW(model.parameters(), lr=cfg.lr_start, weight_decay=cfg.weight_decay)
    elif (optim_opt == 3):
        optim = torch.optim.SGD(model.parameters(), lr=cfg.lr_start, momentum=0.9, weight_decay=cfg.weight_decay, )
    elif (optim_opt == 4):
        optim = Lion(model.parameters(), lr=cfg.lr_start, weight_decay=cfg.weight_decay, )
    elif (optim_opt == 5):
        optim = Lamb(model.parameters(), lr=cfg.lr_start, weight_decay=cfg.weight_decay, )
    return optim


def set_model_dist(net):
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

    ## optimizer
    optim = set_optimizer(net)

    ## SWA initialization
    use_swa = args.use_swa
    if use_swa:
        swa_model = AveragedModel(net)
        swa_start = int(args.swa_start * cfg.max_epochs)
        swa_scheduler = SWALR(optim, swa_lr=args.swa_lr)
        logger.info(f'SWA is enabled. Starting at epoch {swa_start} with lr {args.swa_lr}')

    ## mixed precision training
    scaler = amp.GradScaler()

    ## ddp training
    net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    lr_schdr = CosineLRScheduler(optimizer=optim,
                                 t_initial=cfg.max_epochs,
                                 lr_min=5e-6,
                                 warmup_t=2,
                                 warmup_lr_init=1e-4)

    miou = 0.0
    mprecision = 0.0
    mrecall = 0.0
    gap = int(len(dl) / 10)
    if (gap == 0): gap = 2

    ## train loop
    for epoch in range(cfg.max_epochs):

        # SWA 调度逻辑
        if use_swa and epoch >= swa_start:
            swa_model.update_parameters(net)
            swa_scheduler.step()
        else:
            lr_schdr.step(epoch)

        lr = optim.param_groups[0]['lr']
        writer.add_scalar('lr', lr, epoch)

        net.train()
        for it, (im, lb) in enumerate(dl):
            im = im.cuda()
            lb = lb.cuda()
            lb = torch.squeeze(lb, 1)

            optim.zero_grad()
            aux_weight = 0.5 * (1 - epoch / cfg.max_epochs)
            with amp.autocast(enabled=cfg.use_fp16):
                logits, *logits_aux = net(im)
                loss_pre = criteria_pre(logits, lb)
                loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
                loss = loss_pre + aux_weight * sum(loss_aux)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5, norm_type=2)
            scaler.step(optim)
            scaler.update()
            torch.cuda.synchronize()

            loss_meter.update(loss.item())
            loss_pre_meter.update(loss_pre.item())
            _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

            if (it + 1) % gap == 0:
                print_log_msgs(epoch, cfg.max_epochs, it, len(dl), lr, loss_meter, loss_pre_meter, loss_aux_meters,
                               writer)

        interv, ets = time_meter.get()
        logger.info('ets:{},interv:{:.2f}s'.format(ets, interv))
        time_meter.update()

        torch.cuda.empty_cache()

        # 评估阶段：如果开启 SWA 且在 SWA 阶段，使用 swa_model 进行评估
        eval_net = net.module
        if use_swa and epoch >= swa_start:
            # SWA 评估前需要更新 BN
            update_bn(dl, swa_model)
            eval_net = swa_model.module

        iou_heads, iou_content, f1_heads, f1_content, precision_heads, precision_content, recall_heads, recall_content = eval_model(
            cfg, eval_net, printlabels)

        logger.info('\neval results of miou metric:')
        logger.info('\n' + tabulate(iou_content, headers=iou_heads, tablefmt='orgtbl'))

        current_miou = float(iou_content[-2][-1])
        writer.add_scalar('miou', current_miou, epoch)
        writer.add_scalar('mprecision', float(precision_content[-1][-1]), epoch)
        writer.add_scalar('mrecall', float(recall_content[-1][-1]), epoch)

        if (miou < current_miou):
            miou = current_miou
            mprecision = float(precision_content[-1][-1])
            mrecall = float(recall_content[-1][-1])
            if dist.get_rank() == 0:
                torch.save(eval_net.state_dict(), '../pt/best.pt')
            logger.info("miou:{},mprecision:{},mrecall:{},save model!!!".format(miou, mprecision, mrecall))
        logger.info("best miou:{},mprecision:{},mrecall:{}".format(miou, mprecision, mrecall))

    # 训练结束，若开启了 SWA，最终保存一次 SWA 模型
    if use_swa and dist.get_rank() == 0:
        update_bn(dl, swa_model)
        torch.save(swa_model.module.state_dict(), '../pt/swa_final.pt')
        logger.info("Final SWA model saved to ../pt/swa_final.pt")

    return


def main(writer):
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = int(args.local_rank) if args.local_rank is not None else 0

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger(f'{cfg.model_type}-{cfg.dataset.lower()}-train', cfg.respth)
    train(writer)


if __name__ == "__main__":
    writer = SummaryWriter()
    main(writer)
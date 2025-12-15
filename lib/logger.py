#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os.path as osp
import time
import logging

import torch.distributed as dist


def setup_logger(name, logpth):
    logfile = '{}-{}.log'.format(name, time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and dist.get_rank() != 0:
        log_level = logging.WARNING
    try:
        logging.basicConfig(level=log_level, format=FORMAT, filename=logfile, force=True)
    except Exception:
        for hl in logging.root.handlers: logging.root.removeHandler(hl)
        logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())


def print_log_msg(epoch,epochs,it, max_iter, lr, time_meter, loss_meter, loss_pre_meter,loss_aux_meters):
    t_intv, eta = time_meter.get()
    loss_avg, _ = loss_meter.get()
    loss_pre_avg, _ = loss_pre_meter.get()
    loss_aux_avg = ', '.join(['{}: {:.4f}'.format(el.name, el.get()[0]) for el in loss_aux_meters])
    msg = ', '.join(['{}/{}',
        'iter: {it}/{max_it}',
        'lr: {lr:4f}',
        'eta: {eta}',
        'time: {time:.2f}',
        'loss: {loss:.4f}',
        'loss_pre: {loss_pre:.4f}',
    ]).format(epoch,epochs,
        it=it+1,
        max_it=max_iter,
        lr=lr,
        time=t_intv,
        eta=eta,
        loss=loss_avg,
        loss_pre=loss_pre_avg,
        )
    msg += ', ' + loss_aux_avg
    logger = logging.getLogger()
    logger.info(msg)

def print_log_msgs(epoch,epochs,it, max_iter, lr, loss_meter, loss_pre_meter,loss_aux_meters,writer):
    loss_avg, _ = loss_meter.get()
    loss_pre_avg, _ = loss_pre_meter.get()
    loss_aux_avg = ', '.join(['{}: {:.4f}'.format(el.name, el.get()[0]) for el in loss_aux_meters])
    msg = ', '.join(['{}/{}',
        'iter: {it}/{max_it}',
        'lr: {lr:4f}',
        'loss: {loss:.4f}',
        'loss_pre: {loss_pre:.4f}',
    ]).format(epoch,epochs,
        it=it+1,
        max_it=max_iter,
        lr=lr,
        loss=loss_avg,
        loss_pre=loss_pre_avg,
        )
    msg += ', ' + loss_aux_avg
    logger = logging.getLogger()
    logger.info(msg)
    writer.add_scalar('loss/loss', loss_avg, epoch)
    writer.add_scalar('loss/loss_pre', loss_pre_avg, epoch)

    splitlines = loss_aux_avg.split(',')
    for splitline in splitlines:
        line = splitline.strip().split(':')
        name = line[0]
        score = line[1].strip()
        writer.add_scalar('loss/{}'.format(name), float(score), epoch)

def print_log_msgs_segformer(epoch,epochs,it, max_iter, lr, loss_meter,writer):
    loss_avg, _ = loss_meter.get()
    msg = ', '.join(['{}/{}',
        'iter: {it}/{max_it}',
        'lr: {lr:4f}',
        'loss: {loss:.4f}',
    ]).format(epoch,epochs,
        it=it+1,
        max_it=max_iter,
        lr=lr,
        loss=loss_avg,
        )
    logger = logging.getLogger()
    logger.info(msg)
    writer.add_scalar('loss/loss', loss_avg, epoch)


def print_log_msgsV2(epoch,epochs,it, max_iter, lr, loss_meter, loss_pre_meter,loss_aux_meters,arcfaceloss,writer):
    loss_avg, _ = loss_meter.get()
    loss_pre_avg, _ = loss_pre_meter.get()
    loss_aux_avg = ', '.join(['{}: {:.4f}'.format(el.name, el.get()[0]) for el in loss_aux_meters])
    msg = ', '.join(['{}/{}',
        'iter: {it}/{max_it}',
        'lr: {lr:4f}',
        'loss: {loss:.4f}',
        'loss_pre: {loss_pre:.4f}',
    ]).format(epoch,epochs,
        it=it+1,
        max_it=max_iter,
        lr=lr,
        loss=loss_avg,
        loss_pre=loss_pre_avg,
        )
    msg += ', ' + loss_aux_avg
    msg += ', ' + 'arcfaceloss:{:.4f}'.format(arcfaceloss)
    writer.add_scalar('loss/arcfaceloss', arcfaceloss, epoch)
    logger = logging.getLogger()
    logger.info(msg)
    writer.add_scalar('loss/loss', loss_avg, epoch)
    writer.add_scalar('loss/loss_pre', loss_pre_avg, epoch)

    splitlines = loss_aux_avg.split(',')
    for splitline in splitlines:
        line = splitline.strip().split(':')
        name = line[0]
        score = line[1].strip()
        writer.add_scalar('loss/{}'.format(name), float(score), epoch)




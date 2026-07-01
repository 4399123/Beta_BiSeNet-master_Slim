#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax

from .lovasz_losses import lovasz_softmax
class OhemCELoss(nn.Module):

    def __init__(self, thresh, lb_ignore=255):
        super(OhemCELoss, self).__init__()
        # Bug fix 1: 不在 __init__ 里 .cuda()，避免 DDP 多卡时固化到 cuda:0
        # 只保存原始 float 值，forward 里按 logits.device 动态创建
        self.thresh_val = float(-torch.log(torch.tensor(thresh, dtype=torch.float)).item())
        self.lb_ignore = lb_ignore
        # self.weight_CE = torch.FloatTensor([1, 22, 11265, 2606,520,8602,150,423]).to('cuda')
        # self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='none',weight=self.weight_CE)
        self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='none')

    def forward(self, logits, labels):
        # Bug fix 1: thresh 跟随 logits 的设备，避免跨设备比较报错
        thresh = torch.tensor(self.thresh_val, dtype=torch.float, device=logits.device)
        # Bug fix 2: n_min 至少为 1，防止全 ignore batch 时 topk(0) 返回空张量导致 nan
        n_min = max(labels[labels != self.lb_ignore].numel() // 16, 1)
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)




#FocalLoss
def focal_loss(input_values, gamma):
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, lb_ignore=255, gamma=2):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.lb_ignore = lb_ignore

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', ignore_index=self.lb_ignore), self.gamma)


#PolyFocalLoss
#PolyFocalLoss
class PolyFocalLoss(nn.Module):
    # def __init__(self, alpha=0.25, gamma=2, class_number=8,lb_ignore=255):
    def __init__(self, alpha=0.25, gamma=2, lb_ignore=255):
        super( PolyFocalLoss, self, ).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = 1.0
        # self.classnum = class_number
        self.lb_ignore = lb_ignore

    def forward(self, input, target):
        focal_loss_func = FocalLoss(gamma=self.gamma,lb_ignore=255)
        focal_loss = focal_loss_func(input, target)

        p = torch.sigmoid(input)
        # labels = torch.nn.functional.one_hot(target, self.classnum)
        labels = torch.nn.functional.one_hot(target, input.shape[1]).permute(0, 3, 1, 2)
        # labels = torch.tensor(labels, dtype=torch.float32)
        labels=labels.clone()
        poly = labels * p + (1 - labels) * (1 - p)
        poly_focal_loss = focal_loss + torch.mean(self.epsilon * torch.pow(1 - poly, 2 + 1), dim=-1)
        return poly_focal_loss.mean()


# ----------------- DICE Loss--------------------
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets, mask=False):
        assert len(logits.shape) == 4, "inputs shape must be NCHW"
        targets = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2).float()
        num = targets.size(0)
        smooth = 1.

        # probs = torch.sigmoid(logits)
        probs = torch.softmax(logits, dim=1)
        m1 = probs.contiguous().view(num, -1)
        m2 = targets.contiguous().view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

# ----------------- Generalized Dice Loss--------------------
class GeneralizedDiceLoss(nn.Module):
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, pred, target,epsilon=1e-6):
        """compute the weighted dice_loss
         Args:
             pred (tensor): prediction after softmax, shape(bath_size, channels, height, width)
             target (tensor): gt, shape(bath_size, channels, height, width)
         Returns:
             gldice_loss: loss value
         """
        pred=torch.softmax(pred,dim=1)
        target=torch.unsqueeze(target,dim=1)
        wei = torch.sum(target, axis=[0, 2, 3])  # (n_class,)
        wei = 1 / (wei ** 2 + epsilon)
        intersection = torch.sum(wei * torch.sum(pred * target, axis=[0, 2, 3]))
        union = torch.sum(wei * torch.sum(pred + target, axis=[0, 2, 3]))
        gldice_loss = 1 - (2. * intersection) / (union + epsilon)
        return gldice_loss

# -------------------- BCELoss -----------------------
class BCELoss(nn.Module):
    """binary bceloss with sigmoid"""

    def __init__(self,lb_ignore=255):
        super(BCELoss, self).__init__()
        self.criteria = nn.CrossEntropyLoss(ignore_index=lb_ignore, reduction='none')

    def forward(self, inputs, targets, weights=None, mask=False):
        assert len(inputs.shape) == 4, "inputs shape must be NCHW"
        targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2).float()
        if mask:
            inputs = inputs * targets
        # losses = F.binary_cross_entropy_with_logits(inputs, targets, weights,reduction='none')
        losses =self.criteria(inputs, targets)
        return losses

class CELoss(nn.Module):
    def __init__(self, lb_ignore=255, gamma=2):
        super(CELoss, self).__init__()
        self.lb_ignore = lb_ignore

    def forward(self, input, target):
        return torch.mean(F.cross_entropy(input, target, reduction='none', ignore_index=self.lb_ignore))

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2).float()

        # flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU


class LogCoshDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LogCoshDiceLoss, self).__init__()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets, smooth=1):
        x=self.dice_loss(inputs, targets)
        return torch.log((torch.exp(x) + torch.exp(-x)) / 2.0)



class LogCoshDiceLossWithOhemCELoss(nn.Module):
    def __init__(self):
        super(LogCoshDiceLossWithOhemCELoss, self).__init__()
        self.dice_loss = LogCoshDiceLoss()
        self.ohem_loss = OhemCELoss(0.7)
    def forward(self, preds, targets):
        ohemloss = self.ohem_loss(preds, targets)
        diceloss = self.dice_loss(preds, targets)
        return ohemloss + 3*diceloss

class DiceWithOhemCELoss(nn.Module):
    def __init__(self):
        super(DiceWithOhemCELoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.ohem_loss = OhemCELoss(0.7)
    def forward(self, preds, targets):
        ohemloss = self.ohem_loss(preds, targets)
        diceloss = self.dice_loss(preds, targets)
        return ohemloss + 3*diceloss

class DiceWithFocalLoss(nn.Module):
    def __init__(self):
        super( DiceWithFocalLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
    def forward(self, preds, targets):
        focalloss = self.focal_loss(preds, targets)
        diceloss = self.dice_loss(preds, targets)
        return diceloss+2*focalloss

class OhemWithIoULoss(nn.Module):
    def __init__(self):
        super(OhemWithIoULoss, self).__init__()
        self.iou_loss = IoULoss()
        self.ohem_loss = OhemCELoss(0.7)
    def forward(self, preds, targets):
        iouloss = self.iou_loss(preds, targets)
        ohemloss = self.ohem_loss(preds, targets)
        return 2*iouloss + ohemloss

class OhemWithFocalLoss(nn.Module):
    def __init__(self):
        super(OhemWithFocalLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.ohem_loss = OhemCELoss(0.7)
    def forward(self, preds, targets):
        focalloss = self.focal_loss(preds, targets)
        ohemloss = self.ohem_loss(preds, targets)
        return focalloss + ohemloss


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2).float()

        # flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class GDiceWithOhemCELoss(nn.Module):
    def __init__(self):
        super(GDiceWithOhemCELoss, self).__init__()
        self.gdice_loss = GeneralizedDiceLoss()
        self.ohem_loss = OhemCELoss(0.7)
    def forward(self, preds, targets):
        ohemloss = self.ohem_loss(preds, targets)
        gdiceloss = self.gdice_loss(preds, targets)
        return ohemloss + 3*gdiceloss


def flatten(input, target, ignore_index):
    num_class = input.size(1)
    input = input.permute(0, 2, 3, 1).contiguous()

    input_flatten = input.view(-1, num_class)
    target_flatten = target.view(-1)

    mask = (target_flatten != ignore_index)
    input_flatten = input_flatten[mask]
    target_flatten = target_flatten[mask]

    return input_flatten, target_flatten


class FocalTverskyLoss(nn.Module):
    def __init__(self, ignore_index=255, smooth=1.0, alpha=0.3, beta=0.7, gamma=1.33):
        super(FocalTverskyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, input, target):
        input, target = flatten(input, target, self.ignore_index)
        # 用 fp32 计算，数值更稳（即使关闭 AMP 也无副作用）
        input = softmax(input.float(), dim=1)
        num_classes = input.size(1)
        n_valid = input.size(0)  # 有效像素总数（已过滤 ignore）
        if n_valid == 0:
            return input.sum() * 0.0  # 全 ignore batch，返回 0 梯度

        losses = []
        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]

            # 归一化到 [0,1] 区间：除以有效像素数，消除分辨率对量级的影响
            t_p = (input_c * target_c).sum() / n_valid
            f_p = ((1 - target_c) * input_c).sum() / n_valid
            f_n = (target_c * (1 - input_c)).sum() / n_valid

            tversky = (t_p + self.smooth) / (t_p + self.alpha * f_p + self.beta * f_n + self.smooth)
            # gamma>=1 时 u^gamma 的导数 gamma*u^(gamma-1) 在 u=(1-tversky)→0 处有界,
            # 从数学上根除了 gamma<1 时的梯度奇点(后期模型高度自信会触发 nan)。
            # clamp 仅作防御性保护, 杜绝极端浮点情况下的负底数。
            one_minus_tversky = (1.0 - tversky).clamp(min=0.0, max=1.0)
            focal_tversky = one_minus_tversky ** self.gamma
            losses.append(focal_tversky)

        losses = torch.stack(losses)
        loss = losses.mean()
        return loss

class FocalTverskyWithOhemCELoss(nn.Module):
    def __init__(self, ignore_index=255, thresh=0.7,
                 alpha=0.3, beta=0.7, gamma=1.33, smooth=1.0,
                 tversky_weight=1.0, ohem_weight=1.0):
        super(FocalTverskyWithOhemCELoss, self).__init__()
        self.ft_loss = FocalTverskyLoss(
            ignore_index=ignore_index, smooth=smooth,
            alpha=alpha, beta=beta, gamma=gamma
        )
        self.ohem_loss = OhemCELoss(thresh, lb_ignore=ignore_index)
        self.tversky_weight = tversky_weight
        self.ohem_weight = ohem_weight

    def forward(self, preds, targets):
        ohemloss = self.ohem_loss(preds, targets)
        ft_loss = self.ft_loss(preds, targets)
        return self.ohem_weight * ohemloss + self.tversky_weight * ft_loss



class OhemLovaszLoss(nn.Module):
    def __init__(self, thresh=0.7, lb_ignore=255, ohem_weight=1.5, lovasz_weight=1.0):
        super().__init__()
        self.ohem = OhemCELoss(thresh=thresh, lb_ignore=lb_ignore)
        self.ohem_weight = ohem_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, preds, targets):
        ohem_loss = self.ohem(preds, targets)
        # 注意：Lovász 需要 softmax 输入
        lovasz_loss = lovasz_softmax(
            F.softmax(preds, dim=1),
            targets,
            ignore=255
        )
        return self.ohem_weight * ohem_loss + self.lovasz_weight * lovasz_loss


class LogCoshDiceOhemLovaszLoss(nn.Module):
    def __init__(self, thresh=0.7, lb_ignore=255, ohem_weight=1.5, lovasz_weight=1.0):
        super().__init__()
        self.ohem = LogCoshDiceLossWithOhemCELoss()
        self.ohem_weight = ohem_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, preds, targets):
        ohem_loss = self.ohem(preds, targets)
        # 注意：Lovász 需要 softmax 输入
        lovasz_loss = lovasz_softmax(
            F.softmax(preds, dim=1),
            targets,
            ignore=255
        )
        return self.ohem_weight * ohem_loss + self.lovasz_weight * lovasz_loss


class OptimizedCombinedLoss(nn.Module):
    """优化的组合损失"""
    def __init__(self, lb_ignore=255):
        super().__init__()
        self.ohem_ce = OhemCELoss(0.7)
        self.dice = LogCoshDiceLoss()
        self.lovasz = lovasz_softmax

        # 动态权重
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, pred, target, epoch=0):
        # 基础损失
        loss_ce = self.ohem_ce(pred, target)
        loss_dice = self.dice(pred, target)
        loss_lovasz = self.lovasz(F.softmax(pred, dim=1), target, ignore=255)

        # 动态权重（使用 softmax 归一化）
        w = F.softmax(self.weights, dim=0)

        # 组合
        total_loss = (w[0] * loss_ce +
                      w[1] * loss_dice +
                      w[2] * loss_lovasz )

        return total_loss



if __name__ == '__main__':
    pass


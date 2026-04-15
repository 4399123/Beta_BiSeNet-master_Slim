# 🚀 快速开始优化指南

这是一个精简版的实施指南，帮助你在 **1-2 天内**实现 **+5-8% mIoU** 的提升。

## 📋 准备工作（10 分钟）

```bash
# 1. 备份当前代码
cp -r . ../backup_$(date +%Y%m%d)

# 2. 创建新分支
git checkout -b optimization_v1

# 3. 安装依赖
pip install wandb albumentations
```

## 🎯 第一步：数据增强升级（30 分钟）

### 1. 创建 GridMask
```bash
# 创建文件 lib/data/gridmask.py
```

```python
import numpy as np

class GridMask:
    def __init__(self, d_range=(96, 224), ratio=0.6, prob=0.3):
        self.d_range = d_range
        self.ratio = ratio
        self.prob = prob
    
    def __call__(self, im_lb):
        if np.random.random() > self.prob:
            return im_lb
        
        im, lb = im_lb['im'], im_lb['lb']
        h, w = im.shape[:2]
        d = np.random.randint(*self.d_range)
        
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(0, h, d):
            for j in range(0, w, d):
                mask[i:i+int(d*self.ratio), j:j+int(d*self.ratio)] = 0
        
        mask = mask[:, :, None]
        im = (im * mask).astype(np.uint8)
        
        return dict(im=im, lb=lb)
```

### 2. 修改 transform_cv2.py
```python
# 在 lib/data/transform_cv2.py 中添加
from .gridmask import GridMask

class TransformationTrainOptimized(object):
    def __init__(self, scales, cropsize):
        self.trans_func = Compose([
            RandomResizedCrop(scales, cropsize),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            GridMask(prob=0.3),  # 新增！
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ])
    
    def __call__(self, im_lb):
        return self.trans_func(im_lb)
```

### 3. 更新 get_dataloader.py
```python
# 修改 train_model=0 的部分
if(train_model==0):
    trans_func = T.TransformationTrainOptimized(cfg.scales, cfg.cropsize)  # 改这里
    batchsize = cfg.ims_per_gpu
    annpath = cfg.train_im_anns
    shuffle = True
    drop_last = True
```

## 🎯 第二步：EMA 实现（20 分钟）

### 1. 创建 EMA 模块
```bash
# 创建文件 lib/ema.py
```

```python
import torch

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + \
                             self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
```

### 2. 修改 train_amp.py
```python
# 在文件开头添加
from lib.ema import ModelEMA

# 在 train() 函数中，创建模型后添加
def train():
    # ... 现有代码 ...
    net, criteria_pre, criteria_aux = set_model(dl.dataset.lb_ignore)
    
    # 添加 EMA
    ema = ModelEMA(net, decay=0.9999)
    
    # ... 训练循环 ...
    for it, (im, lb) in tqdm(enumerate(dl)):
        # ... 现有训练代码 ...
        scaler.step(optim)
        scaler.update()
        ema.update()  # 添加这行
        
        # ... 其余代码 ...
    
    # 评估前应用 EMA
    ema.apply_shadow()
    iou_heads, iou_content, f1_heads, f1_content = eval_model(cfg, net.module)
    ema.restore()
```

## 🎯 第三步：优化损失函数（30 分钟）

### 1. 创建组合损失
```bash
# 创建文件 lib/optimized_loss.py
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ohem_ce_loss import OhemCELoss
from .lovasz_losses import lovasz_softmax

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits, targets):
        targets_one_hot = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2).float()
        probs = torch.sigmoid(logits)
        
        num = targets.size(0)
        m1 = probs.contiguous().view(num, -1)
        m2 = targets_one_hot.contiguous().view(num, -1)
        intersection = (m1 * m2).sum(1)
        
        score = 2. * intersection / (m1.sum(1) + m2.sum(1) + 1e-5)
        return 1 - score.mean()

class OptimizedLoss(nn.Module):
    def __init__(self, ohem_weight=1.5, dice_weight=1.0, lovasz_weight=1.0):
        super().__init__()
        self.ohem = OhemCELoss(0.7)
        self.dice = DiceLoss()
        self.ohem_weight = ohem_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
    
    def forward(self, preds, targets):
        loss_ohem = self.ohem(preds, targets)
        loss_dice = self.dice(preds, targets)
        loss_lovasz = lovasz_softmax(F.softmax(preds, dim=1), targets, ignore=255)
        
        total = (self.ohem_weight * loss_ohem + 
                self.dice_weight * loss_dice + 
                self.lovasz_weight * loss_lovasz)
        
        return total
```

### 2. 修改 train_amp.py
```python
# 导入新损失
from lib.optimized_loss import OptimizedLoss

# 修改 set_model 函数
def set_model(lb_ignore=255):
    logger = logging.getLogger()
    net = model_factory[cfg.model_type](cfg.n_cats)
    # ... 现有代码 ...
    
    # 使用新损失
    criteria_pre = OptimizedLoss(ohem_weight=1.5, dice_weight=1.0, lovasz_weight=1.0)
    criteria_aux = [OptimizedLoss(ohem_weight=1.5, dice_weight=1.0, lovasz_weight=1.0)
                   for _ in range(cfg.num_aux_heads)]
    
    return net, criteria_pre, criteria_aux
```

## 🎯 第四步：切换到 AdamW（15 分钟）

### 修改 train_amp.py 中的优化器
```python
def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, 'weight_decay': cfg.weight_decay},
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10, 'weight_decay': cfg.weight_decay},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, 'weight_decay': cfg.weight_decay},
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    
    # 改用 AdamW
    optim = torch.optim.AdamW(
        params_list,
        lr=cfg.lr_start,
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay,
    )
    return optim
```

## 🎯 第五步：更新配置（5 分钟）

### 修改你的配置文件
```python
# configs/bisenetv1_blueface_efficientnet_b3.py
cfg = dict(
    model_type='bisenetv1_efficientnet_b3',
    n_cats=9,
    num_aux_heads=2,
    lr_start=0.001,  # AdamW 通常用更大的学习率
    weight_decay=1e-4,  # 降低 weight decay
    max_epochs=300,
    dataset='BlueFaceDataset',
    im_root='../../BlueFaceDataX2',
    train_im_anns='../../BlueFaceDataX2/train.txt',
    val_im_anns='../../BlueFaceDataX2/val.txt',
    scales=[0.75, 1.5],  # 扩大范围
    cropsize=[640, 640],  # 可选：增大尺寸
    ims_per_gpu=16,  # 根据显存调整
    eval_ims_per_gpu=1,
    use_fp16=True,
    use_sync_bn=True,
    respth='./res_optimized',
)
```

## 🚀 开始训练

```bash
python tools/train_amp.py --config configs/bisenetv1_blueface_efficientnet_b3.py
```

## 📊 预期结果

实施以上优化后，你应该能看到：

- ✅ **GridMask**: +1-2% mIoU
- ✅ **EMA**: +0.5-1% mIoU
- ✅ **组合损失**: +2-3% mIoU
- ✅ **AdamW**: +1-2% mIoU

**总计**: +5-8% mIoU 提升

## 🔍 验证优化效果

### 对比实验
```bash
# 1. 训练基线模型
python tools/train_amp.py --config configs/baseline.py

# 2. 训练优化模型
python tools/train_amp.py --config configs/optimized.py

# 3. 对比结果
python tools/compare_results.py --baseline res/baseline --optimized res_optimized/
```

## ⚠️ 常见问题

### Q1: 显存不足
```python
# 解决方案：
# 1. 减小 batch size
ims_per_gpu=8

# 2. 减小输入尺寸
cropsize=[512, 512]

# 3. 使用梯度累积
accumulation_steps = 4
```

### Q2: 训练不稳定
```python
# 解决方案：
# 1. 降低学习率
lr_start=0.0005

# 2. 增加 warmup
warmup_iters=1000

# 3. 使用梯度裁剪
torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
```

### Q3: EMA 导致性能下降
```python
# 解决方案：
# 1. 调整 decay
ema = ModelEMA(net, decay=0.999)  # 降低 decay

# 2. 延迟启动 EMA
if epoch > 10:  # 前 10 个 epoch 不用 EMA
    ema.update()
```

## 📈 下一步

完成快速优化后，可以继续实施：

1. **CutMix/Mosaic** (+1-2%)
2. **TTA** (+1-2%)
3. **知识蒸馏** (+2-3%)

详见 `COMPREHENSIVE_OPTIMIZATION_GUIDE.md`

---

**预计时间**: 2 小时  
**预期提升**: +5-8% mIoU  
**难度**: ⭐⭐ (中等)

祝你优化顺利！🎉

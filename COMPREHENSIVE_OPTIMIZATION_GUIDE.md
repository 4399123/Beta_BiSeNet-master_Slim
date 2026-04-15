# 🚀 语义分割模型全面优化指南

基于对你整个工程的深入分析，我发现了多个可以显著提升性能的优化点。

## 📊 当前工程分析

### ✅ 做得好的地方
1. **混合精度训练** (AMP) - 已实现
2. **OHEM Loss** - 困难样本挖掘
3. **多种损失函数** - Lovasz, Dice, Focal 等
4. **数据增强** - RandomCrop, Flip, ColorJitter
5. **学习率调度** - Warmup + Poly

### ⚠️ 需要改进的地方
1. **数据增强不够丰富**
2. **缺少现代训练技巧**
3. **损失函数组合不够优化**
4. **缺少模型集成策略**
5. **评估指标单一**
6. **缺少在线难例挖掘**
7. **没有使用知识蒸馏**
8. **缺少测试时增强 (TTA)**

---

## 🎯 优化建议（按优先级排序）


### 🥇 优先级 1: 数据增强升级（预期提升 3-5%）

#### 1.1 添加 Mixup/CutMix
```python
# lib/data/mixup.py
import numpy as np
import torch

class MixUp:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        
        # 生成 lambda
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 随机打乱索引
        index = torch.randperm(batch_size)
        
        # Mixup
        mixed_images = lam * images + (1 - lam) * images[index]
        labels_a, labels_b = labels, labels[index]
        
        return mixed_images, labels_a, labels_b, lam

class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        
        lam = np.random.beta(self.alpha, self.alpha)
        index = torch.randperm(batch_size)
        
        # 生成随机框
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # CutMix
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        labels[:, bby1:bby2, bbx1:bbx2] = labels[index, bby1:bby2, bbx1:bbx2]
        
        return images, labels
```

#### 1.2 添加 Mosaic 数据增强
```python
# lib/data/mosaic.py
class MosaicAugmentation:
    def __init__(self, size=(512, 512)):
        self.size = size
    
    def __call__(self, images_list, labels_list):
        # 4 张图拼接成 1 张
        h, w = self.size
        mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
        mosaic_label = np.zeros((h, w), dtype=np.uint8)
        
        for i, (img, label) in enumerate(zip(images_list, labels_list)):
            # 缩放到 h/2, w/2
            img_resized = cv2.resize(img, (w//2, h//2))
            label_resized = cv2.resize(label, (w//2, h//2), 
                                      interpolation=cv2.INTER_NEAREST)
            
            # 放置到对应位置
            if i == 0:  # 左上
                mosaic_img[:h//2, :w//2] = img_resized
                mosaic_label[:h//2, :w//2] = label_resized
            elif i == 1:  # 右上
                mosaic_img[:h//2, w//2:] = img_resized
                mosaic_label[:h//2, w//2:] = label_resized
            elif i == 2:  # 左下
                mosaic_img[h//2:, :w//2] = img_resized
                mosaic_label[h//2:, :w//2] = label_resized
            else:  # 右下
                mosaic_img[h//2:, w//2:] = img_resized
                mosaic_label[h//2:, w//2:] = label_resized
        
        return mosaic_img, mosaic_label
```

#### 1.3 添加 GridMask
```python
# lib/data/gridmask.py
class GridMask:
    def __init__(self, d_range=(96, 224), ratio=0.6, prob=0.5):
        self.d_range = d_range
        self.ratio = ratio
        self.prob = prob
    
    def __call__(self, im_lb):
        if np.random.random() > self.prob:
            return im_lb
        
        im, lb = im_lb['im'], im_lb['lb']
        h, w = im.shape[:2]
        
        # 随机选择网格大小
        d = np.random.randint(*self.d_range)
        
        # 创建网格掩码
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(0, h, d):
            for j in range(0, w, d):
                mask[i:i+int(d*self.ratio), j:j+int(d*self.ratio)] = 0
        
        # 应用掩码
        mask = mask[:, :, None]
        im = (im * mask).astype(np.uint8)
        
        return dict(im=im, lb=lb)
```

**使用方法**：
```python
# 在 transform_cv2.py 中添加
class TransformationTrainAdvanced(object):
    def __init__(self, scales, cropsize):
        self.trans_func = Compose([
            RandomResizedCrop(scales, cropsize),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            GridMask(prob=0.3),  # 新增
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ])
```

---


### 🥈 优先级 2: 损失函数优化（预期提升 2-4%）

#### 2.1 边界感知损失 (Boundary Loss)
```python
# lib/boundary_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

class BoundaryLoss(nn.Module):
    """边界损失 - 专注于边缘区域"""
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, pred, target):
        # 计算边界
        target_boundary = self.get_boundary(target)
        
        # 计算距离图
        dist_map = self.compute_distance_map(target_boundary)
        
        # 加权 CE Loss
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = ce_loss * (1 + dist_map)
        
        return weighted_loss.mean()
    
    def get_boundary(self, target):
        # 使用形态学操作提取边界
        kernel = torch.ones(1, 1, 3, 3).to(target.device)
        target_float = target.float().unsqueeze(1)
        
        eroded = F.conv2d(target_float, kernel, padding=1)
        boundary = (eroded != 9 * target_float).float()
        
        return boundary.squeeze(1)
    
    def compute_distance_map(self, boundary):
        # 计算到边界的距离
        dist_maps = []
        for b in boundary:
            dist = distance_transform_edt(1 - b.cpu().numpy())
            dist = np.exp(-dist / self.theta)
            dist_maps.append(torch.from_numpy(dist))
        
        return torch.stack(dist_maps).to(boundary.device)
```

#### 2.2 组合损失策略
```python
# lib/combined_loss.py
class OptimizedCombinedLoss(nn.Module):
    """优化的组合损失"""
    def __init__(self, n_classes):
        super().__init__()
        self.ohem_ce = OhemCELoss(0.7)
        self.dice = DiceLoss()
        self.lovasz = lovasz_softmax
        self.boundary = BoundaryLoss()
        
        # 动态权重
        self.weights = nn.Parameter(torch.ones(4))
    
    def forward(self, pred, target, epoch=0):
        # 基础损失
        loss_ce = self.ohem_ce(pred, target)
        loss_dice = self.dice(pred, target)
        loss_lovasz = self.lovasz(F.softmax(pred, dim=1), target, ignore=255)
        loss_boundary = self.boundary(pred, target)
        
        # 动态权重（使用 softmax 归一化）
        w = F.softmax(self.weights, dim=0)
        
        # 组合
        total_loss = (w[0] * loss_ce + 
                     w[1] * loss_dice + 
                     w[2] * loss_lovasz + 
                     w[3] * loss_boundary)
        
        return total_loss, {
            'ce': loss_ce.item(),
            'dice': loss_dice.item(),
            'lovasz': loss_lovasz.item(),
            'boundary': loss_boundary.item(),
            'weights': w.detach().cpu().numpy()
        }
```

#### 2.3 类别平衡损失
```python
# lib/class_balanced_loss.py
class ClassBalancedLoss(nn.Module):
    """类别平衡损失 - 处理类别不平衡"""
    def __init__(self, samples_per_class, beta=0.9999):
        super().__init__()
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / weights.sum() * len(weights)
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
    
    def forward(self, pred, target):
        self.weights = self.weights.to(pred.device)
        return F.cross_entropy(pred, target, weight=self.weights, ignore_index=255)
```

**使用建议**：
```python
# 在训练脚本中
criteria_pre = OptimizedCombinedLoss(cfg.n_cats)

# 训练循环中
loss, loss_dict = criteria_pre(logits, lb, epoch=epoch)
```

---


### 🥉 优先级 3: 训练策略优化（预期提升 2-3%）

#### 3.1 EMA (Exponential Moving Average)
```python
# lib/ema.py
class ModelEMA:
    """模型指数移动平均 - 稳定训练，提升泛化"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 初始化 shadow 参数
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

# 使用方法
ema = ModelEMA(net, decay=0.9999)

# 训练循环中
for it, (im, lb) in enumerate(dl):
    # ... 前向传播和反向传播 ...
    optim.step()
    ema.update()  # 更新 EMA

# 评估时
ema.apply_shadow()
eval_model(cfg, net)
ema.restore()
```

#### 3.2 Stochastic Weight Averaging (SWA)
```python
# lib/swa.py
from torch.optim.swa_utils import AveragedModel, SWALR

# 在训练脚本中
swa_model = AveragedModel(net)
swa_scheduler = SWALR(optim, swa_lr=0.0001)

# 训练循环
for epoch in range(cfg.max_epochs):
    for it, (im, lb) in enumerate(dl):
        # ... 正常训练 ...
        
    # 在最后几个 epoch 使用 SWA
    if epoch >= cfg.max_epochs - 10:
        swa_model.update_parameters(net)
        swa_scheduler.step()

# 最后更新 BN 统计
torch.optim.swa_utils.update_bn(dl, swa_model)
```

#### 3.3 Cosine Annealing with Warm Restarts
```python
# lib/lr_scheduler.py 中添加
class CosineAnnealingWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        self.T_i = T_0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            self.T_cur = epoch
        
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
```

#### 3.4 Gradient Accumulation（大 Batch 模拟）
```python
# 在 train_amp.py 中修改
accumulation_steps = 4  # 模拟 batch_size * 4

for it, (im, lb) in enumerate(dl):
    im = im.cuda()
    lb = lb.cuda().squeeze(1)
    
    with amp.autocast(enabled=cfg.use_fp16):
        logits, *logits_aux = net(im)
        loss_pre = criteria_pre(logits, lb)
        loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
        loss = (loss_pre + sum(loss_aux)) / accumulation_steps
    
    scaler.scale(loss).backward()
    
    # 每 accumulation_steps 步更新一次
    if (it + 1) % accumulation_steps == 0:
        scaler.step(optim)
        scaler.update()
        optim.zero_grad()
```

---


### 🎖️ 优先级 4: 测试时增强 (TTA)（预期提升 1-2%）

#### 4.1 多尺度测试
```python
# tools/tta_inference.py
class MultiScaleTTA:
    def __init__(self, model, scales=[0.75, 1.0, 1.25, 1.5]):
        self.model = model
        self.scales = scales
    
    def __call__(self, image):
        h, w = image.shape[2:]
        final_pred = torch.zeros(1, self.model.n_classes, h, w).cuda()
        
        for scale in self.scales:
            # 缩放图像
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = F.interpolate(image, size=(new_h, new_w), 
                                      mode='bilinear', align_corners=False)
            
            # 推理
            with torch.no_grad():
                pred = self.model(scaled_img)
                if isinstance(pred, tuple):
                    pred = pred[0]
            
            # 缩放回原尺寸
            pred = F.interpolate(pred, size=(h, w), 
                                mode='bilinear', align_corners=False)
            
            final_pred += pred
        
        # 平均
        final_pred /= len(self.scales)
        return final_pred

#### 4.2 翻转测试
class FlipTTA:
    def __init__(self, model):
        self.model = model
    
    def __call__(self, image):
        # 原始
        with torch.no_grad():
            pred1 = self.model(image)
            if isinstance(pred1, tuple):
                pred1 = pred1[0]
        
        # 水平翻转
        image_flip = torch.flip(image, dims=[3])
        with torch.no_grad():
            pred2 = self.model(image_flip)
            if isinstance(pred2, tuple):
                pred2 = pred2[0]
        pred2 = torch.flip(pred2, dims=[3])
        
        # 垂直翻转
        image_vflip = torch.flip(image, dims=[2])
        with torch.no_grad():
            pred3 = self.model(image_vflip)
            if isinstance(pred3, tuple):
                pred3 = pred3[0]
        pred3 = torch.flip(pred3, dims=[2])
        
        # 平均
        final_pred = (pred1 + pred2 + pred3) / 3
        return final_pred

#### 4.3 组合 TTA
class CombinedTTA:
    def __init__(self, model, scales=[0.75, 1.0, 1.25], use_flip=True):
        self.model = model
        self.scales = scales
        self.use_flip = use_flip
    
    def __call__(self, image):
        h, w = image.shape[2:]
        final_pred = torch.zeros(1, self.model.n_classes, h, w).cuda()
        count = 0
        
        for scale in self.scales:
            # 缩放
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = F.interpolate(image, size=(new_h, new_w), 
                                      mode='bilinear', align_corners=False)
            
            # 原始
            pred = self._predict(scaled_img, h, w)
            final_pred += pred
            count += 1
            
            if self.use_flip:
                # 水平翻转
                pred_hflip = self._predict(torch.flip(scaled_img, dims=[3]), h, w)
                pred_hflip = torch.flip(pred_hflip, dims=[3])
                final_pred += pred_hflip
                count += 1
        
        final_pred /= count
        return final_pred
    
    def _predict(self, image, target_h, target_w):
        with torch.no_grad():
            pred = self.model(image)
            if isinstance(pred, tuple):
                pred = pred[0]
        pred = F.interpolate(pred, size=(target_h, target_w), 
                            mode='bilinear', align_corners=False)
        return pred
```

**使用方法**：
```python
# 在评估脚本中
tta = CombinedTTA(model, scales=[0.75, 1.0, 1.25], use_flip=True)

for im, lb in val_loader:
    pred = tta(im.cuda())
    # ... 计算指标 ...
```

---


### 🏅 优先级 5: 知识蒸馏（预期提升 2-3%）

#### 5.1 特征蒸馏
```python
# lib/distillation.py
class FeatureDistillation(nn.Module):
    """特征层蒸馏"""
    def __init__(self, teacher_channels, student_channels):
        super().__init__()
        # 通道对齐
        self.align = nn.Conv2d(student_channels, teacher_channels, 1)
    
    def forward(self, student_feat, teacher_feat):
        student_feat = self.align(student_feat)
        
        # L2 距离
        loss = F.mse_loss(student_feat, teacher_feat.detach())
        return loss

class KnowledgeDistillationLoss(nn.Module):
    """完整的知识蒸馏损失"""
    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
    
    def forward(self, student_logits, teacher_logits, targets):
        # Hard Label Loss
        loss_ce = self.ce_loss(student_logits, targets)
        
        # Soft Label Loss (KD)
        loss_kd = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 组合
        total_loss = self.alpha * loss_ce + (1 - self.alpha) * loss_kd
        return total_loss
```

#### 5.2 自蒸馏 (Self-Distillation)
```python
class SelfDistillation(nn.Module):
    """自蒸馏 - 使用深层特征指导浅层"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.kd_loss = KnowledgeDistillationLoss(alpha=0.7, temperature=3.0)
    
    def forward(self, x, targets):
        # 获取多层输出
        logits, aux_high, aux_mid = self.model(x)
        
        # 主损失
        loss_main = F.cross_entropy(logits, targets, ignore_index=255)
        
        # 自蒸馏：深层指导浅层
        loss_kd1 = self.kd_loss(aux_mid, logits.detach(), targets)
        loss_kd2 = self.kd_loss(aux_high, logits.detach(), targets)
        
        total_loss = loss_main + 0.3 * loss_kd1 + 0.3 * loss_kd2
        return total_loss
```

**使用方法**：
```python
# 训练脚本中
# 1. 加载教师模型
teacher_model = load_pretrained_model('teacher.pth')
teacher_model.eval()

# 2. 训练学生模型
kd_loss = KnowledgeDistillationLoss(alpha=0.5, temperature=4.0)

for im, lb in train_loader:
    # 教师预测
    with torch.no_grad():
        teacher_logits = teacher_model(im)
    
    # 学生预测
    student_logits = student_model(im)
    
    # 蒸馏损失
    loss = kd_loss(student_logits, teacher_logits, lb)
    loss.backward()
    optim.step()
```

---


### 🎨 优先级 6: 优化器和学习率策略（预期提升 1-2%）

#### 6.1 使用 AdamW 替代 SGD
```python
# 在 train_amp.py 中
def set_optimizer_adamw(model):
    # 分组参数
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, 'weight_decay': cfg.weight_decay},
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10, 'weight_decay': cfg.weight_decay},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
        ]
    else:
        params_list = model.parameters()
    
    # AdamW
    optim = torch.optim.AdamW(
        params_list,
        lr=cfg.lr_start,
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay,
    )
    return optim
```

#### 6.2 OneCycleLR 学习率策略
```python
from torch.optim.lr_scheduler import OneCycleLR

# 在训练脚本中
optim = set_optimizer_adamw(net)
scheduler = OneCycleLR(
    optim,
    max_lr=cfg.lr_start,
    total_steps=cfg.max_iter,
    pct_start=0.3,  # 30% 用于 warmup
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=10000.0
)

# 训练循环中
for it, (im, lb) in enumerate(dl):
    # ... 训练代码 ...
    scheduler.step()
```

#### 6.3 Layer-wise Learning Rate Decay (LLRD)
```python
def get_layer_wise_lr_params(model, lr, decay_rate=0.95):
    """为不同层设置不同学习率"""
    params = []
    
    # Backbone 层
    backbone_layers = []
    for name, param in model.backbone.named_parameters():
        backbone_layers.append((name, param))
    
    # 反向遍历，越深层学习率越大
    num_layers = len(backbone_layers)
    for i, (name, param) in enumerate(backbone_layers):
        layer_lr = lr * (decay_rate ** (num_layers - i - 1))
        params.append({'params': param, 'lr': layer_lr})
    
    # Head 层使用最大学习率
    for name, param in model.named_parameters():
        if 'backbone' not in name:
            params.append({'params': param, 'lr': lr})
    
    return params

# 使用
params = get_layer_wise_lr_params(model, lr=1e-3, decay_rate=0.9)
optim = torch.optim.AdamW(params, weight_decay=1e-4)
```

#### 6.4 Lookahead Optimizer
```python
# lib/lookahead.py
class Lookahead(torch.optim.Optimizer):
    """Lookahead 优化器包装器"""
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        
        for group in self.param_groups:
            group["counter"] = 0
    
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)
    
    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

# 使用
base_optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
optim = Lookahead(base_optim, k=5, alpha=0.5)
```

---


### 🔬 优先级 7: 在线难例挖掘增强（预期提升 1-2%）

#### 7.1 动态 OHEM
```python
# lib/dynamic_ohem.py
class DynamicOHEM(nn.Module):
    """动态调整 OHEM 阈值"""
    def __init__(self, initial_thresh=0.7, min_thresh=0.5, max_thresh=0.9):
        super().__init__()
        self.thresh = initial_thresh
        self.min_thresh = min_thresh
        self.max_thresh = max_thresh
        self.ce = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    
    def forward(self, logits, labels, epoch, max_epochs):
        # 动态调整阈值：训练初期宽松，后期严格
        progress = epoch / max_epochs
        self.thresh = self.min_thresh + (self.max_thresh - self.min_thresh) * progress
        
        thresh_value = -torch.log(torch.tensor(self.thresh)).cuda()
        
        n_min = labels[labels != 255].numel() // 16
        loss = self.ce(logits, labels).view(-1)
        loss_hard = loss[loss > thresh_value]
        
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        
        return torch.mean(loss_hard)
```

#### 7.2 Focal Loss with Adaptive Gamma
```python
class AdaptiveFocalLoss(nn.Module):
    """自适应 Gamma 的 Focal Loss"""
    def __init__(self, alpha=0.25, gamma_init=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=255)
        p_t = torch.exp(-ce_loss)
        
        # 使用可学习的 gamma
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        return focal_loss.mean()
```

#### 7.3 难例样本重采样
```python
# lib/data/hard_example_sampler.py
class HardExampleSampler:
    """基于损失的难例采样"""
    def __init__(self, dataset, model, top_k=0.3):
        self.dataset = dataset
        self.model = model
        self.top_k = top_k
        self.sample_losses = []
    
    def update_losses(self, dataloader):
        """计算每个样本的损失"""
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for im, lb, idx in dataloader:
                im, lb = im.cuda(), lb.cuda()
                pred = self.model(im)
                loss = F.cross_entropy(pred, lb, reduction='none')
                loss = loss.view(loss.size(0), -1).mean(dim=1)
                losses.extend(loss.cpu().numpy())
        
        self.sample_losses = np.array(losses)
        self.model.train()
    
    def get_hard_samples(self):
        """返回难例索引"""
        k = int(len(self.sample_losses) * self.top_k)
        hard_indices = np.argsort(self.sample_losses)[-k:]
        return hard_indices
```

---


### 📊 优先级 8: 评估和监控改进

#### 8.1 更全面的评估指标
```python
# lib/metrics.py
class SegmentationMetrics:
    """全面的分割评估指标"""
    def __init__(self, n_classes, ignore_index=255):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.reset()
    
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
    
    def update(self, pred, target):
        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]
        
        for t, p in zip(target.flatten(), pred.flatten()):
            self.confusion_matrix[t, p] += 1
    
    def get_metrics(self):
        # IoU
        intersection = np.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(axis=1) + \
                self.confusion_matrix.sum(axis=0) - intersection
        iou = intersection / (union + 1e-10)
        
        # Dice
        dice = 2 * intersection / (self.confusion_matrix.sum(axis=1) + 
                                   self.confusion_matrix.sum(axis=0) + 1e-10)
        
        # Pixel Accuracy
        pixel_acc = intersection.sum() / self.confusion_matrix.sum()
        
        # Mean Accuracy
        acc_per_class = intersection / (self.confusion_matrix.sum(axis=1) + 1e-10)
        mean_acc = acc_per_class.mean()
        
        # Frequency Weighted IoU
        freq = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum()
        fwiou = (freq[freq > 0] * iou[freq > 0]).sum()
        
        return {
            'mIoU': iou.mean(),
            'IoU_per_class': iou,
            'mDice': dice.mean(),
            'Dice_per_class': dice,
            'Pixel_Acc': pixel_acc,
            'Mean_Acc': mean_acc,
            'FWIoU': fwiou
        }
```

#### 8.2 边界 IoU (Boundary IoU)
```python
def compute_boundary_iou(pred, target, dilation=5):
    """计算边界区域的 IoU"""
    from scipy.ndimage import binary_dilation
    
    # 提取边界
    pred_boundary = binary_dilation(pred) ^ pred
    target_boundary = binary_dilation(target) ^ target
    
    # 扩展边界区域
    pred_boundary = binary_dilation(pred_boundary, iterations=dilation)
    target_boundary = binary_dilation(target_boundary, iterations=dilation)
    
    # 计算边界 IoU
    intersection = (pred_boundary & target_boundary).sum()
    union = (pred_boundary | target_boundary).sum()
    
    return intersection / (union + 1e-10)
```

#### 8.3 Wandb 集成
```python
# tools/train_with_wandb.py
import wandb

# 初始化
wandb.init(
    project="semantic-segmentation",
    config={
        "model": cfg.model_type,
        "dataset": cfg.dataset,
        "lr": cfg.lr_start,
        "batch_size": cfg.ims_per_gpu,
        "epochs": cfg.max_epochs
    }
)

# 训练循环中
for it, (im, lb) in enumerate(dl):
    # ... 训练代码 ...
    
    if (it + 1) % 100 == 0:
        wandb.log({
            "train/loss": loss.item(),
            "train/loss_ce": loss_pre.item(),
            "train/lr": lr,
            "train/epoch": epoch
        })

# 评估时
metrics = eval_model(cfg, net)
wandb.log({
    "val/mIoU": metrics['mIoU'],
    "val/Pixel_Acc": metrics['Pixel_Acc'],
    "val/mDice": metrics['mDice']
})

# 保存最佳模型
if metrics['mIoU'] > best_miou:
    wandb.save('best_model.pth')
```

---


## 🎯 实施路线图

### 阶段 1: 快速提升（1-2 周）
**预期提升: +5-8% mIoU**

1. ✅ 添加 GridMask 数据增强
2. ✅ 使用组合损失 (OHEM + Dice + Lovasz)
3. ✅ 实施 EMA
4. ✅ 切换到 AdamW + OneCycleLR

**实施步骤**：
```bash
# 1. 更新数据增强
cp lib/data/gridmask.py lib/data/
# 修改 transform_cv2.py 添加 GridMask

# 2. 更新损失函数
cp lib/combined_loss.py lib/
# 修改 train_amp.py 使用新损失

# 3. 添加 EMA
cp lib/ema.py lib/
# 修改 train_amp.py 集成 EMA

# 4. 切换优化器
# 修改 train_amp.py 中的 set_optimizer 函数
```

### 阶段 2: 进阶优化（2-3 周）
**预期提升: +3-5% mIoU**

1. ✅ 实施 CutMix/Mosaic
2. ✅ 添加边界损失
3. ✅ 实施 TTA
4. ✅ 添加 SWA

### 阶段 3: 高级技巧（3-4 周）
**预期提升: +2-4% mIoU**

1. ✅ 知识蒸馏
2. ✅ 难例重采样
3. ✅ Layer-wise LR
4. ✅ 完善监控系统

---

## 📈 预期性能提升总结

| 优化项 | 难度 | 时间 | 预期提升 | 优先级 |
|--------|------|------|----------|--------|
| GridMask | ⭐ | 1天 | +1-2% | 🥇 |
| 组合损失 | ⭐⭐ | 2天 | +2-3% | 🥇 |
| EMA | ⭐ | 1天 | +0.5-1% | 🥇 |
| AdamW + OneCycle | ⭐ | 1天 | +1-2% | 🥇 |
| CutMix/Mosaic | ⭐⭐ | 2天 | +1-2% | 🥈 |
| 边界损失 | ⭐⭐⭐ | 3天 | +1-2% | 🥈 |
| TTA | ⭐⭐ | 2天 | +1-2% | 🥈 |
| SWA | ⭐ | 1天 | +0.5-1% | 🥈 |
| 知识蒸馏 | ⭐⭐⭐⭐ | 5天 | +2-3% | 🥉 |
| 难例重采样 | ⭐⭐⭐ | 3天 | +1-2% | 🥉 |
| **总计** | - | **3-4周** | **+12-20%** | - |

---

## 🔧 配置文件示例

### 优化后的配置
```python
# configs/optimized_config.py
cfg = dict(
    model_type='fastefficientformerseg_efficientnetv2_b3',
    n_cats=9,
    num_aux_heads=2,
    
    # 优化器
    optimizer='adamw',  # 改用 AdamW
    lr_start=0.001,
    weight_decay=1e-4,
    
    # 学习率策略
    lr_scheduler='onecycle',  # OneCycleLR
    max_epochs=300,
    warmup_epochs=30,
    
    # 数据增强
    use_gridmask=True,
    use_cutmix=True,
    cutmix_prob=0.5,
    use_mosaic=False,  # 可选
    
    # 损失函数
    loss_type='combined',  # OHEM + Dice + Lovasz + Boundary
    loss_weights=[1.5, 1.0, 1.0, 0.5],
    
    # 训练技巧
    use_ema=True,
    ema_decay=0.9999,
    use_swa=True,
    swa_start_epoch=270,
    gradient_accumulation=4,
    
    # TTA
    use_tta=True,
    tta_scales=[0.75, 1.0, 1.25],
    tta_flip=True,
    
    # 数据集
    dataset='BlueFaceDataset',
    im_root='../../BlueFaceDataX2',
    train_im_anns='../../BlueFaceDataX2/train.txt',
    val_im_anns='../../BlueFaceDataX2/val.txt',
    scales=[0.75, 1.5],  # 扩大范围
    cropsize=[640, 640],  # 增大尺寸
    ims_per_gpu=16,  # 配合梯度累积
    eval_ims_per_gpu=1,
    
    # 其他
    use_fp16=True,
    use_sync_bn=True,
    respth='./res_optimized',
)
```

---

## 💡 额外建议

### 1. 数据质量
- 检查标注质量，修正错误标注
- 增加数据多样性（不同光照、角度）
- 考虑使用伪标签（Pseudo Labeling）

### 2. 模型集成
```python
# 训练多个模型并集成
models = [model1, model2, model3]
predictions = []

for model in models:
    pred = model(image)
    predictions.append(pred)

# 投票或平均
final_pred = torch.stack(predictions).mean(dim=0)
```

### 3. 后处理
```python
# CRF 后处理
from pydensecrf import densecrf

def crf_postprocess(image, pred, n_classes):
    d = densecrf.DenseCRF2D(image.shape[1], image.shape[0], n_classes)
    
    # Unary potential
    U = -np.log(pred + 1e-10)
    U = U.reshape((n_classes, -1))
    d.setUnaryEnergy(U)
    
    # Pairwise potentials
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
    
    Q = d.inference(5)
    return np.array(Q).reshape((n_classes, image.shape[0], image.shape[1]))
```

### 4. 持续监控
- 使用 Wandb/TensorBoard 实时监控
- 定期可视化预测结果
- 分析错误案例，针对性改进

---

## 📚 参考资源

### 论文
1. **Mixup**: Zhang et al., "mixup: Beyond Empirical Risk Minimization"
2. **CutMix**: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers"
3. **Lovasz Loss**: Berman et al., "The Lovász-Softmax loss"
4. **EMA**: Polyak & Juditsky, "Acceleration of Stochastic Approximation"
5. **Knowledge Distillation**: Hinton et al., "Distilling the Knowledge in a Neural Network"

### 代码库
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [Albumentations](https://github.com/albumentations-team/albumentations)

---

## ✅ 检查清单

在实施优化前，请确认：

- [ ] 已备份当前代码和模型
- [ ] 已准备好验证集进行对比
- [ ] 已设置好实验跟踪（Wandb/TensorBoard）
- [ ] 已了解每个优化的原理
- [ ] 已准备好充足的计算资源
- [ ] 已设置好定期保存检查点

---

**最后更新**: 2026-02-06  
**作者**: AI Assistant  
**版本**: v1.0

祝你训练顺利，性能大幅提升！🚀

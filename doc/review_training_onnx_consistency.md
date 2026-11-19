# 语义分割工程源码审阅与整改建议

审阅日期：2026-11-18  
审阅范围：训练主脚本、ONNX 导出与样例推理、BlueFace 数据读取和增强、模型输出、损失函数、训练期评估及相关配置。按要求跳过 `tools/demov3.py`。本次只读审阅并新增本报告，没有修改既有代码。工作区原有未提交改动保留。

## 1. 摘要

最需要优先处理的不是“多卡 sampler 每个 epoch 漏掉样本”，而是以下几项：

1. **LogCoshDice 路径对 `ignore_index=255` 不安全。** `DiceLoss` 对标签直接 `one_hot(..., n_classes)`，只要 batch 中存在 255 就会越界。包装类也没有把构造参数传给内部 OHEM，`thresh`、`lb_ignore` 参数形同虚设。
2. **Focal Tversky 的平滑项量纲不匹配。** TP/FP/FN 先除以有效像素数，再与默认 `smooth=1.0` 相加，使平滑项通常远大于归一化计数，Tversky 接近常数 1，Tversky 梯度可能被严重压小。它不应被视为已验证的“小类增强”损失。
3. **多卡 epoch 放大不是样本覆盖所必需。** 当前 sampler 通过补齐索引使各 rank 获得相同数量的完整 batch，并使每个原始样本至少在每 epoch 出现一次；代价是少量重复样本。再将 epoch 数乘以卡数，会增加数据曝光量并改变训练预算和调度，不是修复漏训的正确手段。
4. **ONNX 导出和训练比较必须锁定同一模型、权重、预处理、空间尺寸和输出定义。** 现有导出脚本有另一套模型/权重默认值、只把 batch 轴设为动态、默认通过模型 `pred` 分支导出 argmax 类别图；样例 `onnxrun.py` 用 PIL resize 和近似均值方差，与训练端 OpenCV resize 及精确归一化不完全一致。若比较的是 logits、标签图或不同 resize 路径，结果不能直接等同。
5. **训练期 mIoU 有口径风险。** 指标实现给每类 IoU 的分母加 1，验证集上缺席类别会被计为 0 并进入均值；分布式验证 sampler 还会为整除长度补样本。报告的 mIoU 因而可能偏离标准“仅对验证集中出现的类别取均值”口径。

## 2. 工程调用脉络

### 2.1 训练链路

`tools/clip_m_trainv2_blackwell_v2.py` 加载配置（默认 `configs/fastefficientbisenet_blueface_mobilenetv5.py`），经 `lib.data.get_data_loader.get_data_loader()` 构造训练数据；配置中的数据集为 `BlueFaceDataset`，它继承 `BaseDataset`。`BaseDataset` 从 annotation 文本读图像/标签路径，OpenCV 读入 BGR 后转 RGB，灰度读取标签；训练变换由 `TransformationTrain` 组成，最后由 `ToTensor` 转为 CHW float、除以 255、按 BlueFace mean/std 归一化。模型按配置工厂创建，训练输出主头及两个辅助头；训练脚本对主头和辅助头分别计算损失，AMP、梯度裁剪后更新 AdamW，之后每个 epoch 调用 `eval_model` 并按 mIoU 保存最佳权重。

### 2.2 导出和推理链路

`tools/export_onnx.py` 按自身 CLI 配置构造模型、加载 state dict、切换 eval、调用 `reparameterize_model`，然后用 512×512 dummy input 导出。默认 `aux_mode='pred'`。本工程的 `fastefficientbisenet_mobilenetv5` 在此模式下输出 `argmax(logits, dim=1)` 的 float 类别图，而不是 `[N,C,H,W]` logits。`tools/onnxrun.py` 读 RGB 图片、缩放和归一化后执行 ONNX，并将输出当作类 ID 上色。

### 2.3 重要边界

训练数据、标注文件及实际训练权重位于工程外部，本次无法直接检查真实 label 值、annotation 是否完整、训练日志、GPU 数量/启动命令或某个 ONNX 文件。因此“是否有 255 标签”“模型与权重是否实际配对”“具体误差大小”均需按本报告的验证流程在目标环境确认；下文对源码可直接证明的问题与条件性风险分别标记。

## 3. 训练与 ONNX 不一致排查

### 3.1 已确认的对齐点

- 训练图像读取为 RGB；BlueFace mean/std 顺序也是 RGB。训练预处理是 `uint8 -> float32 / 255 -> (x - mean) / std`。
- `tools/onnx_segmentation_utils.py` 的 `preprocess_image()` 也读 BGR 后转 RGB，使用同一 mean/std 和相同的归一化表达式。对“固定尺寸 resize 后”的图像而言，该辅助工具的通道次序/归一化和训练定义一致。
- 训练标签 resize 使用 nearest neighbor；ONNX 工具 `load_resized_label()` 也使用 `INTER_NEAREST`，这是类别标签应采用的插值方式。
- 目标模型主头将输出上采样到输入的 1 倍尺寸，`pred` 模式执行 argmax；只要导出时使用相同模型与权重，标签图的语义输出定义是明确的。

### 3.2 已确认的不一致或高风险点

| 优先级 | 位置 | 发现与影响 |
|---|---|---|
| P1 | `tools/export_onnx.py:17-26`、`configs/fastefficientbisenet_blueface_mobilenetv5.py` | 导出脚本默认配置是 `fastefficientbisenet_blueface_inceptionnext_atto_sppf`，默认权重路径也是该模型命名；训练脚本默认配置则是 MobileNetV5。两组默认值彼此可能成对，但**不是训练脚本默认训练对象**。若直接拿主训练产物却忘记显式传 config/weight，模型结构可能不匹配。`load_state_dict(..., strict=False)` 又不检查 missing/unexpected keys，容易在未明显失败时导出部分随机初始化的模型。导出时应严格校验模型/权重配对并打印参数匹配报告。 |
| P1 | `tools/export_onnx.py:41-57` | 导出 dummy input 固定 512×512，dynamic axes 只包含 batch 轴；ONNX 的空间尺寸固定。模型配置 cropsize 是 512×512，因此固定 512 导出与当前配置相符，但不能据此推断支持任意长宽。模型内部按 stride 8/16/32 的 `scale_factor` 上采样，非合适倍数的尺寸可能造成输出尺寸不等于输入；导出前应明确输入尺寸契约，或改成按目标尺寸插值并验证动态空间轴。 |
| P1 | `tools/export_onnx.py:25-26`、`lib/models/fastefficientbisenet_mobilenetv5.py:168-174` | `aux_mode='pred'` 导出的是 argmax 后的类别图，不是 logits。该路径只适用于最终离散 mask 部署，不适合和 PyTorch logits 做逐元素差分、置信度后处理或概率融合。训练脚本每轮 eval 使用 `aux_mode='eval'` 的 logits，再 softmax/插值/argmax。对齐测试必须在 PyTorch `eval` logits 与 ONNX logits 之间进行；要验证 pred ONNX，则 PyTorch 端也执行同样的 argmax，并接受浮点精度下边界像素的 argmax 敏感性。 |
| P1 | `tools/onnxrun.py:35-41` 与 `lib/data/base_dataset.py:52-55`、`lib/data/transform_cv2.py:33-34` | 样例推理用 PIL 的 `Image.resize`；训练用 OpenCV `cv2.resize` 默认线性插值。插值核、坐标取样/舍入可能不同，物体边缘像素会改变。样例使用 `(120,114,104)/(70,69,73)`，与训练均值/方差乘 255 后接近但不完全相同；小偏移会改变边界类别。`tools/onnx_segmentation_utils.py` 是更一致的实现（OpenCV INTER_LINEAR + 精确 mean/std），应统一推理入口，不保留两套预处理实现。 |
| P2 | `tools/export_onnx.py:43` | dummy 输入采样值为 0..255 的 float，而真实训练输入是归一化后大致落在数个标准差内的 float。dummy 仅用于形状追踪时通常不改变静态图，但它不是有效样例输入，不能拿导出 smoke test 的数值输出来判断精度。建议使用归一化范围内 dummy，并增加真实样本数值对齐测试。 |
| P2 | `tools/onnxrun.py:44-47` | `onnxrun.py` 将 batch 的第 0 项当成二维 class-id map。对当前 MobileNetV5 `pred` 输出成立；若换成 `eval` 输出 logits，则这里缺少 `argmax(axis=1)`，索引 palette 会错误。输出契约应作为模型导出元数据/命令行明确区分。 |
| P2 | `tools/evaluate.py:58-60`、`168-182`、`lib/models/fastefficientbisenet_mobilenetv5.py:69,93` | 模型内插值和 eval 额外 resize 均为 bilinear `align_corners=False`，但 eval 对输入尺度和 logits 回原图使用 `align_corners=True`。当前单尺度 1.0 下，`eval_model` 外部输入尺度不变、输出也已是全尺寸，外插值是否造成变化取决于实现路径；多尺度/裁剪评估启用时会明确引入与常见训练/推理对齐约定不同的坐标网格。ONNX 直接输出 full-resolution 预测，不包含 eval 的多尺度融合。应统一单尺度基准或把同一后处理显式用于两端。 |
| P2 | `tools/export_onnx.py:34` | `strict=False` 没有检查 missing/unexpected keys。即使当前 checkpoint 确实匹配，也应使用严格加载或对不匹配键做白名单校验；否则错误权重会掩盖 ONNX 与训练结果不一致的根因。 |

### 3.3 建议的数值一致性验收

1. 固定一组真实验证图像，保存训练 dataloader 产出的归一化 tensor，直接把**同一个 tensor**送到 PyTorch eval 模型与 ONNX Runtime，先对比 logits（最大绝对误差、均方误差、逐像素 top-1 一致率），排除预处理变量。
2. 再分别从原图走统一预处理函数，然后比较两端；若第 1 步一致而第 2 步不一致，问题在读图、通道、resize、归一化或标签尺寸处理。
3. `aux_mode='eval'` 导出 logits 用于诊断和精度评估；部署如需类别图再单独导出 `pred`，验收时比较 argmax mask 及逐类 confusion matrix。保存输入尺寸、模型类型、权重哈希、归一化常量、resize 插值和 ONNX opset 到导出记录。
4. 先验证 eager PyTorch 与 ONNX 未简化版本，再验证简化版；逐阶段比较，防止优化器/简化过程带来差异。若启用 FP16/INT8，另设精度预算和逐类指标门槛，不与 FP32 等价假设混用。

## 4. 多卡采样、漏训与吞吐方案

### 4.1 当前代码真实行为

`get_data_loader.py` 对训练集令 `n_train_imgs = ceil(N / (B*W)) * B*W`，其中 `N` 为原始样本数、`B=ims_per_gpu`、`W=world_size`。`RepeatedDistSampler` 构造确定性 shuffle 序列并循环拼接，截断至该补齐长度，然后用 `indices[rank:total_size:world_size]` 切片。因此每个 rank 获得相同样本数，且该数量可被本地 batch size 整除；原始数据集首轮排列完整，余数由重复样本补齐。**在已初始化 DDP 的训练路径，单个 epoch 不会因 `drop_last=True` 丢掉原始集合尾部样本；会有最多 `B*W-1` 个补齐重复项。** `set_epoch(epoch)` 也会使下一 epoch 改变 shuffle。注意不启动分布式的单卡 DataLoader 直接使用 `drop_last=True`，会丢掉 `N % B` 个尾样本（若余数非零）；建议同样计入覆盖审计。

训练脚本默认 `scale_epochs=True`，多卡时 `max_epochs=cfg.max_epochs*W`。其历史动机可以解释为：每个 DDP step 的本地 batch 与单卡相同，但每 epoch 的 step 数大约缩至 `1/W`，乘 epoch 是为了对齐 optimizer step 数。然而这会使全局样本曝光和总训练计算量明显增加；每轮 sampler 已经覆盖全体原始数据，乘 epoch 并非防漏训所需。它也让 cosine schedule、warmup 和训练时长随 W 变化。

另一个准确性点：为确保全局 batch 整除而补齐，会重复样本；若 N 小于全局 batch，则一个 epoch 可能对样本重复多次。分布式验证使用标准 `DistributedSampler`，它同样可能为整除 rank 数而重复验证样本，稍微改变混淆矩阵权重。

### 4.2 推荐落地方案

采用 PyTorch DistributedDataParallel（当前工程已具备）和显式、可审计的 epoch sampler：

1. 每个 epoch 用同一个 seed+epoch 生成长度 N 的全局随机排列，然后按 rank 切 shard。保证各 rank 的 step 数严格相同：优先使用标准 `DistributedSampler(drop_last=False)` 语义（最多补齐 W-1 个副本），再在 sampler/batch sampler 层只补齐“每 rank 最后一批”的缺额；记录补齐索引数量与样本 ID。这样每个 epoch 所有原始样本至少出现一次，不因 DataLoader `drop_last` 被静默丢弃。若业务要求严格零重复，则采用跨 epoch remainder carry-over 或带有效样本 mask 的等步数 batch，不能简单让 rank 步数不等，否则 DDP collective 会挂起。
2. **关闭 epoch 乘 world size**：`max_epochs=cfg.max_epochs`，按同样 epoch 定义训练。这样数据曝光预算不随卡数膨胀。日志同时输出原始 N、每 rank 样本数、补齐数、全局 batch 和每 epoch optimizer steps，可自动检查覆盖与重复。
3. 固定 `ims_per_gpu` 时，全局 batch 为 `B*W`，每 epoch 更新次数下降；为达到吞吐扩展，保持 epoch 数不变并采用经验证的 LR scaling / warmup。当前代码已经按 AdamW 使用 sqrt scaling，但这是经验规则，不保证所有任务稳定；至少比较固定 LR、线性 scaling、sqrt scaling 三条短跑曲线。按全局 batch 重新设定调度的单位应明确为 optimizer step 或 epoch，不能在两个口径间混用。
4. 若必须保持与单卡相同的**有效全局 batch 和更新次数**，用每卡较小 batch + gradient accumulation 配合 DDP，并正确缩放/同步 BN；这能保持优化统计但不会获得理想线性加速。若目标是固定训练数据曝光下缩短墙钟时间，则采用第 2、3 步的固定 epoch 方案，接受更大的全局 batch/更少更新，并用验证集确认收敛质量。二者是不同目标，不能同时无代价满足。
5. 通过 `torchrun --standalone --nproc_per_node=W` 启动，每个进程独占 `LOCAL_RANK` 对应 GPU；不要在 Python 脚本内硬编码 `CUDA_VISIBLE_DEVICES`，由启动环境决定可见设备。DDP 参数同步、`DistributedSampler.set_epoch`、每 rank 独立随机增强、验证集去重/精确 gather、仅 rank 0 写 checkpoint 和汇总日志。
6. 用单卡与 W 卡基准实测 images/s、GPU 利用率、DataLoader wait、all-reduce 时间和验证开销，报告 speedup=`吞吐(W)/吞吐(1)` 及效率=`speedup/W`。主训练脚本每 epoch 全量验证会占固定墙钟开销，短 epoch 场景下应单独量化；数据解码/存储若成为瓶颈，增加 worker、缓存或调整存储布局。

### 4.3 当前多卡实现的附加风险

- `CUDA_VISIBLE_DEVICES="0,1,2,3"` 在脚本导入 torch 后硬编码，覆盖外部 GPU 选择意图，也容易造成 `LOCAL_RANK` 与物理 GPU 映射困惑；应由启动命令控制。
- `eval_model` 每个 rank 都跑验证分片并 all-reduce，但每个 rank 仍会执行汇总日志；保存 checkpoint 被 rank 0 限制，基本正确。TensorBoard writer 在各进程启动时创建，建议仅 rank 0 创建和写入，避免重复事件/文件混乱。
- 验证 `DistributedSampler` 默认补齐可能重复样本。精确指标可 rank 0 单卡验证，或使用不补齐 sampler 并按样本 ID gather/去重；若多 rank 验证需保证每条验证样本计数恰好一次。
- 训练里每步 `torch.cuda.synchronize()` 会强制主机等待 GPU，可能降低吞吐；计时/性能分析阶段才需要显式同步，普通训练可移除或只在计时窗口使用。

## 5. 损失函数审阅与 mIoU 现象解释

### 5.1 `LogCoshDiceOhemLovaszLoss` 实际组成

其 forward 为 `1.5 * (OHEM CE + 3 * LogCosh Dice) + Lovasz`。训练脚本当前 `loss_opt=8`，因此这正是当前默认实际使用的主损失；`loss_opt=9` 的 `FocalTverskyWithOhemCELoss` 只是可选分支，并未被当前脚本启用。构造参数 `thresh`、`lb_ignore` 没有传给内部 `LogCoshDiceLossWithOhemCELoss`；后者固定创建 `OhemCELoss(0.7)`。Dice 路径对所有类别、所有像素展平后算 micro Dice，且使用 one-hot，没有 ignore mask。Lovasz 使用 `classes='present'` 的默认逻辑，会跳过当前 batch 中 GT 不存在的类别，并过滤 ignore=255。由此它不是一个统一 ignore 语义、可配置的损失组合。

**确定性条件风险：** 若标签中实际含有 255，`DiceLoss` 的 `F.one_hot(targets, logits.shape[1])` 会因类别索引超出类别数而报错。`lb_ignore=255` 的 dataset 声明说明这是允许值，但是否真实出现在训练标注需扫描数据集。若标签绝不含 255，则该具体运行错误不会触发，但损失接口仍不一致。

### 5.2 `FocalTverskyWithOhemCELoss` 合理性和缺陷

合理之处：

- OHEM CE 聚焦高损失像素，能保留稳定的逐像素分类监督；Tversky 的 beta 大于 alpha 时增加 FN（漏检）惩罚，gamma>1 会强调较难类别/难样本区域；ignore label 在 Tversky flatten 阶段会被过滤。
- Tversky 与 IoU/Dice 家族目标相关，对不平衡分割有潜力，尤其当少数类漏检是主要业务成本时。

关键缺陷：

- `FocalTverskyLoss` 把 TP/FP/FN 全部除以 `n_valid`，之后 `smooth=1.0` 加到分子和分母。对于大图，归一化计数在 0 到 1 量级，smooth 则为 1，导致各类 Tversky 接近 1；`(1-T)^gamma` 和其梯度都可能很小。应在**计数空间**使用 smooth，或同步将 smooth 除以有效像素数/使用极小 epsilon，并验证梯度量级。当前注释称其“消除分辨率对量级影响”，却没有同步缩放平滑项，恰好可能破坏损失尺度。
- 对所有类别求均值，包括 GT 缺席类别；这会把“该图/批次中不存在的小类”的 false-positive 惩罚和 present 类同权混合。类别稀少时会改变目标，未必有利于召回。应按 present 类求均值或明确采用全类别 one-vs-rest 目标，并做 ablation。
- `alpha=0.3,beta=0.7` 是漏检优先而不是通用最优；`gamma=1.33`、两个分量权重均为 1，却没有基于梯度/损失量级校准。OHEM 和 Tversky 的量纲、梯度规模差异可能使某一项实质主导。
- OHEM 按像素难度筛选而非类别配额；小类即使重要，也可能因为数量少而在全局 batch 的 hard pixel 集合中占比低。Focal Tversky 的宏平均本意能缓解此点，但当前 smooth 问题可能削弱该分支。
- 当有效像素为空时，FT 分支返回零梯度，OHEM 分支依靠 topk 退化返回零；应保留显式全 ignore 测试。注意 Lovasz 的空张量路径及 Dice 路径应统一处理。

### 5.3 为什么旧损失的实测 mIoU 可能更高

这并不矛盾，也不能据此断言损失定义优劣。当前旧损失包含 OHEM CE 和 Lovasz-Softmax：前者提供稳定的像素级分类梯度，后者直接优化 IoU 的 Lovasz surrogate，并默认只平均当前标签中存在的类别；这与 mIoU 指标更接近。若 Tversky 的 smooth 量纲问题令其项几乎不产生梯度，新损失在实践中可能近似 OHEM CE，反而失去稀有类的 overlap 约束。即使修正 smooth，beta/gamma/权重、类别缺席策略、global batch、LR 和随机种子也会改变结果。

旧损失的 Dice 部分本身是 micro Dice，flatten 覆盖全部类别像素，受背景/大类像素影响，理论上并不天然“保护小类”；而 Lovasz present-class 平均与 hard CE 可能使整个组合的 mIoU 更好。若旧 Dice 未 mask ignore，会有额外风险。评价时应记录逐类 IoU、召回率、预测占比和损失分量/梯度范数，不只看一个 mIoU。

### 5.4 推荐损失实验

1. 先修正并单测 ignore mask、全 ignore、缺席类别、仅有单像素小类、不同输入分辨率的有限性与梯度。
2. 基线固定为现有 LogCosh+OHEM+Lovasz（但给 Dice 补 ignore mask，并把参数真实传递）；候选 A 为 OHEM CE + present-class macro Lovasz；候选 B 为 OHEM CE + 修正后 present-class Focal Tversky；候选 C 为 CE/Focal CE + class-balanced Dice/Tversky。逐项 ablation，不同时更改学习率、增强和 sampler。
3. 每个实验使用相同训练/验证划分、总样本曝光量、优化器、调度、随机种子集合（至少 3 个 seed），比较 mIoU、逐类 IoU、macro recall、稀有类 FN、稳定性及训练耗时。报告均值/方差，避免把单次波动归因于损失。
4. 对各 loss term 记录值和 backbone/head 梯度范数。对比两者量级后再定 `tversky_weight`/`ohem_weight`，不要仅凭 loss scalar 相等判断贡献相等。

## 6. 其他高风险/缺陷清单

| 优先级 | 文件/位置 | 风险 |
|---|---|---|
| P1 | `lib/ohem_ce_loss.py:85-99,358-373` | Dice one-hot 不处理 ignore；组合损失对外暴露的 `lb_ignore` 未贯穿所有子损失。先扫描真实 mask，确认是否触发越界。 |
| P1 | `lib/ohem_ce_loss.py:303-318` | Focal Tversky 归一化计数配 `smooth=1.0`，疑似损失/梯度退化；需做解析尺度检查和梯度单测。 |
| P1 | `tools/evaluate.py:92-96` | `IoU = tp/(tp+fp+fn+1)` 会将缺席类别 IoU 设为 0 并纳入 `nanmean`，使 mIoU 口径不标准且对小类/验证集类别组成敏感。应使用零分母策略并明确是否对 absent class 排除。 |
| P2 | `lib/data/get_dataloader.py:86-90` | 分布式验证使用默认 `DistributedSampler` 补齐，可能重复若干验证样本，轻微偏置 confusion matrix。 |
| P2 | `lib/data/transform_cv2.py:43-44` | RandomResizedCrop 若缩放后图像小于 crop，会以标签值 0 填充，等价于把 padding 监督成 background，而不是 ignore。配置 scales 最小 0.75，若源图尺寸不足 crop（512），边缘 padding 会制造背景标签。建议 image padding 用均值/合理值、label padding 用 ignore_index，并检查实际数据尺寸。 |
| P3 | `lib/data/transform_cv2.py:89-134` | 饱和度矩阵在代数上等价于常见的 saturation factor 与灰度混合，未发现确定性实现错误；训练增强固定按 brightness、contrast、saturation 顺序应用，且各自因子在 `[0.6,1.4]`。是否符合该数据域需要用样例可视化和消融确认，不应单凭实现判为 Bug。 |
| P2 | `tools/clip_m_trainv2_blackwell_v2.py:38`；`tools/evaluate.py:26` | 两脚本硬编码 `CUDA_VISIBLE_DEVICES`，由 Python 代码覆盖运行环境设备选择，且训练和评估映射不一致。统一由启动脚本控制。 |
| P2 | `tools/clip_m_trainv2_blackwell_v2.py:320` | 每个训练 step 调用 `torch.cuda.synchronize()`，会造成 CPU/GPU 同步，限制流水并降低吞吐，不应常驻性能训练路径。 |
| P2 | `lib/data/base_dataset.py:29-35,52-54` | annotation 用逗号无 maxsplit，路径自身含逗号会解析失败；图像读取没有检查 `imread` 返回 None，坏路径会以低信息量的索引异常崩溃。应增加输入清点/明确报错。 |
| P3 | `tools/clip_m_trainv2_blackwell_v2.py:341-342` | 每个 epoch `empty_cache()` 后紧接全量验证，会产生额外 allocator 开销；除非确认显存碎片/峰值需要，不建议把它当作加速措施。 |

## 7. 整改优先级与验收门槛

### P0：先确保比较可信

- 明确实际训练模型 config、checkpoint、`n_cats`、输入 H×W、输出是 logits 还是类别图；导出禁止静默部分加载。
- 用统一 OpenCV RGB 预处理函数固定真实图片；同 tensor 对比 PyTorch eval logits 与 ONNX logits。
- 导出后检查 ONNX 输入/输出名字、dtype、shape、输出语义；至少跑 10 张真实样本并输出全局和逐类一致性统计。

### P1：清理训练目标和指标

- 对 train/val 所有 label 扫描唯一值、类别像素计数、ignore 比例、坏图/尺寸，检查所有值都在 `[0,n_cats-1]` 或等于 ignore。
- 给 Dice/Tversky/Lovasz 同一 ignore 语义；修正 Tversky smooth 尺度后，跑分辨率缩放不变性与梯度测试。
- 明确 mIoU absent-class 规则，验证训练日志和独立 confusion-matrix 脚本得到同一结果。

### P1：恢复正确的分布式训练预算

- 保持 `max_epochs` 不随 world size 扩大；用覆盖审计确认各 epoch 的原始样本覆盖率为 100%，补齐重复数被记录且有界。
- 单卡与多卡统计每 epoch optimizer steps、实际读取样本数、累计样本曝光数、LR 曲线和总 wall time。
- 在 1/2/4 卡上至少跑短基准，确认不存在 DDP hang/重复写 checkpoint，并报告吞吐扩展效率。线性加速是测量目标，不是单靠 DDP 可保证的数学结论。

## 8. 结论边界

源码可以确认：DDP 训练 sampler 的设计是“补齐并重复”，不是简单漏掉数据；单卡 DataLoader 在 `N % B != 0` 时会因 `drop_last=True` 丢尾样本；训练 epoch 乘 world size 会增加曝光预算；当前启用的是 loss 8 而非 Focal Tversky loss 9；Focal Tversky 的 smooth 量纲问题成立；LogCosh Dice 对 255 ignore 有条件性越界；导出和推理存在必须对齐的配置、空间尺寸、插值和输出契约风险。

源码无法单独确认：当前训练标注是否真的含 255/缺类、具体 checkpoint 与 ONNX 是否配错、实际 ONNX 数值误差、当前 GPU 数量下吞吐是否线性、两种损失在控制变量实验下谁更优。上述事项应按 P0/P1 验收项运行数据扫描与数值对照后定论。

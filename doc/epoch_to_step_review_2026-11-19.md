# Epoch-to-Step 训练步数换算审阅报告

审阅对象：`tools/clip_m_trainv2_epoch_to_step.py` 及其训练 DataLoader、分布式 sampler 和本地 timm scheduler 调用。

审阅范围：判断 `total_steps = max_epochs * len(dataloader)` 是否正确，以及当前实现是否会使调度步数、实际训练更新次数或多卡行为失真。未修改任何代码，也未运行真实 GPU 训练。

## 结论

将 epoch 转换为计划步数的公式本身是合理的：在训练集、batch size、world size、sampler 长度和 `drop_last` 均固定，且每个 batch 都恰好执行一次优化器更新时，`len(train_loader)` 就是每个 epoch 每个 rank 的 batch 数，`max_epochs * len(train_loader)` 是预期的 batch/update 数。

但当前代码并没有严格保证“计划步数 = 实际 optimizer update 数”。脚本以 epoch 循环作为停止条件，`global_step` 实际计数的是已处理/跳过的 batch，而不是已成功更新参数的次数。因此，当前写法可用于静态训练配置下设定 scheduler 的大致训练周期，但遇到非有限 loss、GradScaler 跳过更新、恢复训练或动态改变 loader 时，调度位置会与实际参数更新错位。

## 关键链路

1. `steps_per_epoch = len(dl)`，随后计算 `total_steps = max_epochs * steps_per_epoch`（训练脚本约第 217-234 行）。DDP 下 `len(dl)` 是每个 rank 的本地 batch 数，不是所有 rank 的 batch 数之和；这是正确的，因为 DDP 的一次同步优化器更新由所有 rank 的本地 batch 共同组成，只应计为一个全局 update。
2. 训练循环仍为 `for epoch in range(max_epochs)`，每轮完整遍历 `dl`，并在 epoch 开始时调用 sampler 的 `set_epoch(epoch)`（约第 285-292 行）。所以在无异常且 loader 长度固定时，训练实际尝试的 batch 数与 `total_steps` 相符。
3. train DataLoader 使用 `drop_last=True`。DDP sampler 将样本数补齐到 `ims_per_gpu * world_size` 的整数倍，再在各 rank 间分片；这样各 rank 的 batch 数一致，但每轮可能有少量重复样本（`lib/data/get_dataloader.py` 约第 40-60、72-96 行）。因此“每轮步数”是补齐后 loader 的步数，不代表每个样本恰好一次且无重复。
4. scheduler 以 `t_initial=total_steps` 初始化；但代码随后用 `lr_schdr.step(global_step + 1)` 逐 batch 驱动（训练脚本约第 255-264、326-330 行）。该调用把 update 计数传给 epoch-oriented 的 `step()` 接口，接口语义不匹配。

## 发现的问题

### P1：非有限 loss 时，各 DDP rank 可能分叉

训练脚本只检查本 rank 的 `torch.isfinite(loss)`（约第 316-324 行）。若一个 rank 的 loss 为 NaN/Inf、其余 rank 正常，异常 rank 会跳过 `scaler.step(optim)`，而其他 rank 继续更新参数。各 rank 的模型参数和优化器状态可能因此不一致，后续 DDP 训练不再可信。DDP 下必须先跨 rank 汇总 skip 标记，确保所有 rank 对该更新作出同一决定；或者采用能保证所有 rank 一致处理的错误策略。

此外，异常分支递增 `global_step`，却不推进 scheduler；这又使 batch 计数和 scheduler 计数分离。

### P1：`global_step` 不是成功的 optimizer-step 计数

非有限 loss 分支会递增 `global_step`，但不会更新参数；正常分支中，`GradScaler.step(optim)` 也可能因反向传播得到 Inf/NaN 梯度而跳过 optimizer 更新。脚本没有检测 GradScaler 是否实际执行了 `optimizer.step()`，仍无条件调用 scheduler，并在后续增加 `global_step`（约第 312-330、356 行）。

结果是：

- `total_steps` 是计划 batch 数；
- `global_step` 是已处理 batch 的计数；
- scheduler 被调用的次数可能既不等于成功更新数，也可能因 skip 分支而少于 batch 数；
- 学习率可能已经走到训练后段，而参数实际更新次数还没有达到对应位置。

应分别维护 `batch_step` 和 `optimizer_step`。若目标是让余弦周期精确覆盖成功的参数更新，应以成功的 `optimizer_step` 推进 scheduler，并明确处理 AMP 跳步和 DDP 全局 skip 同步。

### P2：scheduler 方法与配置语义不匹配

本地 timm scheduler 的 `step(epoch)` 是 epoch 调度接口，`step_update(num_updates)` 才是按 optimizer update 调度的接口。当前 `CosineLRScheduler` 没有显式设为 update 模式，却通过 `step(global_step + 1)` 传入 batch/update 编号。该余弦调度实现下，现有参数组合可能仍产生看似合理的 LR 数值序列，但接口语义依赖实现细节，不宜视为严格正确；更换 scheduler 配置或 timm 版本时尤其容易出现偏差。

按步调度时应显式采用 `t_in_epochs=False`，并用 `step_update(optimizer_step)` 推进。需在实际安装版本上验证首步 LR、warmup 结束点、最后一步 LR 以及总调用次数。

### P2：warmup 下限可能大于总训练步数

`warmup_steps = max(5, int(0.05 * total_steps))`（约第 256 行）。当训练总步数小于 5 时，warmup 会长于训练周期，短任务可能始终处于 warmup 阶段。建议对 warmup 步数做上限约束，不超过有效总步数，并对 `total_steps <= 0` 做参数校验。

### P2：学习率下限可能高于初始学习率

代码设置 `scaled_lr_min = max(1e-6, scaled_lr * 0.01)`（约第 243 行）。当 `scaled_lr < 1e-6` 时，`lr_min` 会大于初始 LR，余弦下降的“下限”反而高于起点。应验证 `0 <= lr_min <= scaled_lr`，或采用不会违反此条件的计算规则。

### P2：训练终止仍由 epoch 控制，不能保证严格步数预算

即使 scheduler 配置了 `total_steps`，训练停止条件仍是固定 epoch 数，而不是成功更新次数。静态 loader 下二者等价；若未来出现梯度累积、动态采样、跳过 batch、训练中调整数据集/loader，或需要从 checkpoint 精确续训，`epoch × 初始 len(dl)` 就不能代表训练最终完成的成功更新数。

### P2：恢复训练不能续接 step-based 训练状态

脚本的 finetune 入口只恢复模型权重；当前最佳模型也只保存 `state_dict`。没有保存/恢复 optimizer、scheduler、GradScaler、epoch、batch/update 计数及随机状态。因此该机制适用于权重微调，不是严格的断点续训。中断后从该权重重新启动会重新开始 LR 曲线和优化器状态，不能宣称继续原训练轨迹。

### P2：多卡下“每轮总步数”的语义需明确

DDP 每个 rank 的本地 batch size 为 `ims_per_gpu`，全局 batch size 为 `ims_per_gpu * world_size`。固定 epoch 数时，每轮覆盖约一个数据集，但每轮同步 optimizer update 数随 GPU 数增多而减少；这通常使 wall-clock 训练更快，却不等价于单卡相同更新次数的优化轨迹。当前 `--scale-epochs-with-world-size` 会把 epoch 数乘以 world size，补回一部分总 update 数，同时也增加样本暴露和总训练工作量，不应与“固定 epoch、多卡加速”混为一谈。

`len(dl)` 在各 rank 上应相同，因此不能把它再乘 `world_size` 来算 DDP 的全局 update 数。否则会把 scheduler 周期错误地放大 world size 倍。

### P3：日志 step 的编号落后于当前更新

正常分支先用 `global_step + 1` 更新 scheduler，然后用旧的 `global_step` 写 TensorBoard，最后才递增计数（约第 329-356 行）。因此首次更新记为 step 0，而 LR 已按 step 1 计算。零起始编号本身可以接受，但日志中的 LR 与 step 含义不一致，排查曲线时容易误读。建议统一采用清晰定义的零基或一基 update 编号。

## 建议的可落地规则

1. 将训练预算定义为成功 optimizer update 数，而不是笼统称为“总步数”；单独记录 batch 尝试数和成功更新数。
2. 启动时计算计划预算：`planned_updates = epochs * len(train_loader)`。确认 `len(train_loader)` 在所有 rank 相同，并记录数据集大小、batch size、world size、sampler 类型、`drop_last` 与补齐策略。
3. DDP 中对异常 loss/梯度 skip 决策做跨 rank 同步，所有 rank 必须共同执行更新或共同跳过。否则不能继续训练或评估该 checkpoint。
4. 使用明确的 update-based scheduler 配置和调用接口；只在成功 optimizer update 后推进 scheduler。若选择按 batch 尝试推进，则必须将调度器总长度也定义为 batch 数，并接受跳步会改变有效更新预算。
5. 对 AMP 跳步建立可靠检测；不要仅凭 loss 有限就推断 optimizer 一定执行了更新。验证使用的 PyTorch/GradScaler 版本行为，并保证所有 DDP rank 的更新决策一致。
6. 对 warmup、`lr_min`、总步数做边界校验；输出计划步数和实际成功更新数，训练结束时比较二者并报告差额。
7. 若需要断点续训，checkpoint 同步保存模型、optimizer、scheduler、scaler、epoch、batch/update 计数及必要随机状态；若仅恢复模型权重，应明确标注为 finetune/restart。
8. 多卡性能用相同数据、相同全局样本处理预算进行基准测试，分别测量 images/sec、每轮耗时、验证开销和扩展效率。`epochs * len(dl)` 只修正调度器周期，不会自动带来线性加速；通信、同步 BN、数据加载和每 batch 的 `torch.cuda.synchronize()` 都会影响扩展效率。

## 结论分级

- **公式正确性**：有条件成立。静态配置下，它能正确表示计划 batch 数/计划更新数；DDP 下 `len(dl)` 是每 rank 的 batch 数，因此不应乘 world size。
- **当前实现正确性**：不严格。异常 loss 分支、AMP 梯度溢出跳步、scheduler 接口调用与训练终止/计数方式使“计划步数”不等于“成功 optimizer update 数”。
- **严重风险**：DDP rank 对非有限 loss 作出不同更新决定，可能导致模型参数分叉。
- **性能结论**：仅凭代码无法确认线性加速；该公式只设定调度周期，不等于加速方案或性能证明。
- **验证限制**：本次为源码审阅和本地 scheduler 行为核对，没有真实 GPU、多卡吞吐或完整训练实验，因此不对实际加速比作结论。

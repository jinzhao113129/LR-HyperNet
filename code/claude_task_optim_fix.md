## 任务：LR-HyperNet 优化器修复（让模型在 b8/30ep 下真正学得动）

## 背景
当前所有 b8 变体（v1/v3.1/欧氏对照）Train <40%，严重欠拟合，30ep 内看不出组件级差异。证据链：
- 5类 sanity 已到 89%（模型容量没问题）
- baseline b32/150ep 同架构 Train 99.86%（架构能拟合 101 类）
- 最大嫌疑：BatchNorm + b8（统计量噪声大）+ 长尾数据（train 3596 样本 / 101 类，每类 5-333）+ LR 沿用 b32 配方

## 修改要求（全部做成新 flag，默认关闭，不影响正在跑的 4 个作业）

1. `--group_norm`：把 models/FECNet.py 里所有 nn.BatchNorm1d 替换为 nn.GroupNorm。注意 C 可能不是 32 的倍数，用 `num_groups=min(32, C)` 或按 C 动态取最大公约数。需要同时作用于 backbone（FECNet/Linear1Layer）和 LRHyperNet 内部的 BN。
2. `--grad_accum N`：梯度累积，等效大 batch。默认 N=4（等效 b32）。训练循环里 loss 除以 N，每 N 步 optimizer.step() 一次。
3. `--balanced_sampler`：WeightedRandomSampler，每个样本权重 = 1/n_c（n_c 为该样本类别在训练集的样本数）。注意与 drop_last 的交互：WeightedRandomSampler 的 num_samples 应设为 len(dataset) 或 len(dataset)//batch*batch，保持每个 epoch 的 iter 数稳定。
4. `--logit_adj tau`：logit-adjusted CE，loss = CE(logits + tau * log(prior))，prior = 训练集各类占比，tau 默认 1.0。
5. 确认曲率参数组不施加 weight decay（v3.1 已有逻辑，保留）。
6. 所有改动通过新 flag 控制，默认值全部为 False/0，确保不加 flag 时与当前行为逐字节一致（正在跑的 168298/168343/168344/168346 不受影响）。

## 文件
- `/home/jnguan/JiangYouxin/ZhaoJin/train_lr_hyper.py`：加 flag、sampler、grad accum、logit-adj 损失
- `/home/jnguan/JiangYouxin/ZhaoJin/models/FECNet.py`：GroupNorm 分支
- `/home/jnguan/JiangYouxin/ZhaoJin/models/local_remote_grouper.py`：如果内部有 BN 也处理
- 新增：一个 CPU 可跑的验收脚本 `verify_optim_fix.py`

## 注意事项
- 不要动正在跑的作业的代码路径——新 flag 默认关闭即可
- 不要改数据路径、模型结构、超参默认值（150ep/AdamW 1e-3/official_gen 数据）
- 不要提交任何 slurm 作业，只改代码
- 中文注释，PEP 8

## 验收标准
1. `python -c "import train_lr_hyper"` 能通过
2. `--help` 显示 4 个新 flag
3. CPU 上 1 batch 前向+反向无 NaN（用 verify_optim_fix.py 验证）
4. 不加任何新 flag 时，输出 Namespace 与当前完全一致
5. 逐项说明：每处改动的理由 + 与长尾/b8 欠拟合的关系

## 补充说明
跑完验收后，输出一段给豆包的审查话术（按交接格式：变更概要/改动文件/已知问题/测试情况）。

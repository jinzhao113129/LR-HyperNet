# 给 HPC Claude 的复审：CS-Net 可学习选点复现（per-center 适配）

> 复审方式（2026-10-27 起）：在 HPC 上用 Claude 复查，不再用 Codex。
> HPC 工作目录 `/home/jnguan/JiangYouxin/ZhaoJin`。审前必须先把本次改动的
> 3 个文件同步到 HPC（git pull 或 scp），否则 Claude 读到的是旧代码。
> 本单要求给出 P0/P1/P2 分级与「能否 push + 跑 3090 快测」的结论。

## 背景

学长要求「先复现现成的可学习选点」。选 CS-Net（arXiv 2501.10789）：
选择性采样 + 贡献分数 + 熵正则 OT 软 top-k，和「每中心从候选池选 16 个」同构；
无公开代码，按论文实现。CS-Net 是**全局选 K 个**，我们是**每中心选 G 个**，
所以做 per-center 适配（下方「与论文的差异」如实记录）。

## 改动

1. `local_remote_grouper_v2.py`
   - 新增 `eot_soft_topk(scores, k, tau=0.5, iters=20)`：2 列 OT（列 0 选中 / 列 1 未选），
     行边际 1/M、列边际 [k/M,(M−k)/M]、代价 C=[−s,0]，对数域 Sinkhorn，
     返回 w = P[:,0]·M（Σw=k）。
   - 新增 `_sample_candidates`（把 v1/v2 内联的候选采样抽成共用实现，并修一个
     小 N 下的越界兜底：K 缓冲不足时 clamp 到 N−1；生产配置不触发）。
   - 新增 `_candidate_signals`（v2 的四路信号抽成共用实现：cos/XY/时间/密度，
     距离与密度按候选池 z-score）。
   - 新增 `_csnet_select`：候选池 → 候选相对中心特征 → 偏移注意力
     `MLP(F−F_sa)+F`（按中心分块 `cs_chunk=256`，控制 [B,s,M,M] 显存）→
     贡献分数 MLP（OA 特征 + 四路信号）→ EOT 软 top-k。
   - `__init__` 新增 `g16_treatment='csnet'` 分支（q/k/v、OA MLP、score MLP、tau）。
   - forward 分派与 `LocalRemoteGrouper.forward` 的 learned 分支纳入 `csnet`。
2. `train_frequency_ncaltech.py`：choices 增加 `csnet`。

## 与论文的差异（请确认是否可接受）

1. **全局 vs per-center**：论文对整片点云做一次 FE+CA+CS 选 K 个；我们在每个中心
   的候选池（M=256）内做 OA+CS 选 G=16。FE 模块（局部 grouping）由我们 backbone
   已提供的特征替代，未单独复现。
2. **注意力范围**：论文 CA 是全局自注意力（n×n）；我们是候选集内（M×M）自注意力，
   且输入用 `candidate − centroid` 的相对特征。
3. **损失**：论文总损失 = EMD(P_sp, P_in) + L_task；我们只沿用分类损失，
   未加 EMD（EMD 是采样保形正则，在「选远程点」语义下目标不同）。
   是否要加一个「覆盖候选池」的 EMD 正则，请给意见。
4. **点云规模**：论文 N=1024；我们候选池 M=256（与 learned 相同），未对全 N 做注意力。

## 请重点审

1. **EOT 实现正确性**：代价 C=[−s,0] 下，传输计划列 0 的质量和 = k/M、返回
   `P[:,0]·M` 使 Σw=k；对数域 Sinkhorn 更新与 `g` 的 `unsqueeze(-2)` 广播是否正确；
   `clamp(min=-30,max=0)` 是否会影响梯度（P≤1，理论上界 0 成立否）。
2. **tau 的选取**：0.5 是否合理；是否需要退火（论文 ε 固定，但没写具体值）。
3. **OA 的输入**：用相对特征 `cand−centroid` 做自注意力，是否与论文语义偏离过大；
   是否应改为绝对特征 + 位置编码。
4. **分块**：`cs_chunk=256` 在 B=8,S=2048,M=256,C=64 下的显存与数值正确性
   （分块只沿 S 切，注意力在每个中心内部，不应跨块）。
5. **共用实现的回归**：`_sample_candidates` / `_candidate_signals` 重构了 v1/v2 的路径，
   请确认行为与原实现逐位一致（除 clamp 兜底）。

## 自测（CPU，已通过）

- EOT：Σw=k 误差 9.5e-7；top-k 平均权重 0.633 > 其余 0.122；
- selector：形状 (2,8,16)、候选不含 local、cs_q/k/v/mlp/score 梯度非零有限；
  分块路径（cs_chunk=3）Σw 正确；
- grouper 端到端：(2,8,4)/(2,8,6,36)、反传正常；
- `center_xyz=None` 抛 ValueError；
- learned_v2 重构后复测通过。
- 脚本：`D:/tmp/test_csnet.py`、`D:/tmp/test_learned_v2.py`（未入仓）。

## 请给结论

P0/P1/P2 分级，能否 push + 跑 3090 快测。

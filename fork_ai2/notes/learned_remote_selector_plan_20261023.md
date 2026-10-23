# Core Idea 1 落地方案：可学习远程选点（learned scorer + 可微 top-k）

> 2026-10-23 起草。目标：把 G16 的"怎么选这 16 个远程点"从硬编码启发式升级为
> 端到端可学习。技术路线依据 notes/learned_remote_selector_survey_20261023.md §7。

## 1. 研究问题

固定 LRHyperNet/FECNet 骨干与双曲聚合器，**选点打分器从"固定公式"换成"可学习网络"、
选择从"硬 top-k"换成"可微 top-k"**，是否让"远程选点"本身被端到端学出来、并在
N-Caltech101 上超过固定形式 scorer（rp1_scorer / euc_near）？

## 2. 现状（代码实证）

- `RemoteSelector` docstring 自述"第一版：特征距离 top-k，不做 learned scorer"。
- rp1_scorer = 特征距离 + `rp1_lambda`（**固定标量**，非 nn.Parameter）× 空间新颖性，
  硬 `topk(G)` 选点。selector 零可学习参数。
- 选点分两层（都在 `with torch.no_grad():` 内，梯度不回流）：
  1. 块内硬 top-G（`chunk=1024` 分块，`_hyp_dist_chunk`/`square_distance` 打分）；
  2. 候选并集硬 top-G（`cand_points` + `cand_dist.topk(G)`，`local_remote_grouper_v2.py:606-648`）。
- 聚合侧 `HyperbolicRemoteAggregator`（曲率/Möbius/注意力）**已全可学习**。

## 3. 方案：新增 `g16_treatment='learned'`，不碰现有 6 处理

两个核心改动：
1. **可学习 scorer**：小网络给候选点打贡献分，替代硬编码距离。
2. **可微 top-k**：SOFT 的 n→2 熵正则 OT（首选）或 Gumbel-softmax（备选）。

### 3.1 可学习 scorer

- 输入（候选并集层，M≈n_chunks×G≈320，非全 N=10240——沿用现有分块缩小候选，
  scorer 只在并集层打分，代价小 32 倍）：
  - centroid 特征 `c` [B,S,C]
  - 候选点特征 `p` [B,S,M,C]
  - 空间新颖性 `nov` [B,S,M]（复用现有 `_spatial_novelty`）
  - 候选点与 centroid 的特征差 `p - c`（显式对比信号）
- 网络：`[c; p; p-c; nov]` → MLP(4C+1 → C → 1) → score [B,S,M]。
  用 `rp1_lambda` 的固定形式（特征距离+λ·nov）作为 scorer 的初始化先验（好起点）。
- 参数随每个 stage 独立（`stage_idx`），三个 stage 三个 scorer。

### 3.2 可微 top-k（二选一，先 OT）

- **OT 版（SOFT n→2，首选）**：score s_i → 熵正则 OT，源=均匀 M 维，目标=[k/M,(M-k)/M]
  二维，代价 C[i,1]=s_i²、C[i,2]=(s_i-1)²，Sinkhorn 得软权重 a ∈ [0,1]、Σa=k。
  代价矩阵 M×2（M≈320，几 KB），O(M) 可跑，无重复、可微。
- **Gumbel 版（备选）**：s_i → Gumbel key → softmax(温度退火) → 软权重。O(M) 更简单，
  但 0<t<1 时 top-k 一致性不保证（ReparamSubset 自述）。

### 3.3 软选择如何进聚合

硬选流程（现状）是"选 G 索引 → index_points 取特征 → hyper_agg 双曲聚合"。
软选择没有索引，改两条路之一：

- **路 A（软加权聚合，首选）**：`remote_feat = Σ_i a_i · cand_feat_i`（软权重直接加权
  候选特征，可微），替代 hyper_agg 的硬选输入。此时 hyper_agg 可保留做后处理或去掉。
- **路 B（软权重作注意力先验）**：软权重 a 作为 hyper_agg 内注意力 softmax 的 bias，
  保留双曲聚合，软选择变成"软先验 + 硬聚合"。

路 A 更纯粹（真正端到端软选择），路 B 改动小（保留双曲聚合不变）。**先路 B 验证
scorer 有效性，再试路 A**。

## 4. 代码接入点

`local_remote_grouper_v2.py` RemoteSelector.forward：

1. **把 scorer 打分移出 `torch.no_grad():`**（现在 line 528 起整段 no_grad，梯度被切断）：
   - 分块阶段（缩小候选）保持 no_grad 硬 top-G（这块不需要梯度）；
   - 候选并集层（line 606-648）改成：可导 scorer 打分 + 可微 top-k（软权重）。
2. **新增 scorer 模块**：`self.learned_scorer = nn.Sequential(...)`（learned 分支才建）。
3. **新增软 top-k 算子**：`_soft_topk_ot(scores, k, eps)`（Sinkhorn，~20 行）或
   `_soft_topk_gumbel(...)`。
4. **聚合改软**：路 B 用软权重作 hyper_agg 注意力 bias；路 A 用软权重加权替代硬选。
5. **train/inference 分流**：训练用软权重（可微），推理退化成硬 top-k（温度/熵系数→0）。

## 5. 与现有 treatment 的关系

- 新增 `g16_treatment='learned'`，现有 6 处理（random/spatial_fixed/euc_far/euc_near/
  hyp_far/hyp_near/rp1_scorer）一行不动。
- **对照链**：rp1_scorer（固定形式 scorer）是 learned 的"下界"；euc_near/hyp_near 是
  "固定度量"下界。learned 若超它们，证明"学选点"有增益。
- `--g16_treatment learned` 走 argparse choices + 映射 + 断言（同 hyp_near 的加法）。

## 6. 显存/计算评估

- scorer 打 M≈320 候选：B×S×M = 8×2048×320 ≈ 5.2M 次前向，轻量（一个 4C→1 的 MLP）。
- OT n→2 代价 M×2（320×2），Sinkhorn 几轮，可忽略。
- 相比现有 hyp_far（双曲距离矩阵 18.96GB），learned 的 scorer+OT 更便宜；
  但软权重要在 no_grad 外、需保留 scorer 打分图，**新增显存 ≈ 打分图激活，
  估 +1~2GB**，B8 仍有富余（现峰值 ~19.4GB / 24GB）。

## 7. 分阶段实施

- **阶段 1（最小验证）**：可学习 scorer + **硬 top-k**（scorer 打分 → 硬 top-G）。
  验证"学打分"是否优于固定距离，不引入软选择，改动最小、最稳。
- **阶段 2（端到端）**：加可微 top-k（OT 或 Gumbel）+ 软加权（路 B 先），梯度回流 scorer。
- **阶段 3（可选）**：路 A 纯软加权聚合，去掉 hyper_agg 硬选。

## 8. 风险与 Codex 衔接

- **离散选择梯度稳定性**：Gumbel 温度退火 / OT 熵正则系数 ε 的调度，需 tune（SampleNet、
  CS-Net 都有温度退火策略，可抄）。
- **软选择一致性**：Gumbel 版 0<t<1 会选错序，OT 版无此问题——首选 OT。
- **与 hyper_agg 的衔接**：软权重 → 双曲聚合的接口要设计干净，避免"软硬混杂"。
- **双人制**：方案经 Codex 审查通过后再写代码 + 跑训练（学长规矩）。
- 现有 6 处理训练不受影响，learned 是新增独立 treatment，可并行推进。

## 9. 下一步

1. 本方案交 Codex 审查（技术路线 + 接口设计）；
2. Codex 放行后写阶段 1（scorer + 硬 top-k），小样本快测验证"学打分"是否超 rp1_scorer；
3. 阶段 1 有正反馈 → 阶段 2 软 top-k 端到端。

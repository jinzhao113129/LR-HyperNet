# Core Idea 1：可学习远程选点（Learned Remote Selector）论文地图（2026-10-23）

> 学长定位：G16 的"怎么选这 16 个点"是 core idea 1，且强调"如何把 16 个点学习到"与
> "16 个点怎么选"同样重要。本文整理 RP1 原文核心 + 已收集论文 + 领域经典 + 技术路线。

## 1. RP1 原文核心（event-camera-rps/docs/rp1-local-sparse-remote-neighborhood.md）

- 邻域公式：`N_i = L_i ∪ R_i`（稠密时间局部 W384 + 稀疏远程 G16）。
- 关键实验（N-Caltech101，W384-G16 协议）：Oracle 68.03% > Global KNN 66.70% >
  Spatial-uniform 61.08% > Random 59.87% > Local 59.30%。
- 核心结论：**远程点"怎么选"比"选多少"更重要**（16 个好点 > 全局 KNN）。
- 原文明确列出的 selector 六候选：Random / Spatial-uniform / Time-polarity /
  Feature hash / **Learned scorer** / Oracle。其中 **Learned scorer 原文未实现**。
- 原文推荐的流程：`Remote pool → cheap tokenization → Top-G`；评估三维
  Accuracy vs Oracle Recall@G vs Latency。
- 学长亲口确认 RP1 未完成（只证了 headroom，没做完 method）→ 我们接手做 open problem。

## 2. Core Idea 1 定位

现状（代码实证）：G16 两个环节——
- 选点（selector）：**零可学习参数**（rp1_lambda/anchor_lambda 是写死标量，
  选点曲率 c 是 register_buffer），完全不可导。
- 聚合（HyperbolicRemoteAggregator）：曲率/Möbius 变换/注意力全部可学习。

Core idea 1 = 把选点也纳入端到端学习：可学习 scorer + 可微 top-k（Gumbel），
让网络同时学"选哪 16 个点" + "怎么聚合"。离散索引不可导是核心技术障碍。

## 3. 已收集论文（fork_ai2/papers/，5 篇，方向高度契合）

| 论文 | 会议 | 与 core idea 1 的关系 |
|---|---|---|
| Modeling Point Clouds with Self-Attention and Gumbel Subset Sampling（上交+华为诺亚） | CVPR 2019 | **最直接 prior art**：自注意力 + Gumbel 子集采样 = 可微选点 |
| SAMBLE: Shape-Specific Point Cloud Sampling（KIT） | CVPR 2025 | 形状感知的可学习点云采样（局部细节 vs 全局均匀 trade-off） |
| Adaptive Spatial-Temporal Window（清华） | CVPR 2026 | 事件相机的自适应时空窗口（替代固定分组，选点/分组可学习化） |
| Spike-driven Discrete Aggregation for Event-based Object Detection（浙大） | CVPR 2026 | 脉冲驱动**离散聚合**——离散选择与聚合一体，G16 最接近的近期工作 |
| Making Every Event Count: Event Camera Subsampling（代尔夫特） | CVPRW 2025 | 事件子采样对下游任务的影响（数据效率 vs 精度） |

## 4. 建议补充的领域经典（网络受限未实时核验，arXiv ID 供自查）

**可微选择技术基础：**
- Differentiable Top-k with Optimal Transport — Xie et al., NeurIPS 2020（arXiv:2002.06504）
- Stochastic Beams and Where to Find Them（Gumbel-top-k 无放回采样）— Kool et al., ICML 2019（arXiv:1903.06059）
- Categorical Reparameterization with Gumbel-Softmax — Jang et al., ICLR 2017（arXiv:1611.01144）

**可学习采样（点云）：**
- SampleNet: Differentiable Point Cloud Sampling — Lang et al., CVPR 2020（arXiv:1912.03663）
- PointNet++（FPS 启发式基线，可学习采样要替代的对象）— Qi et al., NeurIPS 2017

**Token / 集合选择：**
- TokenLearner — Ryoo et al., NeurIPS 2021（arXiv:2106.11297）
- Set Transformer（Induced Set Attention，可学习集合聚合）— Lee et al., ICML 2019（arXiv:1810.00825）

## 5. 技术路线（从易到难）

1. **可学习加权（最小改动）**：rp1_lambda / anchor_lambda / 选点曲率 c 从标量/buffer
   改 nn.Parameter——网络学"特征距离 vs 空间新颖性"权重与选点曲率。topk 仍不可导，收益有限。
2. **可学习评分器**：硬编码距离（欧氏/双曲）换成可学习小网络 `scorer(centroid, candidate)`。
3. **软选择 / 可微 top-k（正路）**：硬 topk → Gumbel-softmax 或可微 top-k（optimal transport），
   梯度真正回流到评分器，端到端学"选哪 16 点"。改动大，需 Codex 级审查，论文级贡献。

**建议**：先做 1+2（把 rp1_scorer 的固定形式升级为可学习 scorer，顺带让选点曲率可学），
出结果后再上 3（软选择）。1/2 与现有 rp1_scorer treatment 直接衔接，风险低。

## 6. 待补（需学长/用户侧网络）

- 2025-2026 最新工作（CVPR/NeurIPS/ICLR 的 learnable token/point selection）——
  建议搜索词：`learnable point selection`、`differentiable top-k subset selection`、
  `learned token sampling transformer`、`event camera adaptive sampling`。
- RP1 iter7_boundary_rerank / iter9_indexed_b2 的原始 config 与结论（学长服务器 autodl 上）。

## 7. 深入判断（2026-10-23 读 7 篇摘要+方法后，修正 §5 的可扩展性误判）

### 7.1 关键修正：OT top-k 不会爆显存（§5 说"OT 会爆"是误判）

SOFT 把 top-k 建模成 OT 时，**目标集合只有 2 个（{0,1}）**，代价矩阵是 **n×2**
而不是 n×n。构造：源分布 μ=均匀 n 维，目标分布 ν=[k/n, (n-k)/n] 二维，
代价 C_{i,1}=x_i²、C_{i,2}=(x_i-1)²。Sinkhorn 迭代是 **O(n)** 而非 O(n²)。
**n=10240 候选完全可跑**（代价矩阵 10240×2，几 KB）。这推翻了 §5 对 OT 可扩展性的顾虑。

### 7.2 CS-Net 与 SOFT 同源，且 CS-Net 最贴近我们的问题

CS-Net 明确区分 generative（生成新点，S-Net/SampleNet）vs **selective**
（选已存在的点子集）。CS-Net 是 selective：feature embedding → cascade
attention → contribution score → **熵正则 OT top-k**（与 SOFT 同一 trick）。
保证无重复点、可任意采样率、端到端可训。这正是"从 10240 选 16 个已存在
远程点"要的。

### 7.3 首选落地路线

**可学习 scorer + 熵正则 OT top-k（SOFT/CS-Net 的 n→2 构造）**：
1. scorer(centroid, candidate) 给 10240 候选打贡献分 s_i（可学习小网络，替代
   硬编码欧氏/双曲距离）；
2. s_i → SOFT 的 n→2 OT（代价 C_{i,1}=s_i²、C_{i,2}=(s_i-1)²，熵正则 Sinkhorn）
   → 软 top-16 权重 a_i（可微、无重复）；
3. 训练时软权重加权聚合（梯度回流 scorer），推理时硬 top-16。

### 7.4 备选（更轻，O(n)，但一致性略差）

Gumbel-softmax / ReparamSubset 的 RelaxedTopK（Algorithm 2）：
Gumbel key r̂_i = -log(-log(u_i)) + log(w_i) → 温度退火 → relaxed k-hot 向量。
实现简单，但 ReparamSubset 自己承认 0<t<1 时 top-k 一致性不保证（可能选错序）。

### 7.5 SampleNet：可借鉴但本体不对齐

SampleNet 是 generative（生成采样点 Q）+ 软投影到输入点，不是 selective。
它的"软投影 = 加权平均 k 近邻"可借鉴（软权重→硬点的近似），但本体是
"生成新点"不是"选已存在点"，与我们的 selective 需求有偏差。

### 7.6 结论

- **主路线**：可学习 scorer + OT top-k（SOFT/CS-Net，n→2 构造不爆显存）。
- **次路线**：可学习 scorer + Gumbel-softmax 软 top-k，实现简单，快速验证用。
- SampleNet 软投影作"软权重→硬点"的补充技巧。
- 下一步：先把现有 rp1_scorer 的固定形式（特征距离+λ·空间新颖性）升级为
  可学习 scorer（小网络），再接 OT top-k 或 Gumbel softmax。

# 给 Codex 的审查：learned scorer 落地（可微 top-k 选型 + 两个 P0 设计）

## 背景

承接你上一轮审出的两个 P0（候选池偏差、显存估算）和"先做 Gumbel 数值原型"的建议。
可微 top-k 的数值原型已做，选型有变化，两个 P0 的设计如下。请审，不急着写代码。

## 1. 可微 top-k 选型：ST_k（NeurIPS 2024），偏离你建议的 Gumbel-top-k

**排坑过程（数值原型实测）**：
- sigmoid top-k：Σw 爆到 711（δ 与 score 尺度不匹配）；
- sequential softmax：Σw=k 精确但质量不集中（top-16 权重只 0.017）；
- STE（前向硬 + 反向软）：能用但梯度是近似。

**ST_k 核心**：`w_i = SReLU'(s_i - λ) = ½[1 + (s_i-λ)/√((s_i-λ)²+δ²)]`，
λ 是**可训练参数**（Ogryczak-Tamir 的 min 形式，λ 自动收敛到第 k 大元素），δ 平滑系数。

**选 ST_k 而非 Gumbel-top-k 的理由**：
- λ 可训练，自动定位阈值，无需手动调温度 τ；
- Σw=k 精确（δ→0），无需 Gumbel 噪声采样 + RelaxedTopK 的复杂公式
  （ReparamSubset 的 RelaxedTopK 本身无闭式，引用 NeuralSort 的松弛置换矩阵）；
- O(n)，无排序、无额外显存（原文 Table 1 实测 0.0127s vs HeapSort 0.1243s）。

原型 `test_soft_topk.py` 已写，本地 numpy δ sweep 发现 δ 需与 score 尺度匹配
（[0,1] 需 δ~1e-4 才 Σw≈k 且质量集中）。

**请审**：ST_k 选型是否成立？δ 退火（训练中大→小）调度怎么定？λ 初始化为
第 k 大元素（kthvalue）是否够，还是需要别的初始化？

## 2. P0-1 无偏候选池设计

**现状**：候选并集 = 每 chunk 固定距离 top-G 的并集（chunk=512，M≈320）。learned
scorer 若只在这 320 个"固定度量筛过"的点上重排，逃不出 euc_near/rp1 的 59%。

**方案**：learned 分支候选池改用**无偏采样**——从 N=10240 里随机采 M 个非局部
候选（排除该中心点的 local_idx），scorer 在这 M 个无偏候选上打分 + ST_k 软 top-16。

**设计疑问（请审）**：
- M 取多大（256 还是 512）？太小漏好点、太大显存贵；
- 无偏采样"每中心点独立采 M 个" vs "全局共享一个 M 池"？共享池省计算但可能含
  某中心点局部邻居、需额外排除；
- 采样每 epoch 重采还是固定（固定则训练一致、重采则更强随机性）？

## 3. P0-2 显存压缩设计

你算的 `[B,S,M,3C+1]` stage0 ≈4GB、反向 8-10GB。压缩方案：
- scorer 输入只用 `candidate - centroid`（单 C 维）→ `[B,S,M,C]`，M=128 时
  stage0 ≈536MB 前向（8×2048×128×64×4）；
- 按 S 分块（每次 S_chunk 个中心点），峰值再降 S/S_chunk 倍；
- M 从 320 降到 128 先跑通。

**请审**：`candidate-centroid` 单输入是否损失信息（要不要补 novelty 或 centroid 本身）？
按 S 分块的具体粒度（S_chunk=512？）？

## 请给三档结论

1. ST_k 选型 + 两个 P0 设计是否成立，有无更简/更稳的替代；
2. δ 退火调度、M、S_chunk 的具体取值建议；
3. 优先级与分阶段顺序（先跑通 ST_k 原型 + 无偏池，还是先解决显存压缩）。

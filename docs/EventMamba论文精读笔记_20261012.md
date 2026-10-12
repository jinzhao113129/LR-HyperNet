# EventMamba 论文精读笔记

> 论文：Rethinking Efficient and Effective Point-based Networks for Event Camera Classification and Regression
> 期刊：IEEE TCSVT 2025（JCR Q1）
> 作者：Hongwei Ren 等（HKUST Guangzhou）—— 同 SECNet/SpikePoint 一作！
> arXiv：2405.06116
> 本地原文：`D:\论文\2405.06116.pdf`

## 一句话总结

事件点云 + 时空 Mamba，三阶段层级结构（LocalFE + 时间注意力聚合 + Mamba 全局特征），在6个动作识别数据集上 SOTA，同时支持相机位姿重定位和眼动追踪，参数量仅 0.29M（DVS128）。

## 作者系列工作脉络

| 工作 | 会议/期刊 | 核心 | 与我们的关系 |
|---|---|---|---|
| SpikePoint (2023) | ICLR 2024 | SNN + point-based，单阶段 | 轻量级设计参考 |
| EventMamba (2024) | TCSVT 2025 | ANN + point-based + Mamba，三阶段 | 时间建模参考 |
| SECNet (2024) | ICML 2026 | ANN + point-based + 两次傅里叶，三阶段 | **我们的基线** |

**演变路径**：SNN→ANN（追求精度）、单阶段→三阶段（追求容量）、LSTM→Mamba（追求效率）、无傅里叶→两次傅里叶（追求频域特征）。

## 核心方法

### 1. 事件点云预处理
- 滑动窗口切分事件流
- 归一化：PN_i = (X_i/w, Y_i/h, TS_i)，其中 TS_i = (t_j - t_i)/(t_l - t_j)
- **显式空间 (X,Y) + 隐式时间 TS**，时间戳作为第三个坐标
- 点数：1024 或 2048

### 2. 三阶段层级结构

```
输入点云 → Stage 1 → Stage 2 → Stage 3 → 分类/回归头
每个 Stage:
  LocalFE（局部特征提取）→ TAG（时间注意力聚合）→ GlobalFE（Mamba全局特征）
```

#### LocalFE（局部特征提取）
- FPS 采样中心点 + KNN 找邻居（K=24）
- 坐标标准化：PG_i = (PG_i - PS_i) / Std(PG_i)
- 残差块：ResB(x) = BN(MLP(x))，x_out = x + σ(x + M(σ(M(x))))
- 输出：[B, T_i, K, D_i]

#### TAG（时间注意力聚合）—— 关键创新
- 对 K 个邻居按时间顺序排列：S = LocalFE(e_1), ..., LocalFE(e_k)
- 注意力权重：A = SoftMax(MLP(S))，a_k ∈ [0,1]
- 加权聚合：SA = S·A = a_1·S_1 + a_2·S_2 + ... + a_k·S_k
- **作用**：按时间顺序对邻居做注意力加权，显式建模时间顺序
- 输出：[B, T_i, D_i]

#### GlobalFE（全局特征提取）
- **Mamba Block**：用 State Space Model (SSM) 建模 T_i 个 group 之间的时间关系
  - h_t = Ā·h_t + B̄·x_t，y_t = C·h_t
  - Ā = exp(ΔA)，B̄ = (ΔA)^{-1}(exp(ΔA)-I)·ΔB
- 残差块：ST = Mamba(SA)，ST = ResB(ST)
- 输出：[B, T_i, D_i]

### 3. 分类/回归头
- 分类：全连接 + label smoothing
- 回归（CPR/眼动）：全连接 + L2 损失 + 正则化

## 实验结果

### 动作识别（6个数据集）

| 数据集 | 类别 | EventMamba | 之前 SOTA | 参数量 |
|---|---|---|---|---|
| DVS128 Gesture | 11 | **99.2%** | 98.8% (TTR+13D) | 0.29M |
| Daily DVS | 12 | **99.1%** | 99.0% (VAME) | 0.905M |
| DVS Action | 10 | **89.1%** | 92.7% (ST-ResNet)※ | 0.905M |
| HMDB51-DVS | 51 | **56.4%** | 52% (EV-ACT) | 0.916M |
| UCF101-DVS | 101 | **90.3%** | 89.7% (EV-ACT) | 3.72M |
| THU-EACT-50-CHL | 50 | **59.4%** | 47.3% (EV-ACT) | 0.905M |

※ DVS Action 上 ST-ResNet 92.7% 是 frame-based，EventMamba 是 point-based SOTA。

**注意**：EventMamba 没有在 N-Caltech101 上测试。

### 消融实验关键发现

1. **时间模块重要性**：无时间模块（仅空间聚合）在 HMDB51-DVS 上只有 57.5%，加 Mamba 后 82.1%——**时间建模是关键**
2. **Mamba vs LSTM**：Mamba 精度更高（HMDB51 82.1% vs 80.8%），推理更快（6.8ms vs 15.5ms）
3. **点数影响**：HMDB51-DVS 上 1024点→82.1%，2048点→86.6%（+4.5%）
4. **时间注意力聚合(TAG)**：比 MaxPooling 更好，因为 MaxPool 丢弃时间顺序

## 对我们新架构的启示

### 直接可用
1. **TAG 时间注意力聚合**：EventMamba 对 K 个邻居按时间顺序做注意力加权。我们的局部分组目前用 SECNet 的空间傅里叶+注意力，可以考虑加入时间顺序注意力
2. **Mamba 全局特征**：GlobalFE 用 Mamba 建模 group 间关系。我们的远程聚合目前用双曲注意力，可以考虑用 Mamba/SSM 代替（我们的 KDA 就是 gated delta rule，类似 Mamba）
3. **时间表示对比**：EventMamba 用 TS = 时间戳差/窗口长度（简单归一化），我们用多频正弦编码。可以消融对比哪种更好

### 值得探索
1. **Mamba + 双曲结合**：EventMamba 证明 Mamba 在事件序列建模上有效，我们可以探索"双曲空间中的 Mamba"（双曲 SSM）
2. **TAG + 双曲距离**：时间注意力聚合的权重可以用双曲距离计算，而不是 MLP
3. **三阶段一致性**：EventMamba 和 SECNet 都是三阶段，验证了三阶段结构的合理性。我们的 LR-HyperNet 也保持三阶段，正确

### 不适用
1. EventMamba 做动作识别（时序建模为主），我们做物体分类（空间结构为主）
2. EventMamba 没有 N-Caltech101 结果，无法直接对比
3. EventMamba 的 Mamba Block 是标准欧氏 SSM，我们的双曲聚合是不同路线

## 与我们 KDA 的关系

- EventMamba 的 Mamba = 标准 SSM（线性递推 h_t = A·h_{t-1} + B·x_t）
- 我们的 KDA = Gated Delta Rule（S_new = Diag(α)·S_old - β·k·(k^T S_old) + β·k·v^T）
- KDA 是 Mamba 的改进版（Kimi K3 用 KDA 替代 ROPE/Mamba），理论上表达能力更强
- EventMamba 的成功验证了"序列建模 + point-based"路线的有效性，我们的 KDA 是这个路线的进阶

## 关键引用

- Mamba: Gu & Dao, 2023（SSM 选择性扫描）
- SSM 离散化：Ā = exp(ΔA)，B̄ = (ΔA)^{-1}(exp(ΔA)-I)·ΔB
- 时间注意力聚合：MLP + SoftMax 加权

---
**阅读日期**：2026-10-12
**阅读页数**：1-12（全文）
**下一步**：把 SpikePoint + EventMamba + SECNet 三篇作者系列工作的演变整理成对比表，更新飞书论文笔记

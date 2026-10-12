# SpikePoint 论文精读笔记

> 论文：SpikePoint: An Efficient Point-based Spiking Neural Network for Event Cameras Action Recognition
> 会议：ICLR 2024（Spotlight）
> 作者：Hongwei Ren 等（HKUST Guangzhou）—— 同 SECNet 一作！
> arXiv：2310.07189
> 本地原文：`D:\论文\2310.07189_SpikePoint_脉冲点云SNN_ICML2024.pdf`

## 一句话总结

第一个基于点云的脉冲神经网络（SNN），用单阶段结构+速率编码+双分支（局部+全局），在5个事件动作识别数据集上达到 SOTA，参数量只有同精度 ANN 的 0.3%、功耗 0.5%。

## 核心方法

### 1. 事件→点云转换
- 事件流按滑动窗口切分，每个窗口内的事件直接作为点云（不转帧）
- 点坐标 = (x, y, z_m)，其中 z_m = (t_m - t_k)/(t_l - t_k) 是归一化时间戳
- 点云同时携带显式空间信息 (x,y) 和隐式时间信息 z

### 2. 采样与分组
- FPS 采样中心点 + KNN 找局部邻居（和 PointNet++ 一样）
- **关键创新：坐标标准化 + 负值处理**
  - 相对坐标 Δ = (G - Centroid) / SD(G)，服从标准正态 N(0,1)
  - SNN 的速率编码只能处理正值，所以取绝对值 |Δx|,|Δy|,|Δz|
  - 取绝对值后分布变成折叠正态分布，均值从 0 变成 √(2/π)≈0.8
  - 用最小值 x_min,y_min,z_min 补偿（减去最小值使输入重新平衡）
  - 最终输入 = [|Δx|,|Δy|,|Δz|, x_min,y_min,z_min]（6维）

### 3. 单阶段结构（核心区别）
- **不用 PointNet++ 的层级多阶段结构**，因为 SNN 的脉冲特征会随深度变稀疏，梯度消失严重
- 单阶段同时提取局部和全局特征：
  - **局部分支**：每组内的相对坐标 → Conv1D → MaxPool → F_local
  - **全局分支**：Centroid 坐标 → Conv1D → MaxPool → F_global
  - 融合：F_local + F_global（加法优于拼接）

### 4. ResF 残差模块
- 带瓶颈的残差模块，解决 SNN 的梯度爆炸/消失
- LIF 神经元的离散递推：U[n+1] = e^(-Δt/τ_mem) U[n] + I[n] - S[n]
- 残差连接对应 surrogate gradient 中的误差项

### 5. 速率编码
- 坐标值 → 脉冲发射频率（值越大，发射脉冲越多）
- 只有 16 个 timestep（非常少，SNN 通常需要 32+）

## 实验结果

| 数据集 | 类别数 | SpikePoint | 之前 SOTA(SNN) | 备注 |
|---|---|---|---|---|
| DVS128 Gesture | 11 | **98.74%** | 93.6% | 参数量 0.58M |
| Daily DVS | 12 | **97.92%** | 90.3% | 参数量 0.16M |
| DVS Action | 10 | **90.6%** | 78.1% | 参数量 0.16M |
| HMDB51-DVS | 51 | **55.6%** | 54.5% | 参数量 0.79M |
| UCF101-DVS | 101 | **68.46%** | 58.5% | 参数量 1.05M |

**注意**：SpikePoint 没有在 N-Caltech101 上测试（它做的是动作识别，不是物体分类）。

## 消融实验关键发现

1. **ResF 残差模块**：收敛更快、最终精度更高（比无残差高 ~3%）
2. **单阶段 vs 多阶段**：多阶段（PointNet 式）在 SNN 上表现差，单阶段更好
3. **timestep**：16 是最优（8 不够，24/32 边际收益递减）
4. **双分支融合**：加法 > 拼接；局部分支+全局分支 > 单分支
5. **负值处理**：绝对值+最小值补偿 > 简单归一化到 [0,1]

## 对我们新架构的启示

### 直接可用
1. **负值处理方法**：我们的极性 p∈{±1} 可以参考 SpikePoint 的绝对值+补偿思路，但我们是 ANN 不是 SNN，直接用 ±1 没问题
2. **双分支结构**：我们的 local+remote 类似，但可以考虑加一个全局 Centroid 分支（v2 已有 global_anchor，思路一致）
3. **单阶段 vs 多阶段**：我们用三阶段是因为继承 SECNet，SpikePoint 的单阶段论证值得参考——但 SECNet 的三阶段在 N-Caltech101 上有效，不建议改

### 值得探索
1. **速率编码的时间表示**：SpikePoint 把时间戳当 z 坐标，我们 v2 用多频正弦编码。可以考虑混合：时间戳直接作为坐标 + 正弦编码作为额外通道
2. **轻量级设计**：SpikePoint 参数量极小（0.16-1M），我们的 LR-HyperNet 约 2.1M。可以考虑用 SpikePoint 的单阶段+瓶颈设计减少参数
3. **SNN vs ANN**：SpikePoint 证明 SNN 在事件数据上有能效优势，但精度上我们的 ANN 路线（SECNet 82.4%）在 N-Caltech101 上更高。不建议转 SNN

### 不适用
1. SpikePoint 是 SNN，我们是 ANN，速率编码/LIF 神经元不直接适用
2. SpikePoint 做动作识别（时序建模为主），我们做物体分类（空间结构为主）
3. SpikePoint 没有 N-Caltech101 结果，无法直接对比

## 与 SECNet 的关系

- **同一作者**（Hongwei Ren），SpikePoint (ICLR 2024) → SECNet (ICML 2026)
- **技术路线演变**：SpikePoint 是 SNN+point-based，SECNet 是 ANN+point-based（放弃了 SNN，转 ANN 追求更高精度）
- **共同思想**：都是 point-based、都用 FPS+KNN、都强调事件数据的稀疏性处理
- **SECNet 的改进**：从单阶段→三阶段、从 SNN→ANN、加了两次傅里叶变换、参数量更大（~3.5M）

## 关键引用

- 速率编码负值处理：折叠正态分布的均值和方差推导（论文 Eq.5-6）
- ResF 模块：Hu et al. 2021, Fang et al. 2021a, Feng et al. 2022
- 单阶段结构论证：SNN 脉冲稀疏性导致深度网络梯度消失

---
**阅读日期**：2026-10-12
**阅读页数**：1-10（全文）
**下一步**：读 EventMamba / EventMG，对比 Mamba 路线的 point-based 事件分类

# Point-based Event Representation 路线调研与双曲交叉点分析

> 调研时间：2026-10-05
> 目标：梳理 Hongwei Ren 组三篇 point-based backbone（SpikePoint / EventMamba / SECNet）的方法演进，找到双曲空间在 point-based event representation 中的切入点，形成新项目 motivation。

---

## 1. 三篇 backbone 定位

| 论文 | arXiv | 发表 | 核心技术 | 表示 | 时间建模 |
|---|---|---|---|---|---|
| **SpikePoint** | 2310.07189 | ICLR 2024 Spotlight | point-based SNN，单阶段，identity mapping | Point Cloud | 16 timesteps，脉冲时序 |
| **EventMamba** | 2405.06116 | TPAMI | point-based + Mamba/SSM，层次结构 | Point Cloud (x,y,t,p) | implicit(t坐标) + explicit(TAG注意力+Mamba) |
| **SECNet** | 2412.20803 | ICML 2026 Oral | point-based + 频域(Fourier)，Event-based Group & Sampling | Point Cloud | 频域时空特征 |

作者均为 **Hongwei Ren**（HKUST Guangzhou，MICS Thrust，Bojun Cheng 通讯）。三篇构成清晰的演进路线：SNN 脉冲 → Mamba 状态空间 → 频域可扩展。

---

## 2. EventMamba 方法精读（point-based 路线最核心一篇）

### 2.1 整体架构（图3）

```
Event Stream → Downsampling(N=1024/2048) → Hierarchy(m stages) → Classifier/Regressor
                                    ↓
                    Stage i: LocalFE → Aggregation(TAG) → GlobalFE
```

### 2.2 事件预处理（式3-4）

原始事件 $(x_k, y_k, t_k, p_k)$，滑动窗口分割后归一化：

$$PN_i = \left(\frac{X_i}{w}, \frac{Y_i}{h}, TS_i, \frac{t-t_j}{t_l-t_j}\right)$$

- $X_i, Y_i$：归一化空间坐标（implicit spatial）
- $TS_i = \{t_j, ..., t_u\}$：时间戳集合（implicit temporal，作为点的维度）
- $(t-t_j)/(t_l-t_j)$：相对时间差（explicit temporal）
- **关键设计**：时间信息作为点云的一个维度保留，不聚合为帧

### 2.3 LocalFE（局部特征提取，式5,9-13）

1. **FPS 采样**：$P_S = FPS(PN_i, P_S_i)$ —— 最远点采样选中心点
2. **KNN 分组**：$P_G = KNN(PN_i, P_S_i, K)$ —— 每个中心点找 K 个最近邻
3. **标准化**：$PG_S_i = (PG_i - PS_i) / Std(PG_i)$ —— 组内中心化+标准化
4. **Residual Block**：$M(x) = BN(MLP(x))$，$ResB(x) = \sigma(x + M(\sigma(M(x))))$

- 处理**局部空间特征**，K=3 最常用
- 输出维度 $[B, T_i, K, D_i]$

### 2.4 Aggregation / TAG（时间聚合，式14-17）

LocalFE 输出的 K 个最近邻沿**时间轴**聚合：

$$S = LocalFE(x) = (S_{t1}, S_{t2}, ..., S_{tk})$$
$$A = SoftMax(MLP(S)) = (a_{t1}, a_{t2}, ..., a_{tk})$$
$$SA = S \cdot A = a_{t1}S_{t1} + a_{t2}S_{t2} + ... + a_{tk}S_{tk}$$

- **MaxPool + Attention 双路**：MaxPool 鲁棒去噪，Attention 保留细粒度时间信息
- 处理**局部时间特征**，利用 K 个点的严格时间顺序
- 输出 $[B, T_i, D_i]$

### 2.5 GlobalFE（全局特征提取，式18-23）

用 **Mamba Block（SSM）** 替代之前的 Bi-LSTM：

$$h_t = \bar{A}h_t + \bar{B}x_t, \quad y_t = Ch_t$$
$$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A)-I)\cdot \Delta B$$

- 处理**全局显式时间特征**，长序列线性复杂度
- ResB 进一步抽象空间特征
- 输出 $[B, T_{i+1}, D_{i+1}]$

### 2.6 损失函数

- 分类：label smoothing cross-entropy（式24）
- CPR：$\|\rho-\hat{\rho}\|_2 + \beta\|\dot{q}-\hat{\dot{q}}\|_2 + \lambda\sum w_i^2$（式25）
- Eye-tracking：WMSE

---

## 3. SpikePoint 方法要点（ICLR 2024 Spotlight）

- **单阶段结构**：不像 EventMamba 有层次结构，SpikePoint 用 singular-stage
- **Identity mapping feature extractor**：不做复杂特征变换，直接用 identity mapping
- **SNN 脉冲处理**：事件天然是脉冲形式，直接喂给 SNN，surrogate gradient 训练
- **16 timesteps**：时间维度离散化为 16 步
- **极轻量**：参数仅为 ANN 的 0.3%，功耗 0.5%
- 任务：动作识别（5 个数据集 SOTA）

---

## 4. SECNet 方法精读（ICML 2026 Oral）

### 4.1 整体架构（图3）

```
Event Cloud → 随机下采样(T0=1024/10240) → Embedding(Conv1D) → Hierarchy(m stages) → Predict Head
                                                          ↓
                    Stage: G&S → Spatial-FA → AGG → Temporal-FA → RES
```

### 4.2 Event-based Grouping & Sampling (G&S)

- **D-FPS（Differentiation FPS）**：FPS 采样时用可学习缩放因子 α 对坐标加权 $\mathcal{C}_i = FPS(\alpha \cdot \mathcal{EC})$，极性在结构级融入（不只是输入级）
- **EF-KNN（Event Features-based KNN）**：用特征 $F_i$ 找 K 最近邻（不是纯坐标 KNN），$\mathcal{G}_i, \mathcal{F}_{\mathcal{G}_i} = KNN(\mathcal{F}_{\mathcal{C}_i}, F_i, K)$
- **CES（Coordinates Evolution Strategy）**：用组内均值更新坐标 $\mathcal{EC}' = Mean(\mathcal{G}_i), \mathcal{EC}$，特征越深坐标越标准化
- 标准化：$Mean + Std$，输出 $[B, T_i, K, 2D_i+4]$

### 4.3 Frequency-aware Module（频域特征提取）

**核心创新**：用 DFT + Hadamard 乘积替代 MLP，复杂度从 $O(D_i^2)$ 降到 $O(T_i \times K)$。

- **Spatial-FA**：对组内 K 个点的空间维度做 1D DFT → 频域 Hadamard 乘积（可学习滤波器 V）→ 捕获 K 近邻的拓扑相关性
  $$X = FFT(x), \bar{X} = V \odot X, x = iFFT(\bar{X}), \sigma(x)$$
- **AGG**：Linear + SoftMax 注意力聚合组内特征
- **Temporal-FA**：对时间维度做 DFT → 可学习全局滤波器 Hadamard 乘积 → IFFT，捕获长序列时间依赖

### 4.4 关键贡献

1. 极性在结构级（G&S）而非仅输入级集成
2. 频域特征提取解决 MAC 爆炸，可扩展到高分辨率长序列（10240 点）
3. 10 个数据集验证可扩展性

---

## 5. SpikePoint 方法要点（ICLR 2024 Spotlight）

| 维度 | Point-based（SpikePoint/EventMamba/SECNet） | GNN-based（ECAF-GCN/EHGCN） |
|---|---|---|
| 表示 | 事件=点云 $(x,y,t,p)$，直接处理 | 事件→建图（radius_graph/KNN）→节点+边 |
| 局部关系 | KNN 分组（动态，每层重算） | 预建图（固定边，R跳邻居） |
| 时间建模 | 时间作为点维度 + 注意力/SSM | 时间编码进节点特征 + 边属性 |
| 计算开销 | 点云操作（FPS+KNN），可扩展 | 图卷积 + 注意力 $O(N^2)$，难扩展 |
| 信息损失 | 无建图步骤，原始稀疏性保留 | 建图有量化/采样损失 |
| 双曲空间 | **空白**——目前全在欧氏空间 | 已有工作（ECAF-GCN/EHGCN） |

---

## 7. EHGCN 精读（GNN+双曲路线，直接同行）

### 7.1 架构（图2）

```
Event Streams → DAES(分布感知事件筛选) → MAHM(运动感知超图建模) → Hierarchical Euclidean-Hyperbolic Fusion → Perception Head
```

### 7.2 DAES（Distribution-Aware Event Sifting）
- 多尺度体素网格 + 局部高斯分布建模
- 自适应抑制不可靠事件，保留判别性事件，衰减混沌噪声

### 7.3 MAHM（Motion-Aware Hypergraph Modeling）
- MRF（Markov Random Field）优化的 motion-aware hyperedge 生成
- 光流估计 → 能量函数最小化 → 生成超边（连接多个节点，捕获高阶交互）
- 捕获长程时空依赖和运动一致性

### 7.4 HEHE Block（Hierarchical Event Hyperbolic Embedding）
全在 Poincaré ball 上操作：
1. **Cross-space Embedding**：欧氏→双曲（指数映射 $\exp_c^0$）
2. **Hyperbolic Feature Transformation**：双曲矩阵乘法 $\mathbf{W}\otimes_c\mathcal{H}_i \oplus_c \mathbf{b}$（Möbius 加法）
3. **Hyperbolic Aggregation**：$\mathbf{A}\log_c^0(\cdot)$ 聚合
4. **Hyperbolic Activation**：$\exp_c^0 \sigma$
5. **Hyperbolic Embedding**：$\exp_c^0$

### 7.5 融合方式
- 欧氏 GCN（各向同性度量，保留局部类内结构）和双曲 HEHE **交替融合**（BN→HEHE→BN→HEHE，类似残差）
- Cross-space Embedding 用 exp/log 映射在两空间切换

### 7.6 与 ECAF-GCN 的区别
| | ECAF-GCN | EHGCN |
|---|---|---|
| 图结构 | 普通图（radius_graph, R=3） | 超图（motion-aware hyperedge） |
| 双曲运算 | 双曲 GCN（式3-5） | HEHE Block（双曲矩阵乘法+聚合+激活） |
| 融合方式 | 双分支并行 + 跨空间注意力 | 交替融合（BN→HEHE→BN→HEHE） |
| 共同点 | **都是 GNN 路线**（先建图再卷积） | **都是 GNN 路线** |

---

## 8. 双曲空间在 point-based 中的切入点（核心分析）

### 6.1 为什么 point-based event 天然适合双曲？

事件点云具有**多层级结构**：

```
场景 → 物体 → 部件 → 事件簇 → 单个事件
```

PointNet++ 式的 **FPS + KNN 分组**本身就在构建层级树：FPS 选中心点→KNN 分组→下一层再 FPS→再分组，每一层抽象一级。这种层级结构在欧氏空间中需要指数级维度才能表达，而**双曲空间具有指数级容量**，天然适合树状数据。

### 6.2 具体切入点（按模块）

| EventMamba 模块 | 欧氏现状 | 双曲改造方向 |
|---|---|---|
| **FPS 采样** | 欧氏最远点距离 | 双曲距离采样（Poincaré ball 距离） |
| **KNN 分组** | 欧氏 KNN | 双曲 KNN（双曲距离找最近邻） |
| **LocalFE ResBlock** | 欧氏 MLP+BN | 双曲线性层（Möbius 矩阵乘法）+ 双曲 BN |
| **TAG 注意力聚合** | 欧氏点积注意力 | 双曲注意力（双曲距离/相似度） |
| **GlobalFE Mamba** | 欧氏 SSM | 双曲 SSM / 双曲状态空间模型 |
| **分类头** | 欧氏 FC | 双曲分类（双曲 logits + 欧氏解码） |

### 6.3 与 ECAF-GCN/EHGCN 的本质区别

- ECAF-GCN/EHGCN：**先建图→再双曲图卷积**——图构建是中间步骤，有信息损失，且 $O(N^2)$ 注意力难扩展
- 我们的方向：**直接在 point-based 表示上做双曲运算**——不建图，保留事件原始稀疏性，双曲 KNN/分组/注意力天然适配层级结构

### 6.4 Motivation 草案

> 事件相机输出的异步事件流天然具有多层级结构（场景→物体→部件→事件簇→单事件）。现有 point-based 事件处理方法（SpikePoint、EventMamba、SECNet）均在欧氏空间中通过 FPS+KNN 分组建模局部关系，但欧氏空间无法高效表达层级结构，导致深层网络需要大量参数才能拟合事件间的层级依赖。另一方面，现有双曲事件方法（ECAF-GCN、EHGCN）走 GNN 路线，需先将事件构建为图再进行双曲图卷积，图构建步骤引入信息损失且跨空间注意力的 $O(N^2)$ 复杂度难以扩展。
>
> 我们提出直接在 point-based event representation 上引入双曲几何：利用双曲空间的指数级容量建模事件点云的天然层级结构，通过双曲 KNN 分组、双曲注意力聚合和双曲全局建模，在不建图的前提下实现层级感知的事件特征提取。这一方向结合了 point-based 表示的稀疏性保留和双曲空间的层级建模优势，与现有 GNN+双曲路线形成本质区别。

---

## 9. 综述分类体系（arXiv:2606.23078，确认空白）

### 9.1 Event Camera Representation Learning 分类（图3）

```
Single-Representation
├── Dense-based
│   ├── Event Frame
│   ├── Time Surface
│   ├── Binary Map
│   ├── Voxel Grid
│   └── Learned Dense
└── Sparse-based
    ├── Token-based (Event Transformer, EventMamba-TSM, GET...)
    ├── Graph-based (RG-CNN, EV-GCNN, AEGNN, EventMG, SladeGCN...)
    └── Point-based (STEC, EventNet, RTPoint, TPoint, SpikePoint, PEPoint,
                      EventMamba, EModepth, E2B, HSPC, SECNet)  ← 11篇

Multi-Representation
├── Dense-Dense Hybrid
└── Dense-Sparse Hybrid
```

### 9.2 空白确认

- Point-based 类别下 11 篇论文，**全部在欧氏空间**
- Graph-based 类别下有双曲工作（ECAF-GCN/EHGCN 是 GNN 路线）
- **整个综述未提及 "hyperbolic" 与 point-based 的组合**
- 普通 3D 点云+双曲已有 PointCHR (ICML 2026)，但不处理事件时间维度，且只是曲率感知整流而非完整双曲架构

---

## 10. 最终 Motivation 与项目方向

- [ ] SECNet 全文精读（频域具体怎么处理时空特征）
- [ ] SpikePoint 全文精读（SNN 脉冲时序与 point-based 的结合细节）
- [ ] 综述 arXiv:2606.23078 精读（event camera representation learning 全貌，确认 point-based×双曲是否真的空白）
- [ ] 搜索是否已有 "hyperbolic point cloud" / "hyperbolic PointNet" 工作（普通点云领域可能已有，需区分）
- [ ] EHGCN 全文精读（双曲+欧氏运动融合的具体做法，GNN 路线的参考）
- [ ] 双曲 KNN / 双曲注意力的具体实现可行性（Poincaré ball 上的高效最近邻搜索）

---

## 8. 关键引用

- EventMamba: arXiv:2405.06116 (TPAMI) — 本地 `D:\UserData\Desktop\学习\进组\论文\EventMamba-2405.06116.pdf`
- SpikePoint: arXiv:2310.07189 (ICLR 2024 Spotlight)
- SECNet: arXiv:2412.20803 (ICML 2026 Oral)
- 综述: arXiv:2606.23078 "A Systematic Survey on Event Camera Representation Learning"
- ECAF-GCN: ICLR 2027 审稿（本地 PDF）
- EHGCN: arXiv:2504.16616

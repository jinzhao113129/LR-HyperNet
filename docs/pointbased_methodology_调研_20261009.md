# point-based 表征方法论系统调研 + 双曲几何方法调研（2026-10-09）

> 项目：LR-HyperNet（point-based 事件点云分类）｜用途：论文 related work + 后续架构设计
> 采信范围：CCF-A 会议（CVPR/ICML/ICLR/NeurIPS/ECCV/ICCV）优先；非 CCF-A 标参考
> 规则：所有 arXiv ID 经摘要页核对（含多处纠错）；不确定标【待验证】；具体数值

---

## 一句话核心结论

**通用点云方法分六大族；事件 point-based 目前只占据 S1「层级集合抽象」（四个真身的共同骨架 = PointNet++ 式 FPS+KNN+层级+max-pool）与局部 S6（仅 EventMamba 用 Mamba 做时序远程）——S2 点卷积 / S3 图 / S4 Transformer 只作对比基线、S5 掩码预训练因无事件语料未用；而"显式 Local+Remote 空间几何配对"与"双曲远程聚合"在事件点云上都是空白：四个真身的"全局"无一例外是 max-pool / 频域滤波 / 1D 时序扫描等非几何算子，双曲×事件点云分类零工作（EHGCN 走检测、未发表）。**

---

## 一、六族谱系（冻结定义，related work 组织用）

| 族 | 一句话 | 复杂度基准 |
|---|---|---|
| **S1 集合方法** | 排列不变对称聚合（max-pool/对称函数），层级集合抽象 | O(N) / O(NK) |
| **S2 点卷积** | 以点坐标为核函数输入的空间连续卷积，局部邻域 | O(NK) |
| **S3 图方法** | kNN 动态/静态图上的边卷积与图注意力消息传递 | O(NK) |
| **S4 Transformer** | 邻域向量注意力 / 分组向量注意力 / 序列化窗口线性注意力 | O(NK) / O(N) |
| **S5 掩码预训练** | 掩码点建模/自编码重建自监督，下游微调 | O(N²) |
| **S6 混合/SSM** | 点云状态空间模型（Mamba/RWKV）、注意力-卷积混合 | O(N) |

---

## 二、通用 3D 点云方法分族综述表

> 事件适配性判断的共性依据：4D 输入 (t,x,y,p)（t 为时序维度）、事件密度非均匀、点量级远大于静态点云、无大规模事件预训练语料（from-scratch）、对远程/全局依赖的建模需求。

| 家族 | 方法 | 会议/年份 | 核心机制一句话 | 层级聚合方式 | 复杂度 | 事件适配性（理由） | arXiv | 本地 PDF |
|---|---|---|---|---|---|---|---|---|
| S1 | PointNet | CVPR 2017 | 逐点 MLP + T-net 对齐 + 全局 max-pool 对称聚合 | 无层级，逐点变换→全局 max-pool | O(N) | **中**：max-pool 契合无序、O(N) 抗海量点；但无局部结构、t 无显式建模、全局池化压扁时序 | 1612.00593 | PointNet_2017.pdf |
| S1 | PointNet++ | NeurIPS 2017【待验证：摘要页无 Comments，通行标注】 | FPS + 球查询分组 + PointNet 聚合，密度自适应 | 层级集合抽象 SA，全局池化 | O(NK) | **高**：层级局部性契合事件；t 可并入 4D 坐标做 FPS（事件侧 D-FPS 即加可学习缩放）；密度自适应抗非均匀；短板：无全局时序建模 | 1706.02413 | PointNet++_2017.pdf |
| S1 | 张量化点云层（实例 TTPOINT） | ACM MM 2023（CCF-B） | TT 张量分解压缩层级点云权重 | PointNet++ 式层级 + TT 核 | O(NK) | **高（边缘部署）**：张量化降本可直接迁移事件实时推理 | 2308.09993 | 库内已有 |
| S2 | ConvPoint | Computers & Graphics 2020（非CCF-A，参考） | 以点坐标为输入的空间连续核 + 密度加权 | 层级下采样 + 连续核卷积 | O(NK) | **中**：连续核支持任意点数、密度加权抗非均匀；t 需并入、对稀疏运动边缘适配一般、无全局时序 | 1904.02375 | ConvPoint_2020.pdf |
| S2 | KPConv | ICCV 2019 | 可变形核点位于欧氏空间并学偏移，邻域求和 | grid/随机下采样 + 可变形核点卷积 | O(NK) | **中低**：可变形核自适应局部边缘；但面向静态扫描无原生 t、grid 在事件密度波动下效率差、计算偏重不抗海量点 | 1904.08889 | KPConv_2019.pdf |
| S2 | PAConv | CVPR 2021 | Weight Bank + ScoreNet 按点位置动态组装核 | 嵌入 PointNet++ 式 MLP 管线 | O(NK) | **中**：动态核可学 4D 位置核、易嵌入骨架；仍局部卷积、无全局时序 | 2103.14635 | PAConv_2021.pdf |
| S2 | PointConv（参考） | CVPR 2019 | Monte Carlo 连续卷积，weight(MLP)+密度(KDE) | 层级下采样 + 连续卷积/反卷积 | O(NK) | **中**：密度重加权抗事件非均匀；无时序 | 1811.07246 | PointConv_2019.pdf |
| S3 | DGCNN/EdgeConv | ACM TOG (SIGGRAPH) 2019【待验证：摘要页无 Comments；P1 初标 ICCV2019 存疑，通行标注为 TOG】 | 每层特征空间动态 kNN 图，边特征 e_ij=h(xi,xi-xj) | 动态图(kNN)重建→边卷积→堆叠→全局池化 | O(NK) | **中高**：动态图可把时序近邻/相似事件连边、间接建时序；堆叠可扩远程语义；短板：每层 kNN 重建开销大、t 仅当特征故时序隐式 | 1801.07829 | DGCNN_2019.pdf |
| S3 | GAT（及点云图变体） | ICLR 2018 | 图 mask 自注意力，邻域加权、不依赖先验图 | kNN 建图→消息传递+注意力 | O(NK) | **中**：注意力可加权近邻突出运动边缘；面向通用图、点云需先建图、无原生 4D 时序 | 1710.10903 | GAT_2018.pdf |
| S4 | Point Transformer v1 | ICCV 2021 | 邻域(kNN)内向量注意力（含相对位置编码） | kNN 邻域局部向量注意力 + 下采样 | O(NK) | **中**：局部注意力可建模邻域时空、相对位置可并入 t；无远程依赖、kNN 开销大、from-scratch 在事件小数据易过拟合 | 2012.09164 | PointTransformer_2021.pdf |
| S4 | Point Transformer V2 | NeurIPS 2022 | Grouped Vector Attention + partition-based pooling | 分组向量注意力 + 分区池化 | O(NK) | **中**：GVA 降参数、partition 池化抗非均匀；仍局部注意力、依赖大规模预训练（事件无此语料） | 2210.05666 | PointTransformerV2_2022.pdf |
| S4 | Point Transformer V3 | CVPR 2024 | 序列化大点云 + 窗口线性注意力，感受野 16→1024 | 点序列化(空间填充)→窗口/块线性注意力 | O(N) 近似 | **中高**：序列化+线性注意力抗海量点、大感受野捕获长时空依赖；短板：空间填充打乱事件异步 t、需大规模预训练。**思想可迁移：序列化+线性注意力，t 作排序键/时序通道** | 2312.10035 | PointTransformerV3_2024.pdf |
| S5 | Point-BERT | CVPR 2022 | 分 patch + dVAE tokenizer + 掩码恢复离散 token | patch→掩码→Backbone→恢复 | O(N²) | **低（范式可借）**：掩码自监督缓解事件无标注（库内 2212.10368 Masked Event Modeling 即事件版）；依赖大规模点云语料（事件无语料）、重建目标对异步时序意义弱 | 2111.14819 | PointBERT_2022.pdf |
| S5 | Point-MAE | ECCV 2022【待验证：摘要页无 Comments，通行标注】 | 高比例掩码 patch + 非对称 Transformer 自编码重建坐标(CD) | patch 高掩码→轻量解码器重建 | O(N²) | **低**：高掩码自编码可学通用表征；重建坐标对 4D 时序不直接、无事件语料 | 2203.06604 | PointMAE_2022.pdf |
| S5 | Point-M2AE | NeurIPS 2022 | 多尺度金字塔掩码 + 层次自编码 + 跳连 | 金字塔多尺度掩码→解码器上采+跳连 | O(N²) | **低**：多尺度层级最接近事件层级；同缺事件语料与时序目标 | 2205.14401 | PointM2AE_2022.pdf |
| S5 | PointGPT | NeurIPS 2023【待验证：Comments 仅"9p2fig"，据 Scholar/作者页】 | 自回归生成预训练，空间近邻排 patch + decoder 预测下一 patch | patch 有序化→自回归 decoder | O(N²) | **低-中**：自回归序列思想与事件流时序天然呼应；patch 按空间近邻而非时序排序、缺事件语料 | 2305.11487 | PointGPT_2023.pdf |
| S6 | PointMamba | NeurIPS 2024 | 空间填充曲线 tokenization + 非层级 vanilla Mamba，线性全局建模 | 点序列化→单向 Mamba 扫描（无层级） | O(N) | **高**：SSM 线性+全局状态契合事件海量点+长时序；Mamba 选择性扫描天然适配异步时序（EventMamba 已实证）；短板：空间填充排序与事件时序不一致，t 需作扫描序/通道；from-scratch 可行 | 2402.10739 | PointMamba_2024.pdf |
| S6 | PointRWKV | arXiv 2024【会议待验证（疑 ICLR2025）】 | RWKV 线性序列（多头矩阵值状态+递归）并行固定半径近邻图分支 | 多尺度层级 + RWKV(全局)∥近邻图(局部) | O(N) | **高**：线性递归兼顾全局时序扫描与局部几何、抗海量点、from-scratch 友好 | 2405.15214 | PointRWKV_2024.pdf |
| S6 | HydraMamba（库内） | arXiv 2025（库内 2507.19778） | 点云全局 SSM | 层级 + 全局 SSM | O(N) | **高**（事件侧 Mamba 族已实证：EventMamba/PASS 等） | 2507.19778 | 库内已有 |
| S6 | 注意力-卷积混合（PoinTramba 等） | arXiv 2024 起【待验证，未下载】 | Transformer 分析 × Mamba 效率双向排序混合 | 混合编解码 | O(N)~O(NK) | **中-高**：局部与全局线性 SSM 权衡，适配事件时空；具体选型按任务 | — | 无 |

---

## 三、事件点云 4 真身方法论定位（读原文架构节，页码级依据见子产物）

| 方法 | 家族归属 | 核心聚合机制 | 复杂度特征 | 数据集强弱 | 机制缺口 | arXiv | 库内 PDF |
|---|---|---|---|---|---|---|---|
| **SECNet** | **S1 主**＋频域全局＋组内轻注意力 | Event Cloud(x,y,t,p)；m 级迭代 G&S→SFA(空间频域)→AGG(组内注意力)→TFA(时间频域长程)→RES | D-FPS 每级点数减半；EF-KNN 按特征距离；FFT 全局滤波降 MAC | 对象分类最全（N-Cal 82.4 / C10 75.7 / N-MN 99.7 / N-CAR 94.7 / ASL 99.93）；动作 GES 98.9 / DVS12 99.65 / UCF 91.6；**不做 HMDB51** | 全局=频域滤波（非几何）；无显式 Local/Remote 配对；全欧氏；无超几何 | 2412.20803 | 2412.20803.pdf |
| **EventMamba** | **S6 主**（Mamba/SSM）＋**S1 次**（FPS+KNN 骨干） | 滑窗切分→下采样 N=1024/2048→伪点云[x,y,ts] 按 t 严排；m 级 LocalFE(FPS+KNN+ResBlock)→TAG(时序注意力)→GlobalFE(Mamba 选扫) | 每级 FPS+KNN；Mamba 选扫 O(N)；极轻 0.29–0.92M；滑窗需投票 | 动作强（GES 99.2 / DVS12 99.1 / DACT 89.1；UCF 同协议 90.3、HMDB 同协议 60.4）；**不做对象分类** | 远程=1D 时序顺序扫描（非空间配对）；无空间区域级 Local/Remote 分组；全欧氏；无超几何 | 2405.06116 | 2405.06116.pdf |
| **SpikePoint** | **S1**（单级扁平，PointNet 式）——**明确反层级** | 伪点云(x,y 显式/t 隐式)；单级 FPS 质心+KNN 成员；相对坐标+质心；率编码成脉冲；独立质心分支提全局 | 单级无多级下采样（刻意区别于 ANN 层级）；率编码二值；极轻 0.16–0.58M | 动作 GES 98.74 / DVS12 97.92 / DACT 90.6；UCF 68.46 / HMD 55.6（弱）；**不做对象分类** | 无层级/多尺度；全局仅质心分支；无远程几何关系；全欧氏；无超几何 | 2310.07189 | 库内已有 |
| **TTPOINT** | **S1**（层级集合）＋**张量化**（TT 分解） | 4D 事件点云+时序子窗采样(保密度)；层级；Tensor-Train 分解替代 MLP/卷积；max-pool 聚合 | TT 压缩权重（0.33M）；子窗时序采样；max-pool | 动作 GES 98.8 / DVS12 99.1 / DACT 92.7；UCF 72.5 / HMD 56.9；**不做对象分类** | 全局靠 max-pool（丢远程细节）；TT 是低秩压缩非几何；无远程配对；全欧氏；无超几何 | 2308.09993 | 库内已有 |

**机制缺口矩阵（远程几何维度集体空白）**

| 机制维度 | SECNet | EventMamba | SpikePoint | TTPOINT |
|---|---|---|---|---|
| 局部分组 | ✅ EF-KNN | ✅ FPS+KNN | ✅ FPS+KNN（单级） | ✅ 层级 |
| 层级/多尺度 | ✅ m 级 | ✅ m 级 | ❌ 单级 | ✅ 层级 |
| 全局/远程建模 | 频域滤波（非几何） | 1D 时序 SSM 扫描 | 质心分支 | max-pool |
| 显式 Local/Remote 配对 | ❌ | ❌ | ❌ | ❌ |
| 空间远程区域级关联 | ❌ | ❌（仅时序顺序） | ❌ | ❌ |
| 双曲/非欧几何 | ❌ | ❌ | ❌ | ❌ |

**一句话版图位置**：四者全部立在 S1「层级集合抽象」地基上——SpikePoint 是单级扁平特例、TTPOINT 是张量化特例、EventMamba 在其上叠 S6-Mamba 时序全局、SECNet 在其上叠频域全局；**它们的"全局"无一例外是非几何算子**，没有任何一个做显式 Local+Remote 组几何配对，也没有一个进双曲非欧空间。

---

## 四、双曲几何方法调研

### 重要勘误（必读）
**HGCN 的正确 arXiv ID = 1910.12933（Chami et al., NeurIPS 2019）**。1909.12233 实际是 *Deep Ensemble Learning for News Stance Detection*（cs.CL，与双曲无关），related work 引用时禁止使用该 ID；1910.12111 是 metasurface 深度相机论文，同样不是 HGCN。

### 4.1 双曲空间模型与基础方法

| 方法 | 会议/年份 | 空间模型 | 核心机制一句话 | 典型应用 | arXiv | 本地 PDF |
|---|---|---|---|---|---|---|
| Poincaré Embeddings | NeurIPS 2017 (A) | Poincaré 球 | 符号/层次数据嵌入 Poincaré 球，黎曼优化同时编码层级与相似度，半径≈层级深度 | 词汇/图谱层次 | 1705.08039 | PoincareEmbeddings_2017.pdf |
| HNN | NeurIPS 2018 (A) | Poincaré 球 | Möbius gyrovector 推导双曲线性层/前馈/GRU/多分类逻辑回归 | 句子嵌入、文本蕴含 | 1805.09112 | HNN_2018.pdf |
| HGCN | NeurIPS 2019 (A) | Lorentz/hyperboloid | 双曲面上推导 GCN 卷积，欧氏→双曲指数映射，**每层可学习曲率** | 层次/无标度图链接预测、节点分类 | **1910.12933**（勘误见上） | HGCN_2019.pdf |
| HGNN-Liu | NeurIPS 2019 (A) | Lorentz | 可微 exp/log 映射在黎曼流形做 GNN；Lorentz 模型避免 Poincaré 除法的数值不稳定 | 分子性质、图结构 | 1910.12892 | HGNN_Liu_2019.pdf |
| H2H-GCN | CVPR 2021 Oral (A) | Poincaré 球 | 直接在双曲流形卷积（不落切空间）：正交约束双曲线性 + Einstein midpoint 聚合 | 链接预测、节点/图分类 | 2104.06942 | H2HGCN_2021.pdf |
| ACE-HGNN | ICDM 2021 (B) | 通用 | 多智能体 RL（Nash Q-learning）联合探索最优曲率与节点表征 | 图节点分类 | 2110.07888 | ACE-HGNN_2021.pdf |
| Hyperbolic ViT | CVPR 2022 (A)（已亲自核对 arXiv Comments） | Poincaré 球 | ViT 输出映射双曲空间，改进 pairwise cross-entropy 直接在双曲优化 | 图像度量学习 | 2203.10833 | HyperbolicViT_2022.pdf |
| MERU | ICML 2023 (A)（已亲自核对 arXiv Comments） | Poincaré 球 | CLIP 对比学习→双曲，entailment cone 建模"文本蕴含图像"层次偏序 | 图文跨模态层次 | 2304.09172 | MERU_2023.pdf |
| Curvature-Aware Opt | arXiv 2024 v4【待验证会议】 | Poincaré 球 | Riemannian AdamW + 可微调双曲缩放约束 + 曲率感知学习率，解决可学习曲率训练不稳定 | 视觉分类、EEG、层次度量 | 2405.13979 | CurvatureAwareOpt_2024.pdf |

### 4.2 双曲 × 3D 点云（共 5 篇，前 4 篇全为静态点云）

| 方法 | 会议/年份 | 模型 | 任务 | arXiv | 本地 PDF |
|---|---|---|---|---|---|
| HyperIPC | IROS 2024 (B) | Poincaré | 图像-点云双曲对比，ScanObjectNN 分类/少样本 | 2409.15810 | HyperIPC_2024.pdf |
| Hyperbolic Point Cloud Recon | AAAI 2025 (A) | Poincaré | 双曲 Chamfer 距离 + triplet，单 RGB-D→点云重建 | 2412.09055 | HyperbolicPointCloudRecon_2025.pdf |
| Hyperbolic Contrastive 3D Point Cloud | arXiv 2025-01【待验证会议】 | Poincaré | 文本+图像+点云双曲多模态对比预训练 | 2501.02285 | HyperbolicPointCloudCL_2025.pdf |
| HIPO | CVPR 2025 (A) | Poincaré | FSCIL 点云分割，双曲理想原型+不确定性 | CVF 开放获取页【arXiv ID 未定位】 | 未下载（无 arXiv ID） |
| **EHGCN** | arXiv v4 2026-07（未发表、无代码） | 欧氏-双曲双空间 | **事件流**检测/识别（非分类基准） | 2504.16616 | 2504.16616.pdf（库内已有） |

### 4.3 事件相机 × 双曲现状复查（截至 2026-10-09）
检索 `hyperbolic event camera DVS 2026`、`hyperbolic event-based vision`、`hyperbolic neuromorphic` 均未发现新工作。**EHGCN（2504.16616）仍为唯一**：v4 更新于 2026-07-30，无会议标注、无代码，走检测路线。**分类基准上双曲事件点云方法 = 空白，新颖性声明成立。**

### 4.4 "双曲远程聚合"创新点支撑材料要点
1. **双曲半径 ≈ 层级/不确定性**：Poincaré 球中点到原点径向距离天然编码层级深度（Nickel 2017）；HIPO（CVPR 2025）把双曲距离用作类原型不确定性度量——直接支撑事件点云远距离类间边界建模。
2. **树状几何对远程依赖的意义**：双曲体积随半径指数增长（欧氏为多项式），树状结构中远距节点的距离代价远低于欧氏。事件点云同一物体的事件跨数十毫秒、跨大空间范围，欧氏局部邻域聚合无法跨远距离关联；EHGCN 摘要亦明确指出欧氏 pairwise connectivity "struggle to capture long-range dependencies"。
3. **聚合代价与稳定性**：Poincaré 模型每次聚合需 exp/log 映射，深网堆叠数值误差累积；Lorentz 模型避免除法更稳定（HGNN-Liu 2019）；边界附近需 clamping 防发散；Bdeir 2024 专门推导 Riemannian AdamW 解决可学习曲率训练不稳定。**建议远程聚合层选 Lorentz 模型或加 boundary clamping + 曲率感知优化器。**
4. **曲率可学习**：HGCN 首倡每层可学习曲率，ACE-HGNN 用 RL 自适应。建议远程聚合层曲率设为可学习参数：浅层高曲率（强层次）、深层低曲率（平滑聚合）。

---

## 五、对我们的启发（Local+Remote 分组器 + 双曲远程聚合在方法版图上的位置）

1. **方法版图定位**：六族中事件点云只占 S1（共同骨架）+ S6 局部（仅 EventMamba）；S3 图的远程几何关联能力与 S4 Transformer 的远程注意力能力均未被事件点云用作骨架（只作对比基线）；S5 因无事件预训练语料不可直接采用（但自监督范式可借——库内已有 2212.10368 Masked Event Modeling 事件版自监督先例）。
2. **Local+Remote 分组器填补的空白**：四个真身的"全局"全部是非几何算子（max-pool / 频域滤波 / 1D 时序扫描），没有任何一个做"Local 组 + Remote 组"的显式空间几何配对。这正是 S3 图方法"远程几何关联"思想在事件点云上的未应用点：把点云划分为局部密集组与远程关联组，再做跨组几何聚合。
3. **双曲远程聚合填补的空白（两重）**：双曲几何在通用点云上只有对比学习（HyperIPC/CL）、重建（Recon）、增量分割（HIPO）等应用，**没有"双曲远程聚合算子"用于分类**；事件侧更是零工作（EHGCN 检测路线）。双曲优势（半径≈层级、指数体积→远程距离压缩、曲率可学习）直接支撑"远程聚合"——远程点在欧氏距离远、在双曲距离近，正是事件点云跨时空关联所需要的。
4. **复杂度定位**：目标应为 O(NK) 级局部（Local 组）+ O(N) 级远程（双曲聚合或 SSM 序），避开 O(N²) 全局注意力——事件海量点下 O(N²) 不可行。
5. **related work 组织建议（三段落）**：第一段 S1 层级集合骨架族（四真身共同地基，重点对照改进）；第二段 S6 时序建模族（EventMamba/PointMamba，时序远程的现有解）；第三段双曲族（从 Poincaré/HGCN 到双曲×点云，论证"双曲远程聚合"空白——EHGCN 是唯一事件对照且走检测、未发表）。

---

## 六、待验证清单

1. DGCNN 会议标注：通行标注 ACM TOG(SIGGRAPH) 2019，P1 初标 ICCV2019 存疑（摘要页无 Comments）
2. PointNet++（NeurIPS2017）、Point-MAE（ECCV2022）、PointGPT（NeurIPS2023）：摘要页 Comments 空，按通行标注
3. PointRWKV 正式会议（疑 ICLR2025，未确认）
4. CurvatureAwareOpt（2405.13979）、HyperbolicPointCloudCL（2501.02285）正式会议
5. HIPO（CVPR 2025）的 arXiv ID 未定位（有 CVF 开放获取页）
6. PoinTramba 等注意力-卷积混合方法（未下载）
7. EHGCN 代码发布状态（截至复查未见）
8. SECNet 会议届次：按本方已确认事实记为 ICML（子代理标注 ICML 2026 Oral 未独立复核）

---

## 七、来源链接与本地 PDF 清单

### 本轮新增下载（D:\论文，29 篇全部独立校验 %PDF 开头且 >50KB）
**通用点云 17 篇**：PointNet_2017.pdf、PointNet++_2017.pdf、ConvPoint_2020.pdf、KPConv_2019.pdf、PAConv_2021.pdf、PointConv_2019.pdf、DGCNN_2019.pdf、GAT_2018.pdf、PointTransformer_2021.pdf、PointTransformerV2_2022.pdf、PointTransformerV3_2024.pdf、PointBERT_2022.pdf、PointMAE_2022.pdf、PointM2AE_2022.pdf、PointGPT_2023.pdf、PointMamba_2024.pdf、PointRWKV_2024.pdf
**双曲 12 篇**：PoincareEmbeddings_2017.pdf、HNN_2018.pdf、HGCN_2019.pdf、HGNN_Liu_2019.pdf、H2HGCN_2021.pdf、ACE-HGNN_2021.pdf、HyperbolicViT_2022.pdf、MERU_2023.pdf、CurvatureAwareOpt_2024.pdf、HyperIPC_2024.pdf、HyperbolicPointCloudRecon_2025.pdf、HyperbolicPointCloudCL_2025.pdf
**库内已有未重复下载**：EHGCN（2504.16616.pdf）、TTPOINT（2308.09993）、SpikePoint（2310.07189）、EventMamba（2405.06116）、SECNet（2412.20803）、HydraMamba（2507.19778）
**下载失败**：0 篇；库内已有文件未删除/未重命名/未修改

### 本轮主要来源链接（arXiv ID）
通用：1612.00593、1706.02413、1904.02375、1904.08889、2103.14635、1811.07246、1801.07829、1710.10903、2012.09164、2210.05666、2312.10035、2111.14819、2203.06604、2205.14401、2305.11487、2402.10739、2405.15214
双曲：1705.08039、1805.09112、**1910.12933（HGCN，勘误后正确 ID）**、1910.12892、2104.06942、2110.07888、2203.10833、2304.09172、2405.13979、2409.15810、2412.09055、2501.02285、2504.16616

### 子调研产物（分片原始记录）
- sectionD_pointcloud_families.md（通用分族）、sectionB2_methodology.md（4 真身定位）、sectionE_hyperbolic.md（双曲）

---

## 下一步建议

1. related work 按"三段落"组织（S1 骨架 → S6 时序 → 双曲空白），三份子产物可直接引用
2. 架构设计锚点：Local 组沿用 S1 式 FPS+KNN（已验证）、Remote 组做双曲远程聚合（Lorentz 或 Poincaré+clamping、曲率可学习），复杂度目标 O(NK)+O(N)
3. 投稿前复查：待验证清单 1-8 项（尤其 DGCNN 会议标注、PointRWKV/CurvatureAwareOpt 接收状态、双曲方向新 arXiv）

# N-Caltech101 分类赛道 SOTA 调研报告

> 调研时间：2026-10-06
> 调研人：豆包
> 数据集：N-Caltech101（Orchard et al. 2015，101类，8709样本，事件相机）

## 1. 核心结论

| 赛道 | SOTA 方法 | 准确率 | 参数量 | 表示类型 |
|---|---|---|---|---|
| **整体 SOTA** | PASS (NeurIPS 2025) | **94.60%** | 7M | 稠密（Around events） |
| 稠密表示 | EventDance | 92.35% | 26M | Voxel/EST/Frame |
| 稠密表示 | MVF-Net | 93.9% | - | Event Frame |
| 稠密表示 | Generative Event Pretraining | 93.05% | - | 基础模型对齐 |
| Graph-based | AW-GATCN (2025) | 83.77% | - | 图表示 |
| Graph-based | EDGCN | 83.50% | 0.77M | 图表示 |
| **Point-based** | **SECNet** | **~82.4%** | 1.1M | 事件云 |
| Point-based | EventMamba | ~? | 3.28M | 事件云 |
| Point-based | TT-Point | 72.5% | 0.36M | 事件云 |
| SNN-based | STCA-SNN | 80.88% | - | 脉冲 |

**关键洞察：Point-based 赛道与稠密表示存在 ~12% 的巨大差距！这正是我们的机会窗口。**

## 2. 整体 SOTA 详解

### 2.1 PASS (NeurIPS 2025) — 94.60%
- **论文**：PASS: Path-selective State Space Model for Event-based Recognition
- **方法**：Path-selective SSM（状态空间模型），Mamba 类架构
- **配置**：PASS-M-K(2), 7M params
- **表示**：Around events（稠密表示，Voxel/EST 类）
- **对比**：
  - EventDance: 92.35% (26M)
  - S5-ViT-M-K(2): 88.44% (17.5M)
  - PASS-T-K(1): 88.29% (7M)
  - PASS-T-K(2): 89.72% (7M)
- **其他数据集**：N-Imagenet 61.32%, PAF 98.28%
- **来源**：NeurIPS 2025, arXiv 2409.16953

### 2.2 EventDance — 92.35%
- **方法**：多表示融合（stack images + voxel grids + EST）
- **参数量**：26M
- **表示**：稠密表示（多种融合）
- **变体**：EventDance++（语言引导无监督跨模态适应）

### 2.3 MVF-Net — 93.9%
- **论文**：Event Stream based Human Action Recognition (2024)
- **表示**：Event Frame（稠密）
- **对比**：EST 89.7%, EV-VGCNN 83.2%

### 2.4 Generative Event Pretraining — 93.05% (2026)
- **方法**：生成式事件预训练 + 基础模型对齐
- **对比**：ECDP 87.66% → 93.05% (+5.39%)
- **表示**：稠密表示

## 3. Point-based 赛道详解（我们的方向）

### 3.1 SECNet — ~82.4%（我们的基线）
- **论文**：Scalable Event Cloud Network for Event-based Classification (arXiv 2412.20803)
- **方法**：傅里叶频域 + 局部分组（LocalGrouper）+ 空间/时间 FFT
- **参数量**：1.126M
- **表示**：Event Cloud（Point-based，标记为 EC）
- **N-Caltech101 配置**：PointNumber=8192, Dimension=64, Group=2048
- **其他数据集**：DVSGesture 98.9%, DailyDVS 99.65%, UCF101-DVS 91.6%

### 3.2 EventMamba — 待确认 N-Caltech101 数值
- **论文**：EventMamba (Ren et al., 2025)
- **方法**：Mamba (SSM) + HSFC 空间填充曲线 + RWO 随机窗口偏移
- **参数量**：3.28M
- **表示**：Event Cloud（Point-based，标记为 P）
- **UCF101-DVS**：90.3%
- **N-Caltech101**：需查原文确认

### 3.3 TT-Point — 72.5%
- **参数量**：0.357M
- **表示**：Point-based
- **性能较低，早期方法**

### 3.4 SpikePoint — 待确认
- **方法**：Point-based + SNN 脉冲网络
- **特点**：极低功耗，直接处理原始事件点
- **N-Caltech101**：需查原文确认

## 4. Graph-based 赛道（双曲空间已有工作）

### 4.1 EHGCN — 双曲图卷积
- **论文**：EHGCN: Hierarchical Euclidean-Hyperbolic Fusion via Motion-Aware GCN (arXiv 2504.16616)
- **方法**：欧氏-双曲融合 GCN，双曲空间建模层级拓扑
- **N-Cars**：96.9%, **DVS128**：99.3%
- **N-Caltech101**：需查原文确认
- **局限**：必须显式建图（KNN），事件点多时开销大

### 4.2 AW-GATCN — 83.77% (2025)
- **方法**：自适应加权图注意力卷积网络
- **表示**：Graph
- **其他数据集**：CIFAR10-DVS 76.79%, MNIST-DVS 99.3%, N-CARS 96.89%

### 4.3 EDGCN — 83.50%
- **参数量**：0.77M
- **表示**：Graph

## 5. SNN-based 赛道

| 方法 | N-Caltech101 | 特点 |
|---|---|---|
| STCA-SNN | 80.88% | 时空通道联合注意力 |
| TA-SNN | 80.76% | 时间注意力 |
| Evidential SNN | 77.20% | 证据深度学习 |
| Dspike | ~75% | ResNet18 SNN |

## 6. 我们的机会分析

### 6.1 差距来源
Point-based (~82%) vs 稠密表示 (~90%+) 的 ~8-12% 差距来自：
1. **层级拓扑建模不足**：Point-based 方法（SECNet/EventMamba）只做局部时空特征+时序依赖，没有显式建模部件-整体层级拓扑
2. **欧氏空间拥挤**：层级结构在欧氏空间嵌入会发生节点拥挤，远距离层级关联区分不开
3. **稠密表示的优势**：Voxel/Frame 天然保留全局空间结构，CNN 能提取层级特征

### 6.2 我们的方案：Point-based + 双曲空间
- **避免建图开销**：保持 Point-based 范式，不做 KNN 建图
- **双曲空间建模层级**：在局部分组后，把组内点特征映射到双曲空间，用双曲距离计算注意力，在双曲空间做层级聚合
- **目标**：在 Point-based 赛道提升 ~3-5%，缩小与稠密表示的差距
- **创新点**：首次将双曲空间引入 Point-based 事件分类（EHGCN 是 Graph-based，必须建图）

### 6.3 预期目标
| 模型 | 预期准确率 | 说明 |
|---|---|---|
| SECNet 基线 | ~82.4% | 论文报告值 |
| HFECNet（我们的） | ~85-87% | 双曲聚合提升 ~3-5% |
| 稠密表示 SOTA | 94.60% | 上界参考 |

## 7. 参考文献

1. PASS: Path-selective State Space Model for Event-based Recognition. NeurIPS 2025. arXiv:2409.16953
2. EventDance: arXiv:2409.12778 (EventDance++)
3. SECNet: Scalable Event Cloud Network for Event-based Classification. arXiv:2412.20803
4. EHGCN: Hierarchical Euclidean-Hyperbolic Fusion via Motion-Aware GCN. arXiv:2504.16616
5. AW-GATCN: Adaptive Weighted Graph Attention Convolutional Network. arXiv:2505.11232
6. N-Caltech101 原始数据集: Orchard et al. 2015
7. MVF-Net: Event Stream based Human Action Recognition. arXiv:2408.09764
8. Generative Event Pretraining: arXiv:2603.23032
9. MEM: Masked Event Modeling. arXiv:2212.10368
10. RBE: Retain, Blend, and Exchange. arXiv:2406.18845

## 8. 待确认项

- [ ] EventMamba 在 N-Caltech101 上的具体准确率（需查原文）
- [ ] SpikePoint 在 N-Caltech101 上的具体准确率
- [ ] EHGCN 在 N-Caltech101 上的具体准确率
- [ ] SECNet 论文中 N-Caltech101 的准确数值（0.824 需确认）
- [ ] 2026 年最新的 Point-based 方法（可能有更新的工作）

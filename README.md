# SECNet × 双曲空间 — 事件相机分类项目

> 基于 rp1（SECNet/FECNet + 局部/远程邻域）加入双曲空间建模，探索事件点云的层级运动拓扑。

## 目录结构

```
secnet-hyperbolic/
├── papers/                          # 核心论文（7篇）
│   ├── 01-ECAF-GCN-双曲事件点云检测.pdf      # 核心idea来源：跨空间注意力+双曲GCN+YOLOX检测
│   ├── 02-SECNet-傅里叶频域Point分类.pdf      # rp1基线：FECNet，FFT频域+事件分组采样
│   ├── 03-EHGCN-双曲图卷积事件分类.pdf         # 双曲GCN事件分类（Graph-based，可扩展性差）
│   ├── 04-EventMamba-PointMamba基线.pdf        # Point-based+Mamba长时序基线
│   ├── 05-SpikePoint-PointSNN基线.pdf          # Point-based+SNN低功耗基线
│   ├── 06-事件相机综述.pdf                       # 事件相机数据表示与方法综述
│   └── 07-PoincareEmbeddings-双曲嵌入基础.pdf   # 双曲几何基础（Nickel 2018，庞加莱嵌入）
│
├── code/
│   ├── secnet-official/              # SECNet官方代码（ICML 2026）
│   │   ├── train_frequency.py        # 训练入口（AdamW, CosineAnnealing, label smoothing）
│   │   ├── provider_data.py          # H5数据加载
│   │   └── models/
│   │       ├── FECNet.py             # 核心模型：Embed→LocalGrouper→空间FFT→注意力→时间FFT→分类
│   │       └── modules.py            # LocalGrouper（FPS+KNN分组）、工具函数
│   │
│   ├── rp1/                          # rp1项目资产（局部+稀疏远程邻域，iter3-iter9）
│   │   ├── README.md                 # rp1说明
│   │   ├── settings/                 # 数据配置、资产地图、实验记录
│   │   └── runs/                     # 11轮实验结果（oracle/sparse_remote/global_plus/boundary_rerank等）
│   │
│   └── hyperbolic-utils/             # 双曲几何工具（从ECAF-GCN复现工程复用）
│       ├── hyperbolic.py             # 庞加莱圆盘全套：指数映射/对数映射/双曲GCN/曲率学习
│       ├── models.py                 # ECAFGCN完整模型（双分支+跨空间注意力+YOLOX头）
│       └── spline_conv.py            # 纯PyTorch B-spline卷积（替代C++ CUDA kernel）
│
├── docs/                            # 研究文档
│   ├── 01-ECAF-GCN解读与竞争格局.md       # 论文逐式解读+赛道分析
│   ├── 02-Point-based-backbone调研.md      # SpikePoint/EventMamba/SECNet/EHGCN横向对比
│   ├── 03-项目提案.md                       # 原始项目提案
│   ├── 04-项目状态页.md                     # 进度状态
│   └── 05-ECAF-GCN-paper-xray深度拆解.md   # paper-xray 15节深度拆解
│
└── README.md                         # 本文件
```

## 关键配置

### 数据（N-Caltech101）
- 格式：H5，shape `[N, 2048, 4]`，channels `[t, x, y, polarity]`
- train: 20901 samples，test: 5226 samples，101类
- 本地原始数据：`eventhyperbolic/data/N-Caltech101/`（.bin格式，需转换为H5）

### SECNet/FECNet 基线配置
- group=512, neighbors=24, base_dim=16
- 特征维度：[32, 68, 140, 284]，3个stage
- 训练：AdamW lr=1e-3, batch=80, 150 epoch, CosineAnnealing

### HPC 环境
- 跳板机：`ssh hpc`（10.160.22.46:2223, user jnguan）
- 工作目录：`~/JiangYouxin/ZhaoJin/`
- 推荐GPU：V100 16GB（4点/小时，16张空闲）
- Claude Code：v2.1.284 已安装

## 核心思路

在 rp1 的 SECNet/FECNet（局部邻域+稀疏远程邻域）基础上，将特征空间从欧氏空间扩展到**双曲空间**，利用双曲几何天然的树状层级嵌入能力，建模事件流中"局部部件→整体物体"的层级运动拓扑，解决欧氏空间节点拥挤问题。

与 EHGCN 的区别：EHGCN 是 Graph-based（必须显式建图，KNN开销大），我们保持 Point-based 范式（不建图），在 FECNet 的特征提取阶段引入双曲空间映射。

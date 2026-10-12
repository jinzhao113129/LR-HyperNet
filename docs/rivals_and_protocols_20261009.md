# 冲榜对手深挖 + 各数据集评测协议汇总（2026-10-09）

> 项目：LR-HyperNet（point-based 事件点云分类）｜用途：复现/冲榜前的对手与协议准备
> 采信范围：CCF-A 优先；所有分数可溯源（页码级）；不确定标【待验证】；具体数值
> 前提（上轮已锁定）：CIFAR10-DVS point 75.7% vs 全范式 87.35%；DailyDVS-200 top-1 51.99% point 空白；HMDB51-DVS point 55.6–60.4%；UCF101-DVS point 91.6%

---

## 一句话核心结论

**冲榜协议已全部锁定：CIFAR10-DVS / N-Caltech101 / UCF101-DVS 直接套 SECNet 协议（C10=10240 点/100ms 窗、N-Cal=8192 点/30ms 短窗+序列级投票、UCF=8192 点/1s 窗+按原视频文件名分组 80/20），HMDB51-DVS 套 EventMamba 文件名分组协议（0.5s 窗/0.25s 重叠/2048 点，对标 60.4%），DailyDVS-200 套数据集原文跨被试 31/8/9 划分（top-1+top-5，对标 51.99%）——且复现 SECNet 82.4% 的决定性因素是"30ms 短窗+多窗投票"而非点数（8192→10240 只差平台期 0.5 个点）。**

---

## 一、威胁方法 × 关键细节表（读原文，页码级）

| 方法 | 目标分数+出处 | 网络结构一句话 | 核心创新 | 训练协议 | 参数量 | arXiv | 库内 PDF |
|---|---|---|---|---|---|---|---|
| **NGN** | CIFAR10-DVS **87.35±0.33%**（T=10），p11 Table 3；N-Caltech101 **84.04%**（T=10） | 事件帧（T=10 分箱）→VGG-SNN 骨干（K=16 群体噪声 LIF 替换神经元）→TET 损失→线性头 | K 个带噪声 LIF 群体平均产生梯度脉冲+群体级同步重置，缓解时空信息丢失与前后向梯度不匹配 | optimizer/lr/batch/epoch/wd **未报告**（正文称"同设置对比"）；epoch≈200（Table 5 注）；窗口 T=10 标准分箱 | 未报告 | 2608.17394 | 2608.17394v1.pdf |
| **MD-Mixer+SDT-V2** | HMDB51-DVS **62.8%** / UCF101-DVS **66.3%**，p7 Table 3（哈希比对确认） | 事件帧（CIFAR/N-Cal 为 T=16）→SDT-V2 脉冲 Transformer，K/V 经 MD-Mixer 多延迟时间混合→脉冲自注意力→头 | 脉冲 Transformer 中引入多支路可学习时间延迟做 K/V 时间混合，补纯空间注意力缺时间交互；soft-to-hard 延迟退火 | 训练超参**未报告**（正文称见 Supplementary，但 Supp 未随主文）；增强继承 SDT-V1 策略；UCF/HMDB 的 T 未标注【待验证】 | SDT-V2-8-512 **55.4M**（ImageNet 配置；UCF/HMDB 配置未单列） | CVPR 2026（UESTC） | MultiDelayMixer_CVPR2026_脉冲Transformer时间交互.pdf |
| **EIS-HAR** | DailyDVS-200 **top-1 51.99% / top-5 78.13%**；HARDVS **53.21%/60.45%**，p7 Table 2 | 运动补偿事件帧（3通道）→IGS 贪心选 K 关键帧→HSTS 四阶段混合 Swin Transformer（Refine+Hybrid Block×2/stage）→GAP+Linear | IMU 驱动自适应运动补偿（AIMC）消低光+抖动模糊；IGS 贪心关键帧采样；HSTS 混合 Swin-Attention+局部时空卷积 | **Adam**，lr=**5e-4**，CosineAnnealingLR(min 1e-5)，wd=**2e-2**，batch=**20**，epoch=**250**，2×4090，隐藏维 96 | **34.0M**（p8 Table 3） | 2605.20680 / CVPR 2026（BIT） | EIS-HAR_2026.pdf |
| **EventMG** | UCF101-DVS **79.6%** / HMDB51-DVS **71.2%** / DVS-Action **93.3%** / THUE-ACT-50 **58.8%**，p9-10 Table 4；检测 Gen1 mAP 53.7 / 1Mpx 47.4 | 原始事件点(x,y,t,p)→固定 N 事件数滑窗→微级：kNN 事件图→GNN→APN 坐标扰动→MSM 多方向扫描→Mamba；宏级：连通域 Component Graph→图聚合→微宏门控融合 | 多级 Mamba-Graph：微级 Mamba 建模单点长程依赖，宏级 Component Graph 建模事件簇拓扑；APN+MSM 解决 Mamba 扫描对稀疏不规则事件的不适应 | **AdamW + OneCycleLR**；初始 lr/batch/epoch/wd **未报告**；事件窗=**固定 N=20000 事件数**（检测侧；动作侧沿用 EventMamba 固定事件数滑窗） | 动作 **0.773M** / 检测 1.96M | NeurIPS 2025（复旦） | EventMG_NeurIPS2025_多级Mamba图.pdf |

**⚠ EventMG 分数协议警告**：EventMG 的 Table 4 同表引用的 EventMamba 数字是 **97.9%/86.4%（宽松随机窗划分）**，而非文件名分组的 90.3%/60.4%——**EventMG 自己的 79.6%/71.2% 大概率也是同套宽松协议，不能与我们锁定的文件名分组标准直接比**（EventMG 未报 N-Caltech101/CIFAR10-DVS/DailyDVS-200/HARDVS）。

---

## 二、五数据集评测协议规范（可直接照做）

### CIFAR10-DVS
| 项 | 内容 |
|---|---|
| 出处 | Li et al. 2017, Frontiers in Neuroscience 11:309，DOI 10.3389/fnins.2017.00309（已下载 CIFAR10-DVS_Li2017.pdf）；10000 流/10 类×1000，128²，~1.2s |
| 划分 | 无官方划分；SECNet 按 Bi 2020/Chen 2022b **随机抽 20% 测试**（附录 D）；SNN 圈另有 90/10 惯例【待验证】 |
| 窗口/T | SECNet 滑窗 **100ms**，整录 ~1280ms 切窗，train 99965 / test 24950 窗；帧式 NGN T=10、MD-Mixer T=4–16 |
| 点数/采样 | SECNet **10240 点/窗**下采样；帧式无点 |
| 增强 | 仅去噪/下采样【待验证】 |
| 口径 | top-1，窗级 |
| 对手交叉 | NGN=87.35%（T=10 帧 SNN）；**SECNet=75.7%（10240 点）**；PointNet++=50.3% |
| **冲榜建议** | **与 SECNet 同协议**：窗 100ms、10240 点、随机 20% 测试、top-1，对标 **75.7%**（point-based 新 SOTA 门槛） |

### N-Caltech101
| 项 | 内容 |
|---|---|
| 出处 | Orchard et al. 2015, Frontiers in Neuroscience 9:437，DOI 10.3389/fnins.2015.00437（已下载 N-Caltech101_Orchard2015.pdf）；ATIS，8709 录/101 类，~0.3s |
| 划分 | 无官方；随机 80/20。**两套口径并存**：录制级（EFV++ 6932/1777）vs 窗级（SECNet 20896/5226） |
| 窗口/T | SECNet **SW=30ms**（Table 1）；但 Table 9 写"300ms sliding window"【待验证，以代码为准】。NGN T=10 |
| 点数/采样 | SECNet **8192 点/窗**；EFV++ 非点=体素 8 帧+事件图像；PointNet++ ≈1024 点 |
| 增强 | 未见强增强【待验证】 |
| 口径 | top-1（录制级或窗级两口径） |
| 对手交叉 | **SECNet=82.4%（8192 点，窗级）**；**EFV++=89.7%（体素+图像，录制级）**；NGN=84.04%（T=10）；PointNet++=50.3% |
| **冲榜建议** | **与 SECNet 同协议**：8192 点/窗、30ms 短窗+多窗投票、随机 80/20、top-1，对标 **82.4%**。EFV++ 89.7% 是录制级且非点云，不可与点数直接比 |

### HMDB51-DVS
| 项 | 内容 |
|---|---|
| 出处 | Bi et al. 2020（IEEE TIP），arXiv 1910.03579（已下载 Bi_TIP2020_UCF101_HMDB51-DVS.pdf）；DAVIS240，320×240，6766 clip/51 类 |
| 划分 | 两套：①Bi 2020 原始；②**文件名分组（EventMamba/Ren TPAMI2025）**。SECNet **不报此数据集** |
| 窗口/T | EventMamba **0.5s 窗/0.25s 重叠**，均长 8s，train 48916 / test 12230 窗 |
| 点数/采样 | EventMamba **2048 点/窗**（FPS）；EventMG 沿用固定事件数滑窗 |
| 增强 | 去噪为主【待验证】 |
| 口径 | top-1，窗级 |
| 对手交叉 | EventMamba：易 **86.4%** / 文件名分组 **60.4%**；EventMG=71.2%（协议待核实）；MD-Mixer+SDT-V2=62.8%（p7 Table 3，帧-SNN）；TTPOINT=56.9% |
| **冲榜建议** | **与 EventMamba 文件名分组同协议**：0.5s 窗/0.25s 重叠、2048 点、按原视频分组 80/20、top-1，对标 **60.4%**（point-based 新 SOTA 门槛） |

### UCF101-DVS
| 项 | 内容 |
|---|---|
| 出处 | 同 Bi et al. 2020（arXiv 1910.03579）；DAVIS240，13320 clip/101 类 |
| 划分 | 两套（同上）；SECNet Table 6 明确"adopt filename-based partitioning (Ren et al., 2025)" |
| 窗口/T | EventMamba **1.0s 窗/0.5s 重叠**，均长 6.6s，train 110050 / test 27531 窗；SECNet 同 1s 窗 |
| 点数/采样 | EventMamba **2048 点**；**SECNet 8192 点**（窗数几乎一致但点数更多） |
| 增强 | 去噪为主【待验证】 |
| 口径 | top-1，窗级 |
| 对手交叉 | EventMamba：易 **97.9%** / 文件名分组 **90.3%**；**SECNet=91.6%（文件名分组，8192 点）**；EventMG=79.6%（协议待核实）；MD-Mixer+SDT-V2=66.3%（p7 Table 3）；TTPOINT=72.5% |
| **冲榜建议** | **与 SECNet 同协议**：1.0s 窗/0.5s 重叠、**8192 点**、按原视频文件名分组 80/20、top-1，对标 **91.6%**（point-based 新 SOTA 门槛） |

### DailyDVS-200
| 项 | 内容 |
|---|---|
| 出处 | Wang et al. **ECCV 2024**，arXiv 2407.05106（已下载 DailyDVS200_2024.pdf）；DVXplorer Lite 320×240，22046 序列/200 类/47 人，1–20s |
| 划分 | **跨被试**：47 人→31 train / 8 val / 9 test（按 subject ID） |
| 窗口/T | 原文帧式基线：帧间隔 0.5s 默认（0.25s/0.125s 消融）；点云标准未建立【待验证】 |
| 点数/采样 | 点云标准未建立 |
| 增强 | 原文未强调事件增强 |
| 口径 | **top-1 + top-5 同报**（200 类） |
| 对手交叉 | **EIS-HAR=51.99%/78.13%**；Swin-T=48.06%/74.47%（原文最强基线）；EVMamba=49.65%；**SECNet/EventMG/NGN 均不报本集**（它们的"Daily DVS"是另一个 12 类旧集 Liu 2021，非本集） |
| **冲榜建议** | **与数据集原文跨被试同口径**：31/8/9 被试划分、top-1+top-5，对标 **51.99%**。与 SECNet 不同口径（SECNet 没测过本集） |

---

## 三、"标准文件名分组划分"专项（UCF101-DVS / HMDB51-DVS）

- **来源 = EventMamba，Ren et al.《Rethinking efficient and effective point-based networks》，IEEE TPAMI 2025，arXiv 2405.06116**（TTPOINT→SpikePoint→EventMamba 一脉），**不是 Bi et al. 2020**。原文 §4.3.5："we implement the dataset partitioning method based on filename, achieving **60.4% HMDB51-DVS / 90.3% UCF101-DVS**"；SECNet Table 6 脚注转引同一来源。
- **定义**：按**原 RGB 视频文件名分组**，同一视频切出的所有 clip/滑窗整体落 train 或 test，绝不跨集——堵死随机窗划分把同视频相邻窗泄漏进 test 的捷径。
- **每视频切窗**：UCF101-DVS 1.0s 窗/0.5s 重叠→110050/27531 窗；HMDB51-DVS 0.5s 窗/0.25s 重叠→48916/12230 窗；每窗下采样固定点数（EventMamba 2048 / SECNet 8192）。
- **口径落差（切勿自欺）**：随机易协议 UCF=97.9%、HMDB=86.4%；文件名分组（难、无泄漏）UCF=EventMamba 90.3% / SECNet 91.6%、HMDB=60.4%。引用 EventMG Table 4 时必须注明抄录的是哪套。

---

## 四、N-Caltech101 点数版图与双口径说明

| 方法 | 范式 | 点数/输入 | 划分 | top-1 |
|---|---|---|---|---|
| **SECNet** | 点云 | **8192 点/窗** | 随机 80/20 窗（20896/5226） | **82.4%** |
| **EFV++/RBE** | 体素+事件图像（**非点云**） | 非点=8 帧 | **录制级** 6932/1777 | **89.7%** |
| NGN | 帧式 SNN | T=10 步 | 通行 80/20 | 84.04% |
| PointNet++ | 点云 | ~1024 点 | 同 SECNet 窗 | 50.3% |
| EventMamba/EventMG | 点云 | **未测 N-Caltech101** | — | — |

**双口径**：录制级（每录一个样本）与窗级（每窗一个样本）并存——EFV++ 89.7% 是录制级、SECNet 82.4% 是窗级，跨口径对比不成立。

---

## 五、SECNet 自身复现协议核对结论（含开源代码确认）

| 项 | 论文声明 | 代码确认（github.com/rhwxmx/SECNet_ICML） |
|---|---|---|
| Optimizer | AdamW（p6） | AdamW, lr=1e-3, **wd=1e-4** |
| lr / schedule | 0.001 / Cosine（p6） | 一致 |
| Epoch | 150（p6） | 一致 |
| Batch | "动态自适应显存"（p6） | **80**（单卡，argparse 默认） |
| 增强/正则 | 未报告 | **无几何增强**（离线预生成 h5）；唯一正则是 **label smoothing ε=0.2** |
| 数据准备 | Event Cloud=按 t 严格时序排列的有序点云，随机下采样，Min-Max 归一化（p4 §3.1）；**非滑窗帧/voxel** | 官方提供 train.h5/test.h5 |
| 各数据集点数 | Table 1：N-Caltech101 **8192**、CIFAR10-DVS **10240**、N-MNIST 4096、N-CARS 8192、ASL 4096、Gesture 1024、DailyDVS(12类) 8192、UCF101-DVS 8192 | README 印证 8192/10240 |
| 滑窗 | Table 1：N-Cal 30ms、C10 100ms、UCF 1s… | README："单窗训练 + 跨样本多数投票（most_common(1)）" |
| 划分 | §D：无官方划分数据集按 Bi 2020/Chen 2022b 随机 20% 测试；Table 6 caption：filename-based (Ren 2025)——实质同一（Bi 2020 的 UCF101-DVS 划分本就按文件名分组） | README：filename 划分 + 直接提供 h5 |

**关键结论（与 HPC 复现一致性）**：
1. **点数**：N-Caltech101 论文用 **8192**（非 10240）；10240 是 CIFAR10-DVS 的设置。但 Table 9（300ms 单窗消融）显示 8192→10240 仅 60.04→60.60（平台期），**点数差异影响很小**。
2. **决定性因素是窗口协议**：82.4% 靠 **30ms 短窗 + 序列级多窗投票**；若用 300ms 单窗，8192 点也只有 ~60%。**若 HPC 目前是"整段单窗"配置，这才是复现失败的真因。**
3. **必须对齐**：① Bi et al. 2020 官方 filename 划分（直接用官方 h5，别自做随机 20%）；② N-Caltech101=8192 点+30ms 滑窗+投票；③ AdamW+lr 0.001+Cosine+150 epoch+batch 80+wd 1e-4+label smoothing 0.2。

---

## 六、协议交叉警告（引用分数前必读）

1. **EventMG Table 4 抄录的是宽松协议**：其引用的 EventMamba 97.9/86.4 是随机窗划分虚高数；EventMG 自报的 UCF 79.6/HMDB 71.2 大概率同套协议，**不可与文件名分组标准（90.3/91.6、60.4）直接比**。
2. **EventMamba 双数**：UCF 97.9（易）/90.3（文件名分组）、HMDB 86.4（易）/60.4（文件名分组）——引用必须注明协议。
3. **N-Caltech101 双口径**：EFV++ 89.7%（录制级、非点云）vs SECNet 82.4%（窗级、8192 点）——跨口径不成立。
4. **DailyDVS-200 ≠ 12 类 Daily DVS**：SECNet/EventMamba 的"Daily-DVS 8192 点/99.65%"是 Liu 2021 的 12 类旧集；DailyDVS-200 是 ECCV2024 的 200 类新集，point-based 零结果，冲榜目标按原文跨被试协议。
5. **MD-Mixer 存在 UCF/HMDB 结果**：p7 Table 3 已哈希比对确认（66.3/62.8），早期"仅报 s-CIFAR"的读法不成立；但它是帧-SNN 范式，不影响 point-based 目标线。

---

## 七、更高分数发现（2026-2027）

- **EventMamba 97.9/86.4（HMDB/UCF）**：**不是新发现**——即第一轮已证伪的宽松划分虚高数，EventMG 表内转引。文件名分组实值 90.3/60.4。
- **EvDistill：N-Caltech101 90.2%**（sota2.com 聚合站记录，2026-06-19，>89.76% 阈值）——**仅聚合站记录，未获原文核实【待验证】**；且疑似蒸馏/非 point-based from-scratch，需核实后再决定是否构成全范式威胁。
- **未发现** CIFAR10-DVS >87.35%、DailyDVS-200 >51.99% 的新工作（截至 2026-10-09）。

---

## 八、待验证清单

1. SECNet N-Caltech101 窗口：Table 1（30ms）与 Table 9（300ms）内部矛盾——以官方代码/数据为准
2. NGN 在 CIFAR10-DVS 的具体训练超参（正文未报告 DVS 超参）
3. MD-Mixer+SDT-V2 在 UCF/HMDB 的 T 与参数量（Table 3 无此列）、训练超参（Supp 未随主文）
4. EventMG 具体 lr/batch/epoch/wd，及其动作分数所用划分协议（文件名 vs 随机窗）
5. EIS-HAR 的 IGS 选帧数 K 与增强策略
6. CIFAR10-DVS 录制级划分 80/20 还是 90/10（SECNet 代码最终确认）
7. 动作集测试聚合口径：窗级 top-1 还是整录投票后的视频级 top-1
8. 点云系是否有未写入正文的训练增强（翻转/旋转/抖动）
9. DailyDVS-200 上点云方法的标准点数/窗长（目前全为帧式基线，待我们自建）
10. EvDistill N-Caltech101 90.2% 原文核实
11. SECNet 会议届次：按本方已确认事实记 ICML（子代理标注 ICML 2026 Oral 未独立复核）

---

## 九、来源链接与本地 PDF

### 本轮新增下载（D:\论文，4 篇全部独立校验 %PDF 且 >50KB）
| 文件 | 大小 |
|---|---|
| DailyDVS200_2024.pdf（Wang et al., ECCV2024, arXiv 2407.05106） | 18,633,715 B |
| Bi_TIP2020_UCF101_HMDB51-DVS.pdf（Bi et al., arXiv 1910.03579） | 1,766,113 B |
| CIFAR10-DVS_Li2017.pdf（Li et al., Frontiers 2017, DOI 10.3389/fnins.2017.00309） | 5,055,324 B |
| N-Caltech101_Orchard2015.pdf（Orchard et al., Frontiers 2015, DOI 10.3389/fnins.2015.00437） | 2,814,580 B |

### 库内引用（未重复下载、未改动）
2608.17394v1.pdf（NGN）、MultiDelayMixer_CVPR2026_脉冲Transformer时间交互.pdf（MD-Mixer）、EIS-HAR_2026.pdf、EventMG_NeurIPS2025_多级Mamba图.pdf、2412.20803.pdf（SECNet）、2405.06116.pdf（EventMamba）、2406.18845_EFV++_RBE_NCaltech101全范式SOTA.pdf（EFV++）

### 本轮主要来源链接
2608.17394（NGN）｜2605.20680（EIS-HAR, CVPR2026）｜2407.05106（DailyDVS-200, ECCV2024）｜1910.03579（Bi TIP2020）｜2405.06116（EventMamba, TPAMI2025）｜2412.20803（SECNet）｜2406.18845（EFV++/RBE）｜CVF CVPR2026 MD-Mixer 官网页｜github.com/rhwxmx/SECNet_ICML（SECNet 官方代码）

### 子调研产物（分片原始记录）
- sectionF_rivals.md（威胁方法深挖，位于 D:\UserData\Desktop\学习\进组\artifacts\）
- sectionG_protocols.md（数据集协议规范，位于代理 artifacts/）
- sectionB3_secnet_repro_protocol.md（SECNet 复现协议核对+代码确认）

---

## 下一步建议

1. **复现优先级调整**：先对齐 SECNet 窗口协议（30ms 短窗+投票）再谈点数；CIFAR10-DVS 用 10240 点正确、N-Caltech101 改 8192 点
2. **冲榜实验盘**：CIFAR10-DVS（75.7）→ N-Caltech101（82.4）→ UCF101-DVS（91.6）→ HMDB51-DVS（60.4）→ DailyDVS-200（51.99），按 §二 的协议逐一搭盘
3. 投稿前复查：待验证清单（尤其 EvDistill 90.2%、EventMG 划分协议、SECNet 30ms/300ms 窗口矛盾）

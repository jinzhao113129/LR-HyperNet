# 调研更新：待验证裁决 + 未饱和数据集新进展 + 线性注意力调研（2026-10-09）

> 项目：LR-HyperNet（point-based 事件点云分类）｜本报告为连续第四轮调研的综合交付
> 三包并行：A=上轮待验证项收尾裁决｜B=未饱和数据集新进展扫描｜C=Delta rule/门控线性注意力应用调研
> 全部结论已交叉裁决；CCF-A 优先；数字附页码级出处或链接；未下载原文标【无原文】

---

## 一句话核心结论

**①DailyDVS-200 / HARDVS 目标线上修：HoloEv-Net（arXiv 预印本、非 CCF-A）刷到 54.55% / 54.69%（EIS-HAR 51.99% 不再是线），但 point-based 仍是零结果；②新蓝海 EventKitchen（268 类、Swin-Base 仅 24.69%、point 完全空白）建议作差异化主推数据集；③GLA/Delta rule 用于 point-based 事件点云远程聚合仍是空白（最接近的 SpikMamba 是脉冲窗口线性注意力、非 GLA），v3 架构新颖性成立；④上轮 6 项待验证全部裁决完毕：LSFormer 为预印本、87.60% 属实、非 point；EIS-HAR 34.0M 为原文自报；DVS-Gesture 99.3% 为 SpikeVision 与 EHGCN 并列。**

---

## 一、包 A：上轮待验证项收尾裁决（6 项）

| # | 项 | 裁决 | 页码级证据 | 来源链接 | 库内 PDF |
|---|---|---|---|---|---|
| 1 | **LSFormer**（2605.13887） | 真实存在；**arXiv 预印本【待验证】**；N-Caltech101 **87.60%±0.10（T=10）属实**；范式=稠密脉冲视觉 Transformer（Spiking Tokenizer），**非 point-based** | Table IV（p.8）Ours 行；正文 p.8 "surpasses TIM(79.00%) by 8.60%" | arxiv.org/abs/2605.13887 | 2605.13887_LSFormer.pdf（库内已有） |
| 2 | **SL-Animals-DVS** | Spiking Patches **Transformer 98.1%**（GNN 95.5 / PCN 96.6）——上轮 88–92% 参考值**上修**；该文为预印本非 CCF-A，限定 CCF-A 则【无原文】 | Spiking Patches Table 2 | arxiv.org/html/2510.26614 | 2510.26614_SpikingPatches.pdf（库内已有） |
| 3 | **N-TIDIGITS** | 近期可引精确值 **DGN 99.10%**（Wu et al. 2018 的 99.4% 为旧值、协议不同），量级 99.1–99.4% | DGN 摘要 | arxiv.org/abs/2509.03281 | **本轮新增 DGN_2025_脑启发门控脉冲.pdf（720.5KB，%PDF✓）** |
| 4 | **MorphSNN / SpikePool-B** | **均为 arXiv 预印本，无 CCF-A 接收**（MorphSNN 2603.14285 经 arXiv 列表+Scholar 确认；SpikePool 2510.12102 未查到接收【待验证】） | arXiv 列表 / Google Scholar | 2603.14285 / 2510.12102 | 2603.14285v1.pdf、2510.12102.pdf（库内已有） |
| 5 | **EIS-HAR 参数量 34.0M** | **属实且为原文自报**（非转引）；EIS-HAR 已确认 CVPR 2026 | **Table 3（p.8）#P 列 "Ours"=34.0M**（对照 SlowFast 33.6M、VMamba 76.5M） | CVF CVPR2026 / arxiv.org/abs/2605.20680 | EIS-HAR_2026.pdf（库内已有） |
| 6 | **DVS-Gesture 99.3% 归属** | **SpikeVision 与 EHGCN 并列都报 99.3%**：EHGCN Table 2（p.19）DVS128=0.993（point-based，0.11 GFLOPs，超 EventMamba 0.992）；SpikeVision 摘要（IEEE 10942764）"99.3% on DVS128 Gesture" | EHGCN Table 2 p.19；IEEE 摘要 | arxiv.org/abs/2504.16616 / IEEE 10942764 | 2504.16616.pdf（库内已有） |

**裁决细节**：LSFormer ①论文存在（天津大学，2026-05-12）②PDF 页眉无会议戳、OpenReview/CVF 未查到接收 → 预印本 ③87.60% 属实 ④非 point-based。EIS-HAR"转引"说法经核为原文自报值，上轮记录修正。

---

## 二、包 B：未饱和数据集新进展扫描（2026-09 下旬 ~ 10-09）

### 1. 刷新检查

| 靶子 | 上轮目标线 | 本轮 | 是否被刷 |
|---|---|---|---|
| **DailyDVS-200 top-1** | EIS-HAR 51.99%（CVPR2026） | **HoloEv-Net-Base 54.55%** | ✅被刷（预印本） |
| **HARDVS top-1** | EIS-HAR 53.21% | **HoloEv-Net-Base 54.69%** | ✅被刷（预印本） |
| UCF101-DVS（滑窗） | MD-Mixer 66.3% | 无新高 | ❌ |
| UCF101-DVS（文件名分组, point） | SECNet 91.6% | 无新高（EventMamba point 90.3% 次之） | ❌ |
| HMDB51-DVS（滑窗） | MD-Mixer 62.8% | 无新高 | ❌ |
| HMDB51-DVS（文件名分组 point>60.4%） | — | 未找到该协议新 point 工作 | ⚠️待验证 |
| CIFAR10-DVS point>75.7% | SECNet | 无新高 | ❌ |
| CIFAR10-DVS 全范式>87.35%（NGN T=10） | NGN | 扫到 SNN/ANN ~81–82.7%，未超 | ❌ |

**关键**：**HoloEv-Net**（arXiv 2602.04182，摘要无 Comments=**arXiv 预印本、非 CCF-A**）Table I：DailyDVS-200 **54.55%**、HARDVS **54.69%**、THU-EACT-50-CHL **69.85%**，12.5M params，frame-based CHSR+GSG，from scratch（库内 2602.04182_HoloEvNet.pdf 已有）。**point-based 在这两个集仍完全空白。** 引用时须注明其为非 CCF-A 预印本。

### 2. 横向扩展：未刷满 / point 空白数据集

| 数据集 | 规模 | 全范式 SOTA | point 覆盖 | 饱和度 |
|---|---|---|---|---|
| **EventKitchen**（2608.04865，2026 新） | 第一人称立体厨房，**268 类**/10,762 段/5.5h/跨厨房 split | Swin-Base（K400 预训练）**24.69%**；TSM 19.23% | **完全空白** | 远未饱和（~25%） |
| **DVS-Lip** | 唇读，40 人 DAVIS346 | TVTA **77.49%**（2607.08236） | 基本空白 | 未饱和 |
| THU-E-ACT-50（full） | 50 类高分辨率 | SECNet **97.25%**（ICML2026） | point 有 | 已饱和 |
| THU-EACT-50-CHL | 50 类/2330 跨光照 | HoloEv-Net 69.85% | 空白 | 中等 |
| N-CARS | 2 类 | SECNet 94.7% | point 有 | 饱和 |
| Event ActivityNet（2608.01948） | 仿真 200 类，未 trim | 刚发布 | 空白 | 未饱和（时序定位） |
| Falling-DVS / MNIST-DVS / ECG-DVS / GESTURE-DVS | 小/专项 | 未核到 CCF-A 新作 | 零星 | 【待验证】 |

交叉注意已遵守：DailyDVS-200（ECCV2024, 200 类）≠ 12 类 Daily DVS（Liu 2021, SECNet 99.65%）。

### 3. 更新后可冲清单与目标线

| 数据集 | point SOTA | 全范式 SOTA | 可冲性 | 目标线（超谁立 point 新线） |
|---|---|---|---|---|
| **EventKitchen** | 完全空白 | Swin 24.69%（K400 预训练） | ★★★ 新蓝海 | 首条 point 基线；超 24.69% |
| **DailyDVS-200** | 空白 | HoloEv-Net 54.55%（预印本）/ EIS-HAR 51.99%（CCF-A） | ★★★ 首选 | 首条 point 即破；超 54.55%（注明预印本） |
| **HARDVS** | 基本空白 | HoloEv-Net 54.69%（预印本） | ★★★ | 超 54.69%（需同协议） |
| HMDB51-DVS（滑窗） | SpikePoint 55.6 / TTPOINT 56.9 | MD-Mixer 62.8%（CVPR2026） | ★★ | point>56.9 即新线，再冲 62.8 |
| UCF101-DVS（滑窗） | TTPOINT 72.5 | Event-LSTM 77.6 | ★★ | point 需 >72.5 |
| UCF101-DVS（文件名分组） | SECNet 91.6% | SECNet 91.6% | ★难 | >91.6 才破，不建议主冲 |
| DVS-Lip | 空白 | TVTA 77.49% | ★★ 差异化 | point 首条；超 77.49 |
| CIFAR10-DVS | SECNet 75.7% | NGN 87.35% | ★ | point>75.7 |
| THU-ACT-50(full) / ASL-DVS / N-CARS | 99% 量级 | 99% 量级 | ✗ | 不建议 |

---

## 三、包 C：Delta rule / 门控线性注意力应用调研

### 1. 机制源头谱系（10 篇）

| 方法 | 会议/年份 | 核心机制一句话 | arXiv | 库内 PDF |
|---|---|---|---|---|
| Linear Attention | arXiv 2020 | softmax 注意力按特征核分解为线性内积，复杂度 O(N)→O(N)（Katharopoulos） | 2006.16236 | LinearAttention_2020.pdf（库内已有） |
| **DeltaNet** | NeurIPS 2024（CCF-A） | 用 delta rule 更新线性注意力记忆：新记忆=记忆+β(errors)，可学习遗忘 | 2406.06484 | DeltaNet_2024.pdf（库内已有） |
| **GLA** | ICLR 2024（CCF-A） | 门控线性注意力：逐 token 可学习门控控制记忆写入/衰减，Chunk-wise 高效实现 | 2312.06635 | **新增 GLA_2024.pdf（803KB，%PDF✓）** |
| Based | 预印本 | 简单线性注意力基线：核近似+门控，挑战"必须复杂"的假设 | 2402.18668 | **新增 Based_2024.pdf（2.86MB，%PDF✓）** |
| **Gated DeltaNet** | ICLR 2025（CCF-A） | DeltaNet+逐头门控混合：窄头低秩（长程）+宽头 delta rule（最近信息） | 2412.06464 | GatedDeltaNet_2024.pdf（库内已有） |
| Titans | NeurIPS 2025（CCF-A） | 记忆即参数：神经长期记忆（线性注意力/Delta rule）+ 持久记忆 + 短期上下文，可学习记忆衰减 | 2501.00663 | **新增 Titans_2025.pdf（3.66MB，%PDF✓）** |
| RWKV-7 | 预印本（arXiv 2025） | delta rule 式状态更新+多路门控，"动态递归"线性序列模型 | 2503.14456 | **新增 RWKV7_2025.pdf（3.97MB，%PDF✓）** |
| Kimi Linear / KDA | 预印本（arXiv 2025） | Kilo-Delta-Attention：线性注意力+解耦键值状态（Mixture of Experts 集成） | 2510.26692 | KimiLinear.pdf（库内已有） |
| Kimi K3 | 预印本（arXiv 2025） | 多模态/多尺度 token 预训练+线性注意力 | — | KimiK3.pdf（库内已有） |

### 2. 视觉 / 事件应用（关键参考）

| 方法 | 会议/年份 | 核心机制一句话 | 与 LR-HyperNet v3 的关系 | arXiv | 库内 PDF |
|---|---|---|---|---|---|
| **ViG** | AAAI 2025（CCF-A） | **首个把 GLA 引入视觉主干**：双向扫描+2D 局部注入；ViG-S 匹配 DeiT-B 精度但仅 27% 参数 | **v3 最直接参考**（GLA 视觉化模板） | 2405.18425 | **新增 ViG_2024.pdf（2.04MB，%PDF✓）** |
| DiG | arXiv 2024 | 扩散生成+门控线性注意力（ViG 的生成版） | 生成侧参考 | 2405.18428 | **新增 DiG_2024.pdf（7.73MB，%PDF✓）** |
| MambaVision | arXiv 2024 | 纯 Mamba 视觉主干（conv+SSM 混合），多分辨率 | hybrid 架构参考 | 2407.08083 | **新增 MambaVision_2024.pdf（15.4MB，%PDF✓）** |
| RMT | arXiv 2023 | Retentive Network 视觉版：窗口+块内线性递推 | 线性注意力视觉参考 | 2309.09023 | **新增 RMT_2023.pdf（1.02MB，%PDF✓）** |
| Demystify Mamba | 预印本 | Mamba 注意力的局部/全局分解分析 | 分析工具参考 | 2403.03804 | **新增 DemystifyMamba_2024.pdf（978KB，%PDF✓）** |
| **SpikMamba** | arXiv 2024 | **事件 HAR**：脉冲化窗口线性注意力+SSM，做时序帧序列 | **最接近的事件侧工作，但非 GLA/Delta rule、非 3D 点云** | 2410.16746 | **新增 SpikMamba_2024.pdf（3.83MB，%PDF✓）** |
| SSLA | ECCV 2026（CCF-A） | 空间稀疏线性注意力（事件目标检测） | 机制谱系不同（稀疏近似 vs 门控递推） | 2603.06228 | SSLA.pdf（库内已有） |

### 3. 事件点云线性注意力是否空白 —— **基本仍空白**

- 最接近的 SpikMamba：脉冲化**窗口**线性注意力（时序帧序列，非 3D 点云、非 GLA/Delta rule）；
- SSLA：稀疏近似谱系，非门控递推，且任务为检测；
- ViG：GLA on 2D 图像，未推广到 3D 事件点云；
- **结论：用 GLA/Delta rule 做 point-based 事件点云远程聚合 = ViG × EventMamba 的交叉，无直接先例——v3 架构新颖性成立。**

### 4. 待验证（包 C 侧）
DGEViT（"双门控线性注意力"）多次检索未找到该名论文，疑似名字记混【待验证】；RWKV-6 无独立论文（delta rule 相关为 RWKV-7）；Gated DeltaNet-2（NeurIPS 2026）未定位 arXiv ID。

---

## 四、交叉裁决记录（整合时已核对）

1. **SECNet CIFAR10-DVS 75.7%**：包 B 标【待验证】，但上轮 point-based 覆盖图谱已页码级确认（p7 Table 2 "Without Pretraining" 栏 0.757）——**以 75.7% 采信**。
2. **NGN 87.35%**：包 B 称"用户给定未独立核对"，但上轮威胁方法深挖已页码级确认（p11 Table 3，NGN+TET/VGG-SNN 行，87.35±0.33, T=10）——**采信**。
3. **HoloEv-Net 刷新**：与包 A 无交集；其 arXiv 摘要无 Comments → **非 CCF-A 预印本**，目标线更新但引用需注明（论文已宣称超越 EIS-HAR）。
4. **EHGCN DVS-Gesture 99.3%**（包 A）与上轮 point 矩阵 EventMamba GES 99.2% 自洽——DVS-Gesture point 上界再确认（已饱和，不可冲）。
5. **LSFormer**（包 A：脉冲 Transformer 非 point）与包 C 事件点云空白结论无冲突。
6. **DVS-Gesture 99.3% 双归属**（SpikeVision 帧式 vs EHGCN point）——写作何一处 99.3% 均需注明归属与输入范式。
7. DailyDVS-200 与 12 类 Daily DVS 的区分在包 B 中已遵守，与包 A 的 EIS-HAR 裁决一致（EIS-HAR 报的是 200 类集）。

---

## 五、本轮下载清单（D:\论文，13 篇，全部独立校验 %PDF 且 >50KB）

| 文件 | 大小 | 归属 |
|---|---|---|
| EventKitchen_2026.pdf | 20,276,006 B | 包 B：EventKitchen 数据集原文 |
| TVTA_2026.pdf | 6,954,389 B | 包 B：DVS-Lip 当前 SOTA |
| DGN_2025_脑启发门控脉冲.pdf | 737,803 B | 包 A：N-TIDIGITS 99.10% |
| GLA_2024.pdf | 803,395 B | 包 C |
| Titans_2025.pdf | 3,657,065 B | 包 C |
| Based_2024.pdf | 2,859,995 B | 包 C |
| RWKV7_2025.pdf | 3,967,870 B | 包 C |
| ViG_2024.pdf | 2,040,422 B | 包 C |
| DiG_2024.pdf | 7,734,100 B | 包 C |
| MambaVision_2024.pdf | 15,417,049 B | 包 C |
| SpikMamba_2024.pdf | 3,831,088 B | 包 C |
| DemystifyMamba_2024.pdf | 978,425 B | 包 C |
| RMT_2023.pdf | 1,015,330 B | 包 C |

库内已有未重复下载（含 GatedDeltaNet_2024、DeltaNet_2024、LinearAttention_2020、KimiLinear、KimiK3、SSLA、2602.04182_HoloEvNet 等）；未删改任何库内文件；下载失败 0。

---

## 六、待验证清单（更新后）

**已关闭（本轮裁决）**：LSFormer 归属/87.60/范式；SL-Animals-DVS 98.1%；N-TIDIGITS 99.10%；MorphSNN/SpikePool 归属；EIS-HAR 34.0M；DVS-Gesture 99.3% 归属。

**仍待验证**：
1. SECNet N-Caltech101 窗口 30ms（Table 1）vs 300ms（Table 9）矛盾——以官方代码/数据为准
2. EventMG 的 lr/batch/epoch/wd 与动作分数所用划分协议（文件名 vs 随机窗）
3. MD-Mixer+SDT-V2 在 UCF/HMDB 的 T 与参数量（Table 3 无此列）；训练超参（Supp 未随主文）
4. EIS-HAR 的 IGS 选帧数 K 与增强策略
5. 动作集测试聚合口径（窗级 vs 视频级投票）
6. DailyDVS-200 上点云方法的标准点数/窗长（全为帧式基线）
7. EvDistill N-Caltech101 90.2% 原文核实（仅聚合站记录）
8. HoloEv-Net 正式会议归属（预印本，待接收）
9. DGEViT 论文名核实（疑似记混）；Gated DeltaNet-2（NeurIPS 2026）ID
10. Falling-DVS/MNIST-DVS/ECG-DVS/GESTURE-DVS 的 2025-2026 CCF-A 新分数

---

## 七、对 v3 架构与投稿的启发（汇总）

1. **v3 远程聚合**：GLA/Delta rule 事件点云应用为空档（包 C）；最直接模板是 **ViG（GLA 视觉主干）+ EventMamba 的 FPS+KNN 点云骨架**；机制源头可引 DeltaNet→GLA→Gated DeltaNet→Titans→RWKV-7 谱系。
2. **冲榜主推建议**：EventKitchen（268 类、point 全空白、Swin 仅 24.69%）作差异化新蓝海；DailyDVS-200 目标线更新为 54.55%（注明 HoloEv-Net 预印本）；CIFAR10-DVS/N-Caltech101/UCF101-DVS 仍锁 SECNet 协议（见上轮 rivals_and_protocols 报告）。
3. **related work 可引用**：SECNet 为 ICML2026 Oral（CCF-A）已确认；EIS-HAR 为 CVPR 2026（参数量 34.0M 自报）；EventKitchen/TVTA/HoloEv-Net 均为 2026 新（后两者预印本）。

---

## 八、子调研产物路径（分片原始记录）

- sectionA2_verification.md（包 A 裁决）
- sectionB3_fresh_scan.md（包 B 新进展）
- sectionC_linear_attention.md（包 C 线性注意力）

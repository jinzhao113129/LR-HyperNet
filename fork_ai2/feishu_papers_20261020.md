# 飞书论文笔记更新 — 2026-10-20（ai2/Claude 提供，豆包审核后更新飞书）

> 本文件按协作规范由 ai2（Claude）生成，请金朝转给豆包。豆包核对后更新到飞书论文笔记。
> 内容：SECNet 主线组件调研中接触到的全部论文，按「做成组件 / 赛道内 / 跨赛道 / 理论基础」分类。
> 重点标注：**哪些做成组件、哪些是查新先例、哪些是 Related Work 必引**。

## 一、做成组件的论文（3 篇，均为跨赛道借思想）

> 注意：这三篇都不是「事件点云分类」赛道，是我们从 2D 图像 / 3D 普通点云借机制、自己在 SECNet 上重实现的。论文 Related Work 里要如实写"借鉴自 XX（非本赛道）"。

| 论文 | 赛道 | 借的思想 | 做成组件 |
|---|---|---|---|
| **FcaNet**: Frequency Channel Attention Networks（Qin et al., ICCV 2021, arXiv:2012.11879） | 2D 图像 CNN | 用多频 DCT 分量（非仅均值）学通道门控 | **fca**（频域通道门控，融合层） |
| **PointCHR**: Point Cloud Analysis via Curvature-Aware Hyperbolic Rectification（arXiv:2607.24052, 2026-07） | 3D 普通点云 | 高曲率/精细点投影到双曲边界（大容量） | **chr**（曲率感知双曲径向校正，特征层） |
| **Rethinking the Fully Hyperbolic ViT in Polar Coordinates**（arXiv:2609.32899, 2026-09） | 2D 图像 ViT | 方向/半径分离映射，半径可独立学 | **polar**（方向/半径分离，特征层，chr 的欧氏对照） |

**chr 的实现关键**：径向校正必须在 Lorentz 流形内部做（exp→缩放 x0→log），否则 exp/log 首尾恒等、双曲失效（我们曾踩坑，已修）。**polar 是 chr 的严格欧氏镜像**，配对差可回答"双曲 vs 欧氏增量"。

## 二、我们赛道的论文（事件相机 / 神经形态）

> 这些是真正同赛道的，但**没有能直接做成组件的**：要么赛道细分不同，要么是先例（要避开），要么路线不搭。

| 论文 | 赛道/贡献 | 和我们的关系 |
|---|---|---|
| **SECNet**（Ren et al.） | 事件点云分类 N-Caltech101，**82.4% SOTA** | 底座（我们 fork 的原版，被改的对象），对标靶子 |
| **EHGCN**（arXiv:2504.16616） | 事件 × 双曲融合，但做**检测** + **图卷积** | **最接近的邻居**——事件×双曲，但检测≠分类、图≠点云，只能当 Related Work 的"最接近者"，抄不了组件 |
| **GSS**（Gumbel subset sampling, CVPR 2019） | 可学习子集采样，DVS128 Gesture（神经形态） | **采样层旧先例**（共识 §2 必引，要避开，不能当我们的创新） |
| **PAT**（CVPR 2019） | 可学习采样 | 同上，可学习采样先例 |
| **EventRPG**（SNN） | 事件相机识别 85.62% | 共识里作为"数据增强背书"引用，SNN 路线 |
| **Active Spiking Perception**（2026-08） | SNN + 3D 点云识别（膜电位作信念状态） | 事件/脉冲相关，但 SNN 路线，不搭 |
| **Spiking Point Cloud PEFT**（2026-07） | 脉冲点云模型微调 | SNN 路线，不搭 |

**关键结论**：同赛道没有能借的组件，能借的组件都来自跨赛道（2D/3D）。「双曲 × 事件点云分类」细分是空白，尤其我们的「远程选择 + 双曲 + 事件」三交叉，比 EHGCN 的"双曲融合检测"更细。

## 三、跨赛道查新背景（扫过摘要，判断可移植性，大多否掉）

| 论文 | 赛道 | 结论 |
|---|---|---|
| **PointLearner**（Point-Focused Attention Meets Context-Scan State Space, arXiv:2610.11342, 2026-10） | 3D 普通点云 | 「可学习诱导点 + 注意力」逼近 G16 远程选择，**待查新确认是否构成先例** |
| **Point-Selection Fine-Tuning**（arXiv:2607.19711, 2026-07） | 3D 点云 | "选点"但方向相反（丢离群点做鲁棒），作 Related Work 对照 |
| **On-the-Fly Object-aware Representative Point Selection**（2025-08） | 3D 点云 | 代表点选择，采样层背景 |
| **RadiusFPS / FlashFPS**（2026） | 3D 点云 | FPS **效率加速**（剪枝/缓存），非机制创新 |
| **Hyper²**（2026-08） | 双曲 VLM | "双曲是否装饰"的审计，方法论参照（支撑 polar 对照的意义） |
| **HypOProto / Hyperbolic Concept Bottleneck / Hyp2Former**（2026） | 医学/分割/多模态 | 双曲分类头，无可移植组件 |
| **Riemannian-Lorentz Fusion**（2026-09） | 模型合并 | 训练后权重合并，非训练架构 |

## 四、组件设计的理论基础（经典，非本赛道，但必须引）

| 论文 | 支撑 |
|---|---|
| **Ganea et al. 2018**《Hyperbolic Neural Networks》 | 双曲算子/MLP/原型基础（hsel/hmlp/hclf 的根基） |
| **Gulcehre et al. 2019**《Hyperbolic Attention Networks》（ICML） | ha（双曲注意力） |
| **SENet**（Hu et al., CVPR 2018） | SECNet 门控的原型（hgate/fca 替换的对象） |
| **Xiong et al. 2020**《On Layer Normalization》（ICML） | NLP2 的 Pre-LN 依据；也支撑对 Post-LN 梯度问题的理解 |

## 五、待坐实的出处（之前只从共识转述，没亲自读原文）

- **EventRPG**：具体 arXiv 编号/会议待查（共识里写"SNN 85.62"）；
- **EHGCN**：arXiv 2504.16616 待复核原文赛道（检测 vs 分类）；
- **Hyper²**：编号待复核（共识里写 2608.22238）；
- **GSS / PAT**：CVPR 2019 的具体标题、作者待查证。

以上「待坐实」的，豆包如需可让 ai2 逐个查证原文。

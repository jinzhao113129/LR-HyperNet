# 组件调研与架构缺口整理 — 2026-10-18

> ai2/Claude 起草。承接「继续准备新组件」的工作流（已产出 hgate/fca/chr 三个
> 融合层/特征层候选）。本文整理 2026-10-18 的 arXiv 检索结果，回答「替换路线
> 还剩哪些缺口、最新论文里还有没有可移植的新组件、下一步该投哪里」。
> 检索渠道：arXiv API（WebSearch 当日不可用，用 curl 直连 export.arxiv.org）。

## 一、一句话结论

**最新论文里可移植的"新组件"已经稀缺**——点云/事件分类的架构机制在 2022 前后
基本成熟，2024-2026 的增量集中在应用、微调、效率、硬件。但检索到 **两篇高价值
工作**：一篇印证并升级了 chr 的设计（双曲极坐标），一篇逼近论文灵魂「16 点远程
选择」（可学习诱导点），后者**必须查新**。

## 二、替换路线 5 步的组件覆盖现状

| 位置 | 组件 | 状态 |
|---|---|---|
| 编码层 | spike（快测中）+ time_enc/kda（已实现） | 覆盖 |
| **特征层** | **chr**（HyperbolicRadialRectify，PointCHR 思路） | ✅ 定稿待审 |
| ①聚合层 | ha / hmlp | 快测中 |
| ②采样层 | hsel / rp1（快测中）+ G16 远程选择（已实现未训练） | 覆盖 |
| ③融合层 | hgate / fca | ✅ 定稿 |
| ④分类头 | hyperbolic_classifier（Lorentz 双曲原型） | 已实现未进快测 |
| ⑤结构层 | **无** | **真空** |

唯一真空是 ⑤结构层；而共识 §1 写死的论文灵魂「16 点远程选择」本质是结构层的
创新，已有实现（RemoteSelector 五种选法 + 44 断言 + 显存实测）但零训练证据。

## 三、关键发现（分方向，含 arXiv ID）

### 3.1 双曲 Transformer 最新：极坐标表述（高价值，印证并升级 chr）

**[Rethinking the Fully Hyperbolic Vision Transformer in Polar Coordinates](https://arxiv.org/abs/2609.32899)**（2026-09-26）

- **核心观察**：当前 Lorentz transformer 用 ambient 坐标（环境坐标）表述，**半径
  大时 Lorentz 内积数值不稳定，很多模型因此限制半径**，等于放弃了双曲空间最有
  价值的"边界大容量"区域。
- **解法**：在**极坐标**里重写 transformer 组件——① polar fully connected layer
  （**方向/半径分离映射**，半径可学习而非由线性映射 norm 决定）；② horospherical
  shifts（相对位置编码，query-key 距离随 token 间隔对数增长）；③ residual 改写为
  average radius Lorentz boosts。
- **对我们的意义**：
  1. 它印证了我们 chr/hgate 里已经踩到的数值问题——`lorentz_exp_origin` 的
     `norm_max=10` 保护、HyperbolicEmbedding 的"进球前尺度归一"、Codex 抓的
     "LayerNorm 后范数≈8 导致距离 7-12"都是同一个病根；
  2. 它的"方向/半径分离"正是 chr（径向校正）的思想，但更系统、有理论化；
  3. **chr 组件的潜在升级方向**：若要深化，可把 chr 的"径向校正"对齐到"极坐标
     方向/半径分离"，但这是后续优化，不是现在的候选。

### 3.2 论文灵魂（远程选择）的最新逼近（高价值，必须查新）

**[PointLearner: Point-Focused Attention Meets Context-Scan State Space](https://arxiv.org/abs/2610.11342)**（2026-10-08）

- **核心机制**（生物视觉启发）：① **point-focused attention**——模拟中央凹视觉，
  在"局部 KNN 邻居 vs 空间下采样特征"之间做**竞争归一化注意力**；② 下采样特征
  由 **learnable inducing points（可学习诱导点）** 提取，诱导点数量可控、直接与
  点云交互、适应非均匀分布；③ context-scan state space——Hilbert 曲线引导的双向
  S6（Mamba）模拟眼睛扫视。
- **为什么关键**：**"learnable inducing points + 注意力"和我们的"16 点远程选择"
  是同一类思想**——用少量可学习点，通过选择/注意力机制捕捉长程依赖，这正是
  学长反复强调的论文灵魂。
- **必须查新**：它是否构成「事件点云分类中，双曲/时空联合的远程邻居选择准则」
  （共识 §2 查新口径）的**先例**？三点要厘清：① 它的 inducing points 是"远程
  邻居选择"还是"可学习子集采样"（后者 PAT/GSS CVPR 2019 已做，是旧先例）；
  ② 它是否用双曲；③ 它是否在事件点云上。**这三点的答案决定我们论文灵魂的
  查新结论要不要改写。**

### 3.3 无颠覆性新机制的方向（低价值，列出备查）

- **门控/通道注意力**：2022 后无新地基，SENet/CBAM/ECA/FcaNet 仍是代表（fca 已
  采用 FcaNet）。
- **分类头**：HypOProto（2606 序数原型）、Hyperbolic Concept Bottleneck、Hyp2Former
  均为医学/分割/多模态应用，无事件点云分类可移植的"分类头新机制"。
- **结构层**：Point Ladder Tuning（2607，PEFT 微调）非新结构。
- **采样层**：RadiusFPS、FlashFPS 是 FPS 的**效率加速**（剪枝/缓存），非机制创新。
- **频域点云**：NPNet（2601，高斯-傅里叶位置编码）为遥感应用；频域鲁棒性（2511）
  是鲁棒性非分类。
- **事件/神经形态**：脉冲 Transformer、SNN 硬件为主，与 ANN 点云路线不同。
- **模型合并**：Riemannian-Lorentz Fusion（2609）是训练后权重合并，非训练架构。

## 四、结论

1. **组件准备的边际收益递减**：融合层（hgate/fca）、特征层（chr）已补齐，分类头
   有 hyperbolic_classifier 兜底，继续搜"可移植新组件"不太可能再有 PointCHR 那种
   高价值命中。
2. **真正的空白不是"缺组件代码"，而是"缺训练证据"**：论文灵魂 G16 远程选择已
   实现未训练，这是比再写新组件更该投的事。
3. **两个具体行动**（见下）：PointLearner 查新（可能改写查新结论）、chr 极坐标
   升级（后续优化）。

## 五、建议（按优先级）

1. **查新 PointLearner**（最高优先）：它逼近论文灵魂，必须厘清是否构成先例——
   交 Codex 查新（见 §六）。
2. **把 G16 推进训练验证**：论文灵魂缺的是"跨 seed 配对差"，不是组件代码。
3. **④分类头（hyperbolic_classifier）纳入快测候选**：双曲贯穿三件套最后一环，
   随时可排链。
4. chr 的"极坐标方向/半径分离"升级：记入 backlog，等 chr 有正反馈再深化。

## 六、给 Codex 的查新任务

针对 [PointLearner (arXiv 2610.11342)](https://arxiv.org/abs/2610.11342)，回答：

1. 它的「learnable inducing points + point-focused attention」属于**远程邻居选择
   准则**还是**可学习子集采样**？（后者 PAT/GSS CVPR 2019 已做，是已知先例）
2. 它是否用**双曲**空间？
3. 它是否在**事件点云 / N-Caltech101** 上评测，还是普通 3D 点云（ModelNet 等）？
4. 结论：它是否构成共识 §2 查新口径「事件点云分类中，双曲/时空联合的远程邻居
   选择准则」的**先例**？我们论文灵魂的查新结论要不要改写、怎么改写？
5. 顺带扫一遍它的引用/被引，看有没有更近的"远程选择 + 双曲 + 事件"三交叉工作。

产出：`notes/codex_survey_pointlearner_20261018.md`。

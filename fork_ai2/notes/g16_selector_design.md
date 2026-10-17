# G16 Remote Selector 设计分析：原文框架 × 我们架构（2026-10-14）

用户问题：**"怎么选这 16 个点"——双曲距离选点、可学习 scorer、还是其他？**
本文对照 RP1 原文（`event-camera-rps/docs/rp1-local-sparse-remote-neighborhood.md`）与我们的
RemoteSelector 实现，给出现状映射、关键差距、和推荐实验。

---

## 1. 原文的证据链（N-Caltech101，W384-G16 协议）

| Selector | Seq Acc | 含义 |
|---|---:|---|
| Local W384 | 59.30% | 快，信息不够 |
| Random remote | 59.87% | (+0.57) remote 不能乱选 |
| Time-polarity | 60.10% | (+0.80) 简单时间先验不够 |
| Spatial-uniform | 61.08% | (+1.78) 空间补充有价值 |
| **Oracle G16** | **68.03%** | (+8.73) 16 个好点已超 KNN |
| Global Feature-KNN | 66.70% | (+7.40) 强但昂贵（11.38ms） |

三条核心信息：
1. **怎么选比选多少更重要**：16 个 Oracle 点 > 全局 KNN 的 K 个点。瓶颈是选点质量不是数量。
2. **Random 只 +0.57pp**：远程信息不是"随便补充什么都有用"，选点差等于白加。
3. **原文的 KNN 语义是"最近"**（特征相关、空间远程），Oracle profiling 表里明确列
   "Feature cosine similarity | 是否与 center 保持特征相关性？"——原文假设远程点可能
   **保持特征相关性**（同一语义对象在别处的证据），而非"最不相似"。

原文 §4.2 的推荐流程：**Remote pool → cheap tokenization → Top-G**，selector 矩阵
（Accuracy / Oracle Recall@G / Latency）六候选：Random、Spatial-uniform、Time-polarity、
Feature hash、Learned scorer、Oracle。

## 2. 我们架构的现状映射（HPC 实测确认，2026-10-14）

RemoteSelector（`models/local_remote_grouper.py:136`）现有能力 × 原文候选对照：

| 原文候选 | 我们的对应 | 接线状态 |
|---|---|---|
| Random | `selector_mode=random` | 已实现（含局部排除） |
| Spatial-uniform | `selector_mode=spatial_uniform` | 已实现但有 Codex #8/#9 缺陷；修正版 `ai2_g16/g16_selector_fixed.py` 就绪未接入 |
| Time-polarity | — | 未实现（原文本身只 +0.80pp，优先级低） |
| Feature hash | — | 未实现（原文未测，无先验） |
| Learned scorer | `use_rp1_selector`（固定形式 scorer：特征距离+λ·空间新颖性） | 已实现；纯可学习版未实现，原文/Codex 均列为高方差 P3 |
| Global KNN（基线） | `selector_mode=topk`，**但方向相反** | 见 §3.1 |
| **（超出原文）双曲距离选点** | `use_hyperbolic_selector` + topk | 已接线（lorentz/poincare exp map + 双曲距离），**从未跑过** |

## 3. 关键差距分析

### 3.1 方向问题（最重要，原文对照直接暴露）

我们 topk 模式 `cand_dist.topk(G, largest=True)` 选的是**特征距离最远**的 16 个点
（"与中心最不相似"）；原文 Global KNN 66.70% 是**特征最近**（cosine 相似度最高，
"特征相关但空间远程"）。两者是相反的选点假设：

- 假设 A（我们现状）：远程点价值 = **最大对比**（最不相似，补充正交信息）
- 假设 B（原文 KNN）：远程点价值 = **语义远程同源**（相似特征在别处的证据——
  同一物体移动/重复模式/全局上下文）

原文 Oracle profiling 表明确怀疑假设 B（"是否与 center 保持特征相关性？"）。
**原文数据不能直接判定哪种对**——Oracle 68.03% > KNN 66.70% 说明 KNN 不是最优，
但 KNN 是已知最强 cheap selector。我们从未测过假设 B。
→ 一行代码的对照：`largest=False`（或加 `selector_mode=topk_near`）。

### 3.2 双曲距离选点：为什么它可能恰好是"方向问题"的答案

双曲空间距离放大"语义层级远但同源"的点（特征空间呈树状层级时，双曲距离
把浅层共祖的点拉近、深层分叉的点推远）。这给了它一个与欧氏不同的中间立场：
- 欧氏最远（假设 A）：选正交方向，可能选到纯噪声事件
- 欧氏最近（假设 B）：选几乎复制的点，信息增量小（KNN 的 66.70% 上限即此）
- **双曲距离**：层级结构下自动区分"真分叉"与"浅层同源"——可能比欧氏两极都好

这正好是 CVPR 叙事需要的：G16 双曲选点 = 把"选什么远程点"变成几何问题。
且 `use_hyperbolic_selector` 接线已存在（exp map → 双曲距离 → topk），缺的只是：
a) 方向对照（远/近）、b) 从未跑过（三个历史 G16 run 全 False）、c) Codex #4 的
诊断（越界 clamp 比例、格碰撞率）——双曲距离选点同样吃 Codex #5 的逐 stage
坐标漂移问题。

### 3.3 其他确认的差距（Codex 审查发现，与原文框架一致）

- **列错位**（#8）：现 spatial_uniform 按 t×x 分格（y 被忽略）→ 修正版已交付。
- **池构造混淆**（#9）：topk 全局打分 vs random/spatial 随机小池 → 修正版
  `build_shared_candidate_pool` 已交付（三选择器共用池）。
- **无 Oracle membership**：原文 Oracle Recall@G 列在我们协议下没有 ground truth
  （Oracle 68.03% 是 W384-G16 协议的数，不可直接比，Codex #7）。我们用
  Accuracy + Latency 两维 + 选择分布诊断代替。

### 3.4 历史三个 G16 run 的真相（本次查证）

`lr_hyper_kda_v32_hyper_light`（topk 默认，36.5% ep40）/ `hyper_random`（27.1% ep10）/
`hyper_spatial`（26.3% ep9）——sacct 确认**全部 CANCELLED**（人为取消释放 GPU，非代码
崩溃），且都 `use_hyperbolic_selector=False`、`seed=None`、无固定 seed NPZ。轨迹参考：
同 epoch（9-10）topk ≈ 28.5% > random ≈ 27.1% > spatial ≈ 26.3%，但时长不等、代码
版本混合（light run 的 Namespace 里连 selector_mode 键都没有——更早版本提交），
如 Codex #8 所言不可作 selector 对比证据。

## 4. 推荐的下一个 G16 实验（研究问题 + 设计）

**研究问题**：固定 LRHyperNet 骨干与候选排除，"以什么准则选 16 个远程点"
（随机 / 空间均匀 / 欧氏最远 / 欧氏最近 / 双曲最远）是否改变 N-Caltech101 准确率？
其中双曲距离选点是否优于欧氏两极（CVPR 叙事核心假设）？

**设计**（对齐 Codex 交付标准 + 原文 selector 矩阵）：
- 共享骨干：LRHyperNet + 固定训练 seed（42，历史 run 全是 None——顺带补齐）。
- ~~（2026-10-15 更新）~~ **口径已定稿为全量**：数据 official_gen、epoch 150、
  batch 32 + grad_accum 2、A800——本 §4 原写的 "kda=True, group_norm=True,
  grad_accum=4, epoch=60, baidu" 是**旧口径，已废弃**（consensus §7.4 学长拍板
  直接对标 82.4 全量，2026-10-14 深夜，晚于本段）。骨干最终定为**默认（无
  kda/group_norm/dropout/anchor）**——查证唯一有健康早期轨迹的官方全量
  LRHyperNet 是默认骨干（v1 ep8 56.3% seq 斜率健康，该 job 也曾被取消过）；
  kda 系列全在 baidu 60ep 上 CANCELLED（最高 28% seq）。详见
  `../code_changelog.md §21`。
- 共享候选池：`build_shared_candidate_pool`（M=256，非局部，零局部重叠）。
  注意：topk-near 需要全局打分，按 Codex 建议作为独立高成本参照（不进 selector-only 对照）。
- 五处理：
  1. `random`（下界锚点）
  2. `spatial_uniform_fixed`（原文最好 cheap selector 的修正版）
  3. `topk_far`（我们现状：欧氏最远 16）
  4. `topk_near`（原文 KNN 方向：欧氏最近非局部 16——一行 `largest=False`）
  5. `hyp_far`（`use_hyperbolic_selector=True`：双曲最远 16——已接线未跑）
- 测量：固定 seed42 推理 + 934 样本 McNemar（vs matched baseline）、每 epoch 选择
  分布诊断（格覆盖、坐标 clamp 率——Codex #4/#5 的 stage 级仪表）、选择延迟。
- 分离声明：结果与 SECNet stage-center 消融表分开报告（不同骨干不同协议，Codex #7）。

**不做的**：Learned scorer（P3：原文顺序也是 cheap selector 先行，且高方差；
在 1-5 出结果、确认哪个几何准则有效后再谈学习它）；Time-polarity（原文只 +0.80pp）；
Oracle Recall（无 membership GT）。

**实现路径（不动运行中作业）**：
- ✅ 五处理的独立实现全部完成（`ai2_g16/g16_selector_fixed.py` v2，2026-10-14）：
  - `random_select`（池版随机，处理 1）
  - `spatial_uniform_select_fixed`（修正版，处理 2）
  - `euclidean_topk_select(near=False/True)`（处理 3/4：欧氏最远/最近）
  - `hyperbolic_topk_select(near=False/True)`（处理 5：Lorentz 双曲距离选点，
    公式与 `local_remote_grouper.lorentz_exp_origin` v3.1 逐项对齐——scale 方向、
    sinh 因子、流形投影兜底、arccosh clamp——独立实现供 Codex 逐行比对）
- ✅ 测试：`test_g16_selector_v2.py` 26 项断言本地+HPC 双侧 ALL PASS
  （流形约束 ⟨x,x⟩_L=-1/c 误差<1e-3、远切向量距离更大、near/far 方向语义、
  五选择器形状/界内/唯一/零重叠/池成员/可复现）；旧 18 项回归测试未破坏。
  HPC md5：g16_selector_fixed.py `effc2f7f...`、test_v2 `45fbe377...`
- ⏳ 训练接线：五处理跑训练仍需接入 `train_lr_hyper.py` 路径（改
  `local_remote_grouper.py` 或薄训练脚本 monkeypatch）——**等 Codex 审阅
  设计与实现后提交**（progress 14:05 交接点；FECNet import 此文件，
  改前需协调）。

**优先级判断**：这个实验把用户三个候选方向（双曲选点 / learned scorer / 其他）里的
前两个都覆盖（双曲=处理5，learned scorer=暂缓），"其他"= 原文最强的 cheap selector
家族（spatial_uniform 修正版 + KNN 方向）。每个处理都有明确研究问题，符合规则 10。

---

## 5. 原文复读补充（2026-10-15 09:20，用户要求"跑之前做到极致"）

复读原文 + progress-20260812.md + RP1_DATA_SETTING.md 后，**四个此前
未被充分消化的关键事实**：

### 5.1 原文数字的协议与我们全量协议根本不同（不可比，但结构结论可用）

| 维度 | RP1 原文协议 | 我们全量协议 |
|---|---|---|
| 每样本点数 | **2048** | 10240 |
| 训练样本 | 20901 | 62703 |
| W384 含义 | **候选池 = local 窗口外的全部 2048-384≈1664 点**（小池！） | 我们池 M=256/中心（从 10240 采） |
| 骨干 | SECNet 轻量（group=512, base_dim=16） | LRHyperNet 大配置 |
| G 的来源 | 每个事件自己的远程邻居（per-event） | 每个中心点的远程邻居（per-center） |

**含义**：Oracle 68.03/KNN 66.70/spatial 61.08 这些**绝对数字全部
不可迁移**（Codex #7 已警告过）——可迁移的只有**排序结构**：
Oracle > KNN > spatial-uniform > random > local。且原文自己的
progress 文档承认（5.1 原话）："W/G 不完全相同，最终论文表格必须
以各 run 的 config.json 重新统一口径"。

### 5.2 原文自己的 status 判定：RP1 只有 headroom claim 没有 method claim

progress-20260812.md 明确记录（2026-08 时点）：
- ✅ 可靠：Oracle 远程邻居显著有效（headroom）；
- ❌ 不可靠：block retrieval/segment prototype/XYTP farthest 等
  selector 未完成有效 full run；多 seed 未做；latency 未核；
- 结论原话："**还不能声称已有 cheap selector 是最终方法**"。

**含义**：RP1 当时连自己的 selector 都没验证完——我们现在的
五处理实验不是"复现已验证的东西"，而是**接着原文没做完的
工作继续做**（正好是查新空位：原文只证明了 headroom）。

### 5.3 原文推荐的顺序：Oracle profiling 先于 selector

原文 §6 工作顺序：①先 Oracle profiling（搞清高价值远程点的
性质）→ ②再按 profiling 设计 selector → ③统一评测。
我们跳过了 ① 直接做五处理 selector 对照——**风险**：如果
五处理全都不显著，没有 profiling 数据就无法解释为什么。
**修正**：五处理训练时同步跑 Oracle profiling 的代理指标
（选择分布诊断已设计：格覆盖/clamp 率/特征相似度分布/
与 local 的空间距离分布），把 §4.1 原文 profiling 表的特征
逐项落进我们的诊断输出——不用 Oracle GT（我们没有），但
"高价值远程点长什么样"的描述性统计能拿。

### 5.4 Learned scorer 的原文定位（比我们想的更明确）

原文矩阵里 Learned scorer 与 Feature hash 并列为"待测"，
且原文 progress 说优先 block retrieval 或 feature/novelty
scorer。**我们的 `use_rp1_selector`（特征距离+λ·空间新颖性）
其实就是原文说的 feature/novelty scorer 的固定形式版**——
它已经在 P0 跑过（+2.46pp 单 seed，绑坏基线）。
全量重测时它应该作为**第六处理**一起进对照（成本=一个 flag）。

### 5.5 对五处理实验的最终修订（吸收 5.1-5.4 后）

1. 六处理（新增 use_rp1_selector 作处理 6：feature/novelty
   scorer 固定形式版）；
2. 训练时同步落 profiling 代理指标（选择分布 h5，每 N epoch
   采样一批，记录 Codex #4/#5 的逐 stage 坐标范围/clamp 率 +
   原文 §4.1 特征表的描述统计）；
3. 报告口径：只报排序与差距（vs random 锚点），不引用原文
   绝对数字；写论文时 RP1 引用其 headroom claim 而非 method；
4. topk_near 保持"高成本全局参照"地位（Codex 建议）不进
   selector-only 主对照，另列。

### 5.6 一个新的意外发现：RP1 的 iter7/8/9 已经探索过 boundary rerank / indexed selector

runs 目录里 iter7_boundary_rerank、iter9_indexed_b2 ——这些是
原文五候选之外的延伸 selector，**我们没有读过它们的 config 和
结果**。若其中有已验证有效的 selector，五处理清单还要再扩。
下一步：让学长确认这些 runs 的结果结论（服务器上 autodl 的
原始记录本地没有 config.json 内容）。

---

## 6. 学长确认（2026-10-15 09:35）：RP1 是未完成的工作

学长亲口确认：**这篇（RP1）论文是没做完的**。

这与 §5.2 的文档证据完全一致——progress-20260812.md 自己写着
"还不能声称已有 cheap selector 是最终方法"。两个独立来源互证：

1. 文档层：Oracle headroom 已证（68.03 > Local 59.30），
   但 selector 验证未完成（无多 seed、无 latency Pareto、
   block retrieval 等未完成有效 run）；
2. 学长层：直接确认未完成。

**这对我们的战略意义（重大利好，三个层面）**：

### 6.1 查新定位升级

RP1 = 一个**有 headroom 证据、没做完方法**的开放问题。
我们的 G16 六处理实验 = **直接接手把这个开放问题做完**——
不是"跟随已有方法"，是"完成一个悬而未决的问题域"。
CVPR 叙事从"我们提出 X selector"升级为：
"Remote event selection 是真实的 representation bottleneck
（引用 RP1 headroom），我们给出首个完整的多 selector 严格
对照 + 双曲几何准则（新维度）+ 在官方全量协议上的验证"。

### 6.2 学长反复强调"16 个点是论文灵魂"的深层含义

学长要的不是复现 RP1 的某个 selector——RP1 本身没做完。
学长要的是**把这个他未完成的核心想法做出来**：
oracle 证明了"16 个好点 > 全局 KNN"这个惊人事实，但
"怎么找到这 16 个好点"整个方法是空的。**我们的双曲距离
选点正是填补这个空位的候选答案**——这也是为什么
"双曲 + 16 点"的组合是愿景公式的灵魂。

### 6.3 iter7/9 延伸 selector 的价值重估

既然 RP1 未完成，iter7_boundary_rerank / iter9_indexed_b2
大概率是作者（学长线）探索到一半的尝试，**其结论对我们
极有价值**（可能已接近有效，也可能已证伪某方向）——
优先问学长要这两条的原始记录。

### 6.4 行动修订

- G16 实验叙事框架从"复现原文"彻底转为"完成 RP1"；
- 六处理对照 = RP1 §4.2 selector 矩阵的**首次完整实现**；
- 我们的双曲准则是 RP1 六候选之外的第七个（原创维度）；
- 论文 Related Work：RP1 作为 headroom 引用（内部/未发表
  来源按学长口径处理）。

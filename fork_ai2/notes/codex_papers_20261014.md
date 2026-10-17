# SECNet 新论文原文与主线实验筛选（Codex，2026-10-14）

范围：本地 `D:\论文` 按文件名查重，再从 Computer Vision Foundation (CVF) 官方 Open Access 下载未收藏的原文。仅在 `fork_ai2/papers/` 保存，不动豆包 docs、Claude 的进度/脚本或 HPC 模型。2026-10-14 核验首页、标题、会刊页面、页数和 SHA256；已能从正文提取方法与实验，不把检测 mAP 当本项目分类准确率。CVPR 2019 的 PAT 是为严谨查新补收的历史先例，不冒充新论文。

## 原文与出处

| 优先级 | 原文（本地） | 官方源 | 验证 |
|---|---|---|---|
| A | `papers/CVPR2025_SAMBLE.pdf` | https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_SAMBLE_Shape-Specific_Point_Cloud_Sampling_for_an_Optimal_Trade-Off_Between_CVPR_2025_paper.pdf | CVPR 2025 主会；11 页；SHA256 `68FA6CEECCD21B26185C4D9534A418DD9D97776DBD30440861725CD82E9F0569` |
| A- | `papers/CVPRW2025_MakingEveryEventCount.pdf` | https://openaccess.thecvf.com/content/CVPR2025W/EventVision/papers/Araghi_Making_Every_Event_Count_Balancing_Data_Efficiency_and_Accuracy_in_CVPRW_2025_paper.pdf | CVPR **Workshop** 2025，非主会；11 页；SHA256 `303E029F90AA3D83D50024370DB500ABEF0E3563AC5BB6DA119BEC72969CED6C` |
| B | `papers/CVPR2026_AdaptiveSpatialTemporalWindow.pdf` | https://openaccess.thecvf.com/content/CVPR2026/papers/Sui_Adaptive_Spatial-Temporal_Window_Unlocking_the_Potential_of_Event_Cameras_in_CVPR_2026_paper.pdf | CVPR 2026 主会；10 页；SHA256 `6C285B86EBCF336215FE6C98D9D9BAA4B6F6BFABF002CED5647ED3E7D7665F2A` |
| B- | `papers/CVPR2026_SpikeDrivenDiscreteAggregation.pdf` | https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Spike-driven_Discrete_Aggregation_for_Event-based_Object_Detection_CVPR_2026_paper.pdf | CVPR 2026 主会；10 页；SHA256 `4B30C73B22600238D467A1D562DBCE1549F9C3AA7E2434BD8EC93065FA09400E` |
| A (prior art) | `papers/CVPR2019_PAT_GumbelSubset_prior_art.pdf` | https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Modeling_Point_Clouds_With_Self-Attention_and_Gumbel_Subset_Sampling_CVPR_2019_paper.pdf | CVPR 2019 主会；10 页；SHA256 `C3D793BE55253C48FE4B9204DA78DDC365203CE474557146FE96D3B138232A3B` |

查重仅对两个本地论文目录文件名进行；它不能排除改名文件内容重复。PDF 下载 URL 在下载时可用，且五篇均已通过 pdfinfo/pdftotext 校验。随后从 CVF 官方 Open Access 检索到四篇新文的正式着陆页和会刊 BibTeX：SAMBLE `https://openaccess.thecvf.com/content/CVPR2025/html/Wu_SAMBLE_Shape-Specific_Point_Cloud_Sampling_for_an_Optimal_Trade-Off_Between_CVPR_2025_paper.html`；Araghi 等 `https://openaccess.thecvf.com/content/CVPR2025W/EventVision/html/Araghi_Making_Every_Event_Count_Balancing_Data_Efficiency_and_Accuracy_in_CVPRW_2025_paper.html`；ASTW `https://openaccess.thecvf.com/content/CVPR2026/html/Sui_Adaptive_Spatial-Temporal_Window_Unlocking_the_Potential_of_Event_Cameras_in_CVPR_2026_paper.html`；SDA `https://openaccess.thecvf.com/content/CVPR2026/html/Li_Spike-driven_Discrete_Aggregation_for_Event-based_Object_Detection_CVPR_2026_paper.html`。本机对 CVF HTML 的 HEAD 请求超时，但官方搜索索引与本机下载原文相互印证。不要把与文献相近的方法包装成首创；会议发表不等于已证明可迁移。

### 查新勘误：已有本地双曲事件工作

初始扫描曾写“事件相机 × 双曲 = 空”，这是错误的，已撤回。`D:\论文\2504.16616.pdf` 是 EHGCN（arXiv:2504.16616v4，2026-07-30），标题为 *EHGCN: Hierarchical Euclidean-Hyperbolic Fusion via Motion-Aware GCN for Hybrid Event Stream Perception*。它把分布感知事件筛选、运动感知超图和欧氏-双曲融合用于事件感知；正文包含 N-Cars/DVS128 Gesture 识别和 N-Caltech101/Gen1/EVMars 检测。N-Cars 的 Table 1 在其他两个组件都关闭的作者协议下由 0.945 提升到 0.954；但表注说明这并非单一 embedding 替换，而是整个 Euclidean-hyperbolic fusion 取代 AEGNN 基线，因此不能当成纯几何机制的独立因果效应。它是 arXiv 预印本，不是已核实的会议正式发表证据，也不是 SECNet/N-Caltech101 单窗口分类的可比成绩，但足以证明“事件 + 双曲”不是空白。

对 SECNet 的影响：双曲贯穿叙事必须明确区别于 EHGCN 的 dual-space GCN/运动超图；我们的可辩护空位应改为“面向事件点云分类的双曲远程选点/跨阶段采样归纳偏置”，而非泛称“首次把双曲用于事件”。引用该工作前，应在正式稿中核对其 arXiv 版本、代码和实验协议；不得只引用文件名。

**对协作调研的再核验（不要写成已证实的新颖性/SOTA）**：`D:\论文\2610.09873v1.pdf` (*Bringing BNNs to Fast Event Processing*，arXiv 2026-10-07) 的最佳 BNN 90.58% 是 Table 5 中 BNext + PEV6、ImageNet 预训练、300ms 全窗，且 Section 4.3 将 EventBind 的 8:2 **验证集当 test**，并使用 EventBind 教师蒸馏；同一原文 Table 1 已列 TMKT 97.93% 和 GEP 96.47%（协议不同），所以 90.58% 不能称“最新最高成绩”或无预训练基线。`D:\论文\2608.22238v1_Hyper2_Dual_Space_Consistency.pdf` (Hyper^2，2026-08 预印本) 在**点云补全**只验证共享 arcosh 距离的 encoder 位置偏置和 Chamfer loss 两端一致性；其自身 Related Work 明说迁移到其他 3D 任务尚未检验。不能据此断言 SECNet 的双曲选点+聚合+分类头“三件套”优于单点组件，也不能断言分类头应改成何种公式。若要探究几何一致性，应先给共享曲率、相同计算预算、同训练种子做最小 2x2 对照；分类标签层级是否存在需独立证据。官方原稿可核 `https://arxiv.org/abs/2610.09873` 与 `https://arxiv.org/abs/2608.22238`。

## 逐篇可迁移性与限制

1. **SAMBLE（Wu 等，CVPR 2025）**：稀疏注意力图估计逐点采样分数，再用可学习 bin 边界和 bin 配额平衡局部细节/全局均匀，不是简单 top-M。主文 Table 2 在 ModelNet40 原始坐标分类报告 94.2% OA（APES global 93.8%）；**CAD 形状数据，不是事件点云**。我们的真实事件批次 G16 诊断显示空间均匀选择覆盖强、池内最近邻聚团，但覆盖是中间指标而非准确率。可借鉴“局部相关性评分 + 覆盖配额”的低成本选择器，不能直接移植其稠密注意力图或声称 94.2% 会转移。它对 FPS 中心采样也有启发，但与当前 G16 邻域选择是两个不同干预点。
2. **Making Every Event Count（Araghi 等，CVPR 2025 EventVision workshop）**：比较随机、时空、角点、因果局部密度等事件子采样，N-Caltech101/DVS-Gesture/N-Cars 的稀疏区间显示密度选择有潜力，同时报告空间相位偏置与视频事件数方差的失败模式。任务虽涉及 N-Caltech101，却是 CNN 事件视频、不同输入/划分与采样预算；不能拿其曲线直接和 SECNet 934 样本数字比。EHGCN v4 **已引用** Araghi 等 (2025)，所以“密度选择是新颖性空白”的说法也不能成立。给我们的明确反证机会：如果统一输入点数后分类错误集中在低密度/短事件序列，简单密度优先会压掉边界/少数类信息。密度特征需与覆盖约束联合，而非绝对偏向高密度。
3. **ASTW（Sui 等，CVPR 2026）**：以 patch 事件密度选择时间窗口，形式 `clamp(gamma/(smoothed_density+epsilon), min, max)`，使用空间平滑和历史 EMA，原文是检测/跟踪和速度异质场景。我们的 10240 点固定输入经预处理，改窗口意味着更换训练/测试样本本身，难与网络组件效果分离。先审计原始流、时间跨度、每图的事件数与早晚窗类别混淆；若不存在速度/时间分布漂移则止步，不直接加新模块。
4. **SDA（Li 等，CVPR 2026）**：门控递归脉冲单元按放电稀疏选择有信息的事件，加多时间尺度融合用于**检测**。Gen1 同 20ms 分辨率的离散聚合对连续聚合在作者架构上 +1.6 mAP50:95；不能外推为 N-Caltech101 分类 +1.6pp。若 SECNet 的错误与噪声/时间冗余关联，可先试简洁无参噪声/极性诊断；否则与已有 Spike/time_enc/KDA/注意力路径重叠，排在后面。
5. **PAT/GSS（Yang 等，CVPR 2019）**：这是必须纳入 Related Work 的直接先例，不能把“可学习 16 点选择”泛称为空白。Gumbel Subset Sampling 在高维特征上学习可微的硬子集，声称避免 FPS 的初始点依赖和低维欧氏坐标敏感性；其 DVS128 Gesture 表 5 中 PAT(FPS+GSS) 为 97.4/96.0（10/11 类），但这不是 N-Caltech101，也不是当前 LRHyperNet 的 G16 远程邻居问题。可借鉴为**候选池内可学习选择器**的后续对照：先固定池和 K=16，比较无梯度 top-k 与 Gumbel straight-through；必须匹配参数/训练预算、固定 eval_seed，并把它与空间覆盖约束分开。由于方法已有发表，SECNet 的新颖性应落在事件点云中双曲/时空联合的远程邻居准则与机制证据，而不是“第一次 learnable sampling”。

## 推荐近期仅做两项诊断（均先不提交训练）

**A. G16 的“相关性 × 覆盖”联合机制（优先于新训练）。** 先等 173720 V100 显存测试并核对完整 LRHyperNet 前向；该作业目前仅测池 + selector，不是完整模型显存。共享同一远程候选池、16 点预算和固定 RNG，在冻结特征上比较已设计的 random、pool-near、pool-far、spatial-uniform、hyper-far；增加一条**受覆盖约束的 pool-near**候选，仅在诊断支持后写代码。对每例/每层记录中心-邻居空间距离分布、特征相似性、16 格覆盖、重复/与局部 KNN 重叠、选择耗时及 V100 峰值显存；在已有验证划分或独立留出样本上评估，**不得用 934 测试集筛方案**。若联合规则并不改善相关性-覆盖权衡或成本超过已实测资源门槛，即止损；再由 Claude 评估是否需要同训练 seed 7/13 的配对训练、固定 eval_seed 42 的标签顺序校验和 exact McNemar；多次候选比较须注明校正策略。这里引用 SAMBLE 为归纳偏置，不宣称等价于其学习式分箱，也不混淆 pool-near 与全局 KNN。

**B. 输入密度/时间异质性的零训练审计。** 在不改变训练集和已运行作业的前提下，从可用原始事件元数据或缓存提取每样本局部密度、各时段事件数、正负极性、空间占用和输入裁剪/补齐率。只在缓存能证明有原始元数据时统计真正原始事件数；若不可得，明确记录数据缺口，绝不能用截断后点数假冒原始密度。用现有固定 seed42 的**训练外**预测分组分析误差，作为生成假设而非选超参数的测试集；密度/时间段差异若与错误关联，再在训练/验证划分预注册固定预算的随机与密度/覆盖兼顾采样对照，随后在封存测试集只评一次。记录宏平均类别准确率、总体准确率、选择耗时和类别/事件数分层。若稀疏样本被进一步丢失，或差异仅来自预处理截断，则停止。文献依据为 workshop 2025 和 ASTW；不引入 SDA 整个 SNN 检测器。

**HPC 只读核验**：`ncaltech101_official_baidu/test.h5` 的键只有 `data: (934,10240,4) float32`、`label: (934,) int16`、`mark: (934,) int16`，`mark` 前十个互不相同，因此不应未经验证就解释为“原始事件数”。样例的时间维取值 [0,1]、极性 [-1,1]，但这仅是前 16 个样本，不代表全体/原始动态范围。若要按“原始事件数”分层，应找到生成 H5 的源数据与预处理脚本、核实 `mark` 的定义；目前最多对**缓存后的空间密度/时间分布**作诊断。

**现有方案的一个算术/统计审查点**：`notes/coords_pilot_design.md` 写 `{0.5,1,1.64,2}^3` 却称 27 点；实际上是 **64** 组合。若只做 27，必须预先定义 3 值/轴或具名稀疏子集。对同一 934 测试集扫 64 个尺度并选最好会泄漏测试集/造成多重检验；应在训练划分出的验证集做筛选，最终封存测试仅评预注册候选。还需确认逐层 fps_parameter 与原始输入坐标缩放的 hook 是否严格等价，记录各 stage 的坐标传播和 clamp 比率。此诊断不可替代跨训练 seed 复核。尤其所谓“固定 checkpoint 扫尺度”的推理优化可能把模型推到训练未见状态，只能提供敏感性/机制线索，不能替代训练对照。

## 当前 HPC 快照（2026-10-14 晚间再次核查，以 squeue/sacct 为准）

- RUNNING：173249 seed7 baseline、173236 hyper-MLP、173237 curvature、173238 GroupNorm；后二者中 173236 缺 attention 开关、173237 curvature no-op，不能作为有效组件结论。
- PENDING：173250 seed7 attention+RP1，173351 frozen FPS，173366/173367 seed13 配对，173386/173387 scaled/raw 选点。**173720 G16 V100 显存测试已 COMPLETED (exit 0, 19s)**：池峰值 1968 MiB、单批池构造 204ms；选择器峰值 random 1182 MiB / spatial 1344 / Euclidean far-near 3134-3136 / Lorentz far 5306 MiB，选择器延迟 2.6/8.7/10.0/47.2ms 每批，`ALL_TESTS_PASS`。此脚本用合成随机 xytp/特征/索引，只测候选池 + selector，不是完整 LRHyperNet 前向/反向；GPU 余量不能由 5306 MiB 单独决定，接线训练前需要一次完整 B=8 前向+反向测峰值，并验证候选特征与中心语义对齐、梯度路径和参数预算。**不取消、不重提、不为占 GPU 随意运行**。
- 本次没有新固定 seed 预测或 exact p；已有基线 46.1456%，RP1 48.6081% (p=.0398)、KDA 47.7516% (p=.2246)、attention+RP1 50.1071% (p=.0023)、scale-normalized FPS 47.9657% (p=.1250)，均仅为单训练 seed、未做多重检验调整，不能视为跨 seed 定论。检验和标签对齐见 `fork_ai2/codex_review_20261014.md`。

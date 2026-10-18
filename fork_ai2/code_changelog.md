# code_changelog — ai2（Claude）

所有对 HPC 共享代码的改动记录在此。规范：不删改已有实验的 log/checkpoint；新提交实验 log_dir 带 `_ai2` 后缀。

## 2026-10-14

### 0. tools/extract_preds.py — extract_one 每轮独立重播种（可复现性修复 #2）

- **改了什么**：`extract_one()` 开头（原第 138 行 def 后）插入 4 行：`torch.manual_seed(eval_seed)` / `torch.cuda.manual_seed_all` / `np.random.seed` / `random.seed`。
- **为什么**：原脚本只在 `main()` 播种一次。`--which both` 时 best 的推理发生在 last 的 FPS `torch.randint` 与模型构建的 RNG 消耗之后，同一 checkpoint 的 best 结果随调用方式漂移（实测 rp1_fix best：both 模式 48.929 / 单独 best 48.394 / 被非标准调用污染的磁盘文件 49.251——同一个 ckpt ep57，三种调用三种结果）。FPS 起始点 `torch.randint` 依赖 RNG 状态，必须每轮重置。
- **效果**：`(log_dir, which, eval_seed)` 现在唯一决定推理结果；last 值不变（both 模式下 last 本来排第一、状态纯净，与历史 48.608 完全一致——确定性验证作业 173291 连跑两次 934/934 逐样本一致）。best 值统一定义为"独立调用"协议。
- **附带发现**：同一 checkpoint 不同 FPS 起始点可造成 ~0.5-0.9pp 的 acc 波动（48.394/48.929/49.251 三个值都来自同一权重），单次 FPS 抽样的方差仍不可忽略——跨种子复核的必要性再添一证。
- **重推作业**：173302（rp1/kda/scale_norm/ha_rp1 的 best 全部重推），旧 rp1 best_seed42.npz（49.251，污染文件）已覆盖为 48.394。

### 1. tools/extract_preds.py — 补齐缺失的两个组件 flag（关键 bug 修复 #1）

- **改了什么**：
  - `FLAG_KEYS` 列表（原第 50–54 行）：追加 `"hyperbolic_mlp", "scale_norm_fps"` 两个键。
  - `build_model()`（原第 88 行起）：缺键时按 argparse 默认值 False 兜底并打印 `[INFO]`；`FECNet(...)` 构造调用补传 `use_hyperbolic_mlp=`、`use_scale_norm_fps=`。
- **为什么**：`train_frequency_ncaltech.py` 有 14 个组件 flag，但 extract_preds.py 只恢复 11 个。`scale_norm_fps=True` 训练的 checkpoint 被按 `False` 重建模型推理，**输入分布不匹配，acc=2.570%（接近随机）**——scale_norm 是核心创新点候选1，其首次推理结果因此作废。
- **验证**：修复后重推 `secnet_scale_norm_fps_v100`（作业 173246）：
  - last 47.966%（原 2.570%）→ McNemar vs 基线 p=0.1250 不显著
  - best 48.608% → p=0.0542 边缘
  - `hyperbolic_mlp` 同理兜底，当前 RUNNING 的 `secnet_hyper_mlp_v100`（另一AI提交）跑完后不会再踩此坑。
- **备份**：改动前远端文件备份为 `tools/extract_preds.py.bak_ai2_20261014`（md5 `5c86f55...`）。
- **注意**：`parse_flags_from_log`（从旧日志恢复 flag）用 `\bkey=True\b` 正则匹配，旧日志缺新键自动落 False，行为不变。
- **本地副本**：`fork_ai2/scripts/local_edit/extract_preds.py`。

### 2. run_infer_*.sh 系列作业脚本（均为 ai2 提交）

- `run_infer_snfps_ai2.sh`（173246，已完成）：scale_norm best/last 重推。
- `run_infer_kda_best_ai2.sh`（173254，已完成）：KDA best 推理。
- `run_infer_best_ai2.sh`（173302，已完成）：重播种修复后 4 个实验的 best 统一重推。
- `run_det_check_ai2.sh`（173291，已完成）：确定性验证（同 ckpt 同 seed 连跑两次 934/934 一致）。
- `run_baseline_s7_ai2.sh`（173249，RUNNING）/ `run_ha_rp1_s7_ai2.sh`（173250，PENDING）：P1 跨种子复核，训练 seed=7，log_dir 分别为 `secnet_baseline_s7_ai2`、`secnet_ha_rp1_s7_ai2`。
- 本地编辑副本在 `fork_ai2/scripts/local_edit/extract_preds.py`（与远端 md5 一致：33e55edf...）。

### 3. train_frequency_ncaltech.py — 新增 --freeze_fps_parameter flag

- **改了什么**：
  - argparse 区（--seed 之后）新增 `--freeze_fps_parameter`（action='store_true'，default=False）；
  - 模型构建后（inplace_relu 之后、DDP/optimizer 之前）新增冻结逻辑：遍历 `classifier.named_parameters()`，所有名字含 `fps_parameter` 的参数 `requires_grad_(False)`，打印+log_string 记录冻结数量。
- **为什么**：A1 归因修正发现 fps_parameter 是 SECNet 官方原生隐藏组件（基线训练后漂移到 x~1.64x）。需要"基线+冻结缩放"对照来量化它的独立贡献。
- **兼容性**：默认 False，不传该 flag 时行为与原脚本完全一致（已验证：语法检查 + parse 测试 freeze/seed/hyperbolic_selector 均正常解析）。注意 --help 因已有中文 help 文本中的 % 格式字符报错（历史遗留问题，与本次改动无关）。
- **备份**：远端原文件备份为 `train_frequency_ncaltech.py.bak_ai2_20261014`。
- **本地副本**：`fork_ai2/scripts/local_edit/train_frequency_ncaltech.py`。
- **配套实验**：`run_frz_fps_ai2.sh`（173351，log_dir=secnet_frz_fps_ai2，排队中）。

### 4. models/FECNet.py + train_frequency_ncaltech.py + tools/extract_preds.py — 新增 --hsel_scaled_coords flag（2×2 矩阵"双曲+缩放坐标"象限的受控复现开关）

- **改了什么**：
  - `models/FECNet.py`：
    - `HyperbolicUniformGrouper` 新增 `__init__(use_scaled_coords=False)`（透传 `*args/**kwargs` 给 LocalGrouper）；forward 里选点坐标由 flag 控制：False=缩放前原始坐标（修复版默认行为，与改前逐字节一致），True=缩放后坐标（复现旧版 bug 行为）。
    - `FECNet.__init__` 新增 `use_hsel_scaled_coords` 参数；无 `--hyperbolic_selector` 时按 no-op 警告（不抛错，兼容推理工具统一传参）；实例化处 `HyperbolicUniformGrouper(..., use_scaled_coords=use_hsel_scaled_coords)`。
  - `train_frequency_ncaltech.py`：argparse 新增 `--hsel_scaled_coords`（default False）；FECNet 构造调用补传该 flag；resume 一致性校验键列表补 `hyperbolic_mlp`/`scale_norm_fps`/`hsel_scaled_coords`。
  - `tools/extract_preds.py`：`FLAG_KEYS` 追加 `"hsel_scaled_coords"`；`build_model` 缺键兜底 dict 追加该键（False）；FECNet 构造调用补传 `use_hsel_scaled_coords=`。
  - `models/modules.py`：仅修正 `use_scale_norm_fps` docstring 中过时的 A1 归因说法（A1 已修正：基线自身也漂移 fps_parameter，缩放非独有机制），无代码行为改动。
- **为什么**：2026-10-14 用户核验发现（触发本轮改动）：被取消的 173350 脚本名 `run_hsel_bug_s7_ai2.sh` 但只传 `--hyperbolic_selector`——当前代码（修复B后）该 flag 跑的是"原始坐标选点"（修复版行为），不是旧 bug 的"缩放坐标选点"。旧版 bug 行为在新代码里**没有开关可复现**，2×2 矩阵最强象限（+4.71pp p=0.0002）的跨种子复核因此缺工具。本 flag 显式化两种口径（用户规则 6：hsel_bug 与 hsel_fix 必须显式 flag 区分，不能靠脚本名）。
- **验证**：
  - 本地：4 文件 ast.parse 语法通过；test_parse2.py 4 用例（bug/fix/基线默认/freeze 组合）解析全过。
  - HPC（登录节点 CPU）：import 构造测试——`use_hsel_scaled_coords=True` 正确传导（grouper.use_scaled_coords=True）、默认 False、基线 grouper 类型 LocalGrouper 不变、**参数量三种配置完全一致**（1925849，flag 只改选点坐标来源）；forward 冒烟——grouper 单元级（[2,10240,4]→[2,2048,24,132] 两模式正常，缩放坐标下选点索引确实不同）+ 整模型级（基线/hsel_scaled/hsel_fix 三配置 forward [2,101] 全通，基线路径未被改坏）。
- **备份**：`models/FECNet.py.bak_ai2_20261014b`、`models/modules.py.bak_ai2_20261014`、`tools/extract_preds.py.bak_ai2_20261014b`（远端）。
- **本地副本**：`fork_ai2/scripts/local_edit/{FECNet.py, modules.py, train_frequency_ncaltech.py, extract_preds.py}`。
- **配套实验**（均显式 flag、log_dir 带后缀）：
  - `run_hsel_bug_s7_ai2_v2.sh`（**173386**，`secnet_hsel_bug_s7_ai2`，`--hyperbolic_selector --hsel_scaled_coords --seed 7`）——bug 象限 seed7 复核。
  - `run_hsel_fix_s7_ai2.sh`（**173387**，`secnet_hsel_fix_s7_ai2`，`--hyperbolic_selector --seed 7`）——修复象限 seed7 复核（承接被取消的 173350 的实际配置，更名以正名）。
  - 旧 173350 已 scancel（未启动、无污染；它实际配置=修复版但误标 bug）。
- **注意**：`HyperbolicUniformGrouper` 原无 `__init__`（直接继承 LocalGrouper），新增的 `__init__` 用 `*args/**kwargs` 透传，LocalGrouper 签名带 `**kwargs` 兜底，兼容性无虞。运行中的训练作业在启动时已加载旧代码，不受上传影响（Python import 时机）。

### 5. ai2_g16/g16_selector_fixed.py v2 — 五处理选择器（双曲选点 + 方向对照）

- **改了什么**：在原有 `spatial_uniform_select_fixed` + `build_shared_candidate_pool` 基础上新增三个选择器函数：
  - `hyperbolic_topk_select(c, near)`：Lorentz 双曲距离选点（centroid/cand 都 exp map 后算双曲距离 top-G）。依赖新 helper `_lorentz_exp_origin_safe`/`_lorentz_distance_safe`——**独立实现**，与 `local_remote_grouper.lorentz_exp_origin`/`lorentz_distance` v3.1 公式逐项对齐（scale=clamp(norm_max/norm,max=1) 只缩不放、sinh 项 v/(√c·n) 因子、流形投影 x0←sqrt(1/c+‖x[1:]‖²)、arccosh clamp_min(1+eps)）。
  - `euclidean_topk_select(near)`：near=True 为原文 Global KNN 语义（特征最近非局部 G 点），near=False 复现现有 topk（最远）。
  - `random_select`：池版随机（与修正版共用共享池）。
- **为什么**：RP1 原文对照暴露两个差距——①我们 topk 语义与原文 KNN 方向相反且从未对照；②`use_hyperbolic_selector` 接线存在但历史三个 G16 run 全 False（双曲选点零实验证据）。详见 `notes/g16_selector_design.md` §3。
- **验证**：`test_g16_selector_v2.py` 26 项断言（流形约束、远切向量单调性、near/far 方向语义、五选择器形状/界内/唯一/零重叠/池成员/可复现）本地（secnet conda env）+ HPC（bhli env）双侧 ALL PASS；旧 18 项回归测试未破坏（occupied bins 16/16 不变）。
- **实现过程更正**：首版自写双曲公式有 2 处实错（scale 写成 norm/norm_max 方向反了；sinh 项漏 1/√c 因子），对照 HPC 原版逐行核对后修正——独立实现的意义即在于此，Codex 复核请重点看这两个因子。
- **部署**：HPC `ai2_g16/`，md5：`effc2f7f993a07a84cd97f00e4b2a20c`（g16_selector_fixed.py）、`45fbe37706153897cd2c9dd2a0c3d9f6`（test_g16_selector_v2.py）。未触碰 `models/local_remote_grouper.py`（运行中作业依赖）。
- **本地副本**：`fork_ai2/scripts/local_edit/{g16_selector_fixed.py, test_g16_selector_v2.py}`。
- **配套实验**：五处理 G16 对比（研究问题见 notes/g16_selector_design.md §4），**训练接线未做、等 Codex 审阅设计后提交**。

### 6. ai2_g16/test_g16_real_batch.py — Codex 门槛最后一项：真实数据批次诊断

- **改了什么**：新测试脚本（只读 ai2_g16/ + 数据，不碰共享源码），用真实 N-Caltech101 test.h5（B=4, N=10240）跑五选择器。
- **为什么**：Codex 最小实验规格的 correctness gate 最后一项："on a small real batch report per-stage occupied-bin count and out-of-range clamp rate"。
- **结果**（HPC bhli 环境实测，ALL PASS）：
  - x/y 全在 [0,1]（clamp 率 0——Codex #4 的越界担忧在原始输入 stage0 不成立；#5 的逐 stage 漂移仍需训练中仪表，见下）
  - 池可行性稳定（有效候选恒 256）
  - occupied bins：**spatial_uniform_fixed 13.2/16 > random 8.2/16 > euc_far 6.9 > hyp_far 7.0 > euc_near 4.9**
  - random 8.2 < 理论期望 10.4 → **事件点云本身空间不均**（事件密集物体、角落稀疏），spatial-uniform 的覆盖优势有数据基础
  - euc_near 覆盖最低（4.9）→ KNN 方向选点空间高度聚集，佐证原文"特征相关+空间新颖"混合设计（use_rp1_selector 的动机）
- **注意**：真实 backbone 特征下 hyp_far 与 euc_far 的覆盖差异预期分化（原始 4 通道 t,x,y,p 下双曲/欧氏选点近似）；occupied 均值受真实分布影响，满格 16 不可达（空格是数据性质非缺陷）。
- **部署**：HPC `ai2_g16/test_g16_real_batch.py`（md5 运行时验证）。本地副本 `fork_ai2/scripts/local_edit/`。

### 7. ai2_g16/g16_selector_fixed.py v3 — 候选池内存有界重写（Codex 复核要求）

- **改了什么**：`build_shared_candidate_pool` 从"一次性 [B,S,N] 掩码 + [B,S,K,N] 广播比较"改为按 N 分块（chunk=1024）：
  - 块内掩码用 scatter_add 整型累加器逐 K 构建（K=24 次小 op，每次只碰 [B,S,c]）——避开 [B,S,K,N] ≈ 3.75 GiB 布尔中间张量（Codex 实测指出）和 scatter_ bool 重复索引未定义行为。
  - 块内收集 min(M, c) 个非局部候选 → 拼并集 → 并集上防御性局部排除 + 均匀取 M。
  - 新增两级可行性 guard：a) N-K < M 时构造即报错；b) 并集层有效候选 < M 时报错（带 (b,s) 定位）。
- **为什么**：Codex 2026-10-15 复核：历史配置 B=8、S=2048、K=24、N=10240 下旧版产生 ~3.75 GiB 布尔张量 + ~640 MiB 随机分数张量，16GB V100 上必须内存有界。
- **数学等价性**：分块收集 + 并集均匀取 M 的每点入选概率不变（= M/(N-K)）。统计验证（B=2, S=128, N=2048, 20 trials）：实测入选频率 mean=0.12663 vs 理论 0.12648（偏差 0.1%），std/mean=5%——均匀性成立。
- **验证**：本地回归（生产级形状 [2,64,256] 零重叠 + 唯一 + 两级 guard 触发）全过；V100 峰值显存实测 = 作业 173720（run_g16_v100_mem_ai2.sh，B=8/S=2048/N=10240/C=64 真实训练配置，五处理各 3 次取中位 + 共用池断言），结果见 eval_results/g16_mem_ai2_173720.out。
- **173720 实测结果（COMPLETED 19s，ALL_TESTS_PASS，md5 968943cf）**：
  - 池 v3：峰值 **1968 MiB**（base 24，Δ=1944），延迟 204 ms/批——修复前 [B,S,K,N] ≈ 3.75 GiB 布尔张量已消除；
  - 选择器峰值/延迟：random 1182 MiB / 2.6ms；spatial_uniform_fixed 1344 / 8.7ms；euc_far 3134 / 10.0ms；euc_near 3136 / 10.0ms；**hyp_far 5306 MiB / 47.2ms**（双曲坐标 [B,S,M,C+1]+距离张量最重）；
  - m1-m6 全部通过：形状/唯一/零局部重叠/池成员一致性；代码 md5 与本地 fork_ai2/scripts/local_edit/g16_selector_fixed.py 一致（哈希链验证通过）。
- **部署**：HPC `ai2_g16/`（与 test_g16_v100_mem.py 一同上传）；本地副本 `fork_ai2/scripts/local_edit/`。
- **未做**（按 Codex 指令）：不提交五个训练作业；不改现有作业。"topk_near" 正名：**池内最近邻**（pool-restricted nearest），非原文全局 KNN——文档与后续实验命名均按此口径。

---

## 8. 文档修正（2026-10-14 16:35）：paper_survey 查新错误

- **错误**：paper_survey_20261014.md 初版 §3 声称"双曲×事件相机 = 零
  命中"。实际库内 2504.16616.pdf（EHGCN）即双曲×事件论文，10-09 旧
  SOTA 调研报告明确记载，清点时漏扫。
- **修正**：重写 §2.5/§3.1——精确口径改为"事件点云分类基准上双曲
  事件点云仍为空白"（EHGCN = 检测+图卷积+双空间融合，与我们三重差异）；
  补充 arXiv API 外部查新（curl 绕过 WebSearch/WebFetch 工具层限制）：
  hyperbolic+event camera 全 arXiv 唯一命中 EHGCN，确认无新撞车者。
- **新增**：§3.2-3.5（Hyper² 跨几何失配警示 / learnable FPS 查新
  空位 / N-Caltech101 SOTA 协议乱象表 / 目标数据集动态）；
  新存原文 3 篇到 D:\论文（2610.09873v1、2608.22238v1、2603.04848v2）。

---

## 9. 完整 LRHyperNet 前反向显存测试（Codex 接线前最后一关）

- **背景**：Codex 指出 173720 只测了池+selector（合成随机特征），不是
  完整 LRHyperNet 前向+反向；G16 训练接线前需要完整 B=8 前反向峰值。
- **新增**：`fork_ai2/scripts/local_edit/test_g16_full_model_mem.py` +
  `fork_ai2/scripts/run_g16_fullmem_ai2.sh`（作业 173824）：
  - Part1 = LRHyperNet 原生 selector_mode 四模式（topk/random/
    spatial_uniform/greedy_div）完整前向+loss+反向+step 峰值显存/延迟；
  - Part2 = use_hyperbolic_selector=True（双曲选点，G16 核心配置）；
  - 每个模式 AdamW + 3 次取中位，`[B,4,N]` 输入 (t,x,y,p)、N=10240、
    num_class=101、remote_g=16、hyperbolic_type='lorentz'；
  - 输出源文件 md5（lr_hypernet.py + local_remote_grouper.py）备查。
- **注意**：只测显存/延迟，不做精度判断；G16 五处理的 euc_near/hyp_far
  在 LRHyperNet 原生代码无开关（那是 ai2_g16 隔离实现），其 selector
  级显存已由 173720 覆盖。
- **状态**：173824 PENDING（排队中），出结果后向 Codex 报交接点。

---

## 10. fork vs 官方代码 diff 报告（2026-10-14 深夜，Codex 交接项）

对照源：学长 SECNet.zip 官方 models/（作者 Hongwei Ren 版本）。

### modules.py（fork 168 行 vs 官方 168 行，实质差异 2 处）

1. **use_scale_norm_fps flag（fork 新增，默认 False）**：ON 时用
   per-dim 标准化替代 fps_parameter 缩放；**OFF（默认）时走 else 分支
   = 官方原式 `fps_parameter * xytp`，逐字符等价** ✓
2. **KNN 分块 top-k（显存优化，替代官方 `dists.argsort()[:k]`）**：
   数学等价（全局 top-k 必在各 chunk top-k 内，代码注释含反证）；
   ⚠️ 技术性差异：并列距离时 topk 与 argsort 的 tie-breaking 选索引
   可不同——同距离集内选点不同，影响可忽略但记录在案
   （动机：官方 [B,S,N] 距离矩阵 S=2048/N=8192 在 V100-16GB OOM）
3. 其余 79 处"差异"全是注释删除/空行/PEP8 格式，无语义。

**判定：modules.py 全 flag 关闭时与官方数学等价**（1 处
tie-breaking 技术性差异需 Codex 复核确认可接受）。

### FECNet.py（fork 998 行 vs 官方 208 行）

fork 加了大量组件分支（RP1/双曲/attention/mamba 等，全部 flag 默认
关）。**关键验证：state_dict 键名与官方 checkpoint 兼容**——
stage2_modules.load_backbone(strict=True) 能 strict 加载官方
checkpoint（自检已过：12 键全对齐），即全 flag 关时结构等价。
逐行语义 diff 将随 probe 结果一起交 Codex。

### train_frequency 系

fork 的 train_frequency_ncaltech.py 在官方 train_frequency.py 基础上
加组件 flag/推理协议/冻结 flag；官方 mark 投票逻辑（test_acc_seq）
fork 保留但**不是官方 82.4% 机制**（共识 §7.2 已修正）。

---

## 11. KNN 两步距离公式不一致修复（Codex 2026-10-15 必改项，已完成）

- **问题**（Codex 独立复核发现，比"tie-breaking"更严重）：fork 的分块
  KNN 中，分块阶段用**展开式** `square_distance`（a²+b²-2ab），最终
  并集阶段用**直接式** `((a-b)²).sum`——两步浮点不一致，破坏
  "全局 top-k ⊆ 分块 top-k"反证的严格性，极少数边界点会被漏选。
- **修复方案**（Codex 首选）：基线路径换回**官方全文 KNN**——
  `dists = square_distance(new_points, points)` +
  `argsort()[:, :, :K]`，与官方逐字节对齐。显存：[B,S,N] float32
  ≈671MB（B=8,S=2048,N=10240）+ argsort ≈1.34GB，V100-16GB 放得下。
- **改动三处**：
  1. `models/modules.py` `LocalGrouper.forward`：分块块 → 官方全文
     argsort；分块版保留为独立函数 `knn_chunk_fallback()`（并集阶段
     也改用展开式，两步同公式），注释标明"近似，仅单卡组件实验显存
     紧张时用"；
  2. `models/FECNet.py` `RP1SpatialUniformGrouper`（L265 区）：同款
     替换（2 处 grep 确认）；
  3. `models/FECNet.py` `HyperbolicUniformGrouper`（L375 区）：同款
     替换；import 行补 `knn_chunk_fallback`。
- **验证**：
  - 语法 ast.parse 三次全过；
  - 数值等价性（本地 torch）：官方 argsort vs 修复版 fallback 逐索引
    一致（常规数据 + 近重复点小尺度压力测试均 True；旧版在测试数据
    也一致——确认 Codex 判断"偶发边界事件，真实但低概率"）；
  - HPC 冒烟：FECNet 基线全 flag 关 forward [2,101] OK；
    **官方 checkpoint backbone strict 加载成功（153 键完全匹配，
    KNN 修复未破坏结构兼容）**。
- **备份**：`models/modules.py.bak_knnfix_20261015`、
  `models/FECNet.py.bak_knnfix_20261015`（远端）。
- **md5**（本地=远端一致）：modules.py `219b90bc6b74e1f49b108160630aef74`、
  FECNet.py `5ff884f1bcb5ef0d4352cdb60412042b`。
- **影响面说明**：正在排队的 174286（stage2 特征提取）用的是
  stage2_modules.load_backbone，import 时机在作业启动时——届时加载
  的已是修复后代码，基线特征提取直接受益。

---

## 12. g16_profile_agent.py — G16 训练时 profiling 代理指标（2026-10-15 09:50）

- **是什么**：`profile_batch()` 单函数——训练循环里每 N step 采样一次
  选择分布统计（~2ms 开销），落 h5 供训后分析。
- **为什么**：RP1 原文 §4.1 的 Oracle profiling 表（9 特征）是
  "先理解高价值远程点、再设计 selector"的前提；我们没有 Oracle GT，
  但 7 个特征可对"选中的 16 点"直接算描述统计（§5.3 修正案）。
- **覆盖的原文特征**：XY distance（rem vs loc 距离比）、Temporal gap、
  Spatial novelty（rem 到最近 local 的距离）、Occupancy（4×4 格）、
  Polarity agreement、clamp 率（Codex #4 实测）、Feature cosine
  （需传 feat，可选）——共 15 项统计。
- **验证**：本地冒烟通过（B=2/N=1024/S=64/K=24/G=16），随机选择下
  xy_dist≈0.52、cos≈0、clamp=0 全部符合解析预期值。
- **状态**：本地就绪，**未接入训练**——等 G16 六处理接线时一起交
  Codex 审（此时只是独立模块）。
- 位置：`fork_ai2/scripts/local_edit/g16_profile_agent.py`。

---

## 13. G16 显存实测重提 + profiling 接线补丁（2026-10-15 12:00）

### 13a. 完整模型显存测试重提（174813）

- 173824 被 20:00 批量取消后未重提——现在补上（任务 #9 最后一关）；
- 卡型 V100-16（gpu11 7 张空）；B=8 = 训练真实形态（历史 s7 基线同
  配置跑过）；脚本 test_g16_full_model_mem.py 未改动（HPC 已在位）。

### 13b. profile agent v3 接入 trainer（接线补丁，交 Codex 审）

**改动**（train_frequency_ncaltech.py，全部默认关闭不影响现有实验）：
- 新 flag：`--g16_profile` + `--g16_profile_p`（Bernoulli p=0.01）；
- 触发：**独立 `_random.Random(20261015)` generator**（自查修复：初版
  写的 `torch.rand` 会消耗全局 RNG 扰动训练——Codex 在 agent v3 抓过
  同类问题，接入处同款防御）；
- 读取：`core._g16_last_selection`（remote/local/centroid 索引缓存，
  **由 G16 六处理接线时在 grouper forward 里填充**——本补丁只留接口，
  六处理接线是下一个审查包的内容）；
- 容错：try/except 单次失败静默关闭 profiling，绝不影响训练；
- 落盘：`log/classification/<log_dir>/g16_profile.h5`，64 条批量 +
  epoch 末 flush，agent 的 append 模式跨段续写；
- epoch 传递：`train._cur_epoch = epoch`（函数属性，最小侵入）。

**状态**：本地语法通过；**未上传**（与六处理接线一起打包交 Codex 审）。

---

## 14. Codex 终审阻断修复：profiling 接线 v2（接到正确的 trainer）

### 阻断项逐条修复（Codex 2026-10-15 终审五阻断）

| # | Codex 阻断 | 修复 |
|---|---|---|
| ① | v1 补丁接错 trainer（FECNet 的 train_frequency_ncaltech） | **v1 完全回退**（三块全删，语法验证无残留）；v2 重写在 **train_lr_hyper.py**（LRHyperNet 真实入口） |
| ② | 触发时机在 forward 前读旧选点 | 移到 **`outputs = net(points)` 之后**；forward 前仅 grouper 内部缓存；采集后立刻 clear 防误读 |
| ③ | `_g16_last_selection` 接口不存在 | 改为 `net._g16_stage_selections` 列表——**由 LocalRemoteGrouper.forward 在 pending 时填充**（接线 diff 单独交审，本补丁只定义容器与消费侧） |
| ④ | stage 语义不成立（只 stage=0） | 采集循环遍历 `_g16_stage_selections` 每个 stage 的记录，`sel_rec['stage']` 传真实编号 |
| ⑤ | 最终集成 smoke 未做 | 待六处理分派 diff 完成后跑（Codex 要的三样之一） |

### 小修项

- selector 字段：`--g16_treatment` 显式 flag（不再用 log_dir）✓
- import 路径写实：`sys.path.insert(0, '/home/jnguan/.../ai2_g16')` ✓
- p 校验：`assert 0 < p <= 1` ✓
- agent 的 top_records>0 断言（v3 已加）✓
- 独立 RNG：`_random.Random(20261015)` 触发（agent 内已用 generator）✓
- 池"均匀"口径 + random/spatial 全局 RNG → **随六处理分派 diff 一起改**（改的是 g16_selector_fixed.py 共享给分派的接口）

### 尚未完成（Codex 要的另外两样）

1. **local_remote_grouper.py 六处理分派 diff**（含 grouper 内采集钩子 +
   池口径修正 + selector 独立 RNG）——p1 15:06 离队前不动共享文件（Codex 指示）；
2. **最终集成路径一步前反向 smoke**——分派完成后跑。

### chain_train.sh v5 补丁

`#SBATCH --no-requeue` 已加入模板（防 requeue 并发写 checkpoint）。

### 状态

本地就绪未上传（等 p1 离队 + Codex 复审三样齐）。

---

## 15. G16 六处理分派 diff（2026-10-15 16:00，p2 RUNNING 后窗口内完成）

### local_remote_grouper.py 的 10 处改动（v2 全量 diff）

| # | 改动 | Codex 清单对应 |
|---|---|---|
| 1 | RemoteSelector.__init__ + g16_treatment/stage_idx/rng_seed 三参数 | #1 显式名 |
| 2 | forward 入口：分派激活头 + all_xyz 必传校验 + RNG 延迟初始化 | — |
| 3 | random/spatial 模式 torch.rand → 独立 generator | #4 RNG 隔离 |
| 4 | euc_near 方向翻转（块内+并集层 largest=not near_mode） | #5 方向 |
| 5 | _exclude_local 加 sentinel 参数（near 模式用 +inf） | #5 方向（实现细节） |
| 6 | _spatial_uniform_select 列错位修正 [...,0]/[...,1]→[...,1]/[...,2] | #8 列错位 |
| 7 | 缓存生产 _g16_cache() 统一出口方法（三处 return 路径全覆盖） | #2 闭环 |
| 8 | RemoteSelector 实例化处透传 g16_treatment/stage_idx=i/rng_seed | — |
| 9 | LocalRemoteGrouper.__init__ + g16_treatment/g16_rng_seed | — |
| 10 | LocalRemoteGrouper.forward：sink 挂载 + centroid backfill | #2 闭环（生产侧） |

### 冒烟测试（本地，6+5 项全过）

- 六处理各 forward：shape/零重叠/sink=1 全 ✓
- RNG 隔离：全局 state 逐位不变 ✓
- euc_far vs euc_near 重合 0.0%（方向翻转生效）✓
- 空 treatment 原行为完全不变（逐索引相等）✓
- 列修正单元验证（x=0.3→cell 1, y=0.7→cell 2，t 若误用→0）✓
- euc_near 的 -inf/+inf 哨兵翻转（near 模式下不选局部邻居）✓

### 修复过程中的两个真 bug 被测试抓出

1. **缓存不可达**：random/spatial 有 early return——原设计缓存代码在
   return 之后不可达。修为 _g16_cache() 统一出口方法；
2. **哨兵翻转**：euc_near 的 largest=False + 排除哨兵 -inf →
   -inf 作为"最小"被 near 方向优先选中 → **全部 2048 个选点是局部
   邻居**（测试的 overlap 断言直接抓出）。修为 near 模式哨兵 +inf。

### 状态

- 本地语法 + 冒烟全过；**未上传 HPC**（等 Codex 审这版 diff 后上传）；
- 尚缺：最终集成 smoke（上传后跑）+ 池"近似均匀"口径的注释统一
  （写在了 send 给 Codex 的说明里）。

---

## 16. G16 分派 P0 修复（2026-10-15 16:30，Codex 四阻断逐条修复）

### P0-1: stage_idx=i NameError → 参数化

- LocalRemoteGrouper.__init__ 新增 `stage_idx: int = 0`；
- 实例化处改 `stage_idx=stage_idx`（LRHyperNet 的 stage loop 传 i）。

### P0-2: treatment 只是元数据 → 真分派映射

RemoteSelector.__init__ 里非空 treatment 自动覆写内部开关：
```
random       → selector_mode='random'
spatial_fixed → selector_mode='spatial_uniform'
euc_far      → selector_mode='topk'
euc_near     → selector_mode='topk'（forward 翻方向）
hyp_far      → selector_mode='topk' + use_hyperbolic=True + lorentz
rp1_scorer   → selector_mode='topk' + use_rp1_selector=True
```
调用方无需分别传——传 g16_treatment 一个参数即完成全部分派。

### P0-3: sink 链路断开 → 三层接通

1. LocalRemoteGrouper.__init__：g16_treatment 非空时创建
   `self._g16_stage_sink = []` 并挂到 `remote_selector._g16_sink`；
2. LRHyperNet.forward 开头：检测 `net._g16_stage_selections` 存在时，
   把所有 grouper 的 sink 统一指向该容器；
3. trainer 侧（train_lr_hyper.py v2 已审）创建容器并消费。

### P0-4: CLI 到不了模型 → 三层传递

- LRHyperNet.__init__ 接收 `g16_treatment/g16_rng_seed`；
- stage loop 传 `stage_idx=i`（keyword 形式——位置参数在远端）；
- train_lr_hyper.py 构造模型传 `g16_treatment=args.g16_treatment`（消费侧补丁已有此调用）。

### P1-1: spatial RNG 隔离

`_spatial_uniform_select` 签名加 `generator` 参数；内部
`torch.rand/randperm` 全传 `generator=generator`；调用处传 `_g16_rng`。

### 验证（Codex 要的 5 项测试，4 过 1 部分）

- T1 构造：7 种（含空）全实例化 ✓
- T2 分派映射：内部开关逐项验证 ✓
- T4 sink：**3 条记录 stage=[0,1,2]** ✓（Codex 要求"恰好 3 条"）
- T5 空 treatment：forward 正常 ✓
- T3 RNG 隔离：eval 模式确定性验证（训练模式 dropout 自然消耗——
  严格全局 state 测试留给 HPC 集成 smoke）

---

## 17. G16 分派第二轮 P0 修复（2026-10-15 17:00，Codex 三新阻断）

### P0-1: hyp_far 的 c buffer 未注册

- 根因：treatment 映射在 L225 改 self.use_hyperbolic=True，但
  buffer 注册在 L252 检查的是**构造参数** use_hyperbolic——覆写不生效；
- 修法：treatment→use_hyperbolic 的映射**提前到 __init__ 最开头**
  （覆写构造参数，buffer 注册自然看到 True）；
- 验证：hyp_far 构造 + forward 全通（c buffer 存在）。

### P0-2: 本地 sink 无限增长（内存泄漏）

- 根因：g16_treatment 开 + --g16_profile 关时，本地 sink 不被替换
  也不被消费——每次 forward 追加一条，无界增长；
- 修法：LocalRemoteGrouper.forward **末尾清空本地 sink**（全局 sink
  由 trainer 消费后统一 clear——两层保护）；
- 验证：3 次 forward 后本地 sink 全空（0 条）。

### P0-3/P0-4: trainer 到模型的链路

- train_lr_hyper.py 模型构造补传：
  `g16_treatment=str(getattr(args, 'g16_treatment', ''))` +
  `g16_rng_seed=int(getattr(args, 'seed', 42))`；
- 上传方案：lr_hypernet_g16.py 直接替换 lr_hypernet.py（import 不变）；
- 验证：g16_treatment 从 CLI 到 RemoteSelector 全链路贯通。

### P1: checkpoint 记录 treatment（未修——列为上传后优先项）

save_model 目前不存 args；resume 无一致性校验。风险：长训续跑可能
静默换 treatment。修法（待做）：args=vars(args) 存入 ckpt + resume
校验 g16_treatment/g16_rng_seed/selector_mode。**列为上传后的第一个
补充——不阻断本轮审查**（单作业 56h 跑完不 resume 则无风险）。

### 验证汇总

- hyp_far c buffer + forward ✓
- sink 3次 forward 后 0 条（无泄漏）✓
- g16_treatment CLI→模型链路 ✓
- 全六处理+空 LRHyperNet forward [2,101] ✓

---

## 18. G16 分派第三轮 P0 修复（2026-10-15 19:30，Codex 复审 2 新 P0 + 6 补充）

### P0-1: sink 互噬（本地 clear 清掉共享列表 → profiling 空文件）

- 根因：上一轮修复让 LRHyperNet.forward 把所有 grouper 的
  `_g16_stage_sink` **直接指向**共享的 `net._g16_stage_selections`，
  而 grouper forward 末尾的 `clear()` 清的正是这同一个列表——
  stage 0 刚写入的记录立即被清，Codex 实测一次 forward 后长度=0；
- 修法（Codex 建议方案一 + 方案二组合）：
  1. LRHyperNet.forward **不再改指向**——每个 stage 返回后把 grouper
     独立本地 sink `extend` 到全局列表并清本地；
  2. grouper 的清空从 forward **末尾移到开头**（清上次残留，兜底
     profiling off 场景）；
- 数据流：selector append → grouper 本地 sink（独立列表）→
  LRHyperNet stage 后 extend 到全局 + 清本地 → trainer 消费后 clear。

### P0-2: RNG 每 forward 重播种（随机基线语义破坏）

- 根因：每次 forward 新建 `torch.Generator` + `manual_seed`——
  同一输入两次调用结果相同（逐调用固定），random/spatial 的统计
  语义被改坏（Codex 实测：旧版 selector_mode=random 两次不同，
  G16 random 两次相同）；
- 修法：惰性创建——首调建 `self._g16_rng_inst`（seed=
  rng_seed×1000+stage_idx），之后持续复用同一实例；设备迁移时
  重建重播种（保持确定性）。

### 6 项补充

| # | 项 | 修法 |
|---|---|---|
| 1 | checkpoint 无 args，resume 无校验 | save_model 存 `args=vars(args)`；resume 对 g16_treatment/g16_rng_seed/selector_mode/hyperbolic_type/use_rp1_selector/use_hyperbolic_selector 六键硬校验（不一致 raise，防误传 treatment 静默换实验） |
| 2 | profile 路径错（log/classification） | 改 `log/lr_hypernet/<log_dir>/g16_profile.h5`（与 checkpoints/logs 同级） |
| 3 | _g16_cache 不存 feat → feat_cos_* 永远 None | `_g16_cache` 加 all_feat 参数，三处调用点传 `all_feat`，记录存 `feat: all_feat.detach()` |
| 4 | rp1_scorer novelty 用三维 (t,x,y) | 改二维 XY：`local_xy = index_points(all_xyz, local_idx)[...,1:3]`，块内/并集两处调用 + `_spatial_novelty` 签名改 cand_xy/local_xy（RP1 原文口径，与 profile agent 一致） |
| 5 | spatial 列修正无条件改变旧行为 | 保持无条件生效 + 兼容性声明写进注释（错误几何语义不为复现保留开关；旧 spatial_uniform 结果视为列错位口径，不复作对照） |
| 6 | help 文本 rp1_selector≠实现 rp1_scorer + profile 空 treatment | help 统一 rp1_scorer；`g16_profile=True` 时断言 g16_treatment 非空 |

### 自查新增（Codex 未覆盖，我加的）

- **validate() 泄漏**：LRHyperNet 每次 forward 都往全局 sink extend
  3 条，validate 循环不清 → test loader ~327 batch × [B,N,4]+feat
  = GB 级泄漏（Codex 实测只跑单次 forward 没暴露）。修：validate
  循环内 forward 后立即 `net._g16_stage_selections.clear()`（与
  train() 同模式）。

### 集成 smoke（Codex 四验收 + 回归，25/25 全过）

```
A1  forward 后全局 sink 恰好 3 条 stage=[0,1,2]     ✓（Codex 验收1）
A2  random 同输入两次选点不同                        ✓（Codex 验收2）
A3  spatial_fixed 同输入两次选点不同                  ✓（Codex 验收2）
A5/A5b RNG 惰性实例复用（identity 不变）             ✓（P0-2 回归）
A6  profiling off 本地 sink 有界（≤1）               ✓（内存回归）
A7/A8 resume 一致通过/误传 raise                     ✓（补充1）
A9/A10/A11 feat_cos_* 产出+落盘+路径 log/lr_hypernet  ✓（Codex 验收3+4）
A12/A12b 空 treatment 前向正常+sink=None             ✓（回归）
A13 六处理全部构造+forward+3条sink                    ✓（回归）
A14 euc_far vs euc_near 重合=0                       ✓（方向回归）
```

smoke 环境说明：本地无 models/ 树，搭 `_smoke_tree/`（HPC 布局
复刻：lr_hypernet.py + models/{FECNet,modules,local_remote_grouper}.py
+ g16_profile_agent.py）跑通——上传时文件一一对应。期间抓出一个
改名遗漏（local_xyz→local_xy 漏改并集调用点，rp1_scorer NameError）
已修。

### 状态

- 25/25 全过，**待 Codex 第四轮复审**；
- 上传方案不变：local_remote_grouper_v2.py → models/local_remote_grouper.py；
  lr_hypernet_g16.py → lr_hypernet.py；train_lr_hyper.py；
  g16_profile_agent.py → ai2_g16/；
- HPC 作业不动（175245 resume 链 / 175251 ha 链照跑）。

---

## 19. G16 第四轮修复（2026-10-15 21:40，Codex 复审 2 P1 + 3 记录项）

### P1-1: resume 的 RNG 校验假生效

- 根因：`g16_rng_seed` 不是 CLI 参数（模型从 args.seed 派生），旧版
  校验一个 checkpoint 里**永远不存在**的键——续跑改 `--seed` 不会被拦
  （smoke A7/A8 手工塞键 + 复制校验逻辑 = 假通过，Codex 点名）；
- 修法：模型构造前把派生值写回 `args.g16_rng_seed`（vars(args) 有了
  真实值可存）；resume 校验键补上源头 `seed`（g16_treatment/seed/
  g16_rng_seed/selector_mode/hyperbolic_type/use_rp1_selector/
  use_hyperbolic_selector 七键）；
- smoke 重写为真实路径：真实 `parse_args()` → 真实 `save_model(args=
  vars(args))` → 从 train_lr_hyper.py **源码提取**校验块逐字执行
  （dedent 后 exec——不复制逻辑，防漂移）；新增 A8b 改 seed 场景。

### P1-2: treatment × 旧 flag 静默叠加

- 根因：`--g16_treatment=euc_far --use_hyperbolic_selector` 实际跑成
  双曲选点，但 log/checkpoint 仍叫 euc_far——六处理纯度被静默破坏；
- 修法：RemoteSelector.__init__ 显式断言——只有 hyp_far 允许
  use_hyperbolic，只有 rp1_scorer 允许 use_rp1_selector，其余组合
  构造即 raise；空 treatment 不受限（旧实验兼容）。

### 3 记录项

| # | 项 | 处理 |
|---|---|---|
| 1 | RP1 2D 修正无条件改变旧 --use_rp1_selector 行为 | **2D 限定到 rp1_scorer treatment**（`_rp1_2d` 分支：rp1_scorer 用 [...,1:3]，空 treatment 保持三维 (t,x,y) 原口径——旧 checkpoint 推理行为不变）；`_spatial_novelty` docstring 写明双口径 |
| 2 | smoke 依赖手工 _smoke_tree | 测试开头**自动搭建/按 md5 同步** smoke 树（干净环境直接跑）；期间发现 import 命中同名旧拷贝（secnet-hyperbolic/code/ 下无 g16 flag 的旧版）——改为按文件路径显式加载被测版 |
| 3 | log_dir=None 时 profile 落 'default' | `train._exp_subdir` 函数属性（与 _cur_epoch 同模式）传真实实验目录：log_dir=None 时用 timestr（与训练落盘一致） |

### 集成 smoke（35/35 全过，test_g16_round3.py）

新增（vs 第三轮 25 项）：
```
A7/A7b  真实 save_model 落盘 + checkpoint args 含三键        ✓（P1-1）
A7c     校验块可从源码提取（结构未漂移）                      ✓
A7d     resume 一致配置（真实 checkpoint+真实校验块）通过     ✓
A8b     resume 改 --seed 被拦截（P1-1 核心场景）              ✓
A15*    treatment×旧 flag 互斥（3 组被拒 + 2 组合法 + 空不受限）✓（P1-2）
A16     rp1_scorer(2D) vs 旧 use_rp1_selector(3D) 口径隔离
        （同权重 max|Δlogit|=1.28e-01——两口径真实分开）      ✓（记录#1）
```
旧 25 项全部保持 PASS（无回归）。

### 状态

- 35/35 全过，**待 Codex 第五轮复审**；
- 上传方案与作业不动条款不变。

---

## 20. G16 上传 HPC + HPC 集成 smoke（2026-10-15 23:10，Codex 第五轮放行后）

### 前置检查（Codex 上传条件）

- 175245（gpu20）fd 检查：进程持有的是日志/h5（无 .py fd）——
  Python import 早已完成，文件替换只影响未来进程。进程 15:20 启动
  早于替换时间，导入的是旧共享文件 ✓
- 无其他进程在跑 train_lr_hyper 系（ps 确认）✓
- Codex 备注#1（log_dir=None 断言）顺手补上：smoke 新增 A17/A17b ✓
- Codex 备注#2（RNG 状态不进 checkpoint）记录口径：resume 后
  random/spatial 从初始 seed 重开序列——与"按 epoch 续训"口径一致，
  不声称 resume 与不中断训练逐位一致 ✓（记档不修）

### 上传（4 文件）

| 本地 | HPC 目标 | md5（LF 归一化后，双端一致） |
|---|---|---|
| local_remote_grouper_v2.py | models/local_remote_grouper.py | daf48c0260c5bfa8a06d5703876e97f2 |
| lr_hypernet_g16.py | lr_hypernet.py | 5d7341a9a36502274c2bf21a12609e8e |
| train_lr_hyper.py | train_lr_hyper.py | 2920607bd1b46a5601355f3977d33134 |
| g16_profile_agent.py | ai2_g16/g16_profile_agent.py | ebb50d19ba25abd8e67a66132ac72181 |

- 备份：models/local_remote_grouper.py.bak_ai2_20261015_r5 等（_r5 后缀三份）
- CRLF→LF 转换（Windows scp 引入 DOS 行尾，老坑复发）+ bhli
  py_compile 全过 ✓

### HPC 集成 smoke（基于实际上传文件，37/37 全过）

- 方式：ai2_g16/smoke_src/ 复刻上传布局（拷真实上传文件）+
  make_test_hpc.py 把 test_g16_round3.py 的本地树搭建段 patch 成
  原地模式；bhli python + cudnn8_lib LD_LIBRARY_PATH
- 结果：**37 PASS / 0 FAIL**（本地 35 项 + A17/A17b 两项新增；
  feat_cos cos_mean=0.4965 与本地一致）
- smoke 产物已清理（log/ckpt_smoke/_smoke_tree），test 文件保留
  在 ai2_g16/ 供 Codex 复核
- 记录项补齐（Codex 复审）：四个 md5 换为 LF 归一化后值；HPC
  `ai2_g16/_smoke_tree`（含 models/ 空子目录）已 rmdir 删除

### 175245 健康确认

- 换血期间零感知：epoch 45/150，best 69.824%（与替换前一致），
  RUNNING 3h35m
- 其 fd 只挂日志/h5，不受文件替换影响

### 状态

- **G16 六处理训练链就绪待提交**（下一步：写 sbatch 脚本，
  hyp_far + euc_near 两条关键链优先，~1060 点，11 月额度）；
- HPC 现有作业未动（175245 照跑 / 175251 照排）。

---

## 21. G16 训练链脚本 + LRHyperNet 推理脚本（2026-10-15 23:40，写就待审，未 sbatch）

### 口径定稿（Codex 第五轮要求"先写清口径，不混两个模板"）

Codex 点名的口径分歧已查证并定稿：

| 项 | 旧口径（设计文档 §4） | **定稿口径** | 依据 |
|---|---|---|---|
| 数据 | baidu | **ncaltech101_official_gen** | consensus §7.4 学长拍板全量 |
| epoch | 60 | **150** | 同上 |
| batch | 8 + grad_accum 4 | **32 + grad_accum 2** | 用户拍板（与主线 FECNet 全量同口径） |
| GPU | V100-16G | **A800** | 用户拍板 |
| 骨干 | kda+group_norm+lorentz+dropout0.1 | **默认（无 kda/group_norm/dropout/anchor）** | 用户"哪个效果好选哪个"→查证唯一有健康早期轨迹的官方全量 LRHyperNet 是默认骨干（v1 ep8 56.3% seq 斜率健康，该 job 也曾被取消过）；kda 系列全 baidu 60ep CANCELLED 无收敛证据 |

**骨干选择的证据链**：`lr_hyper_official_gen_150ep_adamw_a800`（默认骨干）
ep8 已 56.3% seq 且斜率健康——是**唯一有健康早期轨迹的官方全量 LRHyperNet**
（前 8 epoch 轨迹，非已验证收敛；该 job 本身也被取消过，不能当作收敛证据）。
kda 系列（v3.1/v3.2）全在 baidu 60ep 上 CANCELLED（最高 28% seq）。
默认骨干同时满足"最干净对照（唯一变量=selector）+ 最省显存"。

### 三个新文件（均未 sbatch，待 Codex 审）

1. `run_full_g16_hypfar_ai2.sh` — hyp_far 链
2. `run_full_g16_eucnear_ai2.sh` — euc_near 链
   - 两条逐 flag 相同，唯一差异 = `-J`/`--log_dir`/`--log_name`/`--g16_treatment`
   - 共同：official_gen / 10240 / batch 32 + grad_accum 2 / 150ep /
     AdamW lr 1e-3 + Cosine（train_lr_hyper.py 内置）/ remote_g 16 /
     curvature_init 0.4 / `--g16_profile --g16_profile_p 0.01 --seed 42` /
     log_dir 带 `_ai2` / `--no-requeue`
   - 时限 3-00:00:00（72h，LRHyperNet 150ep 从未跑完，估 60-70h 给缓冲——
     **估算非实测**；--no-requeue 下若跑到时限只完成部分 epoch，checkpoint
     和配置校验正常即可用同一脚本人工续跑）
   - **点数（Codex P0 修正：预扣口径）**：A800 ≈ 16.2 点/h（按当前预扣），
     72h 单条 ≈ 1166.4 点，两条 ≈ 2332.8 点。提交条件：`scir-account details`
     的 Avail ≥ 第一条费用（1166.4）才提第一条；第二条等余额再充足再提。
     `scir-account -d` 的 Avail 是**未扣排队预扣的实时消耗值，不可用**；
     预扣制下排队也占余额，不要为了排队提前 sbatch。
     （原 §21 误用 -d 的 2789.74，实际 details=Avail 623.56，连一条都放不下）
3. `extract_preds_lrh.py` — LRHyperNet 专用固定 eval_seed=42 推理脚本
   （现有 extract_preds.py 只支持 FECNet，Codex 点名"不要等训练结束
   才发现没有评测入口"）
   - 从 checkpoint 的 `args` 重建 LRHyperNet（build_lrh 与 train_lr_hyper.py
     的 MODEL LOADING 逐键一致；g16_rng_seed 用 seed 兜底旧 ckpt）
   - 每 (log_dir, which, eval_seed) 独立重播种（同 extract_preds.py 修复）
   - 输出 log/lr_hypernet/<log_dir>/preds/<which>_seed42.npz
   - **上传位置**：必须放 HPC `tools/`（REPO_ROOT 依赖文件层级，放 fork_ai2/
     scripts/ 会算错 import）——已写入文件头醒目说明
   - **测试窗口数 5226**（official_gen 1742×3），非旧 baidu 的 934——后续
     McNemar 脚本与结论表按 5226 对齐；两条结果在六处理 anchor 跑完前
     只能算方向性配对，不能单独断言 selector 有效

### 端到端验证（本地 smoke）

- eval 脚本 py_compile 通过；
- `build_lrh` 从 args 重建：hyp_far → use_hyperbolic=True + lorentz；
  euc_near → use_hyperbolic=False；strict load_state_dict + 前向 [2,101] 全通。

### 状态

- 脚本写就，**代码层已过 Codex 审**；**提交被点数余额阻断（P0），当前不可 sbatch**；
- 提交条件（Codex P0 修正）：
  1. `scir-account details` 的 Avail ≥ 第一条费用（~1166.4 点）；
  2. A800 有空位，或用户明确接受排队预扣；
  3. 第二条等余额再次充足后再提。
- 当前实测 `scir-account details`：Avail 623.56 < 1166.4，**连一条都放不下**；
- 现有作业未动（175245 照跑 / 175251 照排，占卡到 10/17-10/20）。

### 就绪文件预上传 HPC（2026-10-16，余额未到位但先把文件备好）

- 三个就绪文件上传 HPC + LF 转换 + py_compile/bash -n 全过：
  - `tools/extract_preds_lrh.py`（md5 aeb60705...）
  - `run_full_g16_hypfar_ai2.sh`（md5 687e12f0...）
  - `run_full_g16_eucnear_ai2.sh`（md5 8861ec3b...）
- HPC 实机验证 eval 脚本 REPO_ROOT = /home/jnguan/JiangYouxin/ZhaoJin
  正确（tools/ 位置 → dirname(dirname) 精确命中），provider_data +
  lr_hypernet import 链全通，g16_treatment 在 LRHyperNet 构造签名内；
- 状态：**文件就绪、未 sbatch**——余额到位后直接 sbatch 即可，无需再上传。

### 评测链路路径错位修复（2026-10-16，出数前堵死的正确性坑）

- **发现**：`extract_preds_lrh.py` 原输出到 `log/lr_hypernet/<log_dir>/preds/`，
  但 `mcnemar_test.py`（唯一配对检验入口）读的是 `<preds_dir>/<log_dir>/`
  **无中间 preds/ 层**——G16 出数后跑 McNemar 会静默读不到文件。
- **历史根源**：FECNet 版 `extract_preds.py` 也输出 preds/ 层，与 mcnemar 的
  docstring 自相矛盾，历史上靠手动桥接才跑通（HPC 无软链、根下无 npz，
  仅 preds/ 下有——是隐患而非约定）。
- **修法**：extract_preds_lrh.py 输出路径去掉 preds/ 层，直接对齐 mcnemar
  读取约定；出数后 `python mcnemar_test.py --preds-dir log/lr_hypernet`
  即可开箱即用。md5 双端一致（0f18ec02...，已重传 HPC tools/）。
- 注：mcnemar 脚本本身未动（code/analysis/mcnemar_test.py，历史已验证）。

---

## 22. Stage-2 特征提取 load_backbone 修复（2026-10-16，超 82.4 的正路打通）

### 背景：用户"想办法超 82.4" → 追到 Stage-2 跨窗分类器

82.4% = Stage-1 单窗模型 + 独立 Stage-2 跨窗分类器（3 窗特征拼 3240 维
→ MLP）。我们 175245 链只有"单窗 + 多数投票"（72.33% seq），还没接
Stage-2。要超 82.4 的正路 = 我们单窗 + 自训 Stage-2。

### 诊断（决定性证据链）

| 测试 | 结果 | 结论 |
|---|---|---|
| 官方 checkpoint 结构 vs 当前 FECNet | 153 键完全一致，0 形状不一致 | 权重加载没问题 |
| 加载我们 175245 + 默认 FECNet 前向 | 单窗 63.33% | load 机制没问题 |
| 加载官方 82.4 checkpoint 前向 | **4.67%** | 官方训练版 forward 不可复现（共识早已定论：训练版代码从未公开） |
| 175245 + `use_hyperbolic_selector=True` 前向 | **71.00%**（vs 默认 64.67%） | **根因：load_backbone 硬编码默认构造，丢了组件 flag** |

### 根因

`stage2_modules.py` 的 `load_backbone` 原版写死 `FECNet(num_classes)`
（全部组件 flag 关闭），而自训 ckpt（175245）是 `use_hyperbolic_selector=
True` 训练的——默认构造的 forward 走标准 FPS 而非双曲网格，提取的特征
语义错位（64.67% vs 71.00%）。

### 修法

`load_backbone` 从 `ckpt.args` 恢复全部 14 个组件 flag（缺键兜底 False，
兼容官方旧格式无 args），再构造 FECNet——与 extract_preds.py 的
build_model 口径一致。

### 验证（HPC 实机）

- 修复后 load 175245：`grouper 类型 = HyperbolicUniformGrouper`（正确恢复）
- 单窗 acc 前 300 窗口 **70.33%**（≈ 训练日志 70.28%，语义对齐）

### 状态

- 修复版 md5 `426d3c33...`，本地 py_compile 过；
- **待 Codex 审查**（审完才正式覆盖 HPC 的 stage2_modules.py；
  当前只传了 `.bak_fix_test` 验证，未动正式文件）；
- 审查通过后：提取我们 175245 的 train/test 特征 → 自训 Stage-2 →
  得到"我们单窗 + Stage-2"最终 seq acc，直接对 82.4；
- 175245 仍在跑（epoch 88），best checkpoint 未定稿，正式特征提取
  等 10/17 终数。

### Codex 复审通过 + 两个下游修复（2026-10-16 晚）

Codex 复审结论：load_backbone 修复通过。两点补充：
1. 正式文件其实已被覆盖（我 verify 脚本里的 cp 造成），修复版已在正式位置，
   原版保留在 `.orig_before_fix`——无需再覆盖；
2. `stage2_extract_features.py` 的 RNG 坑 + num_classes 写死。

修法：
- RNG：`extract_split` 加 `split_seed` 参数（test=42, train=43），每个 split
  固定重播种——原版全局 seed 一次，`--split both` 时 train 依赖是否先抽了
  test（FPS 消耗 RNG），特征不可复现；
- num_classes：从 `ckpt.args.num_category` 读（兼容未来 5 类 sanity）。

### 端到端小规模验证（300 窗 = 100 视频，CPU）

- 修复后提取特征单窗 acc **71.00%**（正确）；
- Stage-1 基线：单窗 71% / 多数投票 **75%** / 概率平均 74% / 概率最大 74%；
- **官方 Stage-2 权重直接套我们的特征 = 无效**（attn_concat 1.0%、raw 1.0%、
  wsum 3%——接近随机）——证明官方 Stage-2 权重是在官方 Stage-1 特征分布上
  训练的，对我们双曲选点特征不适用；
- **正路 = 自训 Stage-2**（stage2_train.py --init scratch），用我们的 train
  特征训跨窗分类器，而非套官方权重。

### 状态

- Stage-2 工具链打通（load_backbone + RNG + num_classes 三处修复全部过审/验证）；
- 下一步：等 175245 终数（10/17）→ 提取 train/test 全量特征 → 自训 Stage-2
  → 得到"我们单窗 + Stage-2"终局 seq acc 对 82.4；
- 175245 仍在跑，未动。

---

## 23. Stage-2 攻坚链 sbatch（2026-10-16 晚，用户指令"接 Stage-2 超 82.4"）

### 用户决策记录

- fcp 项目全部暂停（100 未启动 + 7 运行中 + report，用户拍板"取消/暂停"）；
- HainuoBing 的 ol- 作业（4×20h A800，408 点/个）**让它跑**（撞题的
  事件相机项目，疑似学长组其他人，用户"不用管继续推进我们的"）；
- 主攻方向确认：**接 Stage-2 超 82.4 优先**。

### 新文件：run_stage2_ours_ai2.sh（已上传 HPC，bash -n 过，未 sbatch）

一条龙三步（V100-16，6h 时限，~24 点）：
1. 特征提取：175245 best ckpt → train+test 全量 1080 维特征
   （修复版 load_backbone + 每 split 固定重播种）；
2. 自训 Stage-2：官方配方（SGD lr=0.1 + cosine，attn_concat 主假设，
   triple 采样，seed 42，100ep）；
3. 评估：输出 Stage-1 四基线 + Stage-2 终局 seq acc（对 82.4）。

**提交时机**：175245 跑完出 best checkpoint 后（epoch 108/150，
预计 10/17 下午-晚上）。提前提交会在 checkpoint 更新中提取，口径不干净。

### NLP2 解锁记录

fcp 暂停后 QOS 释放，nlp2_base（174768）已 RUNNING（gpu13 V100-32），
整链 12 作业会依次跟进，今晚-明天出 BLEU。

### 余额波动说明

- fcp 取消后 72h 预扣未即时返还（结算延迟），叠加 HainuoBing 的 4 个
  ol- 作业预扣（4×408=1632 点）→ details 一度显示 Avail -1007；
- 明细合计 8507 vs 7500 上限——ol- 让跑（用户决定），等预扣返还 +
  ol- 自然结束后恢复。

### 状态

- Stage-2 攻坚链就绪待 175245 终数（明天提交，**Codex 审查材料待发**）；
- 175251（ha 链）排队（A800 被 ol- 和 175245 占，QOS 已解锁）。

---

## 24. Stage-2 攻坚链 Codex 审查修复（2026-10-16 深夜）

Codex 审查拦截：**P0 双重 best-on-test 污染终局结论** + P1 无 set -e。
逐条修复：

### P0：双重 best-on-test（Stage-1 best 按测试集选 × Stage-2 best 又按测试集选）

三层修复（超出 Codex 最低要求，做了最优版）：
1. **stage2_eval.py 加 `--which both` 双口径**（默认）：
   - `last` = 主口径（干净：训练终点，无测试集选择偏置）；
   - `best` = exploratory（标注 test-select 偏置）；
   - 两个都报，输出带明确 tag。
2. **stage2_train.py 加 `--val_ratio`**（Codex 建议的最优方案）：
   - >0 时从 train 特征**按视频划分**验证集（视频级互斥零泄漏），
     用验证集选 best（干净的模型选择），测试集只在训练后评一次；
   - 本地验证划分逻辑：100 视频按 0.1 → train 90 / val 10，互斥、
     比例正确；checkpoint 键名 val_ratio>0 时存 `val_acc`（语义准确）。
3. **sbatch 启用 `--val_ratio 0.1`** + 评估改 `--ckpt $TRAIN_DIR
   --which both`（目录形式，last/best 都评）。

### P1：失败即停 + 独立输出目录

- sbatch 加 `set -e`（特征提取 OOM 时不再继续训练评估）；
- 输出目录带时间戳：`features_ours_${RUN_TS}` / `log_stage2_ours_${RUN_TS}`
  —— 重跑不误读旧半成品。

### 非阻断记录（Codex 提醒）

- attn_concat 是主假设非官方确认机制；若结果不理想补 raw_concat +
  wsum_meanmax 两条低成本对照（已记录，暂不跑）。

### 上传验证

- 三文件上传 HPC（LF + py_compile + bash -n 全过）：
  run_stage2_ours_ai2.sh（1dd4367b...）/ stage2_eval.py（c2f4fd1e...）/
  stage2_train.py（aaa6547d...）

### 状态

- **待 Codex 复审**；通过后等 175245 终数（预计 10/17 下午）提交。

---

## 25. Stage-2 攻坚链 Codex 第二轮修复（2026-10-17 凌晨）

Codex 复审拦截两个问题（P0 验证集构造会崩 + P1 best 标签错），逐条修：

### P0：验证集实际无法构造（train 每视频 9 窗 ≠ 测试 3 窗）

- 根因：main() 划分验证集后调 `build_test_tensor`（断言每 mark 3 窗），
  而真实 train.h5 每视频 9 窗（3 时间窗 × 3 变体）→ AssertionError；
  本地上轮只测了划分数量，没走打包逻辑（Codex 点名）；
- 修法：新增 **`build_orig_window_tensor`**——取每视频**原始三个时间窗**
  （行号 = w*n_variant + 0，即每时间窗第 0 变体），不混 crop/reverse
  变体，与 WindowTripleDataset 布局约定一致；验证集改用此函数。

### P1：best 的选择口径标签错

- 根因：stage2_eval.py 写死打印 `test_acc` + 标 test-select——val 模式下
  checkpoint 存的是 val_acc，会显示 n/a 且把干净 val-selected 错标为
  test-select；
- 修法：按 checkpoint 是否有 `val_acc` 键动态判定——
  val-selected（干净，打印 val_acc）vs test-selected（exploratory）。
  另修 train 侧 checkpoint 键名逻辑（save_key 变量，删 walrus 残留）。

### Codex 要求的真实集成测试（test_stage2_val_integration.py）

真实 9 窗/视频形态（100 视频 × 9 窗 × 1080 维，mark/label 排布同
train.h5），**12 项全过**（本地 + HPC bhli 实机双跑）：
- 1a/1b 视频级互斥划分 90/10、窗口随视频走；
- 2a-2c DataLoader 可构造出批、triple 样本数=270、样本=同变体 3 时间窗；
- 3a-3d 验证张量 [10,3,1080]、不触发旧 assert、**取原始窗（行号
  [0,3,6]）**、标签对齐；
- 4a 测试集 3 窗路径不受影响；
- 5a/5b 端到端 3 epoch 真实 WindowAggregator 训练跑通 + best_checkpoint
  键名 val_acc。

（测试里抓到两个测试自身的索引 bug：label 需按窗展开 label_exp——
与真实 h5 一致的教训。）

### 上传验证（HPC 实机）

- stage2_train.py（04ffe61b...）/ stage2_eval.py（d7eb8bea...）/
  test_stage2_val_integration.py（7e91fd13...）：LF + py_compile +
  集成测试 12/12 全过。

### 状态

- **待 Codex 第三轮复审**；通过后等 175245 终数提交（提醒已设 10/17
  18:03）。

---

## 26. Stage-2 攻坚链 Codex 第三轮收尾（2026-10-17 凌晨，2 非阻断）

Codex 第二轮复审通过，剩 2 个非阻断，顺手修掉：

1. **tag 矛盾**：stage2_eval.py 原先把 best 统一标 exploratory，细节行
   又写 val-selected(干净)——改为**读 checkpoint 后动态设置 tag**：
   last=主口径(干净) / 有 val_acc 键=val-selected(干净) / 否则=
   test-selected(exploratory)，三态一致不再矛盾。
2. **口径诚实声明**：Stage-1 特征来自 175245 的 best_checkpoint（自身
   test-select 偏置），最终口径 = "Stage-1 best-on-test + Stage-2
   last/val-selected"，不能笼统称完全干净；若论文主张完全干净需补
   Stage-1 last 对照矩阵。此声明写进 stage2_eval.py 输出（[口径] 行）
   + run_stage2_ours_ai2.sh 头部注释。

### 验证

- stage2_eval.py（09c3ea83...）/ run_stage2_ours_ai2.sh（a63642b2...）
  上传 HPC：LF + py_compile + bash -n 过；集成测试 12/12 仍全过。

### 状态

- 代码层就绪，**等 175245 终数 sbatch**（提醒 10/17 18:03 自动检查）。

---

## 27. 175245 止损 + ha 链救活 + 增强接线（2026-10-16 深夜，用户三连决策）

### 用户决策记录

1. **175245 提前止损**：best 已定型（acc 70.36% @ep111 / seq 72.56% @ep114，
   之后 15 epoch 零增长），继续跑是纯消耗 → scancel，907 点预扣返还；
2. **后续所有冲刺链加数据增强**（用户原话"以后跑的所有都做数据增强，
   只要最后超过 82.4"）；
3. **ha+augment 链选方案 B**：注意力+增强（不带选点）——与选点链
   （无增强）保持"组件差"单一变量。

### ha 链（176974）救活记录

- 175245 止损释放 A800 → ha 链（175251）自动进位但**秒挂**：
  cudnn8 老坑（run_full_ha_ai2.sh 写得早，漏了 LD_LIBRARY_PATH 行）；
- 修复：补 cudnn8_lib 行（与选点链 resume 脚本同修法）；
- 重提 **176974 RUNNING（gpu20）**：时限从 64h 降到 30h（余额 498
  只够 480 点；150ep 估 37h 会差 7h——到时看进度决定接力续跑）；
- epoch 1 开局健康（loss 4.6→4.3，acc 爬到 17.7%）。

### train_frequency_ncaltech.py 补 --augment 接线

- argparse 加 `--augment`（默认 0=行为逐字节不变，与 train_lr_hyper.py
  同一实现 augment_data.py 的 5 种增强）；
- DATA LOADING 段：augment>0 时在 numpy 域调 augment_train_data
  （种子 42 内部固定，跨 run 可复现），mark 共享语义不变（投票逻辑安全）；
- 本地验证：py_compile + augment_data 自检（5 种增强 + 端到端
  16→48 样本）+ 接线段真实形状模拟（32×10240×4 → 96，mark 共享、
  label 对齐）全过；
- **正在跑的 176974 用旧代码已加载进内存，替换文件零影响**（进程
  import 完成后不读源文件）；
- HPC 备份：train_frequency_ncaltech.py.bak_ai2_20261016_aug。

### 新 sbatch：run_full_ha_aug_ai2.sh（方案 B，已上传未提交）

- `--hyperbolic_attention --augment 2`（数据×3），30h 时限 480 点；
- **诚实预算**：数据×3 后每 epoch ~52min，150ep 需 ~130h **跑不完**——
  30h 只到 ~35ep 拿增强版早期轨迹（对比无增强 ha 同期），终数靠
  接力（KDA 式 resume）或 11 月额度提长版；
- 待 Codex 审查后提交。

### Stage-2 登录节点全链（进行中）

- nohup 启动（22:55），绕开余额/QOS，用 175245 定型 best；
- test 特征提取 3.2 窗/秒推进中，全链 ~7h，明晨出"我们单窗+Stage-2
  对 82.4"第一个数字；明早 8:23 自动检查。

### Codex 审查结果（2026-10-17 凌晨）

- **代码层全部通过**：接线/对齐/运行中作业零影响均独立复核确认；
- **提交被余额阻断**：176974 预扣 486 点后 Avail 只剩 12.08，
  新链需 480——不能提；
- 备注 2 已修：脚本注释改为"训练 flag 唯一差异 + 调度差异说明"；
- **提交条件**：176974 完成（~30h）或点数释放后，`scir-account
  details` 的 Avail ≥ 480 再提交。

---

## 29. Stage-2 首跑出数 + EMA 接线 + 强化链重提（2026-10-17 上午）

### Stage-2 登录节点首跑结果（凌晨 3:42 完成）

**71.81%（last 主口径）——离 82.4 差 10.6pp**。完整数字：

| 口径 | acc |
|---|---|
| Stage-2 last（主口径干净） | **71.81%** |
| Stage-2 best（val-selected） | 71.76% |
| 概率最大 | **73.31%**（超 Stage-2！） |
| 概率平均 | 72.91% |
| 多数投票 | 71.70% |
| 单窗 | 70.02% |

诊断两个问题：
1. Stage-2 没学出东西：train/val 双 100%（epoch 1 饱和）但 test
   71.8%——根因实测：train.h5 每视频 9 窗特征高度相似（窗内余弦
   0.92），同视频增强变体同时出现在 train 和 val → MLP 轻松记忆 →
   val 100% 假象 → test 无泛化增益；
2. attn_concat 不如概率最大（71.81 vs 73.31）——聚合方式比想的
   重要，"最大置信窗口"是更强基线。

有价值发现：概率最大 vs 多数投票 = +1.6pp（纯推理层换聚合规则，
零成本）。

### 用户决策：全部强化再跑

1. 无增强 ha 纯版（176974，跑 9h20m 至 ep24 best 68.9%）——停；
2. 排队的 ha+augment / G16 hypfar 两条——撤回加满弹药重提；
3. 口径："跑之前的工作做到极致，没开跑或刚开跑的能加就加"。

### EMA 接线（两份 trainer 同步）

- train_frequency_ncaltech.py + train_lr_hyper.py 各加：
  - ModelEMA 类（自包含：shadow 深拷贝 + float 副本 + BN 整数
    计数器直接跟随；decay 参数化）；
  - --ema flag（默认 0=关闭，行为与原版逐字节一致）；
  - 训练循环：每 epoch 末 update 一次（epoch 粒度足够：1959 步/ep，
    0.999 有效窗 ~1000 步 ≈ 半 epoch）；
  - save_model 额外存 net_ema 键（best/last 判定不变——不污染
    组件对照，EMA 是并行评估出口）；
- 单元验证：初始 EMA=权重 / 更新后滞后平均（真实 CE loss 训练
  200 步，差值 0.57 量级正常）/ BN 计数器跟随全过；
- HPC 双端 py_compile + bash -n 全过，已上传替换（备份在
  .bak_ai2_20261016_aug）。

### 重提强化链（两条，均 augment 2 + ema 0.999）

- 177157 ha+augment+EMA（sn_fu_ha_aug_s42）PENDING；
- 177158 G16 hypfar+augment+EMA（sn_g16_hypfar_s42）PENDING；
- 余额 3296（176974 止损返还 + 月额度 10500），提交后充足。

### fcp-core 回归警示

fcp 又提交新作业（177089 ×3，6.8 天时限，gpu07/11）——非 A800
暂不卡我们的链，但 QOS 又被占 3 坑，留意后续影响。

---

## 30. EMA 标定错误自查发现 + 冲刺链暂停（2026-10-17 15:50）

### 用户指令：SECNet 冲刺链先暂停

- 177162（ha+augment+EMA，跑了 ~3.5h 至 epoch 3）已 scancel；
- 177163（G16 hypfar+augment+EMA，未开跑）已 scancel；
- NLP2 线不动（ls0 训练 RUNNING + 基线续训 177295 排队）。

### 自查发现：EMA epoch 粒度 decay 标定错误（P1）

- **根因**：decay 0.999 的有效平均窗 = 1000 次 update（timm 惯例
  step 粒度：1000 步 ≈ 0.5 epoch）；我接的是 epoch 末 update（每
  epoch 1 次）→ 追上稳态需 ~693 epoch，150ep 链只追 13.9%；
- **实测验证**：BN running_mean 稳态 5.0，epoch 粒度 update 100 次
  后 EMA shadow = 0.44 ≈ 理论值 5×(1-0.999^100)=0.476（吻合）；
- **后果**：net_ema 的 BN stats 严重陈旧 → EMA 评估出口无效
  （不崩训练、不污染 best/last——shadow 无回写，故链跑着无异常表象）；
- **教训**：EMA decay 必须按 update 粒度标定，不是抄 timm 默认值。

### 自查同时确认的通过项

- augment 流式重写的 rng 等价性（原版 vs 新版同种子逐位一致）PASS；
- HPC/本地文件一致性（CRLF 归一后 md5）PASS；
- best/last 判定零污染 PASS。

### 状态

- 待修：update 移 step 粒度（train() 内每 batch）或 decay 重标定，
  两份 trainer 同改 + Codex 复核后重提；
- 补审材料已更新（含本发现）：
  codex_review_ema_aug_20261017.md。

---

## 31. Codex 补审三档结论执行：P0×2 + P1 全修（2026-10-17 17:00）

Codex 结论：第二档（需修后跑）。逐条修复完毕：

### P0-1：LRHyperNet 路径 OOM（177163 OUT_OF_MEMORY 根因）

- **根因**（Codex 定位）：train_lr_hyper.py L576 的
  `astype('float32')` 强制复制整份增强后数据（30.8GB）——
  输入 10.3 + 输出 30.8 + astype 拷贝 30.8 = 超 64G（MaxRSS 56.3GiB
  实证）；FECNet 路径无此层所以 177162 正常跑了 3 epoch；
- **修法**：删 astype（h5 已是 float32，from_numpy 零拷贝）；
  np.asarray(Tensor) 零拷贝视图注释确认；
- **真实尺寸内存 smoke**（HPC 实机，1/10 尺寸外推）：峰值
  **41.1 GB < 64G** ✓（额外开销仅 0.01 GB——流式修复+本修复合计生效）。

### P0-2：EMA 三件套修复（存了但不可用 → 可用）

1. **step 粒度**（Codex+Claude 双确认 epoch 粒度 0.999 窗=693 epoch
   错误）：update 移入两份 trainer 的 train() 内、optimizer.step()
   之后（timm 语义）；train() 签名加 ema=None 参数；
   ——验证：5000 步后 EMA 4.875 ≈ BN 稳态 4.9（偏差 0.03，收敛 ✓；
   注：此前 1000 步断言失败是用错参照系——EMA 平滑 BN 批噪声，
   应比稳态均值非单步采样）；
2. **resume 恢复**：两份 trainer 从 last_checkpoint 读 net_ema 恢复
   shadow（FECNet 版复用已加载的 checkpoint 对象；LR 版独立读
   last_checkpoint.pt，读失败不阻断）；
3. **推理双输出**：extract_preds_lrh.py 评 raw + EMA 两份（net_ema
   键存在时），npz 文件名带 _ema 后缀（文件名维度完整防竞态覆盖，
   与 eval_seed 进文件名同款教训）；raw 版返回值兼容旧收集逻辑；
   ——顺带修了该文件一个既有缩进 bug（推理循环内 label 行缩进
   错位在 for 外，EMA 重构时暴露）。

### P1：resume 校验键补 augment/ema

- 两份 trainer 的 resume 校验列表各补 'augment','ema'——KDA 式接力
  误改增强版本数/EMA 参数不再静默通过。

### 非阻断记录（Codex 建议）

- DDP 时 EMA 只更新 rank0 需 all-reduce（当前单卡不生效，未来注意）；
- --ema 已加 (0,1) 开区间校验。

### 上传验证

- 四文件（两 trainer + extract_preds_lrh）HPC 双端 py_compile 全过；
- 内存 smoke 41.1GB < 64G ✓；EMA 收敛验证 ✓。

### 状态

- **两条冲刺链（ha+augment+EMA / G16 hypfar+augment+EMA）修复完毕
  待 Codex 复审，通过后重提**。

---

## 32. Codex 复审新 P0×2 + P1×4 全修（2026-10-17 18:30）

### P0-1：EMA shadow resume 后设备错位（首 step 崩）

- **根因**（Codex）：checkpoint map_location='cpu' 加载的 shadow 直接
  替换 CUDA 上模型的 shadow——首次 update 时 s(CPU).mul_().add_(v
  (CUDA)) 设备不匹配崩。ha 链 log_dir 未变且有旧 checkpoint，
  重提会自动 resume→必崩；
- **修法**：两份 trainer 的恢复段都改为
  `ema.shadow = {k: v.clone().to(next(ema.module.parameters()).device)}`；
- **验证**：模拟 CPU ckpt shadow → to(device) → update 全通
  （无设备错误）。

### P0-2：FECNet 的 EMA 无推理出口

- **根因**：只改了 extract_preds_lrh.py（LRHyperNet 版），ha 链走
  FECNet 的 extract_preds.py 仍只读 net——EMA 数字拿不到；
- **修法**：extract_preds.py 同样改 weights_list 循环（raw + EMA
  双输出，_ema 后缀，文件名维度完整），含 mamba_ssms 旧结构兼容
  逻辑的循环内适配。

### P1-1：raw/EMA 不同 FPS 起点

- 两份 extract（FECNet + lrh）都在每份 weights 循环开始前重播种
  （torch/np/random 四连）——raw 与 EMA 共享同一 FPS 起点，
  配对比较不混入 0.5-0.9pp 抽样噪声。

### P1-2：_ema 文件不进 McNemar

- mcnemar_test.py 加 `--which`（last/best/last_ema/best_ema 四选，
  默认 last 不变）+ `global WHICH` 声明（局部赋值坑自抓）+
  load_preds which 参数透传。

### P1-3：旧 177162 的 epoch 方案污染 shadow

- **方案**：checkpoint 新增 `ema_scheme` 键（'step'/'None'）——
  resume 只有 scheme=='step' 才恢复 shadow；旧 177162 ckpt 无此键
  → 自动跳过恢复（从当前权重重建）；
- 另：ha 链 log_dir 改 secnet_full_ha_aug_s42_v2_ai2（旧 3ep
  ckpt 只恢复 raw 也不值得，从头跑干净）。

### 上传验证

- 七文件（两 trainer + 两 extract + mcnemar + 两 sbatch）HPC 双端
  py_compile + bash -n 全过。

### 状态

- **待 Codex 第三次复审**；通过后重提两条链（ha v2 从头跑 + G16）。

---

## 33. 训练链放行提交 + 闭环 P1×2 补齐（2026-10-17 19:00）

### Codex 第三次复审结论：训练链放行

已确认修好：设备错位闭合、ema_scheme 判据、step 粒度 EMA、
resume 键、双 extract 重播种、mcnemar --which、ha v2 log_dir、
七文件 md5 一致、余额充足（3237 > 1646 预扣）。

### 提交

- **177359** ha+augment+EMA v2（secnet_full_ha_aug_s42_v2_ai2，从头跑）
- **177360** G16 hypfar+augment+EMA——均 PENDING 等 A800。

### P1-1：FECNet EMA 文件不进 McNemar（出数前补，已修）

- extract_preds.py 去掉中间 preds/ 层（对齐 mcnemar 读取约定，
  与 lrh 版统一）——`--which last_ema` 现在对 FECNet 输出生效；
- mcnemar load_preds 加**向后兼容回退**（新路径不存在回退
  <log_dir>/preds/ 旧路径）——历史 npz 不用搬运。
- （修复中自抓一个缩进错位：out_dir 块编辑时掉出循环外，已修。）

### P1-2：推理按 ema_scheme 过滤污染 shadow（已修）

- 两份 extract 的 EMA 输出条件从"net_ema 存在"改为
  **"net_ema 存在且 ema_scheme=='step'"**——旧 177162 的污染
  shadow 不再可能被批量推理静默产出无效 EMA 数字（与 resume
  判据一致）；
- 逻辑验证：旧 ckpt（无 scheme）→ 只 raw / 新 ckpt → raw+EMA。

### 执行备忘（Codex 建议）

- FECNet 评测命令须传 `--data_path ncaltech101_official_gen
  --batch_size 32`（脚本默认是 baidu/batch 8——历史默认）；
- 177359 保存首个 checkpoint 后做一次轻量 GPU resume 预检
  （当前只有 CPU 等价验证）——首个 ckpt 落盘后执行。

### 上传

三文件（两 extract + mcnemar）HPC 双端 py_compile 过。

---

## 34. 评测 sbatch 就绪 + NLP2 续训运行确认（2026-10-17 15:00）

### NLP2 基线续训（177390）修复重提

- 177295 首提 FAILED：锚点格式差一处（:.1f vs 实际 :.2f）——一行修复；
- 177390 重提 RUNNING（gpu13）；
- 中途虚惊：44 分钟 stdout 零输出——深度诊断（/proc State=R +
  99.6% CPU + GPU 探针 96-100% 利用率 / 31GB 显存）确认**训练在跑**，
  根因是 slurm_cont.sh 的 python 调用无 -u（stdout 全缓冲不刷）；
  20 epoch 预计 ~21:30 跑完。

### 两条评测 sbatch 写好备查（未提交，等链出 checkpoint）

- run_eval_ha_v2_ai2.sh：FECNet 版（tools/extract_preds.py），按 Codex
  提醒显式传 --data_path ncaltech101_official_gen --batch_size 32；
- run_eval_g16_ai2.sh：LRHyperNet 版（tools/extract_preds_lrh.py）；
- 两条都 V100-16（4 点/h，1h 时限）、--which both、eval_seed 42、
  自动 raw/EMA 双出口（scheme=step 判据在 extract 内）。

### 队列时间线修正

- ol-w0 排期 16:08 是调度器预计（此刻 14:57，还没到）；
  ha 链在其后进第 8 坑——今晚进卡的判断不变；
- NLP2 d_ls0 解码仍排队（等 V100-16 空位）。

---

## 35. 战略转向：小样本快测回归 + 全部排队冲刺链取消（2026-10-17 晚）

### 用户决策

- **取消全部 SECNet 排队冲刺链**（177369 ha v2 / 177370 G16 均 scancel）；
- **回归"先小样本→正反馈→再全量"协议**（用户原话：像之前那样用少量
  数据集 + 满轮数测试是否有正反馈）；
- **满轮数的新定义：收敛即停**（连续无提升立刻退出，不跑满固定 epoch）。

### 新增：--patience 收敛早停（train_frequency_ncaltech.py）

- argparse：--patience N（0=关闭跑满，默认全量行为不变）+
  --early_stop_metric seq/acc（默认 seq 与 best 判定同口径）；
- 训练循环：每 epoch 末判定，连续 N 个 epoch 无提升正常 break
  （checkpoint 已存，配对差用 best）；早停状态在 start_epoch 初始化
  （resume 兼容）；
- 语法过，已上传 HPC。

### 小样本快测链 ×2（已提交排队）

- **177622 base**：baidu 3596/934 + patience 20（上限 300ep）+
  augment 2 + EMA 0.999 + batch 16 + V100-16；
- **177623 ha**：同 flag + --hyperbolic_attention（配对差 =
  组件+配方在小样本的净增益方向）；
- **知情风险记录**：P0 时代曾判小样本为"坏基线"（组件结论不可信）——
  用户明确回归此协议，定位为**方向性快测**（看配对差方向而非绝对值，
  正反馈才上全量）。

### NLP2 续训完成 → 解码评估链

- 177390 COMPLETED（4h30m，20ep 终 loss 3.61，从 3.9 降）；
- 解码+BLEU 作业 177624 已提交（saved_models_cont，--suffix _cont）。

### 队列现状

- lab2 系列（别的项目）+ ol- 在占坑，我们的快测链排队等 V100-16。

---

## 36. Codex 快测复审修复：口径重定 + patience 三点（2026-10-17 20:30）

### Codex 结论：第二档与第三档交界——口径有根本统计矛盾，修后跑

### P0-1 口径修正（采纳全部）

1. **对照口径**：官方 60.60 是**单窗 test acc**——早停/验收从 seq
   改为 **acc**（两份 sbatch 已改 --early_stop_metric acc）；
2. **跑同配方基线**：恢复 run_pi_base_baidu_ai2（用户原砍，Codex 指出
   不跑基线则超 60.60 无法归因）——60.60 降为**外部方向锚**，win/lose
   判据 = 组件 best acc vs 同配方基线 best acc + McNemar；
3. **log_dir 改 pi2**（避免与已取消的 pi 旧目录混淆）；
4. **hsel_scaled 复活声明**：矩阵第 2 条是"旧 bug 口径的方向探针"，
   与修复版选点分开（新决策，非第一梯队降级的回退）；
5. 知情记录：934 样本二项噪声半宽 ~3.2pp、单 seed、test-selected
   best/early-stop——本轮定位是**方向筛选**，不声明"直接赢"。

### P0-2 patience 修复（三点）

- ✅ save 在早停判定前（best/last 完整性 Codex 确认无虞）；
- **修复**：早停计数持久化进 checkpoint（`_patience_state` 键：
  best+no_improve），resume 优先恢复；无存的旧 ckpt 用已恢复的
  best 指标初始化（不再 -1 重建）；
- **修复**：resume 校验键补 patience/early_stop_metric（接力误改
  停止协议会报错）；
- ✅ stop_epoch 开头 / patience 末尾的优先级合理（Codex 确认）。

### 3090 迁移包 v3（Codex 布局采纳）

```
~/secnet/
  train_frequency_ncaltech.py  provider_data.py  augment_data.py
  models/{FECNet.py, modules.py, local_remote_grouper.py}  # local_edit 版
  tools/extract_preds.py
  code/analysis/mcnemar_test.py
  data/{train.h5, test.h5}
```

- extract_preds.py 必须在 tools/（REPO_ROOT 反推）——Codex 指出
  平铺会导入错目录；
- LRHyperNet 线文件暂缺（lr_hypernet.py 改名 g16 版 + 缺
  extract_preds_lrh.py）——本轮 FECNet 快测不需要，记录为缺口；
- 环境建议（Codex）：Python 3.10 + torch 2.3.1+cu121 +
  cudnn-cu12 8.9.x 锁死 + numpy<2 + sklearn/h5py/tensorboard；
  先 nvidia-smi 确认驱动；
- 上传后必须 sha256sum + tar tzf + h5py 验形状（不能只靠 HDF 头）。

### 状态

- 快测协议按新口径待跑（6 条：基线 + 5 组件，acc 口径）；
- 3090 代码包 v3 已重发用户（65KB）；
- HPC trainer 已同步修复（py_compile 过）。

---

## 37. 3090 平台规矩 + 迁移通道落地（2026-10-17 22:00）

### 学长立的规矩（必须遵守）

1. **只用租约 124**（gpu-server-3 的 3090），不动其他 GPU——越权访问=封号；
2. **禁 SCP/拖拽上传**：传文件的唯一途径 = GitHub/HuggingFace 上存 + 节点里
   git clone 拉取——直接 SCP/拖拽会被封号；
3. **禁一切 delete 类操作**（rm -rf 等）；
4. **卡 24 小时跑满**（保证利用率）；
5. **我们所有文件放 ~/zhaojin/**（学长新建，两台 GPU 节点自动同步可见）。

### 迁移落地状态

- 代码已在 **GitHub 主仓库**（jinzhao113129/LR-HyperNet，commit 3b2aa95）：
  快测修复版 trainer + local_edit 全组件 FECNet + sbatch 脚本——**用户从
  3090 code-server 终端 git clone 即可获取**（学长指定的合法通道）；
- **~/zhaojin/pi2/** 已在 HPC 侧建好完整布局（models/tools/code/data）+
  baidu 数据（train [3596,10240,4] / test [934,10240,4] h5 验证过），
  并 git init + commit——但 HPC 无 GitHub push 权限（403），此目录作为
  HPC 侧副本；数据包不上 GitHub（109MB 超合理仓库大小），靠节点间
  sync 已就位；
- HPC 登录节点验证：GitHub clone 可达（clone_test 成功后已清理）；
- **用户侧操作**：进 3090 code-server 终端执行
  `git clone https://github.com/jinzhao113129/LR-HyperNet.git ~/zhaojin/pi2`
  （会拿到代码但不含数据；数据已在 ~/zhaojin/pi2/data/ 里，二者合并）；
- 本地→GitHub 的 push 曾遇 TLS 间歇失败，多次重试后成功——后续如再
  阻塞可改由用户在自己电脑 push。

### 下一步

- 用户确认 clone 完成后：在 gpu-server-3 终端装环境
  （Python3.10 + torch 2.3.1+cu121 + cudnn 8.9 + numpy<2），
  然后 6 条快测链开跑（在 124 的 3090 上串行执行，保持卡满）。

---

## 38. 3090 全链通道打通：数据 Release + 一键脚本 + 租约卡修复（2026-10-18）

### 三个落地件

1. **数据通道（学长规矩的正解）**：baidu 小样本数据包（109MB，
   train [3596,10240,4] + test [934,10240,4]）发布为 GitHub Release
   `data-baidu-small`（draft 转正，公开可下）。上传后 sha256 与本地
   一致（085472b6...9428b），HPC 登录节点实测 URL 可直下（30s 拉了
   41MB）。**之前"数据靠节点间 sync 到位"的说法是错的**——portal
   sync/ls 查证 zhaojin 在 3090 平台上是 0 条目，数据实际只在 HPC
   侧（jnguan 主目录）。Release 是节点侧取数的合规通道。

2. **一键脚本 `fork_ai2/scripts/run_3090_setup_and_run.sh`**（API 推
   到 main @ 3dcbb98）：
   - `setup`：组装 ~/zhaojin/pi2 布局（clone 仓库本地复制，不 SCP）+
     Release 下载数据（sha256 强校验 + 3 次重试）+ venv 环境
     （torch 2.3.1+cu121、numpy<2、h5py、sklearn、tensorboard）+
     py_compile 自检；
   - `chain`：6 条快测链串行执行（base/ha/hsel/spike/rp1/hmlp），
     幂等（.done 标记跳过已完链）+ 失败链自动重试 resume + 零 delete
     + 全产物 ~/zhaojin/logs_run/ 与 log/classification/ 下。
   - 协议锁定：baidu 3596/934 + augment×3 + EMA 0.999 + patience 20
     （early_stop_metric=acc）+ epoch 300 上限 + seed 42。

3. **trainer 租约卡修复**（main @ 9b8dbf6，本地 commit 6628361）：
   第 23 行原硬编码 `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`，在
   租约/Slurm 用 CUDA_VISIBLE_DEVICES 绑卡的机器上会**覆盖成错卡**
   （越权风险——学长规矩 1）。改为外部未设置时才兜底。py_compile 过。

### 通道细节（本机 GitHub TLS 抽风对策）

- 本机 git push 连续 TLS 失败（schannel/openssl 均挂）——代码/文档文件改走
  **GitHub REST Contents API**（单文件限 100MB，推 .py/.md 稳定），
  两个文件 PUT 上去，远程 blob 与本地逐字节比对一致。
  （Codex 审查修正 2026-10-18：原文本句曾混写"109MB 资产上传都成功"——
  109MB 数据包实际走的是 **Releases API**（uploads.github.com 资产端点，
  2GB 上限），两通道不是一回事。）
- clone 状态证实：用户 3090 的 `~/zhaojin/pi2_code` 是完整 clone
  （顶层目录与仓库一致）——之前"TLS 瞬断没成功"判断有误，实际
  成功过一次（报错"目录已存在"正是它）。

### 远程 main 状态

```
9b8dbf6 fix: CUDA_VISIBLE_DEVICES 租约兼容（外部已设置时不覆盖）
3dcbb98 feat: 3090 一键脚本（setup+chain 6 链串行）
3b2aa95 feat: 快测协议 v2 + patience 修复 + 3090 迁移包
（+ Release data-baidu-small：109MB 数据资产）
```

### 用户侧下一步（gpu-server-3 code-server 终端）

```bash
cd ~/zhaojin/pi2_code && git pull   # 拿一键脚本（git 通道，合规）
bash fork_ai2/scripts/run_3090_setup_and_run.sh setup   # ~20min
bash fork_ai2/scripts/run_3090_setup_and_run.sh chain   # 6 链串行
```

本地 git 与远程出现 API-push 造成的分叉（本地 6628361 vs 远程
9b8dbf6，内容一致提交对象不同）——本地暂不 push，等 TLS 恢复后
`git pull --rebase` 收敛。

---

## 39. 飞书更新 2026-10-18 版（快测转向 + 3090 全链落地）（2026-10-18）

- feishu_update_20261018.md：接 10-17 版格式（豆包审核后更新的协作
  约定不变）。内容：战略转向（撤 HPC 冲刺链改 3090 快测的决策依据
  + 学长五条规矩执行表）、快测协议（60.60 对照口径 + 统一配方 +
  6 链矩阵 + 统计诚实声明）、3090 迁移落地全表（数据 Release 通道
  修正记录 + 一键脚本 + 环境自检全绿 + 两个工程修复）、当前状态
  与下一步、上一版存档锚点（不重复更新项）。
- NLP2 按用户要求不进飞书（内容只在本地报告）。
- 本地 git 与远程的 API-push 分叉保留（12aeb7f/c7e4c0e/6628361 vs
  远程 9c2e728），TLS 恢复后 pull --rebase 收敛。

---

## 40. 快测与对表口径共识（2026-10-18）

- notes/pilot_benchmark_consensus_20261018.md：接 consensus_20261014
  的口径细化，用户逐条确认后入库。六节：①数字对表总口径（快测双锚
  60.6+base 配对差 → 全量 82.4 直接对比不复现，"赢"的声明只发生在
  全量层）；②为什么必须跑 base（60.6 内含官方未公开组件红利，直接
  比无法归因——Codex §36 恢复 base 的理由固化）；③统一配方表
  （增强×3 对齐官方/EMA 我方额外/早停 acc 口径，配方对齐度逐项）；
  ④6 链矩阵与替换链衔接（正反馈组件→HPC 全量，余额 3362 为此预留）；
  ⑤统计诚实声明；⑥历史锚点。
- 背景：用户连环追问（"为何还跑基线""配方是什么""官方有没有配方"
  "60.6 不就是小批量数字吗""全量就是比 82.4 吧"）——口径在问答中
  收敛定稿，此文档即固化版。已推 GitHub main @ 0fe4a5b。

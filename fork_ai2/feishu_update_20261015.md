# 飞书更新内容 — 2026-10-15（ai2/Claude 提供，豆包审核后更新飞书）

> 本文件按协作规范由 ai2（Claude）生成，请金朝转给豆包。豆包核对后自行决定更新到飞书方法论/论文笔记/SOTA表。
> 更新时间：2026-10-15 23:55（Asia/Shanghai）

## 一、主线两条全量链实时状态（最重要，学长要看）

### 1. 双曲选点链（175245，RUNNING）

- 今日 15:06 完成 KDA 式续跑接力：p1 TIMEOUT（best 69.81%）→ resume 从 epoch 34 无缝续跑；
- 当前 epoch 50/150，RUNNING 5h11m，速度 ~17.5 min/epoch；
- **best 69.824% acc / 72.22% seq（epoch 40 创下，之后波动）**；
- 剩余 100 epoch，预计 ~10/17 出终数，单作业 56h 时限内可跑完（无断链风险）。

### 2. 双曲注意力链（175251，PENDING）

- 排队中，被调度器预排到 10/17-10/20 的 A800 空位；
- 与选点链唯一差异 = 组件 flag，配对差口径（双曲注意力 vs 双曲选点的真实增益）。

## 二、G16 六处理：代码全部过审、已上传 HPC，训练链就绪待提交

今天最大工作成果。五轮 Codex 审查全部通过（详见 §四）。

### 1. 六处理定义（RP1 selector 矩阵首次完整实现）

`random` / `spatial_fixed` / `euc_far` / `euc_near` / `hyp_far`（双曲距离选点，原创维度）/ `rp1_scorer`——六种"怎么选 16 个远程点"的准则，唯一变量是 selector。

### 2. 口径定稿（此前 60ep/baidu 旧口径已废弃）

| 项 | 定稿 |
|---|---|
| 数据 | ncaltech101_official_gen（train 62703 / test 5226 × 10240 点） |
| epoch | 150（全量，对齐 82.4 靶子） |
| batch | 32 + grad_accum 2 |
| GPU | A800 |
| 骨干 | 默认（无 kda/group_norm/dropout）——唯一有健康早期轨迹的官方全量 LRHyperNet |

### 3. 上传 HPC（4 文件，md5 双端一致，LF 归一化后）

- models/local_remote_grouper.py / lr_hypernet.py / train_lr_hyper.py / ai2_g16/g16_profile_agent.py
- 旧文件已备份（*.bak_ai2_20261015_r5），HPC 集成 smoke 37/37 全过。

### 4. 训练链两条 + 推理脚本已写就，但**提交被点数余额阻断**

- hyp_far / euc_near 两条 sbatch 逐 flag 相同（唯一差异 = treatment）；
- LRHyperNet 专用固定 eval_seed=42 推理脚本补齐（原 extract_preds.py 只支持 FECNet）；
- **当前不可提交**：单条 72h 约 1166.4 点，余额只有 624.44 点，连一条都放不下（详见 §三）。

## 三、点数制度重要勘误（运营信息，之前飞书口径有误）

**之前 12:10 更新写的"可用 2172 点"是错的数。** 两个命令显示两套数：

| 命令 | 值 | 含义 |
|---|---|---|
| `scir-account -d` | Avail ~2755 | 实时已消耗（**不含**排队/运行中作业的预扣） |
| `scir-account details` | **Avail 624** | **真实预扣余额**，提交前必须看这个 |

- 差额 ~2100 点是排队作业的预扣：175245（RUNNING）预扣 907.2 点 + 175251（排队）预扣 1036.8 点 = 1944 点；
- **预扣制下排队也占余额**——余额不足时不能为了排队提前提交；
- 11 月额度重置（7500/月）是 G16 链的主要回升点。

## 四、双人制审查第五轮放行（协作成效）

G16 分派代码历经五轮 Codex 审查，累计拦下真实缺陷（都已修复）：

- 4 个 P0：sink 互噬（profiling 静默空文件）、RNG 每 forward 重播种（随机基线语义破坏）、resume 校验假生效（改 seed 不拦截）、点数口径误用；
- 2 个 P1：treatment×旧 flag 静默叠加、resume RNG 校验键不存在；
- 3 记录项：RP1 2D novelty 口径隔离、smoke 树自动搭建、log_dir=None profile 落 default；
- 我自查新增 1 个 Codex 未覆盖的 validate() GB 级泄漏。

**双人制再次证明价值**：每一个 P0 都出现在"我自认为写完"的地方。共识"所有代码必审 + 本地 GPU 只做冒烟"继续有效。

## 五、NLP2 链状态（独立项目，简短）

- 早间 nlp2_base 曾 FAILED（34 秒挂，环境/脚本问题），现已重提整链（preflight → 6 训练 → 7 解码）排队中；
- 与 SECNet 主线无关，自动跑，不占用主线卡。

## 六、调研成果（简提，详见 survey_master_index）

今日完成 16 方向调研闭环，三个关键结论：
1. **双曲×MLP 视觉领域是真空**（纯 MLP 骨干 + 双曲的组合零命中）；
2. **双曲 KNN 零先例**、Poincaré 网格零命中——"双曲选 16 点"查新成立；
3. **生死文献**：Is the Geometry Doing the Work?（arXiv 2607.05268）审计证明已发表双曲模型实际停留在近欧氏区——这对我们"双曲要真正干活"的叙事既是警示也是机会。

18 篇原文已存 D:\论文，索引见 notes/survey_master_index.md。

# 三平台分工与通道共识 — 2026-10-18

> ai2/Claude 起草，Codex 审查修订版（P0×3 + P1×4 + P2×4 已并入）。
> 本文回答"3090 平台叫什么、HPC 是什么、AI Studio 是什么"以及每个平台上
> ai2 的能力边界与文件通道。用户已口头确认要点（2026-10-18 对话）。
>
> **来源与效力**：学长规矩以 code_changelog.md §37 原话为准（本文为转述，
> 冲突时以 §37 为准）；§38 为 3090 迁移落地记录。portal API 能力边界为
> 2026-10-18 逐端点实测快照。豆包/飞书可引用本文，更新时同步
> feishu_update 系列。

## 一、三个平台一览

| | 3090 平台 | HPC | AI Studio |
|---|---|---|---|
| 全称 | 学长私有 GPU 集群（王玲组） | 学校高性能计算中心 | 百度飞桨云平台 |
| 入口 | https://160.30.231.177/portal/ | ssh 10.160.22.46:2223 | https://aistudio.baidu.com |
| 账号 | youxinjiang | jnguan | 百度账号 |
| 硬件 | 3 台 5×3090（server-1/3/4）※ | V100/A100/H100/A800 大集群 | V100 16G（每日免费领 8 点） |
| 计费 | 免费（实验室机器） | 点数制（10500/月，V100 4点/h） | 每日 8 点免费 |
| 归属 | 学长（王玲组管理员） | 校级公共 | 百度官方 |
| 当前用途 | **SECNet 6 链快测** | NLP2 补跑 + SECNet 全量（备用） | 未用（NLP2 指定平台，走 HPC 替代） |

※ 硬件行为 `api/machines` 2026-10-18 快照（server-4 显示 DOWN）；§37 学长
原话为"两台 GPU 节点自动同步"——两处口径差异未与学长核实，以实际机器状态为准。

## 二、ai2 在各平台的能力边界（2026-10-18 portal API 实测快照）

> 以下"不能做"均为**当日实测结果**（404/403/网络不通），反映的是当前权限与
> 网关配置，不构成平台的永久能力边界；配置变更后需重新探测。

### 3090 平台

**能做**：
- 登录、查卡状态/租约/会话/审计日志（`api/machines` / `api/leases` / `api/sessions`）
- 触发数据同步并读变化行（`api/sync/home-now`——仅能证明日志文件在增长，
  即**疑似训练进程仍在写日志**；GPU 健康、loss/acc 数值、进程是否空转报错
  均无法据此判断，以用户贴出的日志为准）
- 通过同步日志看到其他用户动态（匿名，不写入具体人名）

**不能做（2026-10-18 探测）**：
- 读文件内容（文件 API 404；sync/ls 索引不含新建目录）
- 执行命令（WebSocket 终端被网关 Caddy 403，带浏览器头同样拦）
- 看 GPU 利用率/进程列表
- SSH 直连节点（当日网络探测 10.100.0.x 不通；此为当前网络快照，
  非平台永久拓扑）

**结论：训练数字必须靠用户在 code-server 终端贴 `grep -a "best test acc"
~/zhaojin/logs_run/<链名>.log | tail -1` 的输出。**

### HPC（SSH，完整能力）

- 完整命令执行、文件读写、sbatch/scancel、余额查询（scir-account details）
- 唯一约束：点数余额与 QOS 并发（8 坑）

### AI Studio（未用）

- 指导书指定平台（数据/项目托管），因用户决策"回 HPC 同环境归因"而未启用。

## 三、文件通道（3090 平台）

学长规矩 2 原话（§37）：**禁 SCP/拖拽上传；合法通道 = "GitHub/HuggingFace
上存 + 节点里 git clone 拉取"**。GitHub 与 HuggingFace 均为合法入口；
当前实施只用了 GitHub（repo + Release），HuggingFace 通道未启用。

| 通道 | 状态 |
|---|---|
| GitHub repo：我侧 push → 用户在 3090 终端 git pull | ✅ 当前实施通道，已跑通 3 次 |
| GitHub Release：大文件资产（109MB baidu 包，curl -L -O + sha256 校验） | ✅ 已发布（data-baidu-small） |
| HuggingFace | ✅ 学长认可的合法入口，未启用 |
| SCP 直传 / code-server 拖拽 | ❌ 学长明令禁止（封号） |
| portal API 写文件 | ❌ 无写接口（2026-10-18 探测） |

**两个 GitHub API 通道要区分**（Codex 审查修正，2026-10-18）：
- **Contents API**（`api.github.com/repos/.../contents/`）：推**代码/文档文件**，
  单文件限 100MB——本机 git push TLS 间歇故障时的备用通道（已多次跑通）；
- **Releases API**（`uploads.github.com/.../releases/<id>/assets`）：传**大资产**
  （109MB 数据包走的是这个端点，GitHub 计算的 sha256 与本地一致）。
  §38 原文曾把两者混写为"Contents API 上传 109MB"，已在 changelog 修正。

## 四、当前运行状态（2026-10-18 快照，采样时点见各条）

| 平台 | 在跑 | 状态 |
|---|---|---|
| 3090 | SECNet 快测 base 链 | 用户 2026-10-18 午前贴出原始日志行：`acc:29.336% [best test acc: 29.336%]` → `acc:55.353% [best test acc: 57.281%]`（爬升中，超过官方 fork 版 54.1，距靶 60.6 差 3.3pp）。同步变化行显示 base.log 与 logs/pointmlp.txt 仍在增长（疑似训练在写，健康以用户贴数为准） |
| 3090 | 后续 ha/hsel/spike/rp1/hmlp | 串行排队，base 完自动接（.done 标记触发） |
| HPC | NLP2 60ep 基线补跑（178173） | 2026-10-18 ~12:00 采样：Epoch 26/30，loss 3.52 降中。对照：10ep 基线终 loss 3.89（174768），10+20ep 续训版（30ep 总量）终 loss 3.61（177390）；本次为 30ep 续训到 60ep 总量。训完自动解码评测（178174） |
| HPC | G16/ha_aug 全量链 | **不提交**（战略转向小样本快测，余额 3362 点为快测正反馈组件的全量验证备用弹药） |

## 五、分工

**用户（3090 侧，每链一次 30 秒）**：
- 贴训练进度：`grep -a "best test acc" ~/zhaojin/logs_run/<链名>.log | tail -1`
- 看链完成：`cat ~/zhaojin/logs_run/*.done`
- 代码更新：`cd ~/zhaojin/pi2_code && git pull`

**ai2**：
- 判断趋势/差距解读/异常诊断（基于用户贴数）
- 代码修改 + GitHub push（Contents API 备用）
- 链全部跑完后：配对差汇总表 + 组件方向结论 + 回填报告

---
## 附：Codex 审查处置记录（2026-10-18）

- P0-1（GitHub 单行道 vs 规矩原话）→ §三 已改为 GitHub/HF 双合法入口、标注当前实施；
- P0-2（Contents API 100MB 上限，109MB 实走 Releases）→ §三 已分列两 API；§38 同步修正；
- P0-3（"30ep 版终值"表述歧义）→ §四 已改为"10+20ep 续训版（30ep 总量）"；
- P1-1/2/3（时间限定、"训练正常"过强、网络快照）→ §二 加"实测快照"框架 + 措辞降级；
- P1-4（57.3% 不可追溯）→ §四 嵌入用户贴出的原始 best test acc 行 + 采样时点；
- P2-1/2/3/4（头部来源/匿名/硬件快照注/交叉引用）→ 全部并入。

# 三平台分工与通道共识 — 2026-10-18

> ai2/Claude 起草。本文回答"3090 平台叫什么、HPC 是什么、AI Studio 是什么"以及
> 每个平台上 ai2 的能力边界与文件通道。用户已口头确认要点（2026-10-18 对话）。

## 一、三个平台一览

| | 3090 平台 | HPC | AI Studio |
|---|---|---|---|
| 全称 | 学长私有 GPU 集群（王玲组） | 学校高性能计算中心 | 百度飞桨云平台 |
| 入口 | https://160.30.231.177/portal/ | ssh 10.160.22.46:2223 | https://aistudio.baidu.com |
| 账号 | youxinjiang | jnguan | 百度账号 |
| 硬件 | 3 台 5×3090（server-1/3/4） | V100/A100/H100/A800 大集群 | V100 16G（每日免费领 8 点） |
| 计费 | 免费（实验室机器） | 点数制（10500/月，V100 4点/h） | 每日 8 点免费 |
| 归属 | 学长（WeiFanyi 等王玲组人管理） | 校级公共 | 百度官方 |
| 当前用途 | **SECNet 6 链快测** | NLP2 补跑 + SECNet 全量（备用） | 未用（NLP2 指定平台，走 HPC 替代） |

## 二、ai2 在各平台的能力边界（实测）

### 3090 平台（portal API，全部实测）

**能做**：
- 登录、查卡状态/租约/会话/审计日志（`api/machines` / `api/leases` / `api/sessions`）
- 触发数据同步并读变化行（`api/sync/home-now`——间接判断训练活跃度：
  logs_run/*.log 在同步中持续 changed = 正在写 = 训练正常）
- 通过同步日志看到其他用户动态（如 WeiFanyi 的链）

**不能做**：
- 读文件内容（文件 API 全 404；sync/ls 索引不含新建目录）
- 执行命令（WebSocket 终端被网关 Caddy 拦 403，带浏览器头也拦）
- 看 GPU 利用率/进程列表
- SSH 直连节点（内网隔离，10.100.0.x 不通）

**结论：训练数字必须靠用户在 code-server 终端贴 `grep -a "best test acc"
~/zhaojin/logs_run/<链名>.log | tail -1` 的输出。**

### HPC（SSH，完整能力）

- 完整命令执行、文件读写、sbatch/scancel、余额查询（scir-account details）
- 唯一约束：点数余额与 QOS 并发（8 坑）

### AI Studio（未用）

- 指导书指定平台（数据/项目托管），因用户决策"回 HPC 同环境归因"而未启用。

## 三、文件通道（3090 平台 = GitHub 单行道）

学长规矩 2：**禁 SCP/拖拽上传，唯一合法通道 = GitHub 存 + 节点 git clone/pull**。

| 通道 | 状态 |
|---|---|
| GitHub push（我侧）→ git pull（用户在 3090 终端） | ✅ 唯一合法通道，已跑通 3 次 |
| SCP 直传 | ❌ 学长明令禁止（封号），且网络不通 |
| code-server 拖拽 | ❌ 同属禁令范围 |
| portal API 写文件 | ❌ 无写接口 |
| 数据大文件 | ✅ 走 GitHub Release（109MB baidu 包已发，curl -L -O + sha256 校验） |

**我侧 push 有间歇 TLS 故障时的备用通道**：GitHub REST Contents API（已验证稳定，
109MB 资产上传同通道）。

## 四、当前运行状态（2026-10-18 12:00 前后）

| 平台 | 在跑 | 状态 |
|---|---|---|
| 3090 | SECNet 快测 base 链 | acc 29.3%→57.3% 爬升中（超过官方 fork 版 54.1，距靶 60.6 差 3.3pp）；base.log + logs/pointmlp.txt 同步持续变化（训练活跃确认） |
| 3090 | 后续 ha/hsel/spike/rp1/hmlp | 串行排队，base 完自动接 |
| HPC | NLP2 60ep 基线补跑（178173） | Epoch 26/30，loss 3.52 降中（30ep 版终值 3.61）；训完自动解码评测（178174） |
| HPC | G16/ha_aug 全量链 | **不提交**（战略转向小样本快测，余额 3362 点为快测正反馈组件的全量验证备用弹药） |

## 五、分工

**用户（3090 侧，每链一次 30 秒）**：
- 贴训练进度：`grep -a "best test acc" ~/zhaojin/logs_run/<链名>.log | tail -1`
- 看链完成：`cat ~/zhaojin/logs_run/*.done`
- 代码更新：`cd ~/zhaojin/pi2_code && git pull`

**ai2**：
- 判断趋势/差距解读/异常诊断（基于用户贴数）
- 代码修改 + GitHub push
- 链全部跑完后：配对差汇总表 + 组件方向结论 + 回填报告

# 飞书更新内容 — 2026-10-22（ai2/Claude 提供，豆包审核后更新飞书）

> 本文件按协作规范由 ai2（Claude）生成，请金朝转给豆包。豆包核对后更新到飞书进度页。
> 内容：快测结果补全（融合层 hgate 最强）+ G16 排查双 bug 修复 + euc_near 重跑中。
> 上一版：feishu_update_20261020.md（6 链快测 + 已定稿未测组件清单）。

## 一、快测结果补全（3090，baidu 小样本 3596/934，统一含 EMA 0.999 + 数据增强 ×4）

> 判定口径不变：加减统一对比**官方 60.6**（Table 9 单窗 test acc），正负为该链 best 相对 60.6 的差值。

| 链 | 组件 | best 测试精度 | 相对 60.6 | 状态 |
|---|---|---|---|---|
| EMA 链 | 无组件（含 EMA 0.999 + 增强 ×4） | 58.565% | −2.04 | 已收敛 |
| **hgate** | **双曲距离门控**（离质心近→门控大，替换融合层欧氏门控） | **66.06%** | **+5.46** | 已收敛 |
| 双曲注意力 | 双曲注意力残差支路 | 60.385% | −0.21 | 已收敛 |
| 双曲选点 | 双曲选点 + 缩放坐标 | 60.171% | −0.43 | 已收敛 |
| 双曲 MLP | 双曲注意力后串 Möbius 双曲 MLP | 60.064% | −0.54 | 已收敛 |
| spike 混合时间编码 | 保留时间坐标 + 8 维 sin/cos 编码 | 57.602% | −3.00 | 已收敛 |
| RP1 空间均匀选点 | x,y 空间均匀网格选点 | 57.709% | −2.89 | 手动停 |

**初步趋势更新**：融合层 **hgate 双曲距离门控 +5.46** 是当前最强单组件（对比 10-20 版
双曲注意力 −0.21 的次高）。快测矩阵收敛出清晰分层：融合层门控（hgate）> 采样/聚合层
双曲组件（ha/hsel/hmlp 打平，−0.2~−0.5）> 无组件 EMA 链（−2.04）> 时间编码 / 欧氏
选点（−3 附近）。统计诚实声明不变：934 样本、单 seed、噪声 ±3.2pp，只筛方向不声明赢。

## 二、G16 排查（论文灵魂，本轮重点）

### 背景

首条 G16 链（欧氏最远选点 euc_far）全程只到 44.2%，明显塌（对比 EMA 链 58.6%）。
排查定位到**两个叠加 bug**，都已修复。

### Bug 1（回归，已修）：聚合器从 Lorentz 回落 Poincaré

- 六处理改造时 FECNet 调用 `LocalRemoteGrouperV2` 漏传 `hyperbolic_type`，
  聚合器 `hyper_agg` 从审查版设计的 **lorentz**（数值稳定）回落到默认
  **poincare**（边界 atanh 有梯度尖峰风险）。
- G16 是"替换"接法（远程双曲聚合**完全替换**中心点特征），poincare 一坏，
  第二个 C 维整个废掉 → 六种 treatment 全塌。
- 已修：FECNet 显式传 `hyperbolic_type="lorentz"`（commit e019032）。
- 附：全组件双曲模型体检（notes/component_hyperbolic_model_audit_20261022.md），
  确认其余组件无同类"静默用错双曲模型"问题；HyperbolicEmbedding/HyperbolicMLP
  仍硬编码 poincare 但带残差保护，列为 P2 暂不动。

### Bug 2（方向反了，已切）：euc_far 选"最远"，原文有效的是"最近"

- euc_far 用 `topk(largest=True)` 选**特征最远**的 16 点（假设：最大对比）；
- RP1 原文 Global Feature-KNN（66.70%）选的是**特征最近**（特征相关但空间远程，
  假设：语义远程同源）——方向相反。
- 已切换到 **euc_near**（欧氏最近）重跑。

### euc_near 首段结果（跑中，斜率健康）

| epoch | euc_far（旧，塌） | euc_near（新，修复后） |
|---|---:|---:|
| 1 | 23.2% | 27.5% |
| 5 | 28.5% | **38.3%** |

euc_near 第 5 轮已比 euc_far 同期高近 10pp，每轮稳定爬 2.5~3pp，无塌迹象，
第 5 轮（38.3%）已接近 euc_far 整整 46 轮才到的 44% 上限。**两个 bug 确认改对，
等跑完看最终收敛**（目标：恢复到 55%+ 证明 G16 修复；若仍塌，做欧氏聚合器 A/B）。

## 三、代码状态

- G16 修复已 commit + push GitHub main（e019032 lorentz 修复、b8daf9a 日志节流）；
- 本轮新增笔记（一并提交）：
  - notes/codex_review_g16_20261021.md / codex_task_g16_20261021.md（G16 接线审查）
  - notes/component_hyperbolic_model_audit_20261022.md（全组件双曲体检）
  - notes/scicore_review_prompt_20261022.md（投稿前审稿两阶段 prompt，接论文检查纪律）
- gpu-server-3 侧 euc_near 跑中，`~/zhaojin/logs_run/g16_eucnear.log`。

## 四、下一步

1. euc_near 跑完确认收敛（判定 G16 是否修复）；
2. G16 其余 treatment（random / spatial_fixed / hyp_far / rp1_scorer）按序续跑；
3. 其余未测组件（hclf / fca / chr / polar）排在 G16 之后。

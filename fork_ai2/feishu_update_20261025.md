# 飞书更新内容 — 2026-10-25（ai2/Claude 提供，豆包审核后更新飞书）

> 上一版：feishu_update_20261023.md（G16 固定选点扫完 + rp1 方向修复 + 转向 learned）。
> 内容：固定选点彻底扫完 + learned scorer 三轮选择器结论 + 论文调研（瓶颈定位）。

## 一段话（可直接发学长/飞书）

G16 固定选点彻底扫完了：欧氏最近（euc_near）59.10%、加空间新颖性（rp1_scorer）59.64% 是最优，双曲选点（hyp 系列）44-46% 塌、双曲分类头（hclf）53% 负贡献——所以"双曲"只在聚合层有效（hgate 66.06% 最强）。基于此做了可学习选点（learned scorer，让网络自己学选哪 16 个点，论文灵魂），三个可微选择器都试了：ST_k 稀疏梯度冷启动慢（ep1=24.7%）、Gumbel 稠密梯度改善但仍追不上固定选点（ep3=32% vs 固定 35%）、LapSum（Σp=k 精确 + 稠密 + 确定）已审过待跑。结论：瓶颈不在"怎么选"的选择器，在"打什么分"——RP1 原文列了 8 个选点特征，我们 scorer 只用了 2 个（缺事件密度、XY 距离等）。同时定位一个关键差异：学长原来 Global Feature-KNN 66.7% 用的是 cosine 相似度，我们实现成了欧氏距离（59%），差的 7 个点很可能就是这个度量。下一步：等学长 RP1 原版代码确认 cosine，补打分信号。

## 对照表（豆包核对数字用）

| 链 | 选点语义 | 结果 |
|---|---|---|
| rp1_scorer（方向修复后） | 特征最近 + 空间新颖 | 59.64% ✅ 最优 |
| euc_near | 欧氏最近 | 59.10% |
| euc_far / hyp_far / hyp_near | 最远 / 双曲 | 44-46% 塌 |
| hclf | 双曲分类头 | 53% 负贡献 |
| 无组件 EMA 链 | 基线 | 58.57% |

判定口径不变：统一对比官方 60.6（Table 9 单窗 test acc）。

## 代码状态

- learned scorer 三轮选择器（ST_k/Gumbel STE/LapSum）+ 归一化 bug 修复，均已 push、Codex 审过；
- 新增笔记：scorer_signal_enhancement_20261025.md（打分信号缺口方案）、
  gaussian_param_event_plan_20261024.md（4DGS 迁移，已证伪）；
- 论文调研 36 篇落本地（可微选点技术路线完整）。

## 下一步

1. 等学长 RP1 原版 selector 代码 → 确认 cosine vs 欧氏 → 跑 cosine_near（可能找回 7 个点）；
2. 补打分信号（事件密度 + XY 距离），方案已写待 Codex 审；
3. learned 结果出来补归因消融（Codex 点出：learned 同时变了选点+聚合+池，需同池同聚合只变 scorer）。

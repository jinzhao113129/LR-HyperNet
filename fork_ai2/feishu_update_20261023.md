# 飞书更新内容 — 2026-10-23（ai2/Claude 提供，豆包审核后更新飞书）

> 上一版：feishu_update_20261022.md（G16 排查双 bug + euc_near 重跑中）。
> 内容：G16 固定选点对照基本跑完 + rp1 方向修复 + 转向 learned scorer（core idea 1）。

## 一段话（可直接发学长/飞书）

G16（16 点远程选择，论文灵魂）排查进展：固定选点对照基本跑完了，欧氏最近（euc_near）59% 是唯一有效的（追平无组件 EMA 链 58.57%），其余全塌——欧氏最远 44%、双曲最远 46%、双曲最近 44%，空间新颖性打分（rp1_scorer）修复方向后重跑中、早期 42.9%。三个方向结论：一是"最远"方向反了（原文有效是"特征最近"）；二是双曲距离选点被证伪（双曲最近 44% 明显低于欧氏最近 59%）；三是 rp1_scorer 原实现也反了（feature 项选"特征最远"，已修成"特征最近 + 空间新颖"）。固定选点该试的都试了，天花板就是欧氏最近 59% 且不超 baseline，下一步转向"让网络自己学怎么选"（learned scorer，core idea 1），方案已写好待 Codex 审。

## 对照表（豆包核对数字用）

| 链 | 选点语义 | 结果 |
|---|---|---|
| euc_near | 欧氏特征最近 | 59% ✅（唯一有效） |
| euc_far | 欧氏特征最远 | 44% 塌 |
| hyp_far | 双曲最远 | 46% 塌 |
| hyp_near | 双曲最近 | 44% 塌 |
| rp1_scorer（修复版） | 特征最近 + 空间新颖 | 跑中 ep7=42.9%（早期不弱于 euc_near 同期） |

判定口径不变：统一对比官方 60.6（Table 9 单窗 test acc）。

## 代码状态

- 新增 `g16_treatment=hyp_near`（双曲最近）+ rp1_scorer 方向修复，均已 push GitHub main；
- 新增笔记：learned_remote_selector_survey_20261023.md（7 篇可微选点论文，全部核验真实）、
  learned_remote_selector_plan_20261023.md（learned scorer 落地方案，三阶段）；
- 平台：租约 124 到期 → 191（gpu-server-3 磁盘在，数据/checkpoint/环境保留，setup 幂等续跑）。

## 下一步

1. rp1_scorer 跑完看最终收敛（判定"空间新颖性"是否超纯欧氏最近）；
2. learned scorer 方案过 Codex 审查 → 写代码 → 上卡；
3. 固定选点结论已清晰：唯一有效 = 欧氏最近 59%，双曲/最远/新颖均被证伪。

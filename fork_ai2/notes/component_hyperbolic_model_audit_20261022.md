# 组件双曲模型体检（2026-10-22）

> 起因：G16 euc_far（44%）/ rp1_scorer（24%）全塌，排查发现六处理改造时
> FECNet 调用漏传 `hyperbolic_type`，聚合器从审查版设计的 lorentz 回落到默认
> poincare（有边界 atanh 梯度尖峰风险）。本笔记系统体检所有双曲组件，防止同类
> "静默用错双曲模型"再发生。

## 一、体检总表

| 组件 | 双曲模型 | 接法 | 风险 | 状态 |
|---|---|---|---|---|
| HyperbolicFusionBranch（hyperfusion） | lorentz | 残差门控 | 无 | ✅ |
| HyperbolicAttention（ha） | lorentz | 0.1 残差 | 无 | ✅ |
| HyperbolicGate（hgate） | lorentz | 替换 gate | 无 | ✅ |
| HyperbolicRadialRectify（chr） | lorentz | 单模块 | 无 | ✅ |
| HyperbolicPrototypeClassifier（hclf） | lorentz | 分类头 | 无 | ✅ |
| HyperbolicUniformGrouper（hsel） | Poincaré 盘选点 | no_grad 选点 | 无 | ✅ |
| **G16 聚合器** | **poincare → 已修 lorentz** | **替换中心点特征** | **致命（已修）** | ✅ |
| HyperbolicEmbedding（hyperembed） | poincare | 残差 + 尺度归一 | 低 | ⚠️ |
| HyperbolicMLP（hmlp） | poincare | 0.1 残差 + 零初始化 | 低 | ⚠️ |
| FcaGate / Polar / Mamba / RP1Grouper | 欧氏 | — | 无 | ✅ |

## 二、G16 回归（本次修复）

- **根因**：`LocalRemoteGrouperV2` 的 `hyperbolic_type` 默认 `"poincare"`
  （[local_remote_grouper_v2.py:990](scripts/local_edit/local_remote_grouper_v2.py:990)），
  六处理改造时 FECNet 调用只传了 `g16_treatment/g16_rng_seed/stage_idx`，漏传
  `hyperbolic_type`，聚合器 `hyper_agg` 落到 `HyperbolicRemoteAggregator(hyperbolic_type="poincare")`。
- **对照**：审查版设计（codex_task_g16_20261021.md:33）明确写
  `hyperbolic_type="lorentz"`（v3.1 数值稳定版），codex_review 复跑也记录
  `hyperbolic_type=lorentz`。
- **修复**：[FECNet.py:1077](scripts/local_edit/FECNet.py:1077) 显式传
  `hyperbolic_type="lorentz"`（commit 589ab23）。选点逻辑不变，只改聚合器。
- **为何致命**：六处理共用这一个聚合器 → 六种 treatment 全塌。且 G16 是
  "替换"接法（见下）。

## 三、关键区分：替换 vs 残差，决定 poincare 是否致命

同样的 poincare，G16 塌、Embedding/MLP 不塌，差别在接法：

- **G16（致命）**：`LocalRemoteGrouperV2.forward` 用"远程双曲聚合特征"**完全替换**
  原 LocalGrouper 的"中心点特征"那个稳定锚点（[local_remote_grouper_v2.py:1253]）
  → poincare 一坏，第二个 C 维整个废掉。
- **Embedding（安全）**：`layer_norm(x + x_out)`（[FECNet.py:545]），原特征兜底。
- **MLP（最安全）**：`x + 0.1 * x_hyper` 且 W/b 零初始化（初始输出恒 0）
  （[FECNet.py:1227]）。6 链 hmlp 60.064% 中性印证。

## 四、遗留 P2（不阻断，暂不动）

`HyperbolicEmbedding` 与 `HyperbolicMLP` 仍在用 poincare，未跟上 v3.1
"lorentz 替代 poincare"的升级。但：

1. 不是"漏传参数的回归"——它们是**硬编码** poincare（直接调
   `poincare_exp_origin` 等函数，无 hyperbolic_type 参数），设计时即如此。
2. 都有残差保护，实测未塌。
3. 理论仍有梯度尖峰风险，属旧版残留。

**行动**：G16 排查告一段落、写论文前，把这两个升级 lorentz 走一轮审查。

## 五、防回归清单（后续任何组件接线都过一遍）

1. 涉及双曲的组件，确认用 lorentz（除非有明确理由用 poincare）；
2. 审查版设计若写了某个参数（如 hyperbolic_type="lorentz"），改造时不得漏传；
3. "替换"接法的双曲模块，必须走 lorentz + 显存/数值实测；
4. 双曲选点（no_grad）用 Poincaré 盘/Lorentz 距离均可，无梯度尖峰问题。

# 调研全景汇总（2026-10-14 ~ 10-15，Claude + Codex 全部成果，一页导航版）

> 两个调研者的全部产出合并视图。原文位置两处：
> Claude 存 `D:\论文\`（10 篇）；Codex 存 `fork_ai2/papers/`（5 篇）。
> 深度分析见各自的 notes 文件（见文末索引）。

## 一、总账

| 调研者 | 轮次 | 方向数 | 新存原文 | 核心产出 |
|---|---|---:|---:|---|
| **Claude**（10-14 晚） | paper_survey | 查新 + SOTA 协议 | 3 篇（BNN/Hyper²/HyperMVP） | 查新口径收窄（EHGCN/PAT 界定）；82.4 协议核查；curl 通路打通 |
| **Codex**（10-14） | CVF 扫描 | 5 篇扫描 | 5 篇（SAMBLE/MEC/ASTW/SDA/PAT） | 可迁移性逐篇判定 + 协议警告 + 查新勘误 |
| **Claude**（10-15 五轮） | arch_survey | **16 个方向** | **8 篇** | 骨架/三件套/竞品/靶子/聚合/采样全证据链 + 审计文献 |
| **合计** | — | **~30 个查询方向** | **18 篇新原文** | — |

## 二、按用途分类的全部关键发现

### A. 查新定位（论文立足点）

| 发现 | 来源 | 对论文的作用 |
|---|---|---|
| **双曲×事件仅 EHGCN 一篇**（检测/图卷积/双空间融合） | Claude 10-14 | 差异化声明：我们做 point-based 分类 + 双曲贯穿 |
| **PAT/GSS（CVPR19）可学习子集采样先例** | Codex 10-14 | Related Work 必引；空位收窄为"双曲/时空联合远程邻居准则" |
| **双曲×MLP 视觉领域零先例** | Claude 轮2 | 新架构"纯MLP×双曲×事件"三轴交集 = 双重真空 |
| **双曲 KNN 检索零先例** | Claude 轮4 | G16 双曲选点 = 方法学空地 |
| **Poincaré 网格零命中** | Claude 轮5 | G16 网格设计同样无先例 |
| **事件点云分类 2026 无新竞品** | Claude 轮3 | 赛道窗口期确认 |
| **RP1 = 学长未完成工作** | 学长确认 10-15 | 我们定位 = 完成 RP1 selector 矩阵的首次完整实现 |

### B. 骨架与组件证据（新架构 v1 弹药）

| 发现 | 来源 | 用途 |
|---|---|---|
| KAN vs MLP：MLP 点云仍占优 | Claude 轮1（2504.13593） | 纯 MLP 骨架选择的文献支持 |
| PointEvent 双序列化（空间+时间两路） | Claude 轮1（2609.20066） | 序列化若走须保时间轴 |
| Non-Local MLP + HPE 位置编码 | Claude 轮3（2603.04099） | 坐标感知设计的"正式化"候选；最近方向邻居（必引） |
| Multi-Timescale log-Bspline 时间编码 | Claude 轮2（2609.05049） | 时间编码组件候选（替代无效的 time_enc/KDA） |
| Ev-DTAD 时间-超图聚合 | Claude 轮4（2605.08825） | 跨窗聚合升级选项（超越官方 3 窗拼接） |
| SAMBLE 相关性×覆盖配额采样 | Codex 10-14 | G16 诊断 A 方向（覆盖约束 pool-near）的文献锚 |
| HexFormer 多层双曲 ViT | Claude 轮2（2601.19849，**Codex 深读已完成**） | 双曲注意力堆叠稳定性 + Hybrid 头设计（结论已整合进 new_architecture_draft §2.2） |

### C. 严谨性武器（防 reviewer 攻击）

| 发现 | 来源 | 防什么 |
|---|---|---|
| **🚨 双曲几何审计（Is Geometry Doing the Work?）** | Claude 轮5（2607.05268，48 页） | "双曲只是装饰"质疑——回应三件套：分布分离（profile agent）+ 受控几何对比（2×2，需全量欧氏对照）+ 半径审计 u=√c·ρ |
| **审计落地方法（Codex 深读 10-16）** | codex_audit_deepread_20261016.md | 可执行：u=√c·ρ 在 selector/聚合器/注意力三处 hook + 阈值 0.84/1 + 曲率扰动 + 欧氏/双曲排序 Kendall τ + floor release 对照 + 5 个 reviewer 攻击逐条回应 |
| Geometry-aware Distance（多样层级结构） | Claude 轮3（2506.18533） | "单一几何假设过强"——与我们 2×2 交互发现同族的文献论证 |
| OmniEvent 批判手工 S-T 权重 | Claude 轮1（2508.01842） | 坐标缩放发现的外部呼应（一） |
| EAS-SNN "adaptive event sampling unaddressed" | Claude 轮4（2403.12574） | 坐标缩放发现的外部呼应（二） |

### D. 协议与靶子

| 发现 | 来源 | 结论 |
|---|---|---|
| SECNet 82.4 = point-based from-scratch 赛道标杆 | Claude 10-14 + 轮2 复核 | **靶子确认无需重估**（EventRPG 85.62 属 SNN+增强组） |
| **82.4 标杆复查（Codex 10-16）** | codex_survey_supplement_20261016.md | 复查确认 82.4 仍是对标；新工作全属 SNN/state-space/BNN预训练/检测族，无 point-based from-scratch 竞品 |
| BNN 90.58 协议缺陷（val 当 test + 蒸馏） | Codex 核验 | 引用分组处理 |
| Hyper² 只验证点云补全 | Codex 核验 | 降级为间接警示 |
| 各家 N-Caltech101 split 互不相同 | Codex + Claude 双重确认 | 协议表必须分组+脚注 |
| 事件增强 EventAug 系统方案 | Claude 轮5（2409.11813） | 过拟合对抗候选（官方协议是简单增强） |
| Hedging on the Frontier 帕累托组件组合 | 学长 10-15（2605.30997） | 阶段四减法消融用帕累托前沿视角选组件组合（多目标：acc/泛化gap/成本），一句话 Related Work 引用 |
| 双曲事件序列化/多尺度双曲聚合/双曲contrastive事件 三方向复查 | Codex 10-16 | 均零直接新命中（EHGCN 仍是唯一邻居；SpikeCLR 是 SNN 非双曲）——三轴真空进一步确认 |

## 三、原文清单（18 篇）

**Claude 存 D:\论文\（10 篇）**：
1. 2610.09873v1.pdf（BNN）
2. 2608.22238v1_Hyper2_Dual_Space_Consistency.pdf
3. 2603.04848v2_HyperMVP_Hyperbolic_Multiview_Pretraining.pdf
4. 2504.13593v4_KAN_or_MLP_点云.pdf
5. 2609.20066v1_PointEvent_序列化运动证据.pdf
6. 2601.19849v1_HexFormer_双曲ViT.pdf（Codex 深读已完成）
7. 2508.01842v1_OmniEvent_统一事件表示.pdf
8. 2609.05049v1_多时间尺度事件表示.pdf
9. 2506.18533v1_几何感知双曲距离度量.pdf
10. 2603.04099v1_非局部MLP_高维位置编码.pdf
11. 2605.08825v5_EvDTAD_表示级时间聚合超图.pdf
12. 2607.05268v3_双曲几何真的在工作吗_审计.pdf ⭐
13. 2409.11813v1_EventAug_多面时空增强.pdf
14. 2605.30997_Hedging_on_the_Frontier.pdf（组件组合帕累托视角，学长 10-15 分享）

**Codex 存 fork_ai2/papers\（5 篇，SHA256 已验）**：
1. CVPR2025_SAMBLE.pdf
2. CVPRW2025_MakingEveryEventCount.pdf
3. CVPR2026_AdaptiveSpatialTemporalWindow.pdf
4. CVPR2026_SpikeDrivenDiscreteAggregation.pdf
5. CVPR2019_PAT_GumbelSubset_prior_art.pdf

## 四、深度分析文件索引

| 文件 | 作者 | 内容 |
|---|---|---|
| notes/paper_survey_20261014.md | Claude | 查新口径 + SOTA 协议表 + 勘误记录 |
| codex_review_20261014.md（内嵌） | Codex | 协议核对 + 拦截记录 |
| notes/codex_papers_20261014.md | Codex | 五篇可迁移性判定 + 调度建议 |
| notes/arch_survey_20261015.md | Claude | 五轮 16 方向全记录 |
| notes/codex_hexformer_stability_20261015.md | Codex | HexFormer 深读：双曲注意力堆叠稳定性的设计规则（已整合进 new_architecture_draft §2.2） |
| notes/codex_audit_deepread_20261016.md | Codex | 生死文献深读：u=√c·ρ 审计落地方法 + 5 个 reviewer 攻击回应（可执行） |
| notes/codex_depth_pilot_design_20261016.md | Codex | 3→6→9→12 深度 pilot 完整规格（9 配置 + 7 诊断 + 门禁） |
| notes/codex_survey_supplement_20261016.md | Codex | 四方向补充调研（三轴真空 + 82.4 复查确认） |
| notes/sota_table_deepdive_20261017.md | Claude | N-Caltech101 SOTA 表逐篇深度整理（用户截图 + 交叉核验，10 篇架构/协议/启示 + 赛道分组） |
| notes/g16_selector_design.md §5/§6 | Claude | RP1 原文复读 + 未完成定位 |
| notes/new_architecture_draft.md | Claude | 骨架倾向 + 决策树（调研弹药的使用方） |

## 五、调研结论对主线的三句话总结

1. **查新**：我们的定位 = 完成 RP1（学长未完成）+ 三轴真空（纯MLP×双曲×事件 双重确认无人占位）；
2. **弹药**：新架构 v1 的骨架/三件套/聚合/时间编码全有文献支撑或候选（等 10/17 数据 + Codex 的 HexFormer 深读收尾）；
3. **防御**：双曲审计文献倒逼的机制验证三件套（分布分离/受控对比/半径审计）必须进论文设计——这是今天调研最高价值的意外收获。

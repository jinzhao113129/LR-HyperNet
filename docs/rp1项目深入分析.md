# RP1 项目深入分析

> 分析时间：2026-10-06
> 分析人：豆包
> 项目路径：`event-camera-rps-main/rp1/`
> 说明：rp1 本地代码多为符号链接（指向 AutoDL 服务器），本文档基于目录结构、命名规范、配置文件和实验记录进行分析

## 1. RP1 定位

**RP1 = Research Project 1**，是基于 SECNet（Scalable Event Cloud Network）的迭代研究项目。

核心问题：**SECNet 只做局部 KNN 邻域，缺乏长距离依赖建模能力。**

RP1 的研究方向：在 SECNet 局部分组的基础上，逐步引入**远程邻域（Remote Neighborhood）**，探索跨组、全局的长距离依赖建模。

## 2. 数据配置

| 配置项 | 值 |
|---|---|
| 数据集 | N-Caltech101 |
| 点数/样本 | 2048 |
| 通道 | [t, x, y, polarity] |
| 训练样本 | 20901 |
| 测试样本 | 5226 |
| 类别数 | 101 |
| 常见基础设置 | neighbors=24, group=512, base_dim=16 |
| W384 | 候选 remote pool / local candidate window 的代表性设置 |
| G16 | 重点 Oracle remote 数量（实际 run 可能用 G4/G8/G16/G24） |

**注意**：我们当前用的是 8192 点/样本（SECNet 论文配置），rp1 用的是 2048 点（更轻量）。

## 3. 迭代历程（iter3 - iter9）

### 3.1 iter3：Oracle 统计与因果链

**目标**：用 oracle 方法（真实标签指导）探索远程邻域的上限。

- `iter3_oracle_stats.py`：统计 oracle 远程邻域的召回率、分布等
- `iter3_neighbor_composition.py`：分析邻居组成（局部 vs 远程）
- `iter3_global_graph_audit.py`：全局图审计
- `iter3_submit_causal_chain.py`：因果链提交

**核心思路**：先搞清楚"理想的远程邻域长什么样"，为后续设计提供指导。

### 3.2 iter4：稀疏远程邻域 + 远程检索器

**目标**：从 oracle 走向可学习的远程邻域选择。

- `iter4_sparse_remote`：稀疏远程邻域（只选少量远程点，降低开销）
- `iter4_remote_retriever`：远程检索器（学习如何检索远程邻域）
- `iter4_r2_distill`：R2 蒸馏（用 oracle 远程邻域蒸馏可学习模型）
- `iter4_gain0`：gain=0 的基线（无远程邻域增益）

**核心思路**：
1. 先做稀疏远程邻域（可扩展）
2. 再做可学习的远程检索器
3. 用 oracle 蒸馏提升性能

### 3.3 iter5：因果远程 + 全局增强 + Oracle G24

**目标**：引入因果性和全局信息，扩大 oracle 规模。

- `iter5_causal_remote`：因果远程邻域（时间因果约束的远程连接）
- `iter5_global_plus`：全局增强（全局信息注入）
- `iter5_oracle_g24_graph`：Oracle G24 图（扩大到 24 个远程点）
- `iter5_oracle_g24_dashboard`：Oracle G24 仪表盘（可视化分析）
- `iter5_cross_eval`：交叉评估

**核心思路**：
1. 远程邻域需要满足时间因果性（不能用未来信息）
2. 全局信息可以补充局部+远程的不足
3. Oracle 规模从 G16 扩大到 G24，探索上限

### 3.4 iter6：Gain Oracle 结构

**目标**：分析 oracle 远程邻域的增益结构。

- `iter6_gain_oracle_structure`：收集 gain oracle 结构（哪些远程点带来增益）
- `iter6_g10_selector_identity`：G10 选择器身份分析

**核心思路**：深入理解"为什么某些远程点有用"，为设计更好的选择器提供指导。

### 3.5 iter7：边界重排序

**目标**：优化远程邻域的边界点选择。

- `iter7_boundary_rerank`：边界重排序（对候选远程点重新排序，选最优）
- `iter7_collect_boundary_rerank`：收集边界重排序结果

**核心思路**：远程邻域的选择不是一次性的，可以通过重排序进一步优化。

### 3.6 iter8：扩展

**目标**：方法扩展性验证。

- `iter8_extension`：扩展实验（验证方法在更大规模/更多设置下的泛化）

### 3.7 iter9：索引化 B2

**目标**：索引化优化，提升推理效率。

- `iter9_indexed_b2`：索引化 B2（用索引加速远程邻域检索）
- `iter9_collect_indexed_b2`：收集索引化 B2 结果

**核心思路**：远程邻域检索需要高效索引，降低推理延迟。

## 4. RP1 核心技术路线总结

```
SECNet（局部 KNN 邻域）
    │
    ▼
iter3: Oracle 统计（探索远程邻域上限）
    │
    ▼
iter4: 稀疏远程 + 远程检索器 + 蒸馏（可学习远程邻域）
    │
    ▼
iter5: 因果远程 + 全局增强 + Oracle G24（引入因果性和全局信息）
    │
    ▼
iter6: Gain Oracle 结构分析（深入理解增益来源）
    │
    ▼
iter7: 边界重排序（优化选择质量）
    │
    ▼
iter8: 扩展（泛化验证）
    │
    ▼
iter9: 索引化 B2（效率优化）
```

## 5. RP1 与我们的双曲方向的关系

### 5.1 互补性

| 维度 | RP1 | 我们的双曲方向 |
|---|---|---|
| 建模目标 | 长距离依赖（空间/时间远程） | 层级拓扑（部件-整体） |
| 方法 | 远程邻域选择 + 检索 | 双曲空间嵌入 + 双曲聚合 |
| 空间 | 欧氏空间 | 双曲空间 |
| 开销 | 远程邻域检索开销 | 双曲运算开销 |
| 互补点 | 远程邻域建图后可用双曲 GCN | 双曲聚合可用于局部分组 |

### 5.2 结合机会

1. **远程邻域 + 双曲 GCN**：RP1 选出来的远程邻域构成图，在双曲空间做图卷积（类似 EHGCN，但邻域是 RP1 选的，不是 KNN）
2. **局部分组 + 双曲聚合**：我们当前的 HFECNet 方案，在 SECNet 局部分组后用双曲聚合
3. **层级池化 + 双曲**：RP1 的 group 从 512→256→128→64 本身就是层级池化，可以在双曲空间做层级合并
4. **因果远程 + 双曲时间**：RP1 的因果远程邻域天然有时间层级，可以用双曲时间建模

### 5.3 建议优先级

1. **先做局部分组 + 双曲聚合**（我们当前的 HFECNet）—— 最简单，直接验证双曲空间的有效性
2. **再做远程邻域 + 双曲 GCN** —— 结合 RP1 的远程邻域选择，在双曲空间做图卷积
3. **最后做层级池化 + 双曲** —— 更深入的双曲层级建模

## 6. 待确认项

- [ ] RP1 各 iter 的具体准确率（本地无实验结果，需从 AutoDL 获取）
- [ ] RP1 最终最佳配置和性能
- [ ] RP1 的远程邻域选择器具体实现（本地为符号链接）
- [ ] RP1 与 SECNet 基线的性能对比
- [ ] RP1 的代码是否可以在 HPC 上运行（需要从 AutoDL 迁移）

## 7. 参考文件

- `rp1/settings/RP1_DATA_SETTING.md` — 数据与实验配置
- `rp1/settings/RP1_ASSET_MAP.md` — 资产映射
- `rp1/settings/source_records/` — 实验记录
- `rp1/code/oracle/` — Oracle 方法（iter3-iter7）
- `rp1/code/selectors/` — 选择器（iter4-iter9）
- `rp1/code/submission/` — 提交脚本（iter3-iter7）
- `rp1/runs/` — 实验运行记录（iter3-iter9）

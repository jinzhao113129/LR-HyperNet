# 论文结论表模板（2026-10-15 定骨架，10/17 填数）

> Codex 上午钉死的统计口径落成表格骨架——数据一到直接填格。
> 每张表的"填什么/不填什么/脚注口径"都预先声明，防临场写错。

## Table 1：主结果表（论文第一张表）

**标题草稿**：Main results on N-Caltech101 (official full protocol, 101-class)

| Method | 壳 | 双曲 | 16点 | #Param | test seq acc (%) |
|---|---|---|---|---:|---:|
| SECNet (paper)¹ | SECNet | ✗ | ✗ | — | 82.4 |
| **Ours-hsel**² | SECNet fork | ✓(选点) | ✗ | {{填}} | **{{10/17 填}}** |
| **Ours-ha**³ | SECNet fork | ✓(聚合) | ✗ | +6 标量 | **{{10/17 填}}** |
| **Ours-G16**⁴ | LRHyperNet | ✓(选点) | ✓ | {{填}} | {{G16 出数填}} |
| **Ours-full**⁵ | **新架构** | ✓✓✓(三件套) | ✓ | {{填}} | {{阶段四五填}} |

**脚注（定死，不可临场改）**：
- ¹ SECNet 引用论文报告值（arXiv 2412.20803 Table 2）；其训练版代码未公开（GitHub 全部 4 提交均为简化发布版——我们核查记录在案），故无法同环境复跑，按学长口径直接对比。
- ² 双曲选点 = Poincaré 网格选中心点（非 16 点远程选择——两者区别已在正文 Table 3 分开报告）。
- ³ 双曲注意力 = 零投影 Lorentz 距离注意力，**6 个标量参数**（curvature×3 + temperature×3，无投影矩阵）。
- ⁴ G16 = RemoteSelector 十六点远程选择（学长核心创新），六处理准则对照见表 3。
- ⁵ 终架构 = 壳全新（非 SECNet 系）+ 双曲贯穿三件套。

## Table 2：两种几何先验的相对表现（Codex 口径：不是组件 vs 基线）

**标题草稿**：Relative performance of two hyperbolic priors (same protocol, same seed)

| 对比 | 差值 (pp) | 口径说明 |
|---|---:|---|
| Ours-hsel vs 82.4 | {{填}} | 选点几何先验到官方 SOTA 的 gap |
| Ours-ha vs 82.4 | {{填}} | 聚合几何先验到官方 SOTA 的 gap |
| Ours-hsel − Ours-ha | {{填}} | **两种几何先验的相对表现**（非"组件相对基线的增益"——无组件基线未跑，学长口径直接对比 82.4） |

**填数规则**：
- 全部用固定 eval_seed=42 推理的 last 口径（McNemar 在有 NPZ 时补 p 值列）；
- 单 seed 数字，**不得写成结论**——脚注声明"单训练 seed，跨 seed 复核见表 X"；
- seq acc（跨窗投票）为口径——与 82.4 同语义（官方 Stage-2 的输出即序列级）。

## Table 3：G16 十六点六处理准则对照（阶段二核心表）

**标题草稿**：Effect of remote-neighbor selection criterion (G=16, shared pool, full protocol)

| Selector 准则 | 协议排名 | test seq acc (%) | 选择分布画像 |
|---|---:|---:|---|
| random（下界锚点） | 1 | {{填}} | occupied 8.2/16 |
| spatial_uniform_fixed | 2 | {{填}} | occupied 13.2/16 |
| euc_far（我们原状） | 3 | {{填}} | occupied 6.9/16 |
| euc_near（原文 KNN 方向） | 4 | {{填}} | occupied 4.9/16 |
| **hyp_far（双曲——我们）** | 5 | **{{填}}** | occupied 7.0/16 |
| use_rp1（feature/novelty scorer） | 6 | {{填}} | — |

**口径声明**：
- 全部 LRHyperNet 骨干 + 共享候选池（池口径：近似均匀共享池——分块实现，Codex 注明非严格均匀）；
- **只报排序与相对差距**，不引用 RP1 原文绝对数（协议不同：原文 2048 点/20901 样本 vs 我们 10240/62703）；
- RP1 引用其 headroom 证据（Oracle 68.03 > Local 59.30 = "选点质量是 bottleneck"），不引用其 method（未完成）。

## Table 4：消融表（阶段三产出，预留骨架）

**标题草稿**：Ablation study（逐组件加/减）

| 配置 | seq acc | Δ | McNemar p |
|---|---:|---:|---:|
| 终架构（全组件） | {{}} | — | — |
| − 双曲选点 | {{}} | {{}} | {{}} |
| − 双曲聚合 | {{}} | {{}} | {{}} |
| − 双曲分类头 | {{}} | {{}} | {{}} |
| − 16 点远程 | {{}} | {{}} | {{}} |

## 通用填数纪律（Codex 钉的，写死）

1. 训练日志 acc 只作趋势，**入表数字 = 固定 eval_seed=42 推理**；
2. 单 seed 数字不作结论，跨 seed 复核后才能定"有效/无效"；
3. 配对差方向明确标注（谁减谁，正负号含义）；
4. 每个表的口径脚注**先于数据写好**（本文件的作用）。

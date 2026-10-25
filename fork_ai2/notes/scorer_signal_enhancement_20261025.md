# learned scorer 打分信号增强方案（补齐 RP1 8 特征）

> 2026-10-25 起草。核心判断：learned 结果若不理想，大概率不是"选择器"（Gumbel/ST_k）
> 的问题，而是"打分信号"太单薄——RP1 原文 profiling 列了 8 个特征，我们 scorer 只用了 2 个。

## 1. 现状：scorer 只喂了 2 个信号

`_learned_select` 里 scorer 输入 = `[candidate − centroid (C维); spatial_novelty (1维)]`。

## 2. 缺口（RP1 8 特征 vs 我们 2 个）

| RP1 profiling 特征 | 我们状态 | 缺口价值 |
|---|---|---|
| Feature cosine similarity | 部分（用欧氏差值；cosine_near 待补） | — |
| Spatial novelty | ✅ 已用 | — |
| **XY distance** | ❌ 缺 | ★★★ 区分"特征远程"vs"空间远程" |
| **Temporal gap** | ❌ 缺 | ★★ 时间跨度 |
| **Polarity agreement** | ❌ 缺 | ★ 极性一致性 |
| **Local density** | ❌ 缺（density_channel 是输入层，非选点层） | ★★★ 事件密度=运动=重要 |
| Contour anisotropy | ❌ 缺（covariance 已证伪） | — 撤 |
| Occupancy gain | ❌ 缺 | ★ 覆盖增益，需全局统计，成本高先放 |

## 3. 关键区分：这是"选点打分信号"，不是"输入层通道"

之前 density_channel / covariance_channel 是**给所有点加输入通道**（embed 层），
covariance 已证伪（加在输入层反而负贡献 50%）。但"事件密度"用在**选点打分**是
不同位置——它回答"该优先选哪个区域的点"，而不是"每个点长什么样"。所以密度信号
**在选点层可能有正价值，即使它在输入层证伪过**。

## 4. 增强方案：先加 2 个最高价值的信号（阶段 1）

scorer 输入从 `C+1` 维扩到 `C+3` 维：

1. **事件密度**（FocusMamba/EventFlash 核心）：候选点所在体素的事件数 log1p。
   直觉：事件密集 = 运动/变化 = 重要，该优先选。复用 density_channel 的体素框架，
   但作为 scorer 的一个标量输入（不是输入层通道）。

2. **XY 空间距离**（RP1 "特征相关但空间远程"）：候选点到中心点的 XY 欧氏距离。
   直觉：好的远程点应该是"特征相关（cosine 高）+ 空间远程（XY 远）"，XY 距离
   正是"空间远程"这个维度。

计算都用 all_xyz（[B,N,3]=t,x,y，selector 已收到），**不需要额外传参**。

## 5. 阶段 2（补齐 RP1 8 特征，需要小改传参）

3. **时间跨度**：|t_cand − t_center|（all_xyz 第 0 维，可用）。

4. **极性一致性**：p_cand 与 p_center 是否同号。**需要把 polarity 传给 selector**
   （当前只传 xytp[:, :, 0:3]，要改成 0:4 或单独传 p）。优先级最低。

## 6. 分阶段与验证

- **阶段 1**：加事件密度 + XY 距离（C+3 维），跑 learned 对照 rp1 59.64%。
- **阶段 2**：若阶段 1 有正反馈，再加时间跨度 + 极性（C+5 维）。

每个阶段都走 Codex 审（双人制）→ 3090 快测 → 看是否超 59.64%。

## 7. 诚实预期

RP1 原文的 Oracle headroom 是 68%（比我们的 59.64 高 8 个点），而这 8 个点大概率
就藏在"选点信号"里——原文 Oracle 是"知道哪些点好"（用了全部 8 特征做 membership）。
我们补齐信号、逼近 Oracle 的选择质量，是通向 68% 的正路。但单信号增益未验证，
阶段 1 先证伪"密度+XY 距离"这 2 个最有价值的。

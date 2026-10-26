# learned scorer v2 复审（learned_v2，GAT + 4 路信号）

审查日期：2026-10-26

结论：**无 P0，可放行 push + 跑 3090 快测。** 1 个 P1 建议 push 前修（center_xyz 兜底），其余为 P2。三个你拿不准的问题，逐一判定如下。

## 三个拿不准的问题（直接回答）

### Q1：z-score 是否损失绝对尺度？

会损失跨中心点的绝对尺度，但**对选点无害**。`_z` 是逐中心候选池（M 维）做的单调变换，完全保留每个中心池内的相对排序；而 LapSum 的 threshold 也是逐中心独立求的，所以选点只看池内相对序，绝对尺度用不上。"全局都很远"的中心点被归一化到同分布，反而正是想要的：scorer 学的是"对本中心相对更优的候选"，不学"这个中心是否全局远"。唯一小坑是 `std→0`（池内全等距）时 `1/(std+1e-5)` 梯度会尖，1e-5 只防 NaN 不防大梯度——P2，可把 eps 提到 1e-3。

### Q2：注意力 + MLP 是否双重计数？

不算。`att` 吃的是变换后的绝对值（`W_c·c`、`W_n·x`），`v2_mlp` 吃的是原始差值 + 4 信号（`x−c` 等），两者输入不同、表达互补：前者学"哪些特征方向交互"，后者学"显式差值和信号怎么预测"。相加是合理默认，网络可分别通过末层缩放/压制任一支路来学平衡。若想更干净可改 gate 或加权，但不是 bug，现在不必动。

### Q3：有放回采样是否该改？

不用改，P2。M=256 从 N=10240 有放回采样，期望唯一候选 ≈252.8，重复率约 1.3%（平均 ~3 个重复），对选点质量影响可忽略。排除 local 的 K=24 缓冲也绰绰有余（280 次采样期望 <1 个 local）。无放回采样要做 B·S 个 randperm、更贵，不值得。

## 逐设计点核对

### 1. GAT 交互：形式对，但"等价分解"措辞不精确

`e = v2_a_c(v2_Wc(centroid)).unsqueeze(-2) + v2_a_n(v2_Wn(cand))` 的广播正确（`[B,S,1,1] + [B,S,M,1] → [B,S,M,1]`），LeakyReLU 作用在标量上也对。但原始 GAT 的 `a^T[Wh_i‖Wh_j]` 用的是**同一个 W**（h_i、h_j 共享），这里用了独立的 `W_c/W_n`，所以是 GAT 的泛化（双线性注意力），不是严格等价分解。这是措辞问题（P2），不影响正确性，且独立 W 更灵活。

### 2. 四路信号：全部正确

- `cos_sim`：`normalize(dim=-1)` 后逐元素乘求和，[-1,1]，不 z-score 合理。
- `d_xy/d_t`：对 `center_xyz` 算 XY 距离/时间跨度，逐池 z-score，正确。
- `_voxel_density`：`no_grad`、fp32 累加后 `.to(xyz.dtype)`（AMP 安全），`key=q0·b²+q1·b+q2` 且 `q∈[0,bins)`，是 [0,4096) 的完美双射、无哈希冲突。逐 batch 逐 dim min-max 归一再分格，处理了 t 与 x,y 量纲不同的问题。正确。

### 3. 位移 MLP 支路

`mlp_in = cat([x−c, cos, d_xy, d_t, dens])` = C+4，与 `v2_mlp` 的 `Linear(channel+4, channel)` 对齐。`scores = att + v2_mlp(...)` 见 Q2，成立。

### 4. center_xyz 来源：生产路径正确，兜底是静默近似

grouper 层 `center_xyz = index_points(xytp[:,:,0:3], fps_idx)`（`:1462`），`fps_idx` 是 [B,S]、与 S 个中心一一对应，坐标正确。`center_xyz=None` 时退化为局部邻居坐标均值，这只是"旧调用/直接调 selector"的兼容兜底，生产不经过它。问题在于它是**静默近似**：若未来有人漏传 center_xyz，`d_xy/d_t` 会对错误中心计算且不报错。这正是过去 `**kwargs` 吞 `g16_treatment` 那类坑的同类风险。

**P1：`learned_v2` 且 `center_xyz is None` 时直接 `raise`，或至少 log 警告，不要静默退化。** 当前 grouper 路径正确、不阻断，但 push 前建议加这一行防御。

### 5. 候选池 / RNG / LapSum：与 v1 同款，已复核

- 候选池、`_g16_rng_inst`、M=256、排除 local、`cand.sort().values[..., :M]` 与 v1 逐行相同，`learned vs learned_v2` 只差 scorer+signal，归因干净。
- LapSum 的上一轮 P1（threshold detach）已修复：二分放进了 `torch.no_grad()`，只留对角梯度 `∂p_i/∂s_i`，到位。

## 三档结论

- **P0：无。**
- **P1：center_xyz=None 的静默退化**（learned_v2 时应 raise，push 前加一行防御即可）。
- **P2**：GAT "等价分解"措辞改"泛化双线性注意力"；z-score 的 eps 1e-5→1e-3 更稳；有放回采样 1.3% 重复可接受。

**放行判定：可 push + 跑 3090 快测；建议先补 P1 的 center_xyz 防御（一行 assert），其余 P2 不阻塞。**

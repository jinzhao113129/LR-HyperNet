# 给 Codex 的复审：learned scorer v2（learned_v2，GAT 式交互 + 4 路信号）

## 背景

learned_v2 只改「打分器网络 + 打分信号」，候选池 / M / RNG / LapSum 软 top-k 与
现有 `learned`（v1）**逐行相同**，目的是让 `learned` vs `learned_v2` 两条链能直接
归因到 scorer/signal，而不是又混进池子差异。

v1 实测瓶颈（ep15≈40%，固定选点 rp1_scorer 59.64%）：薄 MLP 只吃
`[candidate−centroid; novelty]`，既没有 center-candidate 显式交互，也缺 RP1 原文
8 特征里的关键信号（事件密度、XY 距离、时间跨度、cosine）。

## 改动文件

1. `scripts/local_edit/local_remote_grouper_v2.py`
   - `RemoteSelector.__init__`：新增 `g16_treatment='learned_v2'` 分支
     （v2_Wc / v2_Wn / v2_a_c / v2_a_n / v2_mlp）。
   - 新增 `_voxel_density`（静态方法）与 `_learned_v2_select`。
   - `RemoteSelector.forward`：签名新增 `center_xyz=None`；learned 分支改为
     `in ('learned','learned_v2')` 分派。
   - `LocalRemoteGrouper.forward`：learned 分支用 `fps_idx` 取准确中心坐标
     `center_xyz = index_points(xytp[:, :, 0:3], fps_idx)` 并传入 selector。
2. `scripts/local_edit/train_frequency_ncaltech.py`：`choices` 与 help 增加 `learned_v2`。

## 设计点（请重点审）

### 1. GAT 式交互项（GF-Core 2509.16639 Eq.1）

```python
e = self.v2_a_c(self.v2_Wc(centroid_feat)).unsqueeze(-2) + self.v2_a_n(self.v2_Wn(cand_feat))
att = F.leaky_relu(e, 0.2).squeeze(-1)
```

即 `e = LeakyReLU(a_c·W_c·c + a_n·W_n·x)`。请确认这就是
`LeakyReLU(aᵀ[Wh_i ‖ Wh_j])` 的等价分解（a 拆成两半；LeakyReLU 作用在标量上），
以及 `unsqueeze(-2)` 的广播是否在你预期的维上（[B,S,1,1] + [B,S,M,1]）。

### 2. 四路信号与 z-score

```python
cos_sim = normalize(c)·normalize(x)                         # [-1,1]
d_xy = _z(||x_xy − c_xy||)
d_t  = _z(|x_t − c_t|)
dens = _z(log1p(体素事件数))
```

- `_z` 是**按每个中心点的候选池（M 维）做 z-score**，理由是信号只提供相对排序、
  且 t 未归一化时量级会压过其余三路。请确认：z-score 的 mean/std 在计算图里
  （可微、会回传梯度）是否有害；以及候选池 z-score 会不会把"全局都很远"的中心点
  信号也归一化到同分布、损失绝对尺度信息（我认为对选点排序无影响，但请判断）。
- `cos_sim` 没有 z-score（本身有界）。
- 密度对**全部 N 点**算一遍再 gather（候选池每个中心点不同，逐中心算不现实）；
  代价 O(B·N) + `scatter_add_`（fp32 累加、no_grad）。请确认 AMP 下 fp16 输入
  的 dtype 处理（返回值 `.to(xyz.dtype)`）与 `bins**3=4096` 的哈希无冲突。

### 3. 位移 MLP 支路 + 注意力项相加

```python
mlp_in = cat([x−c, cos, d_xy, d_t, dens])   # C+4
scores = att + v2_mlp(mlp_in)
```

保留显式差值是想让网络能表达「cosine 最近」这类度量（cosine_near 是目前唯一
接近学长 66.7% 的线索）。请判断两路相加是否会双重计数、是否应改成 gate/加权。

### 4. center_xyz 的来源

`center_xyz = index_points(xytp[:, :, 0:3], fps_idx)`，与传给 selector 的
`all_xyz = xytp[:, :, 0:3]` 同一张量、同一坐标系（(t,x,y)，x/y 归一化到 [0,1]）。
`center_xyz=None` 时退化为局部邻居坐标均值（兼容旧调用）。请确认 fps_idx 与
S 维中心点一一对应、以及该兜底路径不会有静默错误。

### 5. 候选池采样（与 v1 相同，但请一并看）

`randint(0, N, (B,S,M+K))` **有放回**采样，然后排除 local。M=256 时重复候选会
让「同一候选点被重复计入软权重」，v1 也这样。请判断这对 v2 是否构成 P0/P1
（是否应改无放回采样或按唯一候选去重）。

## 本地自测（CPU，已通过）

1. **Selector 单测**：`w.shape=(2,8,16)`、`Σw−G` 最大误差 1.4e-6、候选不含
   local、索引合法、v2_Wc/Wn/a_c/a_n/mlp0 五个参数梯度非零且有限、
   密度对"16 点挤一格 vs 散点"给出 `2.83 > 1.61`、重复调用 cand_idx 不同
   （RNG 无重播种）、`center_xyz=None` 兜底路径 finite。
2. **Grouper 端到端**：`new_xytp=(2,8,4)`、`new_points=(2,8,6,36)`、
   train 反向后五个 v2 参数梯度非零、eval 模式 finite。

自测脚本：`D:/tmp/test_learned_v2.py`、`D:/tmp/test_grouper_learned_v2.py`
（未入仓）。

## 请给结论

按 P0 / P1 / P2 列，并明确「能否 push + 跑 3090 快测」。若不通过，请给出最小修法。

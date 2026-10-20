# Codex 任务：审查 HyperbolicRadialRectify 组件（曲率感知双曲径向校正，代码审查）

> ai2/Claude 起草。本次是**组件代码定稿审查**，不涉及训练提交（组件不进快测 6 链，
> 是特征层双曲化的候选）。按共识双人制：逐行查 + 独立复跑 + 找边界，明确「放行」
> 前不 push / 不跑训练。产出独立文件 `notes/codex_review_hyperbolic_radial_20261018.md`。

## 一、背景（研究问题）

替换路线的特征嵌入层（编码层之后、聚合层之前）还没有双曲组件。新近论文
**PointCHR: Point Cloud Analysis via Curvature-Aware Hyperbolic Rectification**
（arXiv:2607.24052，2026-07-27）给出一个方向：点云里高曲率/精细区域（物体
边缘、复杂结构）在欧氏空间被低曲率平滑区域淹没，应利用双曲流形边界「指数
体积增长」把高复杂度点投影到边界、给更大容量。

本次新增 `HyperbolicRadialRectify`（双曲径向校正，CHR）。**关键适配**：事件
点云（t,x,y,p）无几何曲率，故用可学习复杂度分数 κ 作代理（κ 大的点=离群/
精细，获得更大径向半径）。与已定稿的 HyperbolicGate（离质心近→门控大、抑制
离群）形成相反设计，需要并排对照。

## 二、改动清单 + md5

| 文件 | md5 | 改动 |
|---|---|---|
| `scripts/local_edit/FECNet.py` | `07c2c06f595d41d376172b80a2170146` | 新增 `HyperbolicRadialRectify` 类 + `use_hyperbolic_radial` 开关 + embed 后 forward 接线 |
| `scripts/local_edit/train_frequency_ncaltech.py` | `37e6112c6c8636b5f4315a267368e8bc` | argparse 加 `--hyperbolic_radial` + 实例化 + resume 校验键 |
| `scripts/local_edit/extract_preds.py` | `c9b4042371d64b14384bd84672775a61` | FLAG_KEYS / defaults / FECNet 构造三处接 `hyperbolic_radial` |
| `scripts/local_edit/test_fusion_gate.py` | `967e8041328c2ee8ec353fd444611797` | 新增 chr 三测试（forward/梯度非零/残差幅度） |
| `scripts/run_3090_setup_and_run.sh` | `314f1a7aed7f9941aa1c3b70dcb7b186` | `chain_flags` 加 `chr` 条目 |

## 三、组件设计

挂载：embed 后、第一个 stage 前，输入 `x` 为 `[B, N, feature_list[0]=64]`。

```python
def forward(self, x):
    h     = LayerNorm(in_proj(x))        # [B,N,64] -> [B,N,64] 瓶颈 d_h=64
    kappa = sigmoid(complexity(x))       # [B,N,1] 复杂度分数（曲率代理）
    alpha = softplus(alpha_raw)          # ≈0.1 恒正（避免 abs 零点梯度坑）
    r0    = softplus(r0_raw)             # ≈0.693 恒正
    norm  = h.norm(-1, keepdim=True).clamp_min(1e-6)
    r     = r0 * (1.0 + alpha * kappa)   # 目标半径，κ大→边界
    v     = r * h / norm                 # 调半径、保方向
    h_hyp = lorentz_exp_origin(v, c)     # [B,N,65]
    h_euc = lorentz_log_origin(h_hyp, c) # [B,N,64]
    return x + out_proj(h_euc)           # 残差
```

设计决策：α/r0 用 softplus 参数化（上一轮 HyperbolicGate 的 `alpha.abs()` 在
零点梯度为 0 卡死 alpha，本次直接避开）；残差注入使初始输出≈x+小扰动。

## 四、我侧测试输出（已跑，可独立复跑）

环境：本机 `secnet` conda（torch 2.14.1+cu126）。`test_fusion_gate.py` 11 项全过，
其中 chr 三项：

```
chr forward          out=(1,101) params=1934365 finite=True（base 1925849，+8516）
chr gradient nonzero alpha_raw/r0_raw/curvature/in_proj/complexity/out_proj 全非零
chr residual scale   残差 mean abs = 0.0736（1e-3~10 区间）
```

## 五、审查要点

### P0 正确性（生死，逐行查）

1. **形状链**：`[B,N,64] → in_proj [B,N,64] → 径向校正 → lorentz_exp [B,N,65]
   → lorentz_log [B,N,64] → out_proj [B,N,64] → 残差`，逐行核对。
2. **梯度全非零**：α/r0 用 softplus（非 abs），backward 后 alpha_raw/r0_raw/
   curvature/in_proj/complexity/out_proj 六者梯度是否都非零（我侧已验证，请独立复核）。
3. **数值稳定**：r=r0*(1+ακ) 落在 r0~r0*(1+α)，α≈0.1/r0≈0.693 → r∈(0.693,0.76)，
   lorentz_exp_origin 的 norm_max=10 保护是否覆盖；会不会 NaN/inf。
4. **挂载位置正确性**：embed 后 x 是 `[B,N,feature_list[0]]`，hyper_radial 输入
   channels=feature_list[0]=64 是否与 forward 实际维度一致（我侧已验证 forward 过）。
5. **推理工具接线**：extract_preds 的 FLAG_KEYS/defaults/构造三处是否齐（已同步）。

### P1 设计风险

6. **曲率代理的合理性**：事件点云无几何曲率，用 `sigmoid(Linear(x))` 学复杂度
   分数作代理，这是本组件最大的设计自由度。是否合理？有没有更贴合事件点云的
   「复杂度/离群度」度量（如局部事件密度、离质心距离）？
7. **与 HyperbolicEmbedding 重叠**：两者都是「特征层欧氏→双曲→回欧氏+残差」，
   本组件增量只在「曲率引导径向」。是否值得独立成组件，还是应作为
   HyperbolicEmbedding 的变体开关？
8. **挂载位置**：PointCHR 原文的 rectification 挂在哪？我们挂 embed 后是否是最
   优位置（可否挂每个 stage 的聚合后）？
9. **与 HyperbolicGate 的语义冲突**：hgate 抑制离群、chr 增强离群，两者是相反
   先验。同时开会不会抵消？（代码目前未互斥，同开=组合变体）

### P2 建议

10. 命名/注释一致性；`chr` 链是否需在 CHAIN_ORDER 预留。

## 六、输出要求

- 按 P0 / P1 / P2 三档列发现（文件:行号 + 问题 + 修法）；
- 独立复跑 `test_fusion_gate.py`（11 项），贴出结果；重点验证 chr 三测试的
  梯度非零与残差幅度；
- 若无 P0，明确写「可放行（定稿）」。

产出：`notes/codex_review_hyperbolic_radial_20261018.md`。

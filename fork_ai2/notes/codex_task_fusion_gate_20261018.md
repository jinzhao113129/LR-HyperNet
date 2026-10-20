# Codex 任务：审查融合层双候选组件（HyperbolicGate + FcaGate，代码审查）

> ai2/Claude 起草。本次是**组件代码定稿审查**，不涉及训练提交（两个组件都不在快测
> 6 链内，是替换路线③「融合层 gated→双曲加权」的备选，等快测筛完方向才登场）。
> 按共识双人制：逐行查 + 独立复跑 + 找边界，明确「放行」前我不 push / 不跑训练。
> 产出独立文件 `notes/codex_review_fusion_gate_20261018.md`，不改共享文件。

## 一、背景（研究问题）

替换路线（consensus_20261014 §0.5）第三步「融合层 gated→双曲加权」要把 SECNet
每 stage 末尾的 gated fusion 替换掉。SECNet 的 gated fusion 本质是 SENet 的 SE 通道
注意力（global avg pool = 最低频 DCT 分量 → FC → sigmoid → 逐通道门控）。本次准备
**两个替换候选**（用户 2026-10-18 拍板「都做」）：

1. **HyperbolicGate**（双曲距离门控）——门控信号由双曲距离学出，无直接论文背书，
   靠「双曲贯穿」理念；
2. **FcaGate**（频域通道门控）——按 FcaNet（Qin et al. ICCV 2021,
   arXiv:2012.11879）思路，把「仅 DC 分量」扩展为「多低频 DCT 分量」，有论文背书，
   且与 SECNet 的频域身份天然契合。

两者同层互斥（都替换 gated fusion 的 global 分支），快测里应各自单开一条链对比。

## 二、改动清单 + md5

| 文件 | md5 | 改动 |
|---|---|---|
| `scripts/local_edit/FECNet.py` | `b10f4716b1b943115fc64d6faaec218f` | 新增 `HyperbolicGate` + `FcaGate` 类；`use_hyperbolic_gate` / `use_fca_gate` 开关；forward 里 gated fusion 改 if/elif 分支；互斥检查 |
| `scripts/local_edit/train_frequency_ncaltech.py` | `b1740ad4bd2c10e24712b482e5cedc94` | argparse 加 `--hyperbolic_gate` / `--fca_gate` + 实例化接线 |
| `scripts/run_3090_setup_and_run.sh` | `7cd3620d1f90ee3aa7878af539ac4007` | `chain_flags` 加 `hgate` / `fca` 条目（`CHAIN_ORDER` 未动） |

## 三、组件设计

原 gated fusion（每 stage 末尾，`x` 为 [B,C,S]）：

```python
x_local = conv(x); x_local = bn_local(x_local)
x_global = avg_pool1d(x, 1) -> group_conv -> bn_global -> expand
x = x_local * sigmoid(x_global)                      # 逐通道门控
```

**HyperbolicGate.forward(x_local, x_src)**（门控粒度改为逐组）：

```python
h      = LayerNorm(in_proj(x_src.permute(0,2,1)))    # [B,S,C]->[B,S,64]
h_hyp  = lorentz_exp_origin(h, c)                     # [B,S,65]
centroid = lorentz_exp_origin(h.mean(dim=1,keepdim=True), c)
dist   = lorentz_distance(centroid.expand_as(h_hyp), h_hyp, c)  # [B,S]
gate   = sigmoid(alpha.abs() * (r0 - dist))           # [B,S], alpha=1,r0=2
return x_local * gate.unsqueeze(1)                    # [B,C,S]*[B,1,S] 逐组门控
```

**FcaGate.forward(x_local, x_src)**（保留逐通道门控）：

```python
freq = matmul(x_src, basis.t())          # [B,C,S] @ [S,k] -> [B,C,k], k=4
gate = sigmoid(FC(freq.reshape(B, C*k))) # [B,C]
return x_local * gate.unsqueeze(-1)      # [B,C,S] 逐通道门控
```

`basis` 是预计算的 DCT-II 基矩阵 [num_freq, seq_len]，L2 归一化。`num_freq=1`
退化为原 gated fusion（DC 分量 = 均值）。

**互斥**：`__init__` 里 `use_hyperbolic_gate and use_fca_gate` 显式抛 ValueError。

## 四、我侧测试输出（已跑，你可独立复跑）

环境：本机 `secnet` conda（torch 2.14.1+cu126）。冒烟脚本：

```python
import sys; sys.path.insert(0, '.')
import torch
from FECNet import FECNet, FcaGate
N = 2048; x = torch.randn(1, 4, N)
for name, kw in [('fca_gate', dict(use_fca_gate=True)),
                 ('hgate',   dict(use_hyperbolic_gate=True)),
                 ('base',    dict())]:
    m = FECNet(num_classes=101, **kw); m.eval()
    with torch.no_grad(): y = m(x)
    print(name, 'out=', tuple(y.shape), 'params=', sum(p.numel() for p in m.parameters()))
m4 = FECNet(num_classes=101, use_fca_gate=True, use_spike_time_enc=True, use_rp1_selector=True); m4.eval()
with torch.no_grad(): y4 = m4(x)
print('fca+spike+rp1 out=', tuple(y4.shape))
g = FcaGate(channels=132, seq_len=N, num_freq=1)
print('num_freq=1 basis[0] std=%.2e mean=%.4f (应 std~0, mean~%.4f)' % (g.basis[0].std(), g.basis[0].mean(), 1/N**0.5))
```

输出：

```
fca_gate out= (1, 101) params= 2162585
hgate    out= (1, 101) params= 1986594
base     out= (1, 101) params= 1925849
fca+spike+rp1 out= (1, 101)
num_freq=1 basis[0] std=3.73e-09 mean=0.0221 (应 std~0, mean~0.0221)
```

参数量：hgate +60,745（~+3%）；fca_gate +236,736（~+12%）。

## 五、审查要点（按优先级）

### P0 正确性（生死，逐行查）

1. **FcaGate 形状链**：`[B,C,S] @ basis.t()[S,k] -> [B,C,k] -> reshape(B,C*k) ->
   FC -> [B,C] -> unsqueeze(-1) -> [B,C,1] -> * [B,C,S]`，逐行核对。
2. **FcaGate DCT 基正确性**：DCT-II 公式 `cos(pi*i*(n+0.5)/S)`、L2 归一化、
   `num_freq=1` 是否严格退化为 DC 均值（我侧已验证 std≈0/mean≈1/√S，请独立复核）。
3. **FcaGate seq_len 匹配**：basis 在 `__init__` 用 `group_number[i]` 预计算，
   forward 时 x_src 的 S 维是否恒等于 `group_number[i]`？若数据点数/组数变化会
   matmul 维度不匹配，是否需要在 forward 加显式防御？
4. **FcaGate dtype**：basis 是 float32 buffer，若训练走 AMP 混合精度（fp16），
   matmul 会 dtype mismatch。当前训练脚本无 amp，但请确认此约束是否要写进注释。
5. **HyperbolicGate 形状链 + 广播语义**：`gate [B,1,S] * x_local [B,C,S]` 是否
   真的是「逐组、所有通道共享同一组门控」？
6. **曲率数值稳定**（hgate）：LayerNorm → lorentz_exp_origin（norm_max=10）→
   lorentz_distance（arccosh clamp），会不会 NaN/inf？
7. **替换开关 + 互斥**：if/elif/else 分支是否正确？`use_hyperbolic_gate and
   use_fca_gate` 是否真的抛错（我侧已验证，请复核）？group_conv/bn_global 参数
   保留不调用，推理重建时（extract_preds）state_dict 键是否稳定？
8. **曲率参数化**（hgate）：`log(expm1(0.4))` + `softplus+1e-4` 是否实际曲率≈0.4。

### P1 设计风险

9. **hgate 初始门控中性**：alpha=1、r0=2，初始 `gate=sigmoid(2-dist)`。dist 量级
   取决于 LayerNorm 后 h 范数，若 dist 常 >2 则初始 gate<0.5、特征被系统性抑制
   （替换式主门控不能初始塌缩）。r0=2 这个先验是否成立？
10. **hgate 门控粒度语义变化**（逐通道→逐组）是否合理、需不需要向学长报备。
11. **fca_gate 与 SECNet 已有 FFT 聚合是否冗余**：SECNet 聚合层已经做 FFT，融合层
    再做 DCT 门控，频域信息是否重复提取？这会影响「FcaNet 是否真的带来新信息」的判断。
12. **num_freq=4 是否够**：低频 4 分量能否覆盖通道注意力的主要信息？（可后续扫 k）
13. **与 `--hyperbolic_fusion` 共存**：hgate/fca_gate 是替换、fusion 是叠加，同时
    开会不会语义重叠/冲突（代码无互斥约束）。

### P2 建议

14. 命名/注释一致性；`hgate`/`fca` 链是否需要在 `CHAIN_ORDER` 预留。

## 六、输出要求

- 按 P0 / P1 / P2 三档列出发现，每条给文件:行号 + 问题 + 修法；
- 独立复跑冒烟测试（fca_gate 单开 + hgate 单开 + fca+spike+rp1 组合 + num_freq=1
  退化 + 互斥抛错），贴出复跑结果；
- 若无 P0，明确写「可放行（定稿）」，否则写「暂不放行 + 阻断项」。

产出：`notes/codex_review_fusion_gate_20261018.md`。

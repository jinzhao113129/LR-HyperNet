# Codex 任务：审查 PolarFeatureLayer 组件（方向/半径分离特征层，代码审查）

> ai2/Claude 起草。本次是**组件代码定稿审查**，不涉及训练提交（组件不进快测 6 链）。
> 按共识双人制：逐行查 + 独立复跑 + 找边界，明确「放行」前不 push / 不跑训练。
> 产出独立文件 `notes/codex_review_polar_feature_20261018.md`。

## 一、背景（研究问题）

调研发现 **[Rethinking the Fully Hyperbolic Vision Transformer in Polar
Coordinates](https://arxiv.org/abs/2609.32899)**（2026-09-26），其核心之一：
当前 Lorentz transformer 用 ambient 坐标，半径大时 Lorentz 内积数值不稳，很多
模型因此限制半径，放弃了双曲"边界大容量"区域。解法是 **Polar FC**——把嵌入
分解为方向（单位向量）与半径（标量）分别映射，半径可独立学习（而非由线性
映射 norm 决定）。

本次实现 `PolarFeatureLayer`，作为 **HyperbolicRadialRectify（chr，双曲流形内
径向校正）的欧氏对照**：两者同挂 embed 后，配对差可回答「双曲流形相对欧氏
方向/半径分离的增量」，呼应 Hyper² 对「双曲是否装饰」的审计。这是本组件与
chr 并存的**方法论价值**，不是重复。

## 二、改动清单 + md5

| 文件 | md5 | 改动 |
|---|---|---|
| `scripts/local_edit/FECNet.py` | `91a766b4b3ee9578e30c38ba5a3158a4` | 新增 `PolarFeatureLayer` 类 + `use_polar_feature` 开关 + embed 后 forward 接线 + chr/polar 互斥检查 |
| `scripts/local_edit/train_frequency_ncaltech.py` | `e10a1f619e70a1bab5b4c694cad94215` | argparse `--polar_feature` + 实例化 + resume 校验键 |
| `scripts/local_edit/extract_preds.py` | `4adb5e1754efaa3eca5b92ab3034a1cb` | FLAG_KEYS / defaults / 构造三处接 `polar_feature` |
| `scripts/local_edit/test_fusion_gate.py` | `ecd367cf8782d200993145ff164ffbe4` | 新增 polar 三测试 |
| `scripts/run_3090_setup_and_run.sh` | `720fc51e2da08270a4b1dbb8e556d6df` | `chain_flags` 加 `polar` 条目 |

## 三、组件设计

挂载：embed 后、第一个 stage 前，输入 `x` 为 `[B, N, feature_list[0]=64]`。

```python
def forward(self, x):
    h         = LayerNorm(dir_proj(x))   # [B,N,64] -> [B,N,64]
    direction = F.normalize(h, dim=-1)   # 方向（单位向量）
    radius    = softplus(rad_proj(x))    # [B,N,1] 半径（恒正、可学习）
    out       = out_proj(radius * direction)  # [B,N,64] 合回
    return x + out                       # 残差
```

与 chr 对照：chr 在 Lorentz 流形内 `exp→径向校正→log`；本组件在欧氏空间直接
方向/半径分离，不进双曲。两者互斥（`__init__` 显式 ValueError）。

## 四、我侧测试输出（已跑，可独立复跑）

环境：本机 `secnet` conda（torch 2.14.1+cu126）。`test_fusion_gate.py` 14 项全过，
其中 polar 三项：

```
polar forward              out=(1,101) params=1934362 finite=True（base 1925849，+8513）
polar gradient nonzero     dir_proj/rad_proj/out_proj 全非零
polar/chr mutual exclusion 抛 ValueError
残差 mean abs ≈ 0.0757（1e-3~10 区间）
```

## 五、审查要点

### P0 正确性（生死，逐行查）

1. **形状链**：`[B,N,64] → dir_proj [B,N,64] → normalize → radius [B,N,1] 广播
   × direction [B,N,64] → out_proj [B,N,64] → 残差`，逐行核对。
2. **梯度全非零**：radius=softplus(rad_proj(x))（非 abs），dir_proj/rad_proj/
   out_proj 三路梯度是否都非零（我侧已验证，请独立复核）。
3. **半径恒正**：softplus 是否保证 radius>0（我理解是），有无边界隐患。
4. **挂载位置 + 互斥**：embed 后 x 维度是否=feature_list[0]=64；chr/polar 互斥
   是否真的抛错（我侧已验证）。
5. **推理工具接线**：extract_preds 三处是否齐（已同步）。

### P1 设计风险

6. **与 chr 的对照价值**：本组件定位为 chr 的"欧氏对照"，用于回答"双曲 vs 欧氏
   增量"。这个定位是否成立？还是它与 chr 过于相似、应合并为一个 flag？
7. **方向/半径分离本身的意义**：在欧氏空间做方向/半径分离，半径独立学，是否
   有实际增益，还是只是等价于一个 Linear 的分解（dir×rad 可被 Linear 吸收）？
8. **与 HyperbolicEmbedding 的关系**：三者（chr/polar/hyper_embed）都在特征层，
   是否构成过度的组件堆叠？

### P2 建议

9. 命名/注释一致性；`polar` 链是否需在 CHAIN_ORDER 预留。

## 六、输出要求

- 按 P0 / P1 / P2 三档列发现（文件:行号 + 问题 + 修法）；
- 独立复跑 `test_fusion_gate.py`（14 项），贴结果；
- 若无 P0，明确写「可放行（定稿）」。

产出：`notes/codex_review_polar_feature_20261018.md`。

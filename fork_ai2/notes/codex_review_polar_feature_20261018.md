# Codex 独立审查：PolarFeatureLayer（2026-10-18）

## 结论

**代码可放行（定稿），无 P0。**

PolarFeatureLayer 的形状链、梯度、半径正值、挂载位置、chr/polar 互斥、
trainer/extract_preds/run 脚本接线均正确，14 项回归全部通过。

下面 3 个 P1 是研究设计风险，不阻断代码定稿，但在训练前必须写进实验
口径并修改对照设计。

## P0

无。

独立逐项核验结果：

- [FECNet.py:827](D:/UserData/Desktop/学习/进组/secnet-hyperbolic/fork_ai2/scripts/local_edit/FECNet.py:827)
  的形状链正确：
  `[B,N,64] -> [B,N,64] -> [B,N,64] -> [B,N,1]` 广播后
  `out_proj -> [B,N,64] -> +x`。
- `dir_proj` / `rad_proj` / `out_proj` 梯度全非零。
- radius 由 `softplus` 生成，实测全正，不存在 `abs(0)` 的梯度死点。
- 挂载位置是 `embed_dim` 后、第一个 stage 前，输入维度确实为
  `feature_list[0]=64`。
- [FECNet.py:984](D:/UserData/Desktop/学习/进组/secnet-hyperbolic/fork_ai2/scripts/local_edit/FECNet.py:984)
  的 chr/polar 互斥确实抛 `ValueError`。
- `train_frequency_ncaltech.py`、`extract_preds.py`、
  `run_3090_setup_and_run.sh` 三处接线齐全。
- `extract_preds.build_model` 后：
  - `hyperbolic_radial` strict load 通过，164 keys；
  - `polar_feature` strict load 通过，161 keys。

## P1

### P1-1：Polar 目前不是 chr 的严格欧氏对照

- 文件：
  [FECNet.py:739](D:/UserData/Desktop/学习/进组/secnet-hyperbolic/fork_ai2/scripts/local_edit/FECNet.py:739)
  和
  [FECNet.py:800](D:/UserData/Desktop/学习/进组/secnet-hyperbolic/fork_ai2/scripts/local_edit/FECNet.py:800)
- chr 的半径是复杂度驱动的有界公式：
  `r0 * (1 + alpha * sigmoid(complexity(x)))`；
  Polar 的半径是独立的无界 `softplus(rad_proj(x))`。
- chr 有复杂度分支、曲率、exp/log；Polar 没有这些对应结构。
- 因此当前配对差不是“双曲 vs 欧氏”，而是
  “带复杂度门控的双曲径向校正 vs 独立可学习半径的欧氏分解”。
- 修法（设计层）：
  1. 让 Polar 完全镜像 chr，只删除 Lorentz `exp/log` 和曲率，保留
     相同的 `in_proj/LayerNorm/complexity/r0/alpha/h/norm/out_proj`；
  2. 或者把两者合成一个 `radial_mode=euclidean|hyperbolic` 的单一代码
     路径，避免结构漂移。

### P1-2：方向/半径分离的表达增益尚未证明

- Polar 的 `radius * direction` 并不是一个单层 Linear 的等价写法，
  因为它引入逐点标量半径与单位方向耦合。
- 但它也没有自动优于一个同参数量的残差 MLP，或者
  `dir_proj -> LayerNorm -> out_proj` 去掉独立半径的版本。
- 在训练前应预注册一个 matched 欧氏 MLP baseline。否则论文里不能写
  “方向/半径分离带来增益”，只能写“相对于 matched MLP 的差值”。
- 当前代码本身没有错误，这是归因缺口。

### P1-3：chr/polar/hyper_embed 存在特征层过度堆叠风险

- chr/polar 挂在点级 64 维 embed 后，且互斥；
- HyperbolicEmbedding 挂在最终 1080 维 backbone 特征后，不在同一位置；
- 三者可以组合成“polar/chr + hyper_embed”两条特征层链。
- 如果最终架构同时启用，单组件归因会非常难拆。
- 建议在替换路线文档里明确：
  - chr/polar 是“点级表示几何”消融；
  - hyper_embed 是“分类前全局表示几何”消融；
  - 组合只在两者单独出数后才进入候选。

## P2

1. `F.normalize(h, dim=-1)` 默认 eps 为 `1e-12`。LayerNorm 后理论上仍可能
   得到全零向量，建议显式 `eps=1e-6` 或增加 norm 下限。
2. `test_fusion_gate.py` 现在包含 chr/polar，但文件名和顶部 docstring
   仍只写 fusion gate；建议改名或更新说明。
3. `run_3090_setup_and_run.sh` 的 CHAIN_ORDER 注释目前只说明 hgate/fca
   未排入，应同时注明 chr/polar 也未排入。
4. 回归测试没有覆盖“polar/chr checkpoint 经 extract_preds 严格重建”。
   本次独立复核通过，但应把该断言固化进测试。
5. Polar 半径没有上限或分布监控；训练时可记录 radius 的均值/最大/最小，
   用于检查是否出现极端尺度。

## 独立复跑

环境：本地 torch 2.12.0+cpu。五文件 md5 与任务书一致：

```text
FECNet.py                  91a766b4b3ee9578e30c38ba5a3158a4
train_frequency_ncaltech.py e10a1f619e70a1bab5b4c694cad94215
extract_preds.py           4adb5e1754efaa3eca5b92ab3034a1cb
test_fusion_gate.py        ecd367cf8782d200993145ff164ffbe4
run_3090_setup_and_run.sh  720fc51e2da08270a4b1dbb8e556d6df
```

`test_fusion_gate.py` 14 项：

```text
forward                         PASS
combo fca+spike+rp1             PASS
hgate initial gate neutral      PASS
hgate gradient nonzero          PASS
fca num_freq=1 degenerate       PASS
mutual exclusion                PASS
fca seq_len guard               PASS
fca amp dtype                   PASS（本地无 CUDA，skip）
chr forward                     PASS
chr gradient nonzero            PASS
chr residual scale              PASS
polar forward                   PASS
polar gradient nonzero          PASS
polar/chr mutual exclusion      PASS
```

额外核验：

```text
Polar residual mean/max: 0.0777 / 0.4514
Polar radius min/max:    0.1621 / 1.6260，全正
Polar gradients:
  dir_proj 4716.87
  rad_proj  470.24
  out_proj 9935.64

extract_preds strict load:
  hyperbolic_radial 164 keys OK
  polar_feature     161 keys OK
```

## 放行判定

**可放行（定稿）。**

代码正确性和链路闭环通过；P1-1 到 P1-3 是实验设计与归因问题，必须在
开始 polar/chr 训练前修订对照协议，但不阻止本组件代码入库。

---

## 复审：chr Lorentz 内校正 + polar 镜像（2026-10-18）

新 md5：

```text
FECNet.py          8de6e3cfa2fb8e2c60ad1f778e978014
test_fusion_gate.py 162e87f90793821b5b55d6b1fdbb6200
extract_preds.py   4adb5e1754efaa3eca5b92ab3034a1cb（未变）
train_frequency_ncaltech.py e10a1f619e70a1bab5b4c694cad94215（未变）
run_3090_setup_and_run.sh  720fc51e2da08270a4b1dbb8e556d6df（未变）
```

### chr 公式复核

- 流形约束解是代数正确的：
  `x0' = x0 * (1 + alpha*kappa)`，
  `x_rest' = x_rest * sqrt((x0'^2 - 1/c) / (x0^2 - 1/c))`。
- alpha 恒正且当前初始 0.1，`1+alpha*kappa` 始终不小于 1，因此
  `x0' >= x0 > sqrt(1/c)`，`clamp_min(1e-6)` 在正常训练点不会进入异常
  分支。
- 独立数值：
  - `h_euc - h` 最大 0.0465，mean 0.00800，exp/log 不再恒等；
  - 三个 stage 训练入口和测试的 forward 均 finite；
  - alpha_raw、curvature、in_proj、complexity、out_proj 梯度全部非零。

### 残留数值风险

1. 当前流形残差
   `|x0'^2 - ||x_rest'||^2 - 1/c|`
   在典型随机输入上最大约 0.0065。原因是分母写成
   `x0^2 - 1/c`，在大 x0 下是两个大数相减。
2. 更稳写法应直接用空间分量的范数：
   `den = x_rest.norm(dim=-1, keepdim=True).pow(2).clamp_min(1e-6)`。
3. 建议测试中增加流形残差断言，例如 `< 1e-3`；当前
   `test_chr_not_identity` 只证明不是恒等，没测流形误差。
4. alpha 无上界，训练后期理论上可把半径推到很大；当前初始和冒烟安全，
   但正式训练应监控 alpha 与 h_euc 范数。

### polar 与 chr 的镜像程度

- 参数结构已真正逐项对齐：polar/chr 都含相同的
  `in_proj/complexity/out_proj/layer_norm/alpha_raw`，alpha_raw 初始值
  完全相同；唯一参数差异是 chr 多一个 curvature。
- 但计算不是“把 chr 的 exp/log 删掉”的精确代数镜像：
  - polar 对全部 h 维同时乘 `1+alpha*kappa`；
  - chr 对 Lorentz 的 x0 乘同一个 factor，再独立缩放 x_rest 以恢复
    流形约束。
  - 在曲率 0.4、h 范数较大的情况下，二者不等价，而这正是期望的
    “欧氏径向缩放 vs 双曲径向校正”差异。
- 结论：可作为 matched 欧氏对照，但文档措辞应从“唯一变量=是否进双曲”
  收紧为“参数结构与复杂度因子逐项镜像；差异同时包含欧氏/双曲几何和
  流形约束实现”。

### 复跑结果

```text
15 项回归全部通过。

chr:
  h_euc-h max=0.0465
  manifold error max=0.0065
  gradients: alpha_raw/curvature/in_proj/complexity/out_proj 全非零

polar:
  forward finite
  gradients alpha_raw/in_proj/complexity/out_proj 全非零

extract_preds strict load:
  hyperbolic_radial 163 keys OK
  polar_feature     162 keys OK
```

### 最终结论

**可放行。** chr 双曲分支已实际生效，polar 参数结构已对齐；上述数值
写法和口径措辞是训练前应完成的两个 P2，不构成代码阻断。

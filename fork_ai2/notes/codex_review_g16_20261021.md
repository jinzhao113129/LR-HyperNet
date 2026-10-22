# Codex 独立审查：G16 FECNet 训练入口接线（2026-10-21）

## 结论

**暂不放行。**

训练模型 forward、参数传递、同名 flag 语义、输出维度和互斥均正确，
但存在两个阻断项：

1. `extract_preds.py` 没有 `remote_g16`，训练后的 G16 checkpoint 无法正确
   重建和评测；
2. batch16 + grad_accum2 在 24GB 3090 上的真实显存尚未验证，简单外推的
   最坏值可能超过 24GB。

## P0

### P0-1：推理工具没有接入 remote_g16

- 文件：
  [extract_preds.py:50](D:/UserData/Desktop/学习/进组/secnet-hyperbolic/fork_ai2/scripts/local_edit/extract_preds.py:50)
- `FLAG_KEYS` 没有 `"remote_g16"`，defaults 和
  [FECNet(...) 构造](D:/UserData/Desktop/学习/进组/secnet-hyperbolic/fork_ai2/scripts/local_edit/extract_preds.py:117)
  也没有传 `use_remote_g16`。
- 后果：G16 checkpoint 含 `local_grouper_list.*.remote_selector/hyper_agg`
  等额外键；extract 的兼容逻辑会把这些键过滤掉，并用基础 LocalGrouper
  推理，结果静默错误。
- 修法：
  1. `FLAG_KEYS` 加 `"remote_g16"`；
  2. defaults 加 `False`；
  3. `FECNet(...)` 传 `use_remote_g16=saved_args["remote_g16"]`；
  4. 增加 G16 checkpoint strict-load 测试。

### P0-2：batch16 + grad_accum2 的显存结论未被真实验证

- 任务书只有 batch2 的 forward/backward 数据：
  - forward 1.13GB；
  - train+backward 2.16GB。
- 简单按 batch 线性外推：
  - batch16 backward 约 17.3GB；
  - `grad_accum=2` 会保留第一微批的图，最坏约两倍激活，
    约 34.6GB 的上界估计。
- 实际会有缓存复用，可能低于上界，但当前数据不足以证明 24GB 安全。
- 修法：在 3090 上跑一次真实
  `batch=16 + grad_accum=2 + N=10240 + G16 单步 forward/backward`，
  用 `nvidia-smi` 或 `torch.cuda.max_memory_allocated()` 记录峰值；
  只有峰值加安全余量小于 24GB 才提交训练。

## P1

### P1-1：FECNet 当前 import 的是旧版 local_remote_grouper.py

- FECNet [line 12](D:/UserData/Desktop/学习/进组/secnet-hyperbolic/fork_ai2/scripts/local_edit/FECNet.py:12)
  import `local_remote_grouper.LocalRemoteGrouper`。
- local_edit 目录同时存在：
  - `local_remote_grouper.py`，md5 `b278d3f80fc9443995ef67c64bcbf04f`；
  - `local_remote_grouper_v2.py`，md5 `f75f1351b79055d2b29d5d836e02111c`。
- v2 才是带 G16 六处理分派、RNG 隔离、sink 链路和 Codex 修复的版本；
  run_3090 setup 也复制的是旧 `local_remote_grouper.py`。
- 当前 topk+lorentz 单入口恰好能跑，但后续六处理/随机/空间均匀入口会拿到
  旧实现。
- 修法：确定哪一份是定稿，然后用 v2 覆盖正式 `local_remote_grouper.py`，
  或把 import/setup 改为显式 v2；必须记录 md5，不能同时留两套未标版本。

### P1-2：首条 G16 链先跑 selector_mode=topk 的归因解释不足

- topk 是“特征距离最远”路径，是原 LocalRemoteGrouper 默认行为；
- 如果目标是验证“16 点远程选择”本身的增益，建议至少同时预注册
  `random` 或 `spatial_uniform` 作为 cheap-selector 基线；
- 否则只能说明“双曲远点聚合 vs 无远程分支”，不能把增益都归给选点器。

### P1-3：remote_g16 与 hyperbolic_fusion 可同时开启

- 两者都会在每 stage 使用双曲远程/并行聚合，单组件归因会混。
- 当前代码没有互斥限制；若后续组合，必须作为“G16+融合”组合变体
  单独登记，不能写进单组件消融表。

## P2

1. `use_remote_g16=True` 时，`use_scale_norm_fps` 被采样层分支忽略；
   当前不传没关系，但建议加警告或互斥，防止误配置。
2. run_3090 的 CHAIN_ORDER 注释只说明 hgate/fca 未排入，应同时注明
   chr/polar/hclf/g16 也未排入。
3. 测试文件建议加入：
   - 每 stage `out_points.shape[-1] == 2*feature_list[i+1]+4`；
   - remote 索引与 local 索引零重叠；
   - G16 extract_preds strict load。
4. `use_remote_g16` 是 CLI 名，类内是 `remote_g`；help 文本已经说明，
   可在 CHAIN 注释再写一句避免把“G16 采样层”和“16 个远程点”混用。

## 独立复跑

md5 与任务书一致：

```text
FECNet.py                  31b0b8c3f5709d3f9d3600a9d8787ba3
train_frequency_ncaltech.py 0a735344a3829cdf6a007bc07204548c
run_3090_setup_and_run.sh  9daf622abd2fb35ecc83d70b517449a5
```

完整模型：

```text
use_remote_g16=True:
  params = 2,114,860
  stage types = [LocalRemoteGrouper] x 3
  remote_g=16, kneighbors=24
  use_hyperbolic=True, hyperbolic_type=lorentz
  selector_mode=topk
  forward out=(2,101), finite=True
```

逐 stage 形状：

```text
stage 0:
  in_points  (2,10240,64)
  out_xytp   (2,2048,4)
  out_points (2,2048,24,132)   # 64*2+4

stage 1:
  in_points  (2,2048,132)
  out_xytp   (2,1024,4)
  out_points (2,1024,24,268)   # 132*2+4

stage 2:
  in_points  (2,1024,268)
  out_xytp   (2,512,4)
  out_points (2,512,24,540)    # 268*2+4
```

互斥：

```text
remote_g16 + rp1_selector        -> ValueError
remote_g16 + hyperbolic_selector -> ValueError
```

小规模直接检查：

```text
B=2, N=256, S=32, C=8, K=16, G=16
out=(2,32,4), (2,32,16,20)
remote shape=(2,32,16), bounds=[4,255]
remote/local overlap count=0/64
```

`py_compile`：

```text
FECNet.py + train_frequency_ncaltech.py 均通过
```

显存：本机没有可用 CUDA，无法独立复跑 batch16 GPU 峰值，故不把“约 17GB”
作为可放行依据。

## 放行判定

**暂不放行。** P0-1 是确定性下游错误；P0-2 是训练提交前的显存 gate。
P1-1 的 v2/旧文件分叉也应在训练入口定稿前闭合。

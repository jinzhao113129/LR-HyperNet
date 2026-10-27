# 给 HPC Claude 的复审：Global Feature-KNN 组件化（RP1 G00_global 复现）

> 复审方式（2026-10-27 起）：在 HPC 上用 Claude 复查，不再用 Codex。
> HPC 工作目录 `/home/jnguan/JiangYouxin/ZhaoJin`；审前先同步改动文件。

## 背景

学长要求「把 RP1 里最好的方法当作一个组件来做」。经核对，RP1 表里最高 68.03% 是
Oracle（上界，不可部署），可复现的最好成品是 **G00_global / Global Feature-KNN 66.70%**。
它在该代码里的配置（`rp1_scripts/iter2_submit.py:26-42`）：

```
candidate_mode=global, num_point=2048, base_dim=16, group=512, neighbors=24,
batch=32, epochs=150, adamw, lr=1e-3, wd=1e-4, label_smoothing=0.2, seed=0
```

关键事实：**该方法就是全量特征 KNN（k=24），与我们现有的基础 LocalGrouper 完全等价**
（`modules.py` 里 `square_distance(new_points, points)` 遍历全部 N 点 + argsort 取前 24，
无时间窗、无 local/remote 拆分）。所以「组件化」不新增算法，只是：① 显式命名，能作为
组件跑、与其它组件组合；② 补上复现 66.70 所需的模型宽度配置。

## 改动

1. `scripts/local_edit/FECNet.py`
   - `__init__` 新增：`g16_treatment == 'global_knn'` 时置 `use_remote_g16=False`
     （于是 grouper 走基础 `LocalGrouper`），并与 rp1/hsel selector 互斥报错；
   - 记录 `self.use_global_knn`。
   - `feature_list` / `group_number` 原本就是构造参数，未改逻辑。
2. `scripts/local_edit/train_frequency_ncaltech.py`
   - `--g16_treatment` choices 增加 `global_knn`；
   - 新增 `--base_dim`、`--fps_groups`（默认 0 = 保持官方 [64,132,268,540]/[2048,1024,512,256]）；
     >0 时按 RP1 公式生成 `feature_list=[d, 2d+4, ...]`（4 项）、`group_number=[G, G/2, ...]`，
     并透传给 FECNet。

**为什么用 `--remote_g16 --g16_treatment global_knn` 这种写法**：保持与其它 G16
treatment 同一入口、同一日志/恢复路径，便于在组件矩阵里作为一行对照；FECNet 内部
自动关掉 remote 分支，实际结构 = 基线。

## 本地自测（CPU，已通过）

- `global_knn` + `use_remote_g16=True` → 3 个 grouper 全是 `LocalGrouper`，
  `m.use_global_knn=True`；
- RP1 宽度 `[16,36,76,156]` / `[512,256,128,64]` 可构建，`(1,4,2048)` 前向得 `(1,101)`、
  全 finite；
- 不传新参数时：`feature_list=[64,132,268,540]`、`group_number=[2048,1024,512,256]`、
  `use_global_knn=False`，旧行为不变；
- `global_knn` 与 `rp1_selector`/`hyperbolic_selector` 同开均报 ValueError。
- 自测脚本：`D:/tmp/test_global_knn.py`（未入仓）。

## 请重点审

1. **覆盖 `use_remote_g16=False` 是否安全**：构造参数被局部覆盖后，除 grouper 选择外
   还有没有别的地方依赖它（当前 grep 只有 `:1002` 互斥检查和 `:1083` grouper 分支）。
   特别是 resume：`--remote_g16` 会写进 checkpoint args，恢复时 args 一致即可，
   但请确认 `extract_preds.py` 用 saved args 重建时也会走同一路由（它传
   `g16_treatment=saved_args.get("g16_treatment","")`）。
2. **标签平滑**：我们现有 `cal_loss(smoothing=True)` 的 eps 已是 0.2，与 RP1 一致，
   所以没加 `--label_smoothing`。请确认这不会与 RP1 的 `label_smoothing=0.2` 有实现差异
   （我们是 one-hot*(1-eps)+eps/(n-1)，与 PyTorch CrossEntropyLoss(label_smoothing) 语义相同）。
3. **宽度配置公式**：`stages=3` → feature_list 4 项、group_number `[G,G/2,G/4,G/8]`；
   RP1 代码是 `for i in range(stages+1)`，若其 stages 与我们不同则公式项数会差一项。
   请对 `server_rp1_reference/models/FECNet.py:112-121` 核一遍。
4. `--base_dim/--fps_groups` 只影响模型宽度，不动数据加载；但复现还需
   `--num_point 2048`（RP1 数据是 2048 events/sample），这取决于服务端 `provider_data`
   是否按 num_point 采样。若它整段加载 H5，则必须用 RP1 的 2048 H5，不能靠改参数。

## 请给结论

P0/P1/P2 分级，明确能否 push。

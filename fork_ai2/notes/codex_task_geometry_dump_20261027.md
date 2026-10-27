# 给 HPC Claude 的复审：selector geometry dump（找点 + 几何分析第一步）

> 复审方式（2026-10-27 起）：在 HPC 上用 Claude 复查，不再用 Codex。
> HPC 工作目录 `/home/jnguan/JiangYouxin/ZhaoJin`；审前先同步改动文件。

## 背景

学长路线：先把「每个中心 + 最合适的 16 个点」找出来，再做 PCA / 欧氏与双曲空间
分析，最后才谈学习。本脚本只做第一步（dump + 快速统计），**不参与训练、不改任何
训练路径**。

## 脚本

`scripts/local_edit/selector_geometry_dump.py`

- 加载指定 run 的 `best_checkpoint.pth`（复用 `extract_preds.build_model` 与 args）；
- 在指定 stage 的 grouper 上挂 forward hook，抓 `(xytp, points)`；
- 按 grouper 的同一口径重算 FPS 中心（`fps_parameter` 缩放后取 FPS）；
- 「最合适的 16 个」= 特征距离 `topk(24+16)` 里排在局部 24 之后的 16 个
  （= 排除局部近邻后的特征最近 16，即 euc_near，等价 RP1 oracle 的选点规则）；
- 随机 16 个做对照（拒绝落在 top40 内的，碰撞率 <0.5%）；
- 每个样本抽 `--centers_per_sample` 个中心，保存 center/local/remote/random 的
  特征与坐标（float16）到 npz；
- 打印三组的 cos/L2 均值方差，以及 `remote-center` / `random-center` 的前 5 主成分
  解释方差。

## 请重点审

1. **FPS 复现口径**：hook 抓到的是 `xytp` 原始（未缩放），脚本里按
   `grouper.fps_parameter * xytp` 缩放后做 FPS——是否与 `LocalGrouper.forward`
   完全一致（含 `use_scale_norm_fps` 分支）？
2. **「最合适 16 个」的定义**：`topk(40)` 的 `[24:]` 是否严格等价于
   `euc_near`（排除 24 个局部后的最近 16）？topk 与 argsort 的 tie-breaking 差异
   是否会引入系统性偏差（本数据无精确相等距离）？
3. **随机对照**：`in_top` 命中时替换成 `remote_idx`，是否引入偏差（被替换的点
   本身偏向"特征较远"）？是否应改为重采样直到不在 top40？
4. **hook 生命周期**：`handle.remove()` 在异常路径是否可能漏掉（当前无 try/finally）。
5. 显存：`square_distance(centroid, points)` 在 B=4/S=2048/N=10240/C=64 下约
   335MB，加上 topk，是否有 OOM 风险；B 默认 4 是否合适。

## 自测

`py_compile` 通过；实际运行需在 gpu-server-3（依赖 `provider_data` 与数据）。

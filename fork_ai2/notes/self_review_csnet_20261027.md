# 自审记录：csnet + global_knn + geometry dump（2026-10-27）

> HPC Claude 复审因 headless 鉴权 401 暂时跑不通，用户指示先自审。以下是逐行审查结果。

## 结论

**无 P0**。发现 2 个 P1（已修）、若干 P2（记录在案）。修复后全部自测通过。

## P1（已修）

### P1-1 csnet 推理未走硬 top-k

- 位置：`local_remote_grouper_v2.py::_csnet_select`
- 问题：注释/设计写「训练软、推理硬」，但实现无条件用 EOT 软权重；grouper 的
  learned 分支也不区分 train/eval，导致推理仍在用软加权，与 CS-Net 论文口径不符。
- 修复：`self.training` 为 False 时改为 `scores.topk(G)` 的 one-hot 权重。
- 验证：eval 下权重严格 0/1、每行恰好 G 个 1。

### P1-2 模型宽度参数无法推理/恢复

- 位置：`extract_preds.py::build_model`、`train_frequency_ncaltech.py` resume 校验键
- 问题：`--base_dim/--fps_groups` 只传给了训练时的 FECNet；`build_model` 不读这两个
  参数，窄模型（如 16/512）checkpoint 重建时仍是 64/2048，strict load 必然失败。
  resume 键列表也没有它们，换宽度恢复不会提前报错。
- 修复：`build_model` 按与训练相同的公式从 `saved_args` 重建
  `feature_list/group_number`（缺键=默认宽度）；resume 校验键加入 `base_dim/fps_groups`。

## P2（记录，暂不改）

1. `_sample_candidates` 的 `clamp(max=N-1)` 兜底：小 N 测试场景可能产生重复/局部索引；
   生产 N=10240/K=24/M=256 时被掩码期望数 <1，不会触发。
2. csnet 的偏移注意力作用在**相对特征** `candidate−centroid` 上，论文用绝对特征；
   属于 per-center 适配的偏离，已写入复审单。
3. 未复现论文的 EMD 保形损失（在「选远程点」语义下目标不同）。
4. geometry dump 的 hook 未用 try/finally 移除（异常路径可能泄漏，脚本一次性运行影响小）。
5. 候选池有放回采样会产生少量重复候选（v1/v2 同款设计，RP1 抽样重复率约 1.3%）。

## 验证

- `py_compile` 四个文件通过；
- csnet 自测：EOT Σw−k=9.5e-7、梯度五项非零有限、分块正确、eval 硬 top-k 正确、
  grouper 端到端反传正常、`center_xyz=None` 报错；
- learned_v2 重构后复测通过；global_knn 路由/互斥/宽度构建通过。

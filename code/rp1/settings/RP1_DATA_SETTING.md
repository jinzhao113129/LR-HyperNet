# RP1 数据与实验 Setting

## 数据入口

数据 wrapper：`rp1/data/ncaltech101_tonic_2048_seed0__RP1_W384_G16/`

- `DATA`：`/root/autodl-tmp/secnet-workspace/SECNet/data/ncaltech101_tonic_2048_seed0`
- `SOURCE`：同一原始数据目录
- 数据文件：`train.h5`、`test.h5`，以及 `prepare_summary.json`、`split.json`、`label_map.json`
- 格式：每个 sample 为 `[num_point, 4] = [2048, 4]`，通道为 `[t, x, y, polarity]`
- samples：train 20901，test 5226；类别数 101

这里的 `W384/G16` 是 RP1 的实验标记，不是 H5 文件名中内置的字段：

- `W384`：候选 remote pool / local candidate window 的代表性设置；
- `G16`：重点 Oracle remote 数量；
- 实际 run 可能使用 `G4/G8/G16/G24` 等不同值，必须读取 `runs/<run>/config.json`；
- `neighbors=24`、`group=512`、`base_dim=16` 是服务器现有 SECNet run 中的常见基础设置。

## RP1 需要保留的实验字段

对每个 run 至少记录：

| 类别 | 字段 |
|---|---|
| 输入 | `train_h5`、`test_h5`、`num_point`、`num_category`、train/test shape |
| 局部邻域 | `candidate_window`、`neighbors`、`group` |
| remote 邻域 | `global_neighbors`、`global_selector`、`num_blocks`、`selected_blocks`、`global_base_neighbors` |
| backend | `knn_backend`、`fps_backend`、`backend_chunk_size` |
| 训练 | `epochs`、`batch_size`、`lr`、`weight_decay`、`label_smoothing`、`seed` |
| 代价 | latency、FLOPs/参数量、显存、Oracle Recall |

## 结果入口

`rp1/runs/` 中已经收集：

- `iter3_oracle_stats_fullish`
- `iter4_sparse_remote_20260725_122133`
- `iter4_remote_retriever_20260725_125541`
- `iter4_r2_distill_20260725_152749`
- `iter5_causal_remote_20260726_102513`
- `iter5_global_plus_20260724_191223`
- `iter5_oracle_g24_graph_20260726_1740`
- `iter5_oracle_g24_dashboard_20260726_1740`
- `iter6_gain_oracle_structure_20260726_185939`
- `iter7_boundary_rerank_20260728_055115`
- `iter8_extension_20260729_000000`
- `iter9_indexed_b2_20260728_141743`

这些目录中的 `config.json` 是逐 run 的最终 setting；不要用 RP proposal 中的汇总数字替代具体 run 配置。

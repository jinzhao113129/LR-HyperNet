# RP1 代码与结果资产地图

## 代码

| Repo entry | 原始位置 | 用途 |
|---|---|---|
| `code/core/` | SECNet workspace 顶层 + `models/` | SECNet/FECNet 基础训练、profiling、邻域配置 |
| `code/oracle/` | SECNet workspace 顶层 | Oracle stats、neighbor composition、G24 graph/dashboard、boundary analysis |
| `code/selectors/` | SECNet workspace 顶层 | sparse remote、retriever、global-plus、indexed selector |
| `code/submission/` | SECNet workspace 顶层 | 对应迭代的 submit / collect 脚本 |

## 结果

| Repo entry | 主要内容 |
|---|---|
| `runs/iter3_oracle_stats_fullish` | Oracle feature statistics |
| `runs/iter4_sparse_remote_20260725_122133` | sparse remote profiling |
| `runs/iter4_remote_retriever_20260725_125541` | remote retriever |
| `runs/iter4_r2_distill_20260725_152749` | remote retriever distillation |
| `runs/iter5_causal_remote_20260726_102513` | local + Oracle remote causal chain |
| `runs/iter5_global_plus_20260724_191223` | global-plus controls |
| `runs/iter5_oracle_g24_graph_20260726_1740` | Oracle graph dump |
| `runs/iter5_oracle_g24_dashboard_20260726_1740` | Oracle dashboard |
| `runs/iter6_gain_oracle_structure_20260726_185939` | Oracle structure/gain analysis |
| `runs/iter7_boundary_rerank_20260728_055115` | boundary rerank |
| `runs/iter8_extension_20260729_000000` | extension experiments |
| `runs/iter9_indexed_b2_20260728_141743` | indexed remote selector |

run 内的 `config.json`、`summary.json`、CSV/JSON/PDF 是结果解释的权威来源。

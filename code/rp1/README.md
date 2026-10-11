# RP1 资产：Local + Sparse Remote Neighborhood

这里收集 RP1 的可运行代码入口、Oracle profiling / remote selector 结果和数据入口。

## 目录

- `code/core/`：SECNet 主干、训练入口与 profiling 基础设施；`models` 链接到原始 FECNet/模块目录。
- `code/oracle/`：Oracle neighbor composition、remote feature profiling、graph/dashboard 分析。
- `code/selectors/`：sparse remote、block retrieval、global-plus、indexed/boundary selector 相关脚本。
- `code/submission/`：对应实验的 submit / collect 脚本。
- `data/`：当前 RP1 使用的 N-Caltech101 H5 入口；见其中的 `SETTING.md`。
- `runs/`：已存在的 profiling、Oracle、remote selector 和 boundary rerank 结果入口。
- `settings/`：setting 汇总和原始记录链接。

## 当前数据 setting

RP1 当前代码路径实际以 N-Caltech101 为主：

- H5 输入为 `[t, x, y, polarity]`，每个 sample 为 2048 events；train/test 分别为 20901/5226 samples。
- 模型基础配置常见为 `group=512`、`neighbors=24`、`base_dim=16`。
- RP1 目标设置是 `candidate_window=W384` 加少量 remote events；Oracle 对照中重点关注 `G16`，服务器现存部分训练 run 也包含 `G24` 或其他 `G`，不能只按目录名推断，须以对应 run 的 `config.json` 为准。
- Oracle profiling 的候选特征包括 spatial novelty、coverage gain、时间/空间距离、polarity、density、feature cosine 和 anisotropy。

详细 setting：[settings/RP1_DATA_SETTING.md](settings/RP1_DATA_SETTING.md)

## 资产来源

所有入口均指向 `/root/autodl-tmp/secnet-workspace/SECNet` 内的原始代码、数据和 runs；收集目录不复制原始文件。Oracle profiling 的原始实验说明见 [source_records/20260722_iter2_experiment_claims_summary.md](settings/source_records/20260722_iter2_experiment_claims_summary.md)。

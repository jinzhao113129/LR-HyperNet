# Codex 审查请求 — 小样本快测协议 + 3090 迁移（2026-10-17 晚）

> 背景：用户战略转向——全量冲刺链已全部取消（177369/177370 scancel），
> 回归"小样本快测正反馈→再全量"协议，且**实验场地从 HPC 迁移到学长的
> 3090 平台**（https://160.30.231.177/portal/，王玲组 gpu-server-1/3，
> 你xinjiang 账号，租约 131 已占 gpu-server-1 一张 3090 到 10/24）。
> NLP2 线不动（续训已完成，解码 177624 RUNNING）。

## 一、快测协议（请审设计）

### 口径（用户拍板，与此前全量共识不同——请确认无内在矛盾）

- **靶子 = 官方 Table 9 的 60.60%**（300ms 单窗简化协议、10240 点——
  作者亲口确认网盘 baidu 数据即此格式，论文该口径成绩 60.60）；
- **不跑基线链**：直接比 60.60（历史 P0 无配方基线 0.4753 作心里参照，
  不作对照锚）；
- **单组件逐测，不测叠加**（组合链等单组件出数后再定）；
- **统一配方**（所有链全开，从论文学到的非组件有效方法）：
  - augment 2（数据×3：crop/reverse/polarity/jitter/dropout）
  - EMA 0.999（step 粒度，上一轮全套修复版）
  - patience 20 收敛即停（连续 20ep best seq 无提升正常退出）
- **数据**：baidu 3596/934（train [3596,10240,4] / test [934,10240,4]
  已核验；101 类、每类 5-333 条、均值 35.6）；
- 配置：batch 16 + grad_accum 2、AdamW lr 1e-3、epoch 上限 300
  （patience 预计 60-120ep 先触发）。

### 快测矩阵（5 条，用户已砍 scale_norm_fps——p=0.125 不显著+
  A1 已证固定标准化劣于可学习缩放，无翻盘先验）

| # | 组件 flag | 来源 | 历史证据（无配方口径） |
|---|---|---|---|
| 1 | --hyperbolic_attention | 双曲迁移（EHGCN 启发） | +3.0pp p=0.0097（McNemar） |
| 2 | --hyperbolic_selector --hsel_scaled_coords | RP1×SECNet 隐藏缩放 | +4.71pp p=0.0002 |
| 3 | --spike_time_enc | SpikePoint（学长指定） | 零实验数据（从未测） |
| 4 | --rp1_selector | RP1 空间新颖性 | +2.46pp p=0.0398 |
| 5 | --hyperbolic_mlp | 自研（补频域局部缺陷） | 日志 +2.68pp 未复验 |

### 验收口径

```
best test acc seq：
  > 0.606  → 直接赢了
  0.55-0.60 → 有希望，看组件间排序定组合
  < 0.52   → 该组件在增强配方下无正反馈，跳过
```

**知情风险记录**：P0 时代曾判小样本"坏基线"（组件结论不可信）——
本轮定位为方向性快测（增强配方是新增变量，配对看组件排序，
数字超 60.60 才算真赢）。

## 二、新增代码（请审实现）

### --patience 收敛早停（train_frequency_ncaltech.py）

```python
# argparse：
--patience N（0=关闭跑满，默认全量行为不变）
--early_stop_metric seq/acc（默认 seq，与 best 判定同口径）

# 训练循环（每 epoch 末，save_model 之后）：
if args.patience > 0:
    metric = test_out["test_acc_seq"] if args.early_stop_metric == 'seq' \
        else test_out["acc"]
    if metric > _best_early_stop:
        _best_early_stop = metric
        _epochs_no_improve = 0
    else:
        _epochs_no_improve += 1
    if _epochs_no_improve >= args.patience:
        log_string('EarlyStop triggered at epoch %d ...')
        break
```

- 状态初始化在 `epoch == start_epoch` 时重建（resume 兼容）；
- 判据与 best 判定同口径（test_acc_seq）——避免双口径；
- **注意**：early stop 后 checkpoint 已保存（save 在判定前），
  best 语义不变。

审查点：
- a) early stop 判定在 save_model 之后 break——best checkpoint 完整性
  是否无虞？
- b) resume 场景：早停状态从 start_epoch 重建（丢历史 epoch 的
  无提升计数）——可接受还是要存进 checkpoint？
- c) patience 与 stop_epoch（链式接力）共存时优先级：seg_stop break
  在循环开头、patience break 在循环末尾——同 epoch 触发时
  seg_stop 先 break，合理吗？

## 三、3090 迁移包（请审内容完整性）

已打包两个 tar.gz（用户浏览器拖拽进 code-server 上传）：

1. **secnet_code.tar.gz**（100KB，16 文件）：
   - local_edit 版 FECNet.py（**注意：不是 code/models 版**——local_edit
     版含全部 14 个组件 flag 含 hyperbolic_mlp/scale_norm_fps/
     hsel_scaled_coords；code/models 版缺这三个，快测必须用 local_edit 版）；
   - modules.py / train_frequency_ncaltech.py（早停版）/ extract_preds.py /
     local_remote_grouper.py / lr_hypernet_g16.py / train_lr_hyper.py /
     g16_profile_agent.py / stage2 全套 4 文件 + 集成测试 /
     augment_data.py / mcnemar_test.py / provider_data.py
2. **baidu_small.tar.gz**（109MB）：train/test.h5（HPC 校验过形状与
   head 字节 \x89HDF 有效）

审查点：
- d) 16 文件清单有缺吗？（3090 上跑 FECNet 快测链的最小集合）
- e) train_frequency_ncaltech.py 的 import 依赖：provider_data/
  models.FECNet——**目录结构怎么摆**？建议解压成
  `~/secnet/{FECNet.py, modules.py, train_frequency_ncaltech.py,
  provider_data.py, augment_data.py, ...}`+ 修 import 路径，还是保持
  local_edit 层级 + sys.path？请给一个明确布局。
- f) bhli 环境不可用（3090 是新机器）——PyTorch 版本/依赖装什么
  最稳？CUDA 驱动未知，建议先 nvidia-smi 确认再装对应 torch wheel。

## 四、不迁移的东西（确认清单）

- HPC 上的 log/checkpoint 不动（规范）；
- official_gen 全量数据不迁（11GB，快测用不到，全量回归 HPC 时用）；
- NLP2 线不动（177624 解码 RUNNING，等 BLEU）。

## 请求

1. 审快测协议设计（口径内在一致性：比 60.60 + 不跑基线 + 全配方
   + 单组件——有没有漏洞）；
2. 审 --patience 实现（a/b/c 三点）；
3. 审 3090 迁移包完整性与目录布局建议（d/e/f）；
4. 结论分三档：设计可直接跑 / 需修后跑 / 有根本矛盾需用户重新拍板。

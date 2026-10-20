# Codex 独立审查：HyperbolicGate + FcaGate（2026-10-18）

## 结论

**暂不放行。** 发现 2 个 P0，必须修复后才能定稿：

1. FECNet 推理工具没有恢复 `hyperbolic_gate` / `fca_gate`，hgate/fca
   checkpoint 会被静默重建成基础 gated fusion，结果无效。
2. 3090 一键脚本声称 augment×3，却传 `--augment 3`；按 trainer 语义
   实际数据是 ×4。

两个组件自身的 forward、互斥、DCT 基和跨组件组合均可运行，下面列出的
其余问题是设计或边界风险。

## P0

### P0-1：extract_preds.py 未接入两个新组件 flag

- 文件：`fork_ai2/scripts/local_edit/extract_preds.py:50`
- 问题：`FLAG_KEYS` 没有 `hyperbolic_gate` / `fca_gate`，
  `build_model()` 在 [line 109](D:/UserData/Desktop/学习/进组/secnet-hyperbolic/fork_ai2/scripts/local_edit/extract_preds.py:109)
  构造 `FECNet` 时也没有传这两个参数。
- 后果：训练 hgate/fca 后，checkpoint 含 `hyper_gates.*` 或
  `fca_gates.*`。基础 `FECNet` 直接 strict load 会失败；更糟的是
  [line 211](D:/UserData/Desktop/学习/进组/secnet-hyperbolic/fork_ai2/scripts/local_edit/extract_preds.py:211)
  的旧结构兼容逻辑会过滤掉多余 gate 键并加载基础模型。
- 此时 `group_conv_list` / `bn_global_list` 在 hgate/fca 训练期间
  从未更新，却会在基础 forward 中被使用，推理结果必然错误且静默。
- 独立验证：
  - `base.load_state_dict(hgate.state_dict())` 报多余
    `hyper_gates.*` 键；
  - base 153 keys、hgate 174 keys、fca 168 keys，基础键集合一致。
- 修法：
  1. `FLAG_KEYS` 加 `"hyperbolic_gate", "fca_gate"`；
  2. `defaults` 中两个都兜底 `False`；
  3. `FECNet(...)` 构造传 `use_hyperbolic_gate=` 和 `use_fca_gate=`；
  4. 增加一个“hgate/fca 训练 checkpoint 经 build_model 后 strict load”
     的集成测试，禁止静默丢弃 gate 键。

### P0-2：run_3090 脚本 augment 参数与协议注释不一致

- 文件：`fork_ai2/scripts/run_3090_setup_and_run.sh:34`
- 注释写“augment×3”，但 [line 170](D:/UserData/Desktop/学习/进组/secnet-hyperbolic/fork_ai2/scripts/run_3090_setup_and_run.sh:170)
  传的是 `--augment 3`。
- trainer 的语义是 `N>0` 时“每个原始样本再生成 N 个增强版本”，
  数据量为 `×(1+N)`，见
  [train_frequency_ncaltech.py:148](D:/UserData/Desktop/学习/进组/secnet-hyperbolic/fork_ai2/scripts/local_edit/train_frequency_ncaltech.py:148)。
  因此 `--augment 3` 实际是 ×4，不是 ×3。
- 影响：当前 base/ha/hsel/spike/rp1/hmlp 六条快测链的数据量、
  每 epoch 时间、内存和预算都按错误协议跑。
- 修法：若协议目标是总数据 ×3，改为 `--augment 2`；若 ×4 是新决策，
  则同步修改协议注释、时间和内存估算，不能只改脚本。

## P1

### P1-1：HyperbolicGate 初始 gate 严重塌缩，不是中性

- 文件：`fork_ai2/scripts/local_edit/FECNet.py:623`
- 注释声称 `r0=2` 时初始 gate“偏中性”，独立实测不符：
  - 单模块随机输入：dist 7.74-8.43，gate 0.0016-0.0032；
  - 完整 FECNet seed0 输入，三个 stage gate 均值分别为
    0.000218 / 0.00796 / 0.01766。
- 原因：`in_proj` 后 `LayerNorm` 的向量范数接近 `sqrt(64)=8`，
  `lorentz_exp_origin` 后与质心距离约 7-12，远大于 r0=2。
- 修法：改为可学习偏置门控
  `gate = sigmoid(bias - alpha.abs() * dist)`，bias 初始 0；
  或把 `r0` 初始化到实测距离量级并写清楚依据。至少加一个
  初始化 gate 统计断言，防止未来重写时再次塌缩。

### P1-2：FcaGate 的 float32 basis 在 AMP 下会 dtype mismatch

- 文件：`fork_ai2/scripts/local_edit/FECNet.py:699`
- `basis` 固定注册为 float32；当前训练未启用 AMP，所以不是即时崩溃。
- 独立验证：`torch.matmul(half_x, float32_basis.t())` 直接报
  `RuntimeError: expected m1 and m2 to have the same dtype`。
- 修法：forward 中执行 `basis = self.basis.to(x_src.dtype)`，或者
  在类注释明确“仅支持 float32，AMP 禁止开启”。

### P1-3：num_freq=1 不是结构意义上的原 gated fusion

- 文件：`fork_ai2/scripts/local_edit/FECNet.py:680`
- 文档说“num_freq=1 时退化为原 gated fusion”。实际 basis 做了 L2
  归一化，DC 投影是 `sqrt(S) * mean`，不是 `mean`；同时原实现是
  `group_conv -> bn_global`，这里是 `Linear-ReLU-Linear`。
- 两者信息上等价，但结构和初始化不同，不能声称“严格退化”。
- 修法：改为“信息上等价于全局平均，DC 投影与 mean 成正比”，或另加
  一个真正的 `num_freq=1` 对齐对照。

### P1-4：FcaGate 与已有 FFT 聚合存在信息冗余风险

- 设计层面问题：每 stage 已在 `fft_sptaial` 和 `fft_temporal` 做两次
  频域变换，FcaGate 再对同一激活做 DCT。低 4 频率可能主要重复已有
  FFT 信息。
- 修法：出训练结果前先做 `num_freq=1/2/4/8` 的诊断，或与 baseline
  和原始 gated fusion 三臂对比；论文中不能把 FcaGate 增益自动归为
  “FcaNet 机制有效”。

### P1-5：hgate/fca_gate 与 hyperbolic_fusion 可同开，但语义重叠

- 代码未互斥，独立 forward 均通过：
  - hgate+fusion：1986594 -> 2122048 参数，输出 finite；
  - fca+fusion：2162585 -> 2298039 参数，输出 finite。
- 设计上 hgate 已经做双曲门控，`hyperbolic_fusion` 又做双曲残差
  门控，两个机制会同时注入双曲先验，单组件归因会混。
- 修法：至少在任务书和文档中写“同开视为组合变体，不用于单组件
  归因”；若不打算组合，增加显式互斥或警告。

### P1-6：hgate 从逐通道门控改成逐组门控需要单列设计声明

- 这是有意语义变化，不是代码错误；但它改变了替换路线的研究假设。
- 建议在替换路线文档中单独说明“gated fusion 的粒度变化”和预期
  spatial outlier 语义，并请学长确认；否则实验报告容易把逐组门控的
  作用误写成原通道注意力的同类增益。

## P2

1. FcaGate forward 建议增加显式 `S == self.basis.shape[1]` 断言；
   当前只会在 matmul 报底层维度错误。
2. 两份 trainer 的 resume 校验键建议补 `hyperbolic_gate` / `fca_gate`。
   strict load 通常能拦截，但显式校验错误信息更清楚。
3. run_3090 脚本的 `CHAIN_ORDER` 有意不含 hgate/fca，建议注释写
   “已定义但未排入快测 6 链”，避免以后误读为遗漏。
4. 把第四节冒烟测试保存为 repo 内可复跑文件；当前只存在于对话中，
  后续 regressions 无法直接回归。
5. FcaGate 的参数量 +236736（约 +12%）已经接近一个较大组件；若后续
  全量模型有参数预算，需把 `reduction` 和 `num_freq` 一起进消融表。

## 独立复跑结果

环境：本地 torch 2.12.0+cpu；md5 与任务书一致：

```text
FECNet.py                  b10f4716b1b943115fc64d6faaec218f
train_frequency_ncaltech.py b1740ad4bd2c10e24712b482e5cedc94
run_3090_setup_and_run.sh  7cd3620d1f90ee3aa7878af539ac4007
```

完整模型 forward（N=2048）：

```text
fca_gate        out=(1,101) params=2162585 finite=True logit_std=0.0607
hgate           out=(1,101) params=1986594 finite=True logit_std=0.0359
base            out=(1,101) params=1925849 finite=True logit_std=0.0873
fca+spike+rp1   out=(1,101) params=2163097 finite=True logit_std=0.0605
hgate+fusion    out=(1,101) params=2122048 finite=True logit_std=0.0358
fca+fusion      out=(1,101) params=2298039 finite=True logit_std=0.0867
```

专项验证：

```text
num_freq=1 basis:
  std=3.73e-09, mean=0.022097, expected=0.022097
常量输入 DC 投影:
  S=256 -> 48.000000, expected=48.000000
  S=512 -> 67.882271, expected=67.882251
  S=1024 -> 96.000000, expected=96.000000
  S=2048 -> 135.764618, expected=135.764502
  其他频率最大绝对值 < 1e-5
HyperbolicGate:
  curvature=0.400099, alpha=1.0, r0=2.0
  随机单模块 dist=7.74-8.43, gate=0.0016-0.0032
  完整模型 stage gate mean=0.000218/0.00796/0.01766
互斥:
  FECNet(use_hyperbolic_gate=True, use_fca_gate=True)
  -> ValueError: hyperbolic_gate 与 fca_gate 互斥
错误 seq_len:
  FcaGate(seq_len=2048).forward(S=1024)
  -> RuntimeError mat1/mat2 shape mismatch
AMP 模拟:
  half x @ float32 basis -> RuntimeError dtype mismatch
py_compile:
  FECNet.py + train_frequency_ncaltech.py 均通过
```

## 放行判定

**暂不放行。** P0-1 必须先修，否则两个组件训练出的 checkpoint
没有正确评测入口；P0-2 必须先定增强倍数，否则当前六条快测链本身
就跑在错误协议上。其余 P1 可在修复 P0 后按优先级处理。

---

## 复审结果（修复版 2026-10-18）

新文件 md5 与提交一致：

```text
FECNet.py                  9527a049107686f5379f919b2e2d2b89
extract_preds.py           e508276aa0e4794a39d419391f37bd25
run_3090_setup_and_run.sh  3557baedc718dbf947248fdf07918571
test_fusion_gate.py        89df5748cba3b7716d9ffe159088b776
```

### 已确认修复

- P0-1：`extract_preds.py` 的 `FLAG_KEYS`、defaults、`FECNet(...)`
  构造均已接 `hyperbolic_gate` / `fca_gate`。独立用 build_model 重建后
  hgate 174 keys、fca 168 keys 均 strict load 通过。
- P0-2：3090 脚本已把协议明确为“原样+3 增强 = ×4”，实际参数
  `--augment 3` 与运行脚本一致。
- P1-1：初始 gate 数值已改为 0.5。
- P1-2：`basis.to(x_src.dtype)` 已加，autocast 下 buffer 会跟随。
- P1-3：num_freq=1 的措辞已改成“信息上等价、非严格退化”。
- P2-1/3/4：seq_len 显式检查、CHAIN_ORDER 注释、独立测试文件已补。

### 新 P0：alpha=0 冻结了整个双曲门控路径

- 文件：`fork_ai2/scripts/local_edit/FECNet.py:641`
- `alpha` 初始化为 0，但 forward 使用 `self.alpha.abs()`。
  `abs()` 在 0 点梯度为 0，因此 `alpha` 永远不更新。
- 同时 `curvature` 和 `in_proj` 只通过 `-alpha*dist` 进入输出，
  alpha=0 时它们的梯度也全部为 0。实际训练中只有 bias 会学习，
 组件会退化成共享标量门控，不再是双曲距离门控。
- 独立验证：

```text
alpha.grad     = 0.0
curvature.grad = 0.0
bias.grad      = -288.2

10 个 SGD step 后：
alpha = 0.0
bias  = -6.08
```

- 修法：不要用 `abs(0)`。推荐
  `alpha = softplus(alpha_raw)`，`alpha_raw` 初始为
  `log(expm1(0.01))`，`bias=0`。这样初始 gate 约 0.48、接近中性，
  同时 alpha、in_proj、curvature 都有非零梯度。测试必须增加两条：
  1. 初始 gate 接近 0.5；
  2. 一次 backward 后 `alpha.grad != 0` 且 `curvature.grad != 0`。

### 残留 P1：×4 口径未在全文档同步

`pilot_benchmark_consensus_20261018.md` §3 表格已写 ×4，但同文件
[§2 line 36](D:/UserData/Desktop/学习/进组/secnet-hyperbolic/fork_ai2/notes/pilot_benchmark_consensus_20261018.md:36)
和 [line 53](D:/UserData/Desktop/学习/进组/secnet-hyperbolic/fork_ai2/notes/pilot_benchmark_consensus_20261018.md:53)
仍写“增强×3”；`feishu_update_20261018.md` 也仍写 ×3。运行脚本已经
正确，但外部引用文档内部仍不一致。建议统一为“原样+3 增强=总数据×4”，
或统一为 ×3，不能两处并存。

### 复跑结果

```text
test_fusion_gate.py:
  forward                         PASS
  combo fca+spike+rp1             PASS
  hgate initial gate neutral      PASS（只测了 gate=0.5，未测梯度）
  fca num_freq=1 degenerate       PASS
  mutual exclusion                PASS
  fca seq_len guard               PASS
  fca amp dtype                   PASS（本地无 CUDA，实际 skip）

extract_preds build_model strict load:
  hgate keys=174 strict OK
  fca  keys=168 strict OK

额外梯度检查:
  alpha.grad=0.0 / curvature.grad=0.0 -> 阻断
```

### 复审放行判定

**仍不放行。** 唯一阻断项是新 P0：HyperbolicGate 当前会训练成“共享
标量偏置门控”，双曲距离分支完全无梯度。修掉 alpha 参数化和对应
梯度测试后，再复跑一遍即可放行。P1-4/5/6 维持上一轮的设计层结论，
不作为代码阻断。

---

## 最终复审（2026-10-18，可定稿）

新 md5 与提交一致：

```text
FECNet.py          1b756b13c5d2e171d642cf57fe82838e
test_fusion_gate.py 9003c107f26d196cfeb8744b3cd95982
extract_preds.py   e508276aa0e4794a39d419391f37bd25（本轮未再改）
run_3090_setup_and_run.sh 3557baedc718dbf947248fdf07918571（本轮未再改）
```

### 阻断项处置

- alpha 已改为 `softplus(alpha_raw)`，alpha_raw 初始
  `log(expm1(0.01))`，alpha≈0.01、初始 gate≈0.48。
- 独立 backward 验证：alpha_raw、curvature、in_proj、bias 的梯度均
  非零，不再出现双曲分支梯度死亡。
- 新增 `test_hgate_gradient_nonzero`，回归套件 8 项全部通过。
- extract_preds 的 hgate/fca 重建路径 strict load 通过（174/168 keys）。
- ×3 与 ×4 的文档口径已在 pilot_benchmark_consensus 和
  feishu_update_20261018 中统一。

### 独立复跑

```text
test_fusion_gate.py:
  forward                         PASS
  combo fca+spike+rp1             PASS
  hgate initial gate neutral      PASS（gate=0.4798）
  hgate gradient nonzero          PASS
  fca num_freq=1 degenerate       PASS
  mutual exclusion                PASS
  fca seq_len guard               PASS
  fca amp dtype                   PASS（本地无 CUDA，实际 skip）

extract_preds strict load:
  hyperbolic_gate 174 keys OK
  fca_gate        168 keys OK
```

### 遗留非阻断项

1. `test_fusion_gate.py` 顶部 docstring 仍写“初始 gate=0.5”，实际断言
   已改为约 0.48；建议顺手改为 0.48，避免以后误读。
2. trainer 的 resume 配置校验列表尚未加入 `hyperbolic_gate` /
   `fca_gate`；strict load 通常能拦住，但建议后续补上更明确的错误信息。
3. GPU autocast 测试在本地跳过，真实 3090 首次跑链时应确认该项输出。

### 最终结论

**可放行（定稿）。** 两个 P0 均已闭环，其余 P1-4/5/6 已记录为设计层
问题，不阻塞本次组件代码定稿。

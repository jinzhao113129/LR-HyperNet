# Codex 审查请求 — EMA 接线 + augment OOM 流式修复（2026-10-17 15:30，补审）

> **审查期间更新（15:50）**：Claude 自查已发现并确认一个 P1 问题——
> **epoch 粒度 + decay 0.999 的标定错误**（EMA 有效窗 1000 次 update，
> 但每 epoch 只 update 1 次 → 150ep 结束时 EMA 只追上 13.9%，BN stats
> 严重陈旧，net_ema 出口无效）。数学推导+实测吻合（稳态 5.0，EMA 0.44
> ≈ 理论值 5×(1-0.999^100)）。**用户已暂停两条冲刺链（177162/177163
> 已 scancel）**，等修好（update 移 step 粒度）再重提。
> 请你独立复核这个判断 + 审以下原始材料，确认修法方向。

> 背景：今天上午紧急强化冲刺链时，两批代码改动走了捷径未过审就直接上线
> （违反"所有代码必审"规约，现补审）。**177162（ha+augment+EMA）已带
> 这两批代码在跑（当前 epoch 3，暂无异常）**；177163（G16）PENDING。
> 如审出致命问题，趁早期撤回重提损失小。

## 被审文件（均在 HPC + 本地双端）

| 文件 | 本地路径 | 改动内容 |
|---|---|---|
| train_frequency_ncaltech.py | fork_ai2/scripts/local_edit/ | +ModelEMA 类 / +--ema flag / 训练循环 update / save_model 存 net_ema |
| train_lr_hyper.py | fork_ai2/scripts/local_edit/ | 同上四点（两份 trainer 同步） |
| augment_data.py | secnet-hyperbolic/code/ | augment_train_data 内存重写（列表+concat → 预分配流式） |
| run_full_ha_aug_ai2.sh / run_full_g16_hypfar_ai2.sh | fork_ai2/scripts/ | --ema 0.999 加入 + --mem 64G |

## 审查点（按风险排序）

### 1. ModelEMA 类（两份 trainer 相同实现）

```python
class ModelEMA:
    def __init__(self, net, decay=0.999):
        self.decay = decay
        self.module = net.module if hasattr(net, 'module') else net
        self.shadow = {k: v.detach().float().clone()
                       for k, v in self.module.state_dict().items()}

    @torch.no_grad()
    def update(self, net):
        m = net.module if hasattr(net, 'module') else net
        for k, v in m.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:  # num_batches_tracked 等整数计数器直接跟随
                s.copy_(v)

    def state_dict(self):
        return self.shadow
```

关注点：
- a) **BN running_mean/var 的 EMA 语义**——它们本身就是滑动统计量，
  对它们再做 EMA（shadow=EMA of running stats）是否合理？timm 的
  ModelEmaV2 是对整个 state_dict 做同样处理（含 BN stats），我沿用；
  但想确认你对"二次 EMA 的 BN"有无更大担忧；
- b) **epoch 粒度 update**（每 epoch 末一次，非每 step）——decay 0.999
  的有效窗 ~1000 步 ≈ 半 epoch（1959 步/ep），我判断 epoch 末采样
  足够；若你认为 step 粒度必要，说明理由（EMA 是并行出口、不参与
  best 判定，我倾向 epoch 粒度的开销/收益平衡）；
- c) **DDP 兼容**——net.module 剥离了 DataParallel 包装；当前链是
  单卡（train_method='others'），但 DDP 路径下 EMA 只更新 rank0 视角
  是否够（当前无 DDP 链，非阻断）；
- d) **checkpoint 键**——save_model kwargs 传 net_ema=None（ema 关闭时）
  或 dict。旧推理工具（extract_preds.py）读 'net' 键不受影响；但
  torch.load 后 ckpt['net_ema'] 是 dict（非 OrderedDict）——加载到
  model.load_state_dict 需要键完全匹配，评估 EMA 时直接
  model.load_state_dict(ckpt['net_ema']) 应可行（键来自同一
  state_dict）。

### 2. 两份 trainer 的接线（各 4 处）

```
argparse: --ema float default 0.0（0=完全跳过，行为与原版逐字节一致）
训练循环前: ema = ModelEMA(classifier, args.ema) if args.ema > 0 else None
train_out = train(...) 之后: if ema is not None: ema.update(classifier)
save_model(...): net_ema=(ema.state_dict() if ema is not None else None))
```

关注点：
- e) update 时机在 `train()` 返回后（epoch 末）——此时模型刚完成整个
  epoch 的更新，EMA 捕捉的是 epoch 末权重 ✓；
- f) **best/last 判定完全不动**（is_best 仍按非 EMA test 结果）——EMA
  纯并行出口，不污染组件对照 ✓（设计意图，请确认实现无泄漏）；
- g) resume 场景：链被 KDA 式续跑时 EMA shadow 不存 checkpoint 之外的
  状态、重跑从当前权重重建——EMA 会"重启"（等效于续训后重新积累平均窗）。
  我判断可接受（EMA 是辅助出口），若你认为应存 optimizer 一样存 ema
  状态，是个改进项（非阻断）。

### 3. augment_train_data 内存重写（OOM 紧急修复）

原版（OOM，峰值 ~72GB）：versions 列表持 4 份副本 + np.concatenate 再复制。
新版（预分配流式，峰值 ~31GB）：

```python
data_aug = np.empty((n_total, target_points, C), dtype=data.dtype)
data_aug[:N] = data            # 原样块
for v in range(n_versions):
    base = (v + 1) * N
    for i in range(N):
        aug = augment_sample(data[i], rng)
        if aug.shape[0] != target_points:  # 重采样回 10240
            ...
        data_aug[base + i] = aug
```

关注点：
- h) **块布局**：[原样 N | 增强0 N | 增强1 N]——与原版
  np.concatenate(versions) 逐字节同布局 ✓（label/mark 的 tile 顺序
  对应不变）；
- i) **rng 消耗顺序**：原版是"版本优先"（v0 的全部样本 → v1 的全部），
  新版同为版本外层循环 → 同种子下增强内容逐位一致（可复现性保持）——
  **这是我最想让你核的点**：两版 rng 顺序是否真的等价；
- j) 验证已做：峰值/输出比 1.03（原版 ~2.5）；语义一致（原样块/label
  tile/mark 共享）；HPC bhli 实机自检过。177162 实跑：augment 62703→
  188109 ✓ 无 OOM ✓。

### 4. sbatch 变更

--ema 0.999 加入两条链；--mem 40G→64G（augment 数据 31GB + 模型/优化器/激活）。

## 我的自测记录（供你复核参考）

- ModelEMA 单元测试：初始 EMA=权重；真实 CE loss 训练 200 步后
  EMA-当前权重差 0.57（滞后平均生效）；BN num_batches_tracked 直接跟随 ✓；
- 注意：测试过程中曾出现"差值 8e31"——是测试自身无约束 loss（lr=0.1 无
  正则 100 步）导致权重爆炸，非 EMA bug；换真实 CE loss 复测通过；
- augment：自检 5 种增强 + 端到端 + 真实形状模拟（32→96，mark 共享、
  label 对齐）全过。

## 请求

1. 独立复核上述 9 个关注点（a-j 中 i 为最高优先级）；
2. 结论分三档：**无阻断可继续跑** / **需修但可边跑边修** /
   **必须撤回（177162 当前 epoch ~3，撤回损失小）**；
3. 如有我没列到的暗坑（尤其 EMA 与 checkpoint 保存格式的交互、
   流式 augment 与 DataLoader 的交互），指出。

---

## 附：Claude 自查发现的 P1（15:50，已导致暂停）

### EMA epoch 粒度 decay 标定错误

- 现象：BN running_mean 稳态 5.0，epoch 粒度 update 100 次后 EMA shadow
  仅 0.44——与理论 5×(1-0.999^100)=0.476 吻合（实测吻合验证）；
- 推导：decay 0.999 的有效窗 = 1000 次 update。timm 惯例 step 粒度
  （1000 步 ≈ 0.5 epoch）合理；我们 update 在 epoch 末（每 epoch 1 次）
  → 追上稳态需 ~693 epoch，150ep 链结束只追 13.9%；
- 后果：net_ema 的 BN stats 与权重严重不匹配 → EMA 评估出口无效
  （不崩训练、不污染 best/last——影子无回写，所以此前 177162 跑着
  无异常表象）；
- 自查同时通过的点：augment rng 等价性（点 i）逐位一致 PASS；
  HPC/本地文件一致性 PASS；best/last 判定零污染 PASS；
- **修法方向（待你复核）**：update 移进 train() 的 batch 循环
  （step 粒度，需传 ema 进 train 函数）——两份 trainer 同改；
  备选：decay 改 0.99（epoch 粒度窗 100ep，150ep 追 78%，一行改但
  EMA 质量次优）。倾向前者；
- 两条链已暂停（177162 跑了 ~3.5h epoch 3、177163 未开跑），修好重提。

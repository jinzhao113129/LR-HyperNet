"""
LR-HyperNet 训练脚本
基于 train_hfecnet.py 修改：模型换成 LRHyperNet（Local+Remote + 双曲远程聚合），
训练协议与 SECNet 基线对齐：
  batch=80, epoch=150, num_point=10240, AdamW lr=0.001 wd=1e-4, label smoothing eps=0.2,
  data=./ncaltech101_official_gen/ (train 62703 / test 5226, 布局 t,x,y,p)
"""
import os
import sys
import torch
import numpy as np
import datetime
import logging
import argparse
import provider_data
import time
import torch.nn.functional as F
from pathlib import Path
import sklearn.metrics as metrics
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('LR-HyperNet training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--batch_size', type=int, default=80, help='batch size in training')
    parser.add_argument('--num_category', default=101, type=int, help='number of categories')
    parser.add_argument('--epoch', '--epochs', dest='epoch', default=150, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument("--data_path", type=str, default='./ncaltech101_official_gen/', help="path to data")
    parser.add_argument("--log_path", type=str, default='./tensorboard_log/', help="tensorboard log path")
    parser.add_argument("--log_name", type=str, default=None, help="tensorboard run name (default: derived from log_dir)")
    parser.add_argument('--num_point', type=int, default=10240, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='AdamW,SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--curvature_init', type=float, default=0.4, help='hyperbolic curvature init')
    parser.add_argument('--remote_g', type=int, default=16, help='remote neighbors per centroid')
    parser.add_argument('--time_enc', action='store_true', default=False,
                        help='temporal causal order encoding (multi-frequency sinusoid on t)')
    parser.add_argument('--kda', action='store_true', default=False,
                        help='KDA causal time encoding (Kimi Delta Attention style gated delta rule, '
                             'mutually exclusive with --time_enc)')
    parser.add_argument('--kda_bidirectional', action='store_true', default=False,
                        help='KDA bidirectional causal scan (v3.1, HydraMamba-inspired, only with --kda)')
    parser.add_argument('--use_global_anchor', action='store_true', default=False,
                        help='global absorbing node for remote selection (PVAG-inspired)')
    parser.add_argument('--anchor_lambda', type=float, default=0.5,
                        help='global anchor mixing weight in remote distance')
    parser.add_argument('--hyperbolic_classifier', action='store_true',
                        help='路线A：双曲原型分类头（欧氏backbone+双曲分类层）')
    parser.add_argument('--hyperbolic_type', type=str, default='lorentz',
                        choices=['poincare', 'lorentz', 'euclidean'],
                        help='hyperbolic space type: lorentz (default, v3.1 数值稳定，无边界梯度尖峰), '
                             'poincare (旧版，有边界 atanh 尖峰风险), or euclidean (消融对照, same backbone/params)')
    parser.add_argument('--remote_dropout', type=float, default=0.0,
                        help='dropout on remote aggregator output (anti-overfit)')
    parser.add_argument('--use_hyperbolic_selector', action='store_true', default=False,
                        help='双曲距离选16个远程点（学长建议）：RemoteSelector 中先 exp map 到双曲空间，'
                             '再用双曲距离代替欧氏距离选远程邻居。配合 --hyperbolic_type lorentz 使用')
    parser.add_argument('--use_rp1_selector', action='store_true', default=False,
                        help='RP1 空间新颖性远程选择器（消融实验）：远程选择得分 = '
                             '特征距离 + rp1_lambda * 候选点与局部邻居的最小空间欧氏距离')
    parser.add_argument('--rp1_lambda', type=float, default=1.0,
                        help='RP1 空间新颖性权重（仅 --use_rp1_selector 时有效）')
    parser.add_argument('--selector_mode', type=str, default='topk',
                        choices=['topk', 'greedy_div', 'random', 'spatial_uniform'],
                        help='远程点选择模式：topk（特征距离独立 top-G，默认）/ greedy_div（贪心 max-min '
                             '多样性）/ random（非局部候选中随机选 G，RP1 论文随机基线）/ spatial_uniform'
                             '（x,y 空间均匀网格采样，RP1 论文最好的 cheap selector）')
    parser.add_argument('--sanity_5class', action='store_true', default=False,
                        help='5-class subset overfit sanity check (Task B): filter label<5 samples '
                             'from train/test h5, num_category forced to 5, epoch capped at 50, '
                             'early stop when train acc > 99%%')
    # ===== b8/30ep 欠拟合修复（全部默认关闭，不影响正在跑的作业）=====
    parser.add_argument('--group_norm', action='store_true', default=False,
                        help='全网 BatchNorm1d -> GroupNorm（b8 修复：BN 批统计量在小 batch 下'
                             '噪声极大，GroupNorm 与 batch size 无关；作用于 backbone 的 '
                             'Linear1Layer/Linear2Layer/bn_local/bn_global/classifier）')
    parser.add_argument('--grad_accum', type=int, default=0,
                        help='梯度累积步数 N（b8 修复：等效大 batch，loss 除以 N，每 N 步 '
                             'optimizer.step() 一次；0/1 = 关闭。b8 想等效 b32 用 --grad_accum 4）')
    parser.add_argument('--balanced_sampler', action='store_true', default=False,
                        help='WeightedRandomSampler 类均衡采样（b8 修复：长尾 5~333 样本/类，'
                             '权重 1/n_c，num_samples=len(dataset) 保持每 epoch iter 数稳定）')
    parser.add_argument('--logit_adj', type=float, default=0.0,
                        help='logit-adjusted CE 的 tau（b8 修复：loss = CE(logits + tau*log(prior))，'
                             'prior 为训练集类占比；0 = 关闭，常用 1.0）')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')
    # 2026-10-15 G16 profiling 接线（Codex 终审修正版 v2——修掉 v1 的四个
    # 硬问题：①接错 trainer（v1 误接 FECNet 的 train_frequency_ncaltech）；
    # ②触发时机（v1 在 forward 前读旧选点——改为 forward 后读取，grouper
    # 内部当 stage 缓存 _g16_last_selection）；③selector 字段用 log_dir
    # ——改为显式 treatment 名；④import 路径写实）。
    parser.add_argument('--g16_profile', action='store_true', default=False,
                        help='G16 选择分布采集（profile agent v3, Bernoulli 触发, '
                             '在 LocalRemoteGrouper 各 stage 内缓存、forward 后统一落盘）')
    parser.add_argument('--g16_profile_p', type=float, default=0.01,
                        help='G16 profiling 触发概率（0<p<=1, Bernoulli）')
    parser.add_argument('--g16_treatment', type=str, default='',
                        help='当前六处理名（random/spatial_fixed/euc_far/euc_near/'
                             'hyp_far/rp1_scorer——显式写入 h5 的 selector 字段）')
    parser.add_argument('--augment', type=int, default=0,
                        help='训练数据增强版本数（0=关闭；2=每样本再生成2个增强版本，数据×3）')
    parser.add_argument('--ema', type=float, default=0.0,
                        help='EMA 权重副本 decay（0=关闭；0.999=典型值）。checkpoint 存 '
                             'net_ema 键，best/last 判定不变（EMA 为并行评估出口）。'
                             '2026-10-17 用户决策：冲刺链全开。')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def cal_loss(pred, gold, smoothing=True):
    """计算交叉熵损失，支持 label smoothing（eps=0.2，与基线一致）。"""
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')
    return loss


def compute_class_prior(labels, num_classes):
    """计算训练集类占比 prior（logit-adjusted CE 用）。

    长尾数据下头部类先验大，log(prior) 会压低头部类 logit、抬高尾部类 logit，
    纠正"模型把一切样本都猜成头部类"的先验偏置。

    Args:
        labels: 训练集标签（np.ndarray 或 torch.Tensor，任意形状）
        num_classes: 类数
    Returns:
        np.ndarray [num_classes]，各类占比（和为 1）
    """
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    counts = np.bincount(labels_np.reshape(-1).astype(np.int64), minlength=num_classes)
    prior = counts.astype(np.float64) / max(counts.sum(), 1)
    return prior


def make_balanced_sampler(labels, num_classes):
    """构造类均衡 WeightedRandomSampler（--balanced_sampler 用）。

    每个样本权重 = 1/n_c（n_c 为其类别样本数），尾部类被过采样、头部类欠采样，
    每个 epoch 的期望类分布接近均匀。num_samples = len(dataset) 且
    replacement=True，保证每 epoch 的 batch 数与原 shuffle + drop_last 一致
    （len(dataset)//batch_size），iter 数稳定不破坏 cosine 调度的总步数假设。

    Args:
        labels: 训练集标签（np.ndarray 或 torch.Tensor）
        num_classes: 类数
    Returns:
        torch.utils.data.WeightedRandomSampler
    """
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.asarray(labels)
    labels_flat = labels_np.reshape(-1).astype(np.int64)
    counts = np.bincount(labels_flat, minlength=num_classes).astype(np.float64)
    # 类样本数为 0 时不参与采样（权重 0，WeightedRandomSampler 不会抽到它）
    safe_counts = np.maximum(counts, 1.0)
    weights = torch.from_numpy(1.0 / safe_counts[labels_flat]).double()
    return torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(labels_flat), replacement=True)


def train(net, trainloader, optimizer, criterion, device, args, curvature_init=0.4, curvature_warmup_epochs=10, ema=None):
    """训练一个 epoch。v3.1 新增：曲率正则化损失 + warmup 期固定曲率。
    优化修复新增：--grad_accum N 梯度累积（loss/N，每 N 步 step 一次，等效大 batch）。"""
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_true = []
    train_pred = []
    nan_skipped = 0
    done_steps = 0
    start_time = time.time()
    # v3.1 曲率优化三件套之二：曲率正则化系数（防止曲率偏离初始值过远）
    curvature_reg_lambda = 0.01
    # 径向约束损失：remote_agg 前向缓存的 last_radial_pen，按 epoch 求均值记日志
    radial_pen_sum = 0.0
    radial_steps = 0
    # 梯度累积步数：0/1 视为关闭（不加 flag 时与原逻辑逐字节一致）
    accum = args.grad_accum if getattr(args, 'grad_accum', 0) and args.grad_accum > 1 else 1
    num_batches = len(trainloader)

    # ---- G16 profiling 初始化（Codex 修正版：接在正确的 LRHyperNet trainer）----
    _prof_on = getattr(args, 'g16_profile', False)
    _prof_p = float(getattr(args, 'g16_profile_p', 0.01))
    _prof_treatment = str(getattr(args, 'g16_treatment', '') or 'unknown')
    _prof_top = 2048
    _prof_buf = []
    _prof_path = None
    _cur_epoch = getattr(train, '_cur_epoch', -1)
    if _prof_on:
        assert 0 < _prof_p <= 1, f'g16_profile_p 须 (0,1]，收到 {_prof_p}'
        # Codex 第三轮补充#6: profiling 必须知道在测哪个处理——
        # 空 treatment 时 h5 的 selector 字段为空串，六处理数据无法区分
        assert str(getattr(args, 'g16_treatment', '') or ''), \
            'g16_profile=True 时必须显式传 --g16_treatment（六处理名）——' \
            '空 treatment 的 h5 selector 字段无法区分处理'
        try:
            import sys as _sys
            for _pd in ('/home/jnguan/JiangYouxin/ZhaoJin/ai2_g16',):  # 写实路径（Codex 小修项）
                if _pd not in _sys.path:
                    _sys.path.insert(0, _pd)
            from g16_profile_agent import profile_batch, profile_to_h5
            import random as _random
            _prof_rng = _random.Random(20261015)  # 独立 RNG，不碰训练全局
            import os as _os
            # Codex 第三轮补充#2 + 第四轮记录#3: profile 跟随**训练真实**实验
            # 目录 log/lr_hypernet/<目录>/——与 checkpoints/logs 同级。
            # log_dir=None 时训练用时间戳目录（main() 的 timestr），此处
            # 通过函数属性 main→train 传递（与 _cur_epoch 同模式），避免
            # 落到 'default' 与训练产物分家。显式传 log_dir 时两者一致。
            _exp_sub = getattr(train, '_exp_subdir', None) or str(
                getattr(args, 'log_dir', '') or 'default')
            _prof_dir = _os.path.join('log', 'lr_hypernet', _exp_sub)
            _os.makedirs(_prof_dir, exist_ok=True)
            _prof_path = _os.path.join(_prof_dir, 'g16_profile.h5')
            # grouper 缓存容器：LocalRemoteGrouper.forward 在
            # net._g16_pending_batch=True 时把当 stage 选点 append 进来
            net._g16_stage_selections = []
            log_string(f'[g16_profile] on: p={_prof_p} treatment={_prof_treatment} -> {_prof_path}')
        except ImportError as _e:
            log_string(f'[g16_profile] agent 导入失败，关闭: {_e}')
            _prof_on = False

    for batch_idx, (points, labels) in enumerate(trainloader):
        points, labels = points.to(device), labels.to(device).long()
        points = points.permute(0, 2, 1).float()  # [B, 4, N]

        # 累积窗口的第一个 batch 前（含每个 epoch 开头）清零梯度
        if (batch_idx % accum) == 0:
            optimizer.zero_grad()
        outputs = net(points)
        # ---- G16 profiling 采集（Codex 修正：forward 后读取，不采旧 batch）----
        # grouper 各 stage 的 forward 在 _g16_pending_batch=True 时缓存
        # 当 stage 的 (xytp, remote_idx, local_idx, centroid_idx) 到
        # net._g16_stage_selections 列表——这里 forward 完立刻采集，
        # 坐标与选点严格同 batch 同 stage。独立 RNG 触发（不碰训练全局）。
        if _prof_on and (_prof_rng.random() < _prof_p):
            try:
                for sel_rec in getattr(net, '_g16_stage_selections', []):
                    st = profile_batch(
                        sel_rec['xytp'], sel_rec['remote'], sel_rec['local'],
                        sel_rec['centroid'], feat=sel_rec.get('feat'),
                        top_records=_prof_top, stage=sel_rec['stage'],
                        epoch=_cur_epoch, global_step=(_cur_epoch * num_batches + batch_idx),
                        selector=_prof_treatment, seed=args.seed)
                    _prof_buf.append(st)
                if len(_prof_buf) >= 64:
                    profile_to_h5(_prof_buf, _prof_path)
                    _prof_buf.clear()
            except Exception as _e:
                log_string(f'[g16_profile] 采集异常（本 epoch 静默关闭）: {str(_e)[:120]}')
                _prof_on = False
        # 采集完清 grouper 缓存（防下一 batch 误读）
        if hasattr(net, '_g16_stage_selections'):
            net._g16_stage_selections.clear()
        loss = criterion(outputs, labels)
        # v3.1 曲率正则化：L_c = λ * Σ(c_i - c_init)²
        curvature_reg = torch.tensor(0.0, device=device)
        for name, p in net.named_parameters():
            if 'curvature' in name:
                curvature_reg = curvature_reg + (p - curvature_init) ** 2
        loss = loss + curvature_reg_lambda * curvature_reg
        # 径向约束损失：模型 forward 时在 remote_agg.last_radial_pen 缓存的惩罚项，
        # 存在且非 None 时加权并入总损失（旧模型无该属性则自动跳过，行为不变）
        radial_pen = getattr(getattr(net, 'remote_agg', None), 'last_radial_pen', None)
        if radial_pen is not None:
            loss = loss + 0.01 * radial_pen
            radial_pen_sum += radial_pen.item()
            radial_steps += 1
        # NaN/Inf 防护：loss 异常时跳过该 step（双曲运算数值风险兜底）
        # 累积模式下连带丢弃本窗口已累积的梯度（宁可少更新也不带入脏梯度）
        if not torch.isfinite(loss):
            print(f"[warn] step {batch_idx}: non-finite loss {loss.item()}, skipped")
            nan_skipped += 1
            optimizer.zero_grad()
            continue
        # 梯度累积：loss 除以 N，使 N 个 batch 的平均梯度等效一个大 batch
        (loss / accum).backward()
        total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        if not torch.isfinite(total_norm):
            print(f"[warn] step {batch_idx}: non-finite grad norm {total_norm}, skipped")
            nan_skipped += 1
            optimizer.zero_grad()
            continue
        # 每 N 步（或 epoch 最后一个不满 N 的尾窗口）才更新参数
        if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == num_batches:
            optimizer.step()
            # Codex P0-2 修复（2026-10-17）：EMA 每 optimizer.step 更新
            # （step 粒度，timm 语义）——原 epoch 末一次 + 0.999 的窗
            # 693 次更新≈693 epoch，150ep 只追 13.9%，BN stats 陈旧。
            if ema is not None:
                ema.update(net)

        train_loss += loss.item()
        done_steps += 1
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        train_true.append(labels.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    acc = 100. * correct / total
    avg_loss = train_loss / max(done_steps, 1)
    elapsed = time.time() - start_time
    if nan_skipped > 0:
        print(f"[warn] epoch skipped {nan_skipped} non-finite steps")

    # G16 profiling 收尾：epoch 末 flush（agent append 模式跨 epoch 续写）
    if _prof_on and _prof_buf:
        try:
            profile_to_h5(_prof_buf, _prof_path)
            _prof_buf.clear()
        except Exception as _e:
            log_string(f'[g16_profile] 收尾写盘异常: {str(_e)[:120]}')

    return {"loss": avg_loss, "acc": acc, "time": elapsed, "nan_skipped": nan_skipped,
            "true": train_true, "pred": train_pred,
            "radial_pen": (radial_pen_sum / radial_steps) if radial_steps > 0 else None}


def validate(net, testloader, criterion, device, mark_test, args):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (points, labels) in enumerate(testloader):
            points, labels = points.to(device), labels.to(device).long()
            points = points.permute(0, 2, 1).float()

            outputs = net(points)
            loss = criterion(outputs, labels)

            # G16 profiling: validate 不采集（Bernoulli 只在 train 侧），但
            # LRHyperNet.forward 每次都会往全局 sink extend 本 batch 的 3 条
            # ——不清的话 test loader ~327 batch × 每条 [B,N,4]+feat 张量
            # = GB 级泄漏（Codex 第三轮未覆盖的场景：他的实测是单次 forward）。
            # 与 train() 同模式：forward 后立即清。
            if hasattr(net, '_g16_stage_selections'):
                net._g16_stage_selections.clear()

            test_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
            test_true.append(labels.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    acc = 100. * correct / total
    avg_loss = test_loss / (batch_idx + 1)
    elapsed = time.time() - start_time

    # 序列准确率（按 mark 分组，多数投票）
    if mark_test is not None:
        mark_test = np.array(mark_test).flatten()
        unique_marks = np.unique(mark_test)
        seq_correct = 0
        for m in unique_marks:
            mask = mark_test == m
            if mask.sum() == 0:
                continue
            preds_m = test_pred[mask]
            true_m = test_true[mask]
            # 多数投票
            majority = np.bincount(preds_m).argmax()
            if majority == true_m[0]:
                seq_correct += 1
        test_acc_seq = seq_correct / len(unique_marks)
    else:
        test_acc_seq = 0.0

    return {"loss": avg_loss, "acc": acc, "time": elapsed,
            "test_acc_seq": test_acc_seq, "true": test_true, "pred": test_pred}


def save_model(net, epoch, path, acc, is_best, **kwargs):
    """保存 last/best checkpoint（含 optimizer/scheduler/best 指标，支持断点续训）。

    Codex 第三轮补充#1: kwargs 由调用方传入 args=vars(args)——resume 时
    校验 g16_treatment 等关键键，防止续跑时误传 treatment（六处理权重
    形状完全相同，误传会静默变成另一个实验）。
    """
    state = {
        'epoch': epoch,
        'net': net.state_dict(),
        'acc': acc,
        **kwargs
    }
    filepath = os.path.join(path, 'last_checkpoint.pt')
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(path, 'best_checkpoint.pt')
        torch.save(state, best_filepath)


class ModelEMA:
    """EMA 权重副本（与 train_frequency_ncaltech.py 同一实现，2026-10-17 接线）。

    --ema 0.999 启用（默认 0 关闭）。checkpoint 存 net_ema 键；best/last
    判定不变（不污染六处理对照），EMA 是并行评估出口。
    """

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
            else:
                s.copy_(v)

    def state_dict(self):
        return self.shadow


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''SET SEED'''
    seed = getattr(args, 'seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('lr_hypernet')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    # tensorboard log_name 默认从 log_dir 派生，避免不同实验曲线混叠
    if args.log_name is None:
        args.log_name = '/' + (args.log_dir if args.log_dir else 'lr_hypernet')
    writer = SummaryWriter(args.log_path + args.log_name)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/lr_hypernet.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    writer.add_text('config', 'lr_hypernet: remote_g=%d, curvature_init=%s, time_enc=%s, kda=%s, kda_bidirectional=%s, use_global_anchor=%s, anchor_lambda=%s, hyperbolic_type=%s' % (
        args.remote_g, args.curvature_init, args.time_enc, args.kda, args.kda_bidirectional, args.use_global_anchor, args.anchor_lambda, args.hyperbolic_type))

    '''DATA LOADING'''
    log_string('Load dataset ...')
    TRAIN_FILES = [args.data_path + "train.h5"]
    TEST_FILES = [args.data_path + "test.h5"]
    current_data_test, current_label_test, current_mark_test = provider_data.load_h5_mark(TEST_FILES[0])
    print("test", len(current_data_test), current_data_test.shape)
    current_label_test = np.squeeze(current_label_test)
    current_data_test = torch.from_numpy(current_data_test)
    current_label_test = torch.from_numpy(current_label_test.astype('int64'))
    current_data_test = current_data_test.reshape(-1, args.num_point, 4)
    # 当 num_point 小于原始点数时，reshape 会把每个样本拆成多个片段，
    # 标签/mark 需相应重复对齐（与 train_hfecnet.py 一致）
    if current_data_test.shape[0] != current_label_test.shape[0]:
        n_seg = current_data_test.shape[0]
        rep = n_seg // current_label_test.shape[0]
        rem = n_seg % current_label_test.shape[0]
        # repeat_interleave(rep) 同样有余数截断：尾部点数不足 num_point 的
        # 片段归属最后一个原始样本，补 label[-1]，与下方 mark 对齐逻辑一致
        current_label_test = torch.cat([
            current_label_test.repeat_interleave(rep),
            current_label_test[-1:].repeat(rem),
        ])
    if current_data_test.shape[0] != len(current_mark_test):
        n_seg = current_data_test.shape[0]
        rep = n_seg // len(current_mark_test)
        rem = n_seg % len(current_mark_test)
        # rep 重复后若有余数（尾部样本点数不足 num_point 的片段），
        # 余下片段归属最后一个原始样本，补 mark[-1]，避免 mark 长度不足/错位
        current_mark_test = np.concatenate([
            np.repeat(current_mark_test, rep),
            np.repeat(current_mark_test[-1:], rem),
        ])
    # 任务 B：5 类子集过滤（label<5）。在 reshape/label 对齐之后过滤，
    # 保证 data/label/mark 三者按同一 mask 筛选（对重复展开的样本同样成立）
    if args.sanity_5class:
        labels_np = current_label_test.numpy() if isinstance(current_label_test, torch.Tensor) \
            else np.asarray(current_label_test)
        mask_test = labels_np < 5
        current_data_test = current_data_test[mask_test]
        current_label_test = current_label_test[mask_test]
        current_mark_test = np.asarray(current_mark_test)[mask_test]
        print(f"[sanity_5class] test subset: {int(mask_test.sum())} samples")
    dataset_test = torch.utils.data.TensorDataset(current_data_test, current_label_test)
    testDataLoader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=0, drop_last=False)
    current_data_sum, current_label_sum, current_mark_sum = provider_data.load_h5_mark(TRAIN_FILES[0])
    current_label_sum = np.squeeze(current_label_sum)
    print("train", current_data_sum.shape, current_label_sum.shape)
    current_data_sum = torch.from_numpy(current_data_sum)
    current_label_sum = torch.from_numpy(current_label_sum.astype('int64'))
    current_data_sum = current_data_sum.reshape(-1, args.num_point, 4)
    if current_data_sum.shape[0] != current_label_sum.shape[0]:
        n_seg = current_data_sum.shape[0]
        rep = n_seg // current_label_sum.shape[0]
        rem = n_seg % current_label_sum.shape[0]
        # 同 test：repeat_interleave(rep) 的余数截断需补 label[-1]，
        # 避免尾部片段 label 长度不足导致 TensorDataset 对齐错位
        current_label_sum = torch.cat([
            current_label_sum.repeat_interleave(rep),
            current_label_sum[-1:].repeat(rem),
        ])
    # 任务 B：训练集同样按 label<5 过滤，并把类数/轮数限制为 sanity 协议
    if args.sanity_5class:
        labels_np = current_label_sum.numpy() if isinstance(current_label_sum, torch.Tensor) \
            else np.asarray(current_label_sum)
        mask_train = labels_np < 5
        current_data_sum = current_data_sum[mask_train]
        current_label_sum = current_label_sum[mask_train]
        print(f"[sanity_5class] train subset: {int(mask_train.sum())} samples")
        # 5 类子集协议：类数 5、最多 50 epoch（model 构造在数据加载之后，读取改后的值）
        args.num_category = 5
        args.epoch = min(args.epoch, 50)
    # 数据增强（--augment N）：每个训练样本再生成 N 个增强版本（随机crop/时间反转/极性翻转/抖动/丢点）
    if args.augment > 0:
        from augment_data import augment_train_data
        # Codex P0-1 补充：augment 分支前的 from_numpy+reshape 路径已把
        # current_data_sum 转为 Tensor；np.asarray(Tensor) 是零拷贝视图
        # （共享同块内存），不再产生 float32 副本。输出 dtype=输入 dtype。
        data_np = np.asarray(current_data_sum)
        label_np = np.asarray(current_label_sum)
        mark_np = np.asarray(current_mark_sum)
        current_data_sum, current_label_sum, current_mark_sum = augment_train_data(
            data_np, label_np, mark_np, n_versions=args.augment)
        # Codex P0-1 修复（2026-10-17）：去掉 .astype('float32') 强制拷贝——
        # train.h5 实测已为 float32，astype 会复制整份增强后数据（30.8GB），
        # 叠加输入+输出后超 64G（177163 OUT_OF_MEMORY 实证，MaxRSS 56.3GiB）。
        # augment_train_data 输出 dtype=data.dtype（float32 in float32 out），
        # from_numpy 零拷贝共享内存。
        current_data_sum = torch.from_numpy(current_data_sum)
        current_label_sum = torch.from_numpy(current_label_sum.astype('int64'))
        log_string('Data augmentation: on (%d versions per sample, train=%d)' % (
            args.augment, current_data_sum.shape[0]))
    dataset = torch.utils.data.TensorDataset(current_data_sum, current_label_sum)
    if args.balanced_sampler:
        # 类均衡采样（--balanced_sampler）：权重 1/n_c，num_samples=len(dataset)，
        # 与原 shuffle+drop_last 的每 epoch iter 数（len//batch）保持一致
        sampler = make_balanced_sampler(current_label_sum, args.num_category)
        trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                      sampler=sampler,
                                                      num_workers=0, drop_last=True)
        log_string('Balanced sampler: on (WeightedRandomSampler, 1/n_c weights, '
                   'num_samples=%d)' % len(dataset))
    else:
        trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                                      num_workers=0, drop_last=True)

    '''MODEL LOADING'''
    from lr_hypernet import LRHyperNet
    # Codex 第四轮 P1-1: g16_rng_seed 是 args.seed 的派生值（非 CLI 参数），
    # 写回 args —— vars(args) 进 checkpoint 后 resume 校验才有真实值可比
    # （原版校验一个 checkpoint 里永远不存在的键，改 --seed 不会被拦截）。
    args.g16_rng_seed = int(getattr(args, 'seed', 42))
    classifier = LRHyperNet(num_classes=args.num_category,
                            remote_g=args.remote_g,
                            curvature_init=args.curvature_init,
                            time_enc=args.time_enc,
                            use_global_anchor=args.use_global_anchor,
                            anchor_lambda=args.anchor_lambda,
                            kda=args.kda,
                            kda_bidirectional=args.kda_bidirectional,
                            hyperbolic_type=args.hyperbolic_type,
                            hyperbolic_classifier=args.hyperbolic_classifier,
                            use_group_norm=args.group_norm,
                            use_hyperbolic_selector=args.use_hyperbolic_selector,
                            remote_dropout=args.remote_dropout,
                            use_rp1_selector=args.use_rp1_selector,
                            rp1_lambda=args.rp1_lambda,
                            selector_mode=args.selector_mode,
                            # G16 六处理分派（P0-4: CLI 到模型）
                            g16_treatment=str(getattr(args, 'g16_treatment', '') or ''),
                            g16_rng_seed=int(getattr(args, 'seed', 42)))
    log_string('Model: LRHyperNet (remote_g=%d, curvature_init=%s, time_enc=%s, kda=%s, '
               'kda_bidirectional=%s, use_global_anchor=%s, anchor_lambda=%s, hyperbolic_type=%s, '
               'use_rp1_selector=%s, rp1_lambda=%s, selector_mode=%s)' % (
                   args.remote_g, args.curvature_init, args.time_enc, args.kda,
                   args.kda_bidirectional, args.use_global_anchor, args.anchor_lambda,
                   args.hyperbolic_type, args.use_rp1_selector, args.rp1_lambda,
                   args.selector_mode))
    criterion = cal_loss
    classifier.apply(inplace_relu)
    device = 'cpu' if args.use_cpu else 'cuda'
    classifier = classifier.to(device)

    # logit-adjusted CE（--logit_adj）：训练损失改为 CE(logits + tau*log(prior))
    # 仅作用于训练损失，验证仍用原始 logits 算 acc/loss（Menon et al. 2021
    # logit-adjusted loss 的标准用法：训练时校正类先验，推理时不加偏置）
    train_criterion = cal_loss
    if args.logit_adj > 0:
        prior = compute_class_prior(current_label_sum, args.num_category)
        log_prior = torch.from_numpy(np.log(prior + 1e-12)).float().to(device)
        tau = args.logit_adj

        def logit_adj_criterion(pred, gold):
            return cal_loss(pred - tau * log_prior, gold)

        train_criterion = logit_adj_criterion
        log_string('Logit-adjusted CE: on (tau=%s)' % args.logit_adj)

    # 统计参数量（验收标准：不超过 3.5M）
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    log_string(f'Total params: {total_params:,}')
    log_string(f'Trainable params: {trainable_params:,}')
    remote_params = sum(p.numel() for n, p in classifier.named_parameters() if 'hyper_agg' in n)
    log_string(f'Remote-hyperbolic params: {remote_params:,} ({100 * remote_params / total_params:.2f}%)')

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate,
                                    momentum=0.9, weight_decay=args.decay_rate)
    elif args.optimizer == "AdamW":
        # v3.1 曲率优化三件套之一：曲率参数单独学习率（×0.01），防止曲率异常增大
        # KDA 24 步递推梯度链放大：单独给 1/10 学习率
        # euclidean 消融对照：可学习温度与曲率同为"每聚合器 1 个标量超参"，
        # 用同一单独参数组（lr×0.01、无 weight decay）保证两种距离度量的标量
        # 训练协议对称（否则 temperature 落入 base_params 会被 weight decay
        # 拉向 0，softmax 饱和成 one-hot，欧氏基线被不公平削弱）。
        # 双曲版的 temperature 是普通浮点属性而非 Parameter，此匹配只可能
        # 命中 euclidean 版，不影响已有 poincare/lorentz 训练路径。
        kda_params, curvature_params, base_params = [], [], []
        for n, p in classifier.named_parameters():
            if 'kda_enc' in n:
                kda_params.append(p)
            elif 'curvature' in n or 'hyper_agg.temperature' in n:
                curvature_params.append(p)
            else:
                base_params.append(p)
        optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': args.learning_rate, 'weight_decay': args.decay_rate},
            {'params': kda_params, 'lr': args.learning_rate * 0.1, 'weight_decay': args.decay_rate},
            {'params': curvature_params, 'lr': args.learning_rate * 0.01, 'weight_decay': 0.0},
        ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)

    # 断点续训：文件不存在/损坏才允许从零；权重形状不匹配必须报错（防止静默降级）
    start_epoch = 0
    best_test_acc = 0.
    best_test_seq = 0.
    ckpt_path = os.path.join(str(checkpoints_dir), 'last_checkpoint.pt')
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        log_string('No existing model at %s (%s), starting from scratch' % (ckpt_path, e))
        checkpoint = None
    if checkpoint is not None:
        start_epoch = checkpoint['epoch'] + 1
        classifier.load_state_dict(checkpoint['net'])  # strict=True，配置不匹配直接报错
        # Codex 第三轮补充#1 + 第四轮 P1-1: resume 一致性校验——六处理的
        # 权重形状完全相同，误传 treatment 会静默变成另一个实验，必须在
        # 权重加载之外显式拦截。seed 直接校验源头（g16_rng_seed 在模型
        # 构造前已派生写回 args，vars(args) 里有真实值，两者都能对上）。
        _ckpt_args = checkpoint.get('args') or {}
        _resume_keys = ('g16_treatment', 'seed', 'g16_rng_seed', 'selector_mode',
                        'hyperbolic_type', 'use_rp1_selector', 'use_hyperbolic_selector',
                        # Codex P1 修复（2026-10-17）：augment/ema 影响训练数据量
                        # 与 EMA 语义，接力误改会静默换实验
                        'augment', 'ema')
        for _k in _resume_keys:
            _ckpt_v = _ckpt_args.get(_k)
            _cur_v = getattr(args, _k, None)
            if _ckpt_v is not None and _ckpt_v != _cur_v:
                # 不一致 → 硬错误（不是警告）：错误口径的续跑数据全部作废
                raise ValueError(
                    f"resume 配置不一致: checkpoint['args']['{_k}']={_ckpt_v!r} "
                    f"!= 当前 args.{_k}={_cur_v!r}——六处理权重形状相同，静默续跑"
                    f"会变成另一个实验。请用与原训练一致的参数续跑，或换 log_dir。")
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'best_test_acc' in checkpoint:
            best_test_acc = checkpoint['best_test_acc']
        if 'best_test_seq' in checkpoint:
            best_test_seq = checkpoint['best_test_seq']
        log_string('Resume from %s (epoch %d)' % (ckpt_path, start_epoch))

    '''TRANING'''
    logger.info('Start training...')
    # G16 profiling 的实验目录（Codex 第四轮记录#3）：log_dir=None 时训练
    # 落在时间戳目录，profile 跟随同一目录（否则会落到 'default'）
    train._exp_subdir = args.log_dir if args.log_dir else timestr
    # v3.1 曲率优化三件套之三：前 curvature_warmup_epochs 轮固定曲率，之后放开学习
    curvature_warmup_epochs = 10
    # EMA 权重副本（--ema 0.999 启用；0=完全跳过，行为与原版一致）
    # Codex P0-2 修复（2026-10-17）：参数域校验 + resume 恢复 shadow
    # + update 已移 train() 内 step 粒度。旧 checkpoint 无 net_ema 键时
    # 从当前权重初始化（不阻断兼容旧 ckpt 的 KDA 式接力）。
    if args.ema > 0 and not (0.0 < args.ema < 1.0):
        raise ValueError(f'--ema 须 (0,1) 开区间，收到 {args.ema}')
    ema = ModelEMA(classifier, args.ema) if getattr(args, 'ema', 0) > 0 else None
    if ema is not None:
        _prev_ema = None
        _prev_scheme = None
        try:
            # resume 恢复：与 optimizer 一样从 last_checkpoint 读 shadow
            _ck = torch.load(os.path.join(str(checkpoints_dir), 'last_checkpoint.pt'),
                             map_location='cpu')
            _prev_ema = _ck.get('net_ema')
            _prev_scheme = _ck.get('ema_scheme')
        except Exception:
            _prev_ema = None  # 无 checkpoint（从零训练）或读失败（如损坏由 resume 段拦截）
        if _prev_ema and _prev_scheme == 'step':
            # P0-1 修复（Codex 复审）：CPU 加载的 shadow 搬到模型设备
            #（否则首次 update 设备不匹配崩）。P1-3：只有 step 方案的
            # shadow 才恢复（旧 epoch 方案的 BN stats 已污染）。
            _target = next(ema.module.parameters()).device
            ema.shadow = {k: v.clone().to(_target)
                          for k, v in _prev_ema.items()}
            log_string(f'EMA: on (decay={args.ema}), shadow resumed from '
                       f'checkpoint (moved to {_target})')
        else:
            log_string(f'EMA: on (decay={args.ema}), shadow initialized from '
                       f'current weights')
    for epoch in range(start_epoch, args.epoch):
        # 曲率 warmup：前 N 轮固定曲率（requires_grad=False），第 N+1 轮起放开
        if epoch == curvature_warmup_epochs:
            for name, p in classifier.named_parameters():
                if 'curvature' in name:
                    p.requires_grad_(True)
            log_string(f'Curvature warmup ended at epoch {epoch+1}, curvature now trainable')
        if epoch < curvature_warmup_epochs and start_epoch < curvature_warmup_epochs:
            for name, p in classifier.named_parameters():
                if 'curvature' in name:
                    p.requires_grad_(False)

        log_string('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, args.epoch))
        classifier = classifier.train()
        train._cur_epoch = epoch  # G16 profiling 元数据（epoch 传入）
        train_out = train(classifier, trainDataLoader, optimizer, train_criterion, device, args,
                          curvature_init=args.curvature_init,
                          curvature_warmup_epochs=curvature_warmup_epochs,
                          ema=ema)
        scheduler.step()
        writer.add_scalar('Loss/train', train_out["loss"], epoch)
        writer.add_scalar('Accuracy/train', train_out["acc"], epoch)
        log_string('Train Accuracy: %f' % (train_out["acc"]))
        print(f"Training loss:{train_out['loss']} acc:{train_out['acc']}% time:{train_out['time']}s")
        if train_out.get("radial_pen") is not None:
            writer.add_scalar('RadialPenalty/train', train_out["radial_pen"], epoch)
            print(f"Radial Penalty: {train_out['radial_pen']}")

        test_out = validate(classifier, testDataLoader, criterion, device, current_mark_test, args)
        # best 按 test_acc_seq（视频级多数投票）判定，与基线一致
        if test_out["test_acc_seq"] > best_test_seq:
            best_test_seq = test_out["test_acc_seq"]
            is_best = True
        else:
            is_best = False
        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        writer.add_scalar('Accuracy/test', test_out["acc"], epoch)
        writer.add_scalar('Accuracy/test_seq', test_out["test_acc_seq"], epoch)
        writer.add_scalar('Loss/test', test_out["loss"], epoch)

        # 记录各 stage 双曲曲率值（euclidean 消融版记录可学习温度）
        for name, p in classifier.named_parameters():
            if "curvature" in name:
                writer.add_scalar(f'Curvature/{name}', p.item(), epoch)
            elif "hyper_agg.temperature" in name:
                writer.add_scalar(f'Temperature/{name}', p.item(), epoch)

        save_model(classifier, epoch, path=str(checkpoints_dir), acc=test_out["acc"], is_best=is_best,
                   best_test_acc=best_test_acc,
                   best_test_seq=best_test_seq,
                   optimizer=optimizer.state_dict(),
                   scheduler=scheduler.state_dict(),
                   net_ema=(ema.state_dict() if ema is not None else None),
                   # P1-3 判据：只有 step 方案的 shadow 才在 resume 恢复
                   ema_scheme=('step' if ema is not None else None),
                   args=vars(args))
        print(f"Testing loss:{test_out['loss']}  acc:{test_out['acc']}% time:{test_out['time']}s "
              f"[best test acc: {best_test_acc}%] [best test seq acc: {best_test_seq * 100.}%]\n\n")
        log_string('Test Accuracy: %f ' % (test_out["acc"]))
        log_string('Test Accuracy seq: %f' % (test_out["test_acc_seq"]))
        # 任务 B：sanity 早停——train acc>99% 即容量验证通过，无需跑满 50 epoch
        if args.sanity_5class and train_out["acc"] > 99.0:
            log_string('Sanity check passed: train acc %.2f%% > 99%% at epoch %d, early stop'
                       % (train_out["acc"], epoch + 1))
            break

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)

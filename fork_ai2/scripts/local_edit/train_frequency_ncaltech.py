import os
import sys
import torch
import numpy as np
import datetime
import logging
import shutil
import argparse
import provider_data
import time
import torch.nn.functional as F
from pathlib import Path
import sklearn.metrics as metrics
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# H100 cuDNN 兼容性修复：禁用 benchmark 避免 CUDNN_STATUS_INTERNAL_ERROR
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# GPU 选择：默认映射到可见的第一块（无 SLURM/GPU 绑定时 = 卡0）。
# lease/Slurm 环境已通过 CUDA_VISIBLE_DEVICES 只暴露租约内的卡，
# 这里再设 "0" 会覆盖错卡——仅在外部未设置时才兜底设 0。
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--train_method', type=str, default='others', help='DDP DP')
    parser.add_argument('--rank_dp', type=list, default=[0], help='rank')
    parser.add_argument('--batch_size', type=int, default = 80, help='batch size in training')
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet2_cls_ssg]')
    parser.add_argument('--num_category', default=101, type=int, choices=[10, 40, 101],  help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=150, type=int, help='number of epoch in training')
    # Codex 2026-10-15 P0 修复：链式接力用——本段跑到 stop_epoch 后正常退出
    # （exit 0），使 afterok 依赖保持有效（TIMEOUT 不触发 afterok，靠超时被杀
    # 会导致下一段永远 pending）。0 = 不启用（单作业跑满 args.epoch）。
    parser.add_argument('--stop_epoch', default=0, type=int,
                        help='>0: 本段训练到此 epoch 正常退出（链式接力用），0=跑满')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument("--data_path", type=str, default='./data/', help="path to eyetracking_log")
    parser.add_argument("--log_path", type=str, default='./tensorboard_log/', help="path to tesnorboard_log")
    parser.add_argument("--log_name", type=str, default='/DVSGesture_32', help="path to tesnorboard_log")
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='AdamW,SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    # ===== 组件版本开关（消融实验，默认全关 = 原始 SECNet，模型侧见 models/FECNet.py）=====
    parser.add_argument('--time_enc', action='store_true', default=False,
                        help='多频 sin/cos 时间编码（频率 2^0..2^3 共 8 维）：embed 前 t 通道'
                             '替换为 8 维编码，输入 4→11 维。与 --kda 互斥')
    parser.add_argument('--kda', action='store_true', default=False,
                        help='KDA 因果时间编码：每事件取 (t,x,y) 空间 K=24 近邻按时间做'
                             'gated delta rule 递推，t 通道替换为 16 维 KDA 编码，输入 4→19 维。'
                             '与 --time_enc 互斥')
    parser.add_argument('--spike_time_enc', action='store_true', default=False,
                        help='SpikePoint混合时间编码：保留t作坐标+额外sin/cos通道（版本7）。'
                             '与 --time_enc / --kda 互斥')
    parser.add_argument('--hyperbolic_selector', action='store_true', default=False,
                        help='双曲选点：Poincaré盘等双曲半径网格选点（外圈更密）（版本6）')
    parser.add_argument('--hsel_scaled_coords', action='store_true', default=False,
                        help='双曲网格选点用 fps_parameter 缩放后坐标（复现旧版 bug 行为，'
                             '2026-10-14 ai2）。2×2矩阵"双曲+缩放坐标"象限（seed42 +4.71pp'
                             ' p=0.0002 最强组件）的受控复现开关。需与 --hyperbolic_selector'
                             ' 同开；单独开按 no-op 警告处理')
    parser.add_argument('--remote_g16', action='store_true', default=False,
                        help='16 点远程选择（论文灵魂）：采样层用 LocalRemoteGrouper 替换'
                             'LocalGrouper，remote_g=16。与 --rp1_selector / '
                             '--hyperbolic_selector 互斥。需配合 --g16_treatment 选准则')
    parser.add_argument('--g16_treatment', type=str, default='',
                        choices=['', 'random', 'spatial_fixed', 'euc_far', 'euc_near',
                                 'hyp_far', 'rp1_scorer'],
                        help='G16 六处理选点准则（需与 --remote_g16 同开）：空串=原 topk'
                             '双曲行为；random/spatial_fixed/euc_far/euc_near/hyp_far/'
                             'rp1_scorer（hyp_far=双曲距离选点，学长核心想法）')
    parser.add_argument('--rp1_selector', action='store_true', default=False,
                        help='RP1 spatial-uniform 选点：group center 由 FPS 改为 x,y 空间'
                             '均匀网格采样（只改选点，不动傅里叶聚合）')
    parser.add_argument('--hyperbolic_classifier', action='store_true', default=False,
                        help='双曲原型分类头（Lorentz 模型）：Linear 分类器替换为'
                             '可学习类原型 + 双曲距离分类')
    parser.add_argument('--hyperbolic_fusion', action='store_true', default=False,
                        help='傅里叶+双曲并行融合：傅里叶聚合不动，每 stage 新增轻量'
                             'Lorentz 双曲聚合支路（64 维瓶颈，曲率可学习初始 0.4），'
                             '残差门控融合 F_out = F_fourier + gate*F_hyper（gate 初始≈0，'
                             '训练初期不影响原始性能）。曲率正则 0.01*(c-0.4)^2、'
                             '径向约束正则 0.01*pen、曲率前 10 epoch 冻结后放开'
                             '（lr 0.01x、wd 0）')
    parser.add_argument('--hyperbolic_embedding', action='store_true', default=False,
                        help='双曲嵌入瓶颈层：backbone输出特征(feature_list[-1]*2=1080维)后，'
                             '欧氏→exp_map进双曲(Poincaré球)→双曲空间线性变换+Möbius加偏置→'
                             'log_map回欧氏→残差+LayerNorm，验证双曲空间表示能力')
    parser.add_argument('--hyperbolic_attention', action='store_true', default=False,
                        help='双曲注意力残差支路：每个stage傅里叶聚合后额外加Lorentz双曲'
                             '距离自注意力残差（x + 0.1*x_hyper），不替换原欧氏聚合路径，'
                             '利用双曲空间层级结构特性')
    parser.add_argument('--hyperbolic_mlp', action='store_true', default=False,
                        help='双曲 MLP（P0-2）：双曲注意力输出先进 Poincaré 球做 '
                             'Möbius 矩阵乘（rank=32 低秩零初始化）+ Möbius 偏置加 + '
                             '双曲激活再 log 回欧氏，然后才做 0.1 残差注入。W/b 零初始化，'
                             '初始输出恒为 0。需与 --hyperbolic_attention 同开才有意义，'
                             '单独开时警告并 no-op')
    parser.add_argument('--hyperbolic_gate', action='store_true', default=False,
                        help='双曲距离门控（替换式，共识③ 融合层 gated→双曲加权）：'
                             '替换 gated fusion 的 global 分支（mean pool->group_conv->'
                             'sigmoid 逐通道门控），门控信号改由双曲距离学出——每个'
                             '空间位置（组）的 C 维特征投影到 Lorentz 流形，离全体组'
                             '质心越近门控越大。门控粒度从逐通道改为逐组/逐空间位置')
    parser.add_argument('--fca_gate', action='store_true', default=False,
                        help='频域通道门控（FcaNet 思路，ICCV 2021）：替换 gated fusion'
                             '的 global 分支（mean pool=最低频 DCT 分量），改用前 4 个'
                             '低频 DCT 分量压缩每个通道学门控。保留逐通道门控语义。'
                             '与 --hyperbolic_gate 互斥')
    parser.add_argument('--hyperbolic_radial', action='store_true', default=False,
                        help='曲率感知双曲径向校正（PointCHR 思路，arXiv 2607.24052）：'
                             'embed 后逐点做径向校正——复杂度高的点（离群/精细）投影'
                             '到双曲边界获得更大容量，复杂度低的点留在中心，残差注入')
    parser.add_argument('--polar_feature', action='store_true', default=False,
                        help='方向/半径分离特征层（Polar FC 思路，arXiv 2609.32899）：'
                             'embed 后把特征分解为方向（单位向量）和半径（标量）分别'
                             '可学习映射，半径可独立学。是 --hyperbolic_radial 的欧氏'
                             '对照，两者互斥')
    parser.add_argument('--scale_norm_fps', action='store_true', default=False,
                        help='尺度归一化 FPS（A1 归因发现）：FPS 采样前对 xytp 做 '
                             'per-dimension 标准化 (x-mean)/std，替代可学习 fps_parameter 缩放。'
                             'A1 实验证明 fps_parameter 缩放让 FPS 中心覆盖率 +2.6%，'
                             '这是 bug 版双曲选点 +4.71pp 的真正原因。本组件用 principled '
                             '标准化替代可学习缩放，对事件密度变化更鲁棒')
    parser.add_argument('--curvature_opt', action='store_true', default=False,
                        help='曲率感知优化三件套（P0-1）：所有曲率参数（名字含 '
                             'curvature/c_raw）归入独立 param group（lr=base_lr*0.01、wd=0），'
                             '总 loss 加 L2 正则 0.01*mean((c_actual-0.4)^2) 锚定初始曲率，'
                             '前 10 epoch 曲率组 lr=0 冻结、第 11 epoch 起恢复 base_lr*0.01。'
                             '模型无可学习曲率参数时警告并 no-op')
    parser.add_argument('--event_mamba_ssm', action='store_true', default=False,
                        help='EventMamba风格SSM：只在stage3（最后一级）做状态空间建模'
                             '（轻量Mamba，S=512控制开销），小权重残差 x + 0.1*x_ssm')
    parser.add_argument('--group_norm', action='store_true', default=False,
                        help='全网 BatchNorm1d -> GroupNorm（b8 小 batch 下 BN 统计量噪声大，'
                             'GroupNorm 与 batch size 无关）')
    parser.add_argument('--grad_accum', type=int, default=0,
                        help='梯度累积步数 N（b8 想等效 b32 用 4：loss/N，每 N 步 '
                             'optimizer.step() 一次；0/1 = 关闭）')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for reproducibility')
    parser.add_argument('--freeze_fps_parameter', action='store_true', default=False,
                        help='冻结所有 local_grouper_list.*.fps_parameter（requires_grad=False，'
                             '保持初始 1.0）。A1归因修正对照实验：SECNet 官方原生带可学习'
                             '逐维缩放（基线训练后漂移到 x~1.64x），本 flag 量化这个'
                             '隐藏组件对基线的独立贡献。')
    parser.add_argument('--augment', type=int, default=0,
                        help='训练数据增强版本数（0=关闭，行为与原版逐字节一致；'
                             'N>0=每个训练样本再生成 N 个增强版本（随机crop/时间反转/'
                             '极性翻转/时间抖动/丢点，数据×(1+N)），test 不增强。'
                             '与 train_lr_hyper.py 的 --augment 同一实现（augment_data.py）。'
                             '2026-10-16 用户决策：后续冲刺链均开增强。')
    parser.add_argument('--ema', type=float, default=0.0,
                        help='EMA 权重副本的 decay（0=关闭；0.999=典型值）。训练中'
                             '维护指数滑动平均权重，checkpoint 额外存 net_ema 键。'
                             'best/last 判定仍用非 EMA 权重（不污染组件对照）；EMA '
                             '作为并行出口，推理时按需选用（通常 +0.5~1.5pp，小数据'
                             '集更稳）。2026-10-17 用户决策：跑之前的工作做到极致。')
    parser.add_argument('--patience', type=int, default=0,
                        help='收敛早停（2026-10-17 用户决策：小样本快测协议——'
                             '"满轮数=只要收敛就立刻停"）。>0 时连续 N 个 epoch '
                             'best_test_seq 无提升则正常退出（checkpoint 已保，'
                             '配对差用 best）。0=关闭（跑满 args.epoch，全量默认）。'
                             '判定口径与 best 一致（test_acc_seq），避免双口径。')
    parser.add_argument('--early_stop_metric', type=str, default='seq',
                        choices=['seq', 'acc'],
                        help='早停判据：seq=test_acc_seq（默认，与 best 判定一致）；'
                             'acc=test acc。仅 --patience>0 时生效。')
    return parser.parse_args()

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def validate(net, testloader, criterion, device,mark_test,args):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        label_seq = [[],[]]
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            # print(data.shape)
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()

            ######calculate the accuracy of all sequence######
            label_seq[0] = list([j for i in test_pred for j in i])
            label_seq[1] = list([j for i in test_true for j in i])
            from collections import Counter
            count = 0
            correct_seq= 0
            index = 0
            mark = mark_test if mark_test is not None else label_seq[1]
            n_seq = len(label_seq[1])
            # 修复G：原 range(n-2) 漏掉最后一个样本（off-by-one，末样本永不
            # 参与投票，且条件里的 i==len-2 是死代码）。改为 range(n)：
            # mark 变化处分组（同一 mark 的样本归一组），最后一个样本强制
            # 收尾闭合末组，所有样本都参与投票
            for i in range(n_seq):
                #### if mark is different, we run this code###
                if (i == n_seq - 1) or (mark[i] != mark[i+1]):
                    ####statistic the most common label in the sequence####
                    tar = Counter(label_seq[0][index:i+1])
                    tar = tar.most_common(1)[0][0]
                    ####if the most common label is equal to the label of the sequence, we count it####
                    correct_seq += 1 if tar==label_seq[1][i] else 0
                    index = i+1
                    count +=1
            count = -1 if count == 0 else count
            if args.train_method =="DDP":
                if dist.get_rank() == 0:
                    progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc_seq: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total,correct_seq/count*100,correct_seq,count))
            else:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc_seq: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total,correct_seq/count*100,correct_seq,count))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "time": time_cost,
        "test_acc_seq":correct_seq/count
    }

def is_curvature_param(name):
    """P0-1 --curvature_opt：曲率参数判定（名字含 curvature 或 c_raw）。"""
    return 'curvature' in name or 'c_raw' in name


def actual_curvature(name, p):
    """曲率参数的"实际曲率"（P0-1 正则与日志共用同一口径）。

    按各模块的真实参数化取值：
    - softplus 族（hyper_attns / hyper_embed，P0-2 HyperbolicMLP 同款）：c = softplus(p)+1e-4
    - abs 族（hyper_fusion_list / 双曲原型分类头 classifier）：c = |p|+1e-4
    spec 公式 L_c=0.01*mean((softplus(raw_c)-0.4)^2) 的意图是惩罚"实际曲率"
    偏离 0.4；对 abs 族参数套 softplus 会把正则拉向错误方向，故按实际语义取值。
    """
    if 'hyper_fusion_list' in name or 'classifier' in name:
        return p.abs() + 1e-4
    return F.softplus(p) + 1e-4


def train(net, trainloader, optimizer, criterion, device,args, ema=None):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    # 梯度累积步数：0/1 视为关闭（不加 flag 时与原逻辑逐字节一致）
    accum = args.grad_accum if getattr(args, 'grad_accum', 0) and args.grad_accum > 1 else 1
    num_batches = len(trainloader)
    for batch_idx, (data, label) in enumerate(trainloader):
        data = data.reshape(args.batch_size,args.num_point,4)
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        # 累积窗口的第一个 batch 前（含每个 epoch 开头）清零梯度
        if (batch_idx % accum) == 0:
            optimizer.zero_grad()
        logits = net(data)
        loss = criterion(logits, label)
        # 版本5 --hyperbolic_fusion：曲率正则（0.01*(c-0.4)^2，锚定初始曲率）
        # + Lorentz 径向约束正则（0.01*pen，forward 时缓存在各分支的
        # last_radial_pen）。关闭 flag 时此项为 0，与原逻辑逐字节一致
        if getattr(args, 'hyperbolic_fusion', False):
            core = net.module if hasattr(net, 'module') else net
            curvature_reg = torch.tensor(0.0, device=device)
            # 限定 hyper_fusion_list 前缀：多 flag 同开时不影响
            # hyper_embed / hyper_attns 等其他模块的曲率参数
            for n, p in core.named_parameters():
                if n.startswith('hyper_fusion_list') and 'curvature' in n:
                    curvature_reg = curvature_reg + (p - 0.4) ** 2
            loss = loss + 0.01 * curvature_reg
            radial_pens = [m.last_radial_pen for m in core.hyper_fusion_list
                           if m.last_radial_pen is not None]
            if radial_pens:
                loss = loss + 0.01 * torch.stack(radial_pens).mean()
        # P0-1 --curvature_opt：曲率 L2 正则 0.01*mean((c_actual-0.4)^2)，
        # 锚定初始曲率 0.4（c_actual 口径见 actual_curvature）。关闭 flag 时
        # 不进入此分支，与原逻辑逐字节一致
        if getattr(args, 'curvature_opt', False):
            core = net.module if hasattr(net, 'module') else net
            reg_terms = [((actual_curvature(n, p) - 0.4) ** 2).mean()
                         for n, p in core.named_parameters()
                         if is_curvature_param(n)]
            if reg_terms:
                loss = loss + 0.01 * torch.stack(reg_terms).mean()
        # 梯度累积：loss 除以 N，使 N 个 batch 的平均梯度等效一个大 batch
        (loss / accum).backward()
        # 每 N 步（或 epoch 最后一个不满 N 的尾窗口）才更新参数
        if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == num_batches:
            optimizer.step()
            # Codex P0-2 修复（2026-10-17）：EMA 改每 optimizer.step 更新
            # （timm 粒度）——原 epoch 末一次 + decay 0.999 的有效窗
            # 693 次更新 ≈ 693 epoch，150ep 链只追 13.9%，BN stats
            # 严重陈旧（Codex 实测 ckpt：raw running_var 均值 75 vs EMA
            # 1.20）。step 粒度下 0.999 窗 = 1000 步 ≈ 半个 epoch ✓。
            if ema is not None:
                ema.update(net)
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]
        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())
        total += label.size(0)
        correct += preds.eq(label).sum().item()
        if args.train_method =="DDP":
            if dist.get_rank() == 0:
                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        else:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "time": time_cost
    }
def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
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

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def save_model(net, epoch, path, acc, is_best, **kwargs):
    state = {
        'net': net.state_dict(),
        'epoch': epoch,
        'acc': acc
    }
    for key, value in kwargs.items():
        state[key] = value
    filepath = os.path.join(path, "last_checkpoint.pth")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(path, 'best_checkpoint.pth'))


class ModelEMA:
    """EMA 权重副本（timm 标准，2026-10-17 接线，--ema 0.999 启用）。

    训练中维护参数的指数滑动平均；评估/保存时用 EMA 权重通常比 last
    更稳（小数据集 + 噪声 batch 下 +0.5~1.5pp，文献与 timm 实践一致）。
    实现自包含：state_dict 深拷贝 + float() 副本（BN running stats 同步
    更新），不动训练路径。checkpoint 里存为 'net_ema' 键（推理工具可
    选用），best/last 语义不变——is_best 仍按非 EMA 的 test 结果判定，
    EMA 作为并行出口，事后按需评估（避免 EMA 改变 best 语义污染对照）。
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
            else:  # num_batches_tracked 等整数计数器直接跟随
                s.copy_(v)

    def state_dict(self):
        return self.shadow

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
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
    writer = SummaryWriter(args.log_path + args.log_name)
    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/pointmlp.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''SEED'''
    import random
    seed = getattr(args, 'seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    '''DATA LOADING'''
    log_string('Load dataset ...')

    TRAIN_FILES = [args.data_path+"train.h5"]
    TEST_FILES = [args.data_path+"test.h5"]
    current_data_test, current_label_test, current_mark_test = provider_data.load_h5_mark(TEST_FILES[0])
    print("test",len(current_data_test),current_data_test.shape)
    current_label_test = np.squeeze(current_label_test)
    current_data_test = torch.from_numpy(current_data_test)
    current_label_test = torch.from_numpy(current_label_test.astype('int64'))
    current_data_test = current_data_test.reshape(-1,args.num_point,4)
    dataset_test = torch.utils.data.TensorDataset(current_data_test, current_label_test)
    testDataLoader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    current_data_sum, current_label_sum,current_mark_sum = provider_data.load_h5_mark(TRAIN_FILES[0])
    current_label_sum = np.squeeze(current_label_sum)
    print("train",current_data_sum.shape,current_label_sum.shape)
    # 数据增强（--augment N，2026-10-16 接线，与 train_lr_hyper.py 同模式）：
    # 每个训练样本再生成 N 个增强版本（数据×(1+N)），test 不增强。
    # N=0 完全跳过（行为与原版逐字节一致）。增强种子 42 固定（augment_data
    # 内部 rng），跨 run 可复现。
    if args.augment > 0:
        from augment_data import augment_train_data
        # augment_train_data 吃 numpy，先在 numpy 域做再转 tensor
        data_np = np.asarray(current_data_sum)
        label_np = np.asarray(current_label_sum)
        mark_np = np.asarray(current_mark_sum)
        data_np, label_np, mark_np = augment_train_data(
            data_np, label_np, mark_np, n_versions=args.augment)
        current_data_sum = torch.from_numpy(data_np)
        current_label_sum = torch.from_numpy(label_np.astype('int64'))
        current_data_sum = current_data_sum.reshape(-1, args.num_point, 4)
        log_string('Data augmentation: on (%d versions per sample, train=%d)' % (
            args.augment, current_data_sum.shape[0]))
        print('[augment] train data: %d -> %d (x%d)' % (
            len(label_np) // (1 + args.augment), current_data_sum.shape[0],
            1 + args.augment))
    else:
        current_data_sum = torch.from_numpy(current_data_sum)
        current_label_sum = torch.from_numpy(current_label_sum.astype('int64'))
        current_data_sum = current_data_sum.reshape(-1,args.num_point,4)
    dataset = torch.utils.data.TensorDataset(current_data_sum, current_label_sum)

    if args.train_method =="DDP":
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])  
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True,sampler = train_sampler)
    else:
        trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    '''MODEL LOADING'''
    num_class = args.num_category
    best_test_acc = 0.  # best test accuracy
    best_test_seq = 0.
    best_train_acc = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")


    from models.FECNet import FECNet
    classifier = FECNet(num_classes=args.num_category,
                        use_group_norm=args.group_norm,
                        use_time_enc=args.time_enc,
                        use_kda=args.kda,
                        use_remote_g16=args.remote_g16,
                        g16_treatment=args.g16_treatment,
                        use_rp1_selector=args.rp1_selector,
                        use_hyperbolic_selector=args.hyperbolic_selector,
                        use_spike_time_enc=args.spike_time_enc,
                        use_hyperbolic_classifier=args.hyperbolic_classifier,
                        use_hyperbolic_fusion=args.hyperbolic_fusion,
                              use_hyperbolic_embedding=args.hyperbolic_embedding,
                              use_hyperbolic_attention=args.hyperbolic_attention,
                              use_hyperbolic_mlp=args.hyperbolic_mlp,
                              use_hyperbolic_gate=args.hyperbolic_gate,
                              use_fca_gate=args.fca_gate,
                              use_hyperbolic_radial=args.hyperbolic_radial,
                              use_polar_feature=args.polar_feature,
                              use_event_mamba_ssm=args.event_mamba_ssm,
                              use_scale_norm_fps=args.scale_norm_fps,
                              use_hsel_scaled_coords=args.hsel_scaled_coords)
    criterion = cal_loss
    classifier.apply(inplace_relu)

    # A1归因修正对照（2026-10-14 ai2）：SECNet 原生可学习缩放冻结为 1.0。
    # 放在 inplace_relu 之后、optimizer 构建之前。注意：冻结参数仍会出现在
    # optimizer 的 param group 里（AdamW 对 grad=None 的参数跳过更新），
    # 效果是保持 1.0 不漂移；训练后需核验 checkpoint 里 fps_parameter 仍为 1.0。
    if getattr(args, 'freeze_fps_parameter', False):
        n_frozen = 0
        for name, p in classifier.named_parameters():
            if 'fps_parameter' in name:
                p.requires_grad_(False)
                n_frozen += 1
        print('[freeze_fps_parameter] 冻结 %d 个 fps_parameter（保持 1.0）' % n_frozen)
        log_string('[freeze_fps_parameter] 冻结 %d 个 fps_parameter（保持 1.0）' % n_frozen)

    if args.train_method =="DDP":
        device = torch.device('cuda', local_rank) 
        classifier = classifier.to(local_rank) 
        classifier = DDP(classifier,device_ids=[local_rank],find_unused_parameters=True)
    elif args.train_method =="DP":
        device = 'cuda'
        classifier = classifier.to('cuda:0')
        classifier = torch.nn.DataParallel(classifier, device_ids=args.rank_dp)
    else:
        device = 'cuda'
        classifier = classifier.cuda()
        

    # P0-1 --curvature_opt：曲率组索引与曲率参数名（None/空 = 未分组/无曲率参数），
    # 供 epoch 循环做 warmup lr 管控与曲率日志使用
    curvature_group_idx = None
    curvature_param_names = []
    if args.optimizer =="SGD":
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    elif args.optimizer =="AdamW":
        if getattr(args, 'curvature_opt', False):
            # 所有曲率参数（名字含 curvature/c_raw，覆盖 hyper_attns/hyper_embed/
            # hyper_fusion_list/classifier）单独一组：lr=0.01x、wd=0（曲率是几何
            # 参数，不该被 weight decay 拖向 0）。该分组是 hyperbolic_fusion 曲率
            # 分组的超集且 lr/wd 一致，两 flag 同开时不冲突
            curvature_params, base_params = [], []
            for n, p in classifier.named_parameters():
                if is_curvature_param(n):
                    curvature_params.append(p)
                    curvature_param_names.append(n)
                else:
                    base_params.append(p)
            if curvature_params:
                optimizer = torch.optim.AdamW(
                    [{'params': base_params},
                     {'params': curvature_params, 'lr': args.learning_rate * 0.01, 'weight_decay': 0.0}],
                    lr=args.learning_rate, weight_decay=args.decay_rate)
                curvature_group_idx = 1
            else:
                # 模型无可学习曲率参数（默认无双曲路径时）：警告并 no-op，
                # 优化器结构与基线完全一致
                log_string('警告: --curvature_opt 开启但模型无可学习曲率参数'
                           '（需 --hyperbolic_attention / --hyperbolic_embedding /'
                           ' --hyperbolic_fusion / --hyperbolic_classifier 路径），'
                           '按 no-op 处理')
                optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
        elif args.hyperbolic_fusion:
            # 版本5：曲率参数单独一组（lr 0.01x、wd 0，参考 train_lr_hyper.py；
            # 曲率是几何参数，不该被 weight decay 拖向 0）
            curvature_params, base_params = [], []
            for n, p in classifier.named_parameters():
                (curvature_params if 'hyper_fusion_list' in n and 'curvature' in n
                 else base_params).append(p)
            optimizer = torch.optim.AdamW(
                [{'params': base_params},
                 {'params': curvature_params, 'lr': args.learning_rate * 0.01, 'weight_decay': 0.0}],
                lr=args.learning_rate, weight_decay=args.decay_rate)
        else:
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)

    '''RESUME FROM CHECKPOINT'''
    # 阶段1：读文件——只有"文件不存在"才允许从零开始；
    # 文件损坏（RuntimeError/EOFError 等）必须报错退出，
    # 否则从零训一个 epoch 就会把唯一的 checkpoint 覆盖掉。
    ckpt_path = os.path.join(str(checkpoints_dir), 'last_checkpoint.pth')
    if not os.path.exists(ckpt_path):
        log_string('未找到 checkpoint，从零开始训练...')
        start_epoch = 0
        checkpoint = None
    else:
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(
                'checkpoint 文件损坏，无法加载：%s\n'
                '（拒绝静默从零训练以避免覆盖唯一 checkpoint，请手动处理 %s）'
                % (str(e), ckpt_path))

    # 阶段2：恢复状态——读到了 checkpoint 但恢复失败（如换了组件 flag
    # 导致 state_dict 键不匹配、optimizer param group 变了）必须报错，
    # 不允许带着"半恢复"的权重静默从头训。
    if checkpoint is not None:
        start_epoch = checkpoint['epoch'] + 1
        # 校验训练配置一致性，防止静默错配resume
        if 'args' in checkpoint and checkpoint['args'] is not None:
            saved_args = checkpoint['args']
            cur_args = vars(args)
            mismatch_keys = []
            for k in ['data_path','num_point','batch_size','epoch','learning_rate',
                      'optimizer','decay_rate','num_category','grad_accum','seed',
                      'time_enc','kda','spike_time_enc','remote_g16','g16_treatment','rp1_selector','hyperbolic_selector',
                      'hyperbolic_classifier','hyperbolic_fusion','hyperbolic_embedding',
                      'hyperbolic_attention','event_mamba_ssm','group_norm',
                      # 2026-10-14 ai2 补：推理工具 FLAG_KEYS 已含此二键，
                      # resume 校验也要覆盖，否则换 flag resume 不报错
                      'hyperbolic_mlp','scale_norm_fps','hsel_scaled_coords',
                      # 2026-10-18：hyperbolic_gate/fca_gate 增删门控模块 + 替换
                      # gated fusion forward 路径，换 flag resume 必须报错
                      'hyperbolic_gate','fca_gate','hyperbolic_radial','polar_feature',
                      # Codex P1 修复（2026-10-17）：augment/ema 影响训练数据量
                      # 与 EMA 语义，KDA 式接力误改会静默换实验，必须校验
                      'augment','ema',
                      # Codex 快测复审（2026-10-17）：patience/early_stop_metric
                      # 决定停止协议，接力误改会静默改变实验终止条件
                      'patience','early_stop_metric']:
                if k in saved_args and k in cur_args and saved_args[k] != cur_args[k]:
                    mismatch_keys.append(f"{k}: saved={saved_args[k]} vs current={cur_args[k]}")
            if mismatch_keys:
                raise ValueError(
                    f"checkpoint 训练配置与当前不一致，拒绝静默resume:\n"
                    + "\n".join(mismatch_keys)
                    + "\n请确认是否要用新配置重跑（换新log_dir），或手动删除checkpoint后重试。"
                )
        classifier.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        if 'best_test_acc' in checkpoint:
            best_test_acc = checkpoint['best_test_acc']
        if 'best_test_seq' in checkpoint:
            best_test_seq = checkpoint['best_test_seq']
        log_string('Resume from checkpoint: epoch %d, best_test_acc %.2f%%, best_test_seq %.4f' % (start_epoch, best_test_acc, best_test_seq))

    global_epoch = 0
    # 版本5 --hyperbolic_fusion：曲率 warmup——前 10 epoch 冻结曲率
    # （requires_grad=False，AdamW 对无梯度参数不更新），之后放开学习
    curvature_warmup_epochs = 10

    '''TRANING'''
    logger.info('Start training...')
    # Codex 2026-10-15 P0 修复：链式接力——到达本段 stop_epoch 时正常 break
    # （保存完 checkpoint 后退出 exit 0，afterok 保持有效）
    seg_stop = int(getattr(args, 'stop_epoch', 0) or 0)
    if seg_stop > 0:
        logger.info('Chain segment: will exit normally at epoch %d (args.epoch=%d)'
                    % (seg_stop, args.epoch))
    '''TRAINING'''
    # EMA 权重副本（--ema 0.999 启用；0=完全跳过，行为与原版一致）
    # Codex P0-2 修复（2026-10-17）：参数域校验 + resume 恢复 shadow
    #（原版 resume 后 EMA 从当前权重重启，长链接力时平均窗清零——
    #  Codex 建议与 optimizer 一样存恢复）。更新已移 train() 内
    #  optimizer.step() 后（step 粒度，timm 语义）。
    if args.ema > 0 and not (0.0 < args.ema < 1.0):
        raise ValueError(f'--ema 须 (0,1) 开区间，收到 {args.ema}')
    ema = ModelEMA(classifier, args.ema) if args.ema > 0 else None
    if ema is not None:
        # P1-3（Codex）：旧 177162 checkpoint 的 net_ema 是 epoch 粒度方案
        # 污染的（BN stats 接近初始值，0.999 需极长洗掉）——ha 链重跑时
        # 从当前权重重建 shadow 而不恢复旧值。新 checkpoint（step 粒度
        # 存的）才恢复。
        _resume_ema = (checkpoint is not None and checkpoint.get('net_ema')
                       and checkpoint.get('ema_scheme') == 'step')
        if _resume_ema:
            # P0-1 修复（Codex 复审）：checkpoint 按 map_location='cpu' 加载，
            # shadow 必须搬到模型所在设备——否则首次 ema.update() 时
            # s(CPU).mul_().add_(v(CUDA)) 设备不匹配直接崩。
            _target = next(ema.module.parameters()).device
            ema.shadow = {k: v.clone().to(_target)
                          for k, v in checkpoint['net_ema'].items()}
            log_string('EMA: on (decay=%g), shadow resumed from checkpoint '
                       '(moved to %s)' % (args.ema, _target))
        else:
            log_string('EMA: on (decay=%g), shadow initialized from current weights'
                       % args.ema)
    for epoch in range(start_epoch, args.epoch):
        if seg_stop > 0 and epoch >= seg_stop:
            log_string('Segment stop_epoch=%d reached (current epoch %d). '
                       'Exiting normally for chain handover.' % (seg_stop, epoch))
            break
        # --patience 早停状态初始化/恢复（首个 epoch 前建立基线）：
        # Codex 快测复审修复：resume 时优先从 checkpoint 恢复早停计数
        #（防丢历史无提升计数导致提前/延后停）；无存的旧 ckpt 用已恢复的
        # best 指标初始化（比 -1 稳妥——至少不会把历史最佳当新高估）。
        if epoch == start_epoch:
            _restored = (checkpoint or {}).get('_patience_state') or {}
            if _restored:
                _best_early_stop = _restored.get('best', -1.0)
                _epochs_no_improve = _restored.get('no_improve', 0)
                log_string('EarlyStop state resumed: best=%.4f no_improve=%d'
                           % (_best_early_stop, _epochs_no_improve))
            else:
                _init_best = best_test_seq if args.early_stop_metric == 'seq' \
                    else best_test_acc
                _best_early_stop = _init_best if _init_best > 0 else -1.0
                _epochs_no_improve = 0
        if args.hyperbolic_fusion:
            if epoch < curvature_warmup_epochs:
                for n, p in classifier.named_parameters():
                    if n.startswith('hyper_fusion_list') and 'curvature' in n:
                        p.requires_grad = False
            elif epoch == curvature_warmup_epochs:
                for n, p in classifier.named_parameters():
                    if n.startswith('hyper_fusion_list') and 'curvature' in n:
                        p.requires_grad = True
                log_string('Curvature warmup ended at epoch %d, curvature now trainable' % (epoch + 1))
        # P0-1 --curvature_opt：曲率组 warmup——前 10 epoch lr=0（冻结曲率），
        # 第 11 epoch 起恢复 base_lr*0.01。放在 epoch 开头是因为
        # scheduler.step() 会在上一 epoch 末把该组 lr 覆写为余弦退火值，
        # 这里每 epoch 重设一次（该组 lr 由本 flag 独占管理，不参与退火）
        if getattr(args, 'curvature_opt', False) and curvature_group_idx is not None:
            if epoch < curvature_warmup_epochs:
                optimizer.param_groups[curvature_group_idx]['lr'] = 0.0
            else:
                optimizer.param_groups[curvature_group_idx]['lr'] = args.learning_rate * 0.01
                if epoch == curvature_warmup_epochs:
                    log_string('curvature_opt: warmup ended at epoch %d, '
                               'curvature group lr restored to %.3e'
                               % (epoch + 1, args.learning_rate * 0.01))
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        classifier = classifier.train()
        train_out = train(classifier, trainDataLoader, optimizer, criterion, device,args, ema=ema)
        scheduler.step()
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss
        writer.add_scalar('Loss/train', train_out["loss"], epoch)
        writer.add_scalar('Accuracy/train', train_out["acc"], epoch)
        log_string('Train Accuracy: %f' % (train_out["acc"]))
        print(f"Training loss:{train_out['loss']} acc:{train_out['acc']}% time:{train_out['time']}s")
        # P0-1 --curvature_opt：每 epoch 打印并记录当前实际曲率值（日志+TensorBoard）
        if getattr(args, 'curvature_opt', False) and curvature_param_names:
            core = classifier.module if hasattr(classifier, 'module') else classifier
            curv_parts = []
            for n in curvature_param_names:
                c_val = actual_curvature(n, core.get_parameter(n)).detach().cpu().item()
                curv_parts.append('%s=%.4f' % (n, c_val))
                writer.add_scalar('Curvature/%s' % n.replace('.', '/'), c_val, epoch)
            log_string('curvature: %s' % ', '.join(curv_parts))
        test_out = validate(classifier, testDataLoader, criterion, device,current_mark_test,args)
        if test_out["test_acc_seq"] > best_test_seq:
            best_test_seq = test_out["test_acc_seq"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc                    
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss            
        writer.add_scalar('Accuracy/test', test_out["acc"], epoch)
        writer.add_scalar('Accuracy/test_seq', test_out["test_acc_seq"], epoch)
        writer.add_scalar('Loss/test', test_out["loss"], epoch)
        save_model(classifier, epoch, path=str(checkpoints_dir), acc=test_out["acc"], is_best=is_best, args=vars(args),
            best_test_acc=best_test_acc,
            best_test_seq=best_test_seq,
            best_train_acc=best_train_acc,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            net_ema=(ema.state_dict() if ema is not None else None),
            # P1-3 判据：标记 EMA 更新方案——只有 'step' 方案的 shadow
            # 才在 resume 时恢复（旧 'epoch' 方案的 BN stats 已污染）
            ema_scheme=('step' if ema is not None else None),
            # Codex 快测复审：早停计数随 checkpoint 持久化（resume 不丢历史）
            _patience_state=({'best': _best_early_stop,
                              'no_improve': _epochs_no_improve}
                             if args.patience > 0 else None))
        print(f"Testing loss:{test_out['loss']}  "f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] [best test seq acc: {best_test_seq*100.}%]\n\n")
        log_string('Test Accuracy: %f ' % (test_out["acc"]))
        log_string('Test Accuracy seq: %f' % (test_out["test_acc_seq"]))
        log_string('loss : %f' % train_out["loss"])
        # --patience 收敛早停（小样本快测协议：满轮数=收敛即停）：
        # 连续 N 个 epoch best_test_seq 无提升则正常退出。
        # 判据与 best 判定同口径（test_acc_seq），checkpoint 已保存。
        if args.patience > 0:
            metric = test_out["test_acc_seq"] if args.early_stop_metric == 'seq' \
                else test_out["acc"]
            if metric > _best_early_stop:
                _best_early_stop = metric
                _epochs_no_improve = 0
            else:
                _epochs_no_improve += 1
                log_string('EarlyStop: no improvement for %d/%d epochs'
                           % (_epochs_no_improve, args.patience))
            if _epochs_no_improve >= args.patience:
                log_string('EarlyStop triggered at epoch %d (best %s=%.4f) — '
                           'converged, exiting normally'
                           % (epoch + 1, args.early_stop_metric, _best_early_stop))
                break
    if args.train_method =="DDP":
        dist.destroy_process_group()
    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)

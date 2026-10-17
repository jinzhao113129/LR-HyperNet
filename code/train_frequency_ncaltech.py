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

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
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

def train(net, trainloader, optimizer, criterion, device,args):
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
        # 梯度累积：loss 除以 N，使 N 个 batch 的平均梯度等效一个大 batch
        (loss / accum).backward()
        # 每 N 步（或 epoch 最后一个不满 N 的尾窗口）才更新参数
        if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == num_batches:
            optimizer.step()
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
                        use_rp1_selector=args.rp1_selector,
                        use_hyperbolic_selector=args.hyperbolic_selector,
                        use_spike_time_enc=args.spike_time_enc,
                        use_hyperbolic_classifier=args.hyperbolic_classifier,
                        use_hyperbolic_fusion=args.hyperbolic_fusion,
                              use_hyperbolic_embedding=args.hyperbolic_embedding,
                              use_hyperbolic_attention=args.hyperbolic_attention,
                              use_event_mamba_ssm=args.event_mamba_ssm)
    criterion = cal_loss
    classifier.apply(inplace_relu)

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
        

    if args.optimizer =="SGD":
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    elif args.optimizer =="AdamW":
        if args.hyperbolic_fusion:
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
                      'time_enc','kda','spike_time_enc','rp1_selector','hyperbolic_selector',
                      'hyperbolic_classifier','hyperbolic_fusion','hyperbolic_embedding',
                      'hyperbolic_attention','event_mamba_ssm','group_norm']:
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
    for epoch in range(start_epoch, args.epoch):
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
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        classifier = classifier.train()
        train_out = train(classifier, trainDataLoader, optimizer, criterion, device,args)
        scheduler.step()
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss
        writer.add_scalar('Loss/train', train_out["loss"], epoch)
        writer.add_scalar('Accuracy/train', train_out["acc"], epoch)
        log_string('Train Accuracy: %f' % (train_out["acc"]))
        print(f"Training loss:{train_out['loss']} acc:{train_out['acc']}% time:{train_out['time']}s")
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
            scheduler=scheduler.state_dict())
        print(f"Testing loss:{test_out['loss']}  "f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] [best test seq acc: {best_test_seq*100.}%]\n\n")
        log_string('Test Accuracy: %f ' % (test_out["acc"]))
        log_string('Test Accuracy seq: %f' % (test_out["test_acc_seq"]))
        log_string('loss : %f' % train_out["loss"])
    if args.train_method =="DDP":
        dist.destroy_process_group()
    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)

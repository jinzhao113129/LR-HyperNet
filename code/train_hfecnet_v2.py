"""
HFECNet 双曲模型训练脚本
基于 train_frequency.py 修改，使用 HyperbolicFECNet 模型。
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
    parser = argparse.ArgumentParser('HFECNet training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--num_category', default=101, type=int, help='number of categories')
    # 改动点：--epoch 同时支持 --epochs 别名（兼容两种命令行写法）
    parser.add_argument('--epoch', '--epochs', dest='epoch', default=150, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument("--data_path", type=str, default='./data/', help="path to data")
    parser.add_argument("--log_path", type=str, default='./tensorboard_log/', help="tensorboard log path")
    parser.add_argument("--log_name", type=str, default='/hfecnet', help="log name")
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='AdamW,SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--curvature_init', type=float, default=0.4, help='hyperbolic curvature init')
    # 审查修复：--use_centroid 参数已失效（锚点策略由消融模式内部决定），标记 deprecated 以免误导
    # parser.add_argument('--use_centroid', action='store_true', default=True, help='deprecated: controlled by ablation mode')
    # 改动点：新增 --ablation 消融实验参数，切换 HFECNetAblation 的消融模式
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'no_hyperbolic', 'fixed_curvature', 'no_centroid'],
                        help='ablation mode: none(full model)/no_hyperbolic/fixed_curvature/no_centroid')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def cal_loss(pred, gold, smoothing=True):
    """计算交叉熵损失，支持 label smoothing。"""
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


def train(net, trainloader, optimizer, criterion, device, args):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_true = []
    train_pred = []
    start_time = time.time()

    for batch_idx, (points, labels) in enumerate(trainloader):
        points, labels = points.to(device), labels.to(device).long()
        points = points.permute(0, 2, 1).float()  # [B, 4, N]

        optimizer.zero_grad()
        outputs = net(points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        train_true.append(labels.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    acc = 100. * correct / total
    avg_loss = train_loss / (batch_idx + 1)
    elapsed = time.time() - start_time

    return {"loss": avg_loss, "acc": acc, "time": elapsed,
            "true": train_true, "pred": train_pred}


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


def save_model(net, epoch, path, acc, is_best, ablation='none', **kwargs):
    """改动点：模型保存文件名带上消融模式后缀（如 best_checkpoint_no_hyperbolic.pt），
    避免不同消融实验的模型互相覆盖。"""
    state = {
        'epoch': epoch,
        'net': net.state_dict(),
        'acc': acc,
        'ablation': ablation,
        **kwargs
    }
    # 'none' 模式不加后缀，与历史文件名保持一致
    suffix = '' if ablation == 'none' else '_%s' % ablation
    filepath = os.path.join(path, 'last_checkpoint%s.pt' % suffix)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(path, 'best_checkpoint%s.pt' % suffix)
        torch.save(state, best_filepath)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('hfecnet')
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
    # 改动点：tensorboard 日志名带上消融模式后缀，避免不同消融实验的曲线混在同一 run 里
    if args.log_name == '/hfecnet' and args.ablation != 'none':
        args.log_name = '/hfecnet_%s' % args.ablation
    writer = SummaryWriter(args.log_path + args.log_name)

    '''LOG'''
    # 审查修复：删除重复的 parse_args() 调用——main(args) 已接收外部解析好的 args，
    # 此处重新解析会覆盖外部传入的参数（如程序化调用 main() 时参数会静默丢失）。
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/hfecnet.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)
    # 改动点：console 与日志文件中记录当前消融模式
    log_string('Ablation mode: %s' % args.ablation)
    # 改动点：tensorboard 中也记录当前消融模式
    writer.add_text('ablation', 'ablation mode: %s' % args.ablation)

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
    # 改动点：当 num_point 小于原始点数(8192)时，reshape 会把每个事件拆成多个片段
    # （如 num_point=1024 时每个样本拆成 8 份），标签需相应重复对齐，否则 TensorDataset 报错
    if current_data_test.shape[0] != current_label_test.shape[0]:
        rep = current_data_test.shape[0] // current_label_test.shape[0]
        current_label_test = current_label_test.repeat_interleave(rep)
    # 改动点：同上，mark（序列标识）也按拆分段数重复对齐，用于序列级多数投票准确率
    if current_data_test.shape[0] != len(current_mark_test):
        rep = current_data_test.shape[0] // len(current_mark_test)
        current_mark_test = np.repeat(current_mark_test, rep)
    dataset_test = torch.utils.data.TensorDataset(current_data_test, current_label_test)
    testDataLoader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=0, drop_last=False)
    current_data_sum, current_label_sum, current_mark_sum = provider_data.load_h5_mark(TRAIN_FILES[0])
    current_label_sum = np.squeeze(current_label_sum)
    print("train", current_data_sum.shape, current_label_sum.shape)
    current_data_sum = torch.from_numpy(current_data_sum)
    current_label_sum = torch.from_numpy(current_label_sum.astype('int64'))
    current_data_sum = current_data_sum.reshape(-1, args.num_point, 4)
    # 改动点：同上，标签按拆分段数重复对齐
    if current_data_sum.shape[0] != current_label_sum.shape[0]:
        rep = current_data_sum.shape[0] // current_label_sum.shape[0]
        current_label_sum = current_label_sum.repeat_interleave(rep)
    dataset = torch.utils.data.TensorDataset(current_data_sum, current_label_sum)
    trainDataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=0, drop_last=True)

    '''MODEL LOADING'''
    # 改动点：改用消融版模型 HFECNetAblation，通过 ablation 参数切换消融模式；
    # ablation='none' 时即为完整双曲模型，与原 HFECNet 行为一致。
    # 注意：use_centroid 由消融模式内部决定（none/no_hyperbolic/fixed_curvature 用质心，
    # no_centroid 用组内第一个点做锚点），故不再单独传入。
    from hyperbolic_fecnet_ablation import HFECNetAblation
    classifier = HFECNetAblation(num_classes=args.num_category,
                                  ablation=args.ablation,
                                  curvature_init=args.curvature_init)
    log_string('Model: HFECNetAblation (ablation=%s)' % args.ablation)
    criterion = cal_loss
    classifier.apply(inplace_relu)
    device = 'cpu' if args.use_cpu else 'cuda'
    classifier = classifier.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    log_string(f'Total params: {total_params:,}')
    log_string(f'Trainable params: {trainable_params:,}')

    # 改动点：断点恢复也按消融模式读取对应文件，避免加载到其他消融版本的权重；
    # 若找不到对应文件则从头训练（原有 try/except 逻辑保留）
    ckpt_suffix = '' if args.ablation == 'none' else '_%s' % args.ablation
    try:
        checkpoint = torch.load('last_checkpoint%s.pt' % ckpt_suffix, map_location=device)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['net'])
        log_string('Use pretrain model (ablation=%s)' % args.ablation)
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)

    global_epoch = 0
    best_test_acc = 0.
    best_test_seq = 0.

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        classifier = classifier.train()
        train_out = train(classifier, trainDataLoader, optimizer, criterion, device, args)
        scheduler.step()
        best_train_acc = train_out["acc"]
        writer.add_scalar('Loss/train', train_out["loss"], epoch)
        writer.add_scalar('Accuracy/train', train_out["acc"], epoch)
        log_string('Train Accuracy: %f' % (train_out["acc"]))
        print(f"Training loss:{train_out['loss']} acc:{train_out['acc']}% time:{train_out['time']}s")

        test_out = validate(classifier, testDataLoader, criterion, device, current_mark_test, args)
        if test_out["test_acc_seq"] > best_test_seq:
            best_test_seq = test_out["test_acc_seq"]
            is_best = True
        else:
            is_best = False
        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        writer.add_scalar('Accuracy/test', test_out["acc"], epoch)
        writer.add_scalar('Accuracy/test_seq', test_out["test_acc_seq"], epoch)
        writer.add_scalar('Loss/test', test_out["loss"], epoch)

        # 记录曲率值
        for name, p in classifier.named_parameters():
            if "curvature" in name:
                writer.add_scalar(f'Curvature/{name}', p.item(), epoch)

        # 改动点：保存模型时传入 ablation，文件名带上消融模式后缀
        save_model(classifier, epoch, path=str(checkpoints_dir), acc=test_out["acc"], is_best=is_best,
                   ablation=args.ablation,
                   best_test_acc=best_test_acc,
                   optimizer=optimizer.state_dict())
        print(f"Testing loss:{test_out['loss']}  acc:{test_out['acc']}% time:{test_out['time']}s "
              f"[best test acc: {best_test_acc}%] [best test seq acc: {best_test_seq * 100.}%]\n\n")
        log_string('Test Accuracy: %f ' % (test_out["acc"]))
        log_string('Test Accuracy seq: %f' % (test_out["test_acc_seq"]))

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)

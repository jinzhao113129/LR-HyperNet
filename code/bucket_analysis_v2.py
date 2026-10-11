"""
分桶分析 v2：v3.1双曲(Lorentz) vs 欧氏对照 vs 基线b8/30ep
三者同配置（b8/30ep/百度单窗数据），按训练集每类样本数分桶
零GPU成本，CPU可跑
"""
import torch
import numpy as np
import h5py
import sys
sys.path.insert(0, '/home/jnguan/JiangYouxin/ZhaoJin')

from lr_hypernet import LRHyperNet
from models.FECNet import FECNet


def load_h5(path, num_point=10240):
    with h5py.File(path, 'r') as f:
        data = f['data'][:].reshape(-1, num_point, 4).astype(np.float32)
        label = f['label'][:].astype(np.int64)
    return data, label


def evaluate_per_class(model, data, label, device='cpu', batch_size=32):
    model.eval()
    model = model.to(device)
    num_classes = 101
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = torch.from_numpy(data[i:i+batch_size]).to(device)
            batch = batch.transpose(1, 2)  # [B, N, 4] -> [B, 4, N]
            preds = model(batch).argmax(dim=-1).cpu().numpy()
            for pred, lbl in zip(preds, label[i:i+batch_size]):
                total[lbl] += 1
                if pred == lbl:
                    correct[lbl] += 1
    acc = np.zeros(num_classes)
    for c in range(num_classes):
        if total[c] > 0:
            acc[c] = correct[c] / total[c]
    return acc, total


def load_lr_hyper(checkpoint_path, hyperbolic_type, time_enc=True):
    """加载 LRHyperNet 模型（v3.1双曲 或 欧氏对照）"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model = LRHyperNet(
        num_classes=101,
        feature_list=[64, 132, 268, 540],
        group_number=[2048, 1024, 512],
        remote_g=16,
        curvature_init=0.4,
        time_enc=time_enc,
        hyperbolic_type=hyperbolic_type,
    )
    model.load_state_dict(ckpt['net'], strict=True)
    epoch = ckpt.get('epoch', '?')
    best_acc = ckpt.get('best_test_acc', '?')
    return model, epoch, best_acc


def load_fecnet(checkpoint_path):
    """加载 SECNet/FECNet 基线模型"""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    model = FECNet(num_classes=101)
    state_dict = ckpt.get('net', ckpt.get('model', ckpt))
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model


def main():
    data_path = './ncaltech101_official_baidu/'
    train_data, train_label = load_h5(f'{data_path}/train.h5')
    test_data, test_label = load_h5(f'{data_path}/test.h5')

    train_counts = np.array([np.sum(train_label == c) for c in range(101)])
    test_counts = np.array([np.sum(test_label == c) for c in range(101)])

    print(f"训练集: {len(train_label)} 样本, 每类 {train_counts.min()}-{train_counts.max()}")
    print(f"测试集: {len(test_label)} 样本, 每类 {test_counts.min()}-{test_counts.max()}")

    # 加载三个模型
    print("\n=== 加载 v3.1 双曲 Lorentz (168298) ===")
    v31_path = './log/lr_hypernet/lr_hyper_v31_fix_v100/checkpoints/best_checkpoint.pt'
    v31_model, v31_ep, v31_best = load_lr_hyper(v31_path, 'lorentz', time_enc=True)
    print(f"  checkpoint epoch={v31_ep}, best_test_acc={v31_best}")
    v31_acc, _ = evaluate_per_class(v31_model, test_data, test_label)
    v31_overall = np.sum(v31_acc * test_counts) / test_counts.sum()
    print(f"  实测总体: {v31_overall*100:.2f}%")

    print("\n=== 加载 欧氏对照 (168343) ===")
    eucl_path = './log/lr_hypernet/lr_hyper_euclidean_v100/checkpoints/best_checkpoint.pt'
    eucl_model, eucl_ep, eucl_best = load_lr_hyper(eucl_path, 'euclidean', time_enc=True)
    print(f"  checkpoint epoch={eucl_ep}, best_test_acc={eucl_best}")
    eucl_acc, _ = evaluate_per_class(eucl_model, test_data, test_label)
    eucl_overall = np.sum(eucl_acc * test_counts) / test_counts.sum()
    print(f"  实测总体: {eucl_overall*100:.2f}%")

    print("\n=== 加载 基线 SECNet b8/30ep (168346) ===")
    base_path = './log/classification/secnet_baidu_b8_30ep/checkpoints/best_checkpoint.pth'
    base_model = load_fecnet(base_path)
    base_acc, _ = evaluate_per_class(base_model, test_data, test_label)
    base_overall = np.sum(base_acc * test_counts) / test_counts.sum()
    print(f"  实测总体: {base_overall*100:.2f}%")

    # 分桶分析
    buckets = [
        (0, 20, '极稀有 (<20)'),
        (20, 50, '稀有 (20-50)'),
        (50, 100, '中等 (50-100)'),
        (100, 1000, '丰富 (>100)'),
    ]

    print("\n" + "=" * 85)
    print("分桶对比（按训练集每类样本数，三者同配置 b8/30ep/百度数据）")
    print("=" * 85)
    print(f"{'分桶':<16} {'类数':<5} {'测试样本':<8} {'基线':<10} {'欧氏':<10} {'v3.1双曲':<10} {'欧-基':<8} {'双-基':<8} {'双-欧':<8}")
    print("-" * 85)

    for low, high, name in buckets:
        mask = (train_counts >= low) & (train_counts < high)
        if mask.sum() == 0:
            continue
        n_cls = mask.sum()
        n_test = int(test_counts[mask].sum())
        base_b = np.sum(base_acc[mask] * test_counts[mask]) / n_test
        eucl_b = np.sum(eucl_acc[mask] * test_counts[mask]) / n_test
        v31_b = np.sum(v31_acc[mask] * test_counts[mask]) / n_test
        print(f"{name:<16} {n_cls:<5} {n_test:<8} {base_b*100:<10.2f} {eucl_b*100:<10.2f} {v31_b*100:<10.2f} "
              f"{(eucl_b-base_b)*100:<+8.2f} {(v31_b-base_b)*100:<+8.2f} {(v31_b-eucl_b)*100:<+8.2f}")

    print("=" * 85)
    print(f"总体:           基线 {base_overall*100:.2f}%  欧氏 {eucl_overall*100:.2f}%  v3.1双曲 {v31_overall*100:.2f}%")
    print(f"      欧-基={(eucl_overall-base_overall)*100:+.2f}pp  双-基={(v31_overall-base_overall)*100:+.2f}pp  双-欧={(v31_overall-eucl_overall)*100:+.2f}pp")

    # 稀有类 vs 丰富类
    median = np.median(train_counts)
    rare = train_counts <= median
    rich = train_counts > median
    base_rare = np.sum(base_acc[rare] * test_counts[rare]) / test_counts[rare].sum()
    base_rich = np.sum(base_acc[rich] * test_counts[rich]) / test_counts[rich].sum()
    eucl_rare = np.sum(eucl_acc[rare] * test_counts[rare]) / test_counts[rare].sum()
    eucl_rich = np.sum(eucl_acc[rich] * test_counts[rich]) / test_counts[rich].sum()
    v31_rare = np.sum(v31_acc[rare] * test_counts[rare]) / test_counts[rare].sum()
    v31_rich = np.sum(v31_acc[rich] * test_counts[rich]) / test_counts[rich].sum()

    print(f"\n稀有类(≤中位数{median:.0f}, {rare.sum()}类): 基线 {base_rare*100:.2f}%  欧氏 {eucl_rare*100:.2f}%  双曲 {v31_rare*100:.2f}%")
    print(f"丰富类(>中位数{median:.0f}, {rich.sum()}类): 基线 {base_rich*100:.2f}%  欧氏 {eucl_rich*100:.2f}%  双曲 {v31_rich*100:.2f}%")

    # 结论
    print("\n" + "=" * 85)
    print("结论判断:")
    v31_rare_diff = (v31_rare - base_rare) * 100
    v31_rich_diff = (v31_rich - base_rich) * 100
    eucl_rare_diff = (eucl_rare - base_rare) * 100
    eucl_rich_diff = (eucl_rich - base_rich) * 100
    hyper_vs_eucl_rare = (v31_rare - eucl_rare) * 100
    hyper_vs_eucl_rich = (v31_rich - eucl_rich) * 100

    print(f"  双曲vs基线: 稀有 {v31_rare_diff:+.2f}pp, 丰富 {v31_rich_diff:+.2f}pp")
    print(f"  欧氏vs基线: 稀有 {eucl_rare_diff:+.2f}pp, 丰富 {eucl_rich_diff:+.2f}pp")
    print(f"  双曲vs欧氏: 稀有 {hyper_vs_eucl_rare:+.2f}pp, 丰富 {hyper_vs_eucl_rich:+.2f}pp")

    if abs(v31_overall - eucl_overall) * 100 < 1.0:
        print("  → 双曲与欧氏总体差<1pp，双曲无明显增益")
    if hyper_vs_eucl_rare > 1.5:
        print("  → 双曲在稀有类比欧氏高>1.5pp，双曲翻盘有微弱证据")
    elif hyper_vs_eucl_rare < -1.5:
        print("  → 双曲在稀有类比欧氏低>1.5pp，双曲在稀有类上更差")
    else:
        print("  → 双曲vs欧氏在稀有/丰富类上差异均<1.5pp，无显著差异")
    print("=" * 85)


if __name__ == '__main__':
    main()

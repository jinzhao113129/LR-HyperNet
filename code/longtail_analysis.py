"""
长尾分析脚本：按每类训练样本数分桶，比较双曲版 vs 欧氏版 vs 基线在稀有类上的表现
原理：双曲空间的树状度量可能对稀有类（样本少、层级结构更重要）更友好
用法：加载训练好的模型，在测试集上计算每类准确率，按训练集每类样本数分桶统计
"""
import torch
import numpy as np
import h5py
import sys
sys.path.insert(0, '/home/jnguan/JiangYouxin/ZhaoJin')

from lr_hypernet import LRHyperNet


def load_data(data_path, num_point=10240):
    """加载训练集和测试集"""
    with h5py.File(f'{data_path}/train.h5', 'r') as f:
        train_data = f['data'][:].reshape(-1, num_point, 4).astype(np.float32)
        train_label = f['label'][:].astype(np.int64)
    with h5py.File(f'{data_path}/test.h5', 'r') as f:
        test_data = f['data'][:].reshape(-1, num_point, 4).astype(np.float32)
        test_label = f['label'][:].astype(np.int64)
    return (train_data, train_label), (test_data, test_label)


def get_class_counts(labels, num_classes=101):
    """统计每类训练样本数"""
    counts = np.zeros(num_classes, dtype=int)
    for c in range(num_classes):
        counts[c] = np.sum(labels == c)
    return counts


def evaluate_per_class(model, test_data, test_label, device='cpu', batch_size=32):
    """计算每类准确率"""
    model.eval()
    model = model.to(device)
    num_classes = 101
    correct = np.zeros(num_classes)
    total = np.zeros(num_classes)

    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch_data = torch.from_numpy(test_data[i:i+batch_size]).to(device)
            batch_label = test_label[i:i+batch_size]
            logits = model(batch_data)
            preds = logits.argmax(dim=-1).cpu().numpy()
            for j, (pred, label) in enumerate(zip(preds, batch_label)):
                total[label] += 1
                if pred == label:
                    correct[label] += 1

    per_class_acc = np.zeros(num_classes)
    for c in range(num_classes):
        if total[c] > 0:
            per_class_acc[c] = correct[c] / total[c]
    return per_class_acc, total


def bucket_analysis(per_class_acc, train_counts, test_counts, buckets=None):
    """
    按训练集每类样本数分桶，统计各桶的平均准确率
    buckets: [(min, max, name), ...]，默认按四分位分桶
    """
    if buckets is None:
        # 默认分桶：极稀有(<20)、稀有(20-50)、中等(50-100)、丰富(>100)
        buckets = [
            (0, 20, '极稀有 (<20)'),
            (20, 50, '稀有 (20-50)'),
            (50, 100, '中等 (50-100)'),
            (100, 1000, '丰富 (>100)'),
        ]

    results = []
    for low, high, name in buckets:
        mask = (train_counts >= low) & (train_counts < high)
        if np.sum(mask) == 0:
            continue
        bucket_acc = per_class_acc[mask]
        bucket_test_total = test_counts[mask].sum()
        # 加权平均（按测试样本数）
        weighted_acc = np.sum(per_class_acc[mask] * test_counts[mask]) / bucket_test_total if bucket_test_total > 0 else 0
        results.append({
            'bucket': name,
            'num_classes': int(np.sum(mask)),
            'train_samples_range': f'{low}-{high}',
            'avg_train_per_class': float(train_counts[mask].mean()),
            'test_samples': int(bucket_test_total),
            'avg_acc_simple': float(bucket_acc.mean()),
            'avg_acc_weighted': float(weighted_acc),
            'min_acc': float(bucket_acc.min()),
            'max_acc': float(bucket_acc.max()),
        })
    return results


def print_comparison(results_dict, train_counts, test_counts):
    """
    打印多个模型的长尾对比
    results_dict: {'模型名': per_class_acc, ...}
    """
    buckets = [
        (0, 20, '极稀有 (<20)'),
        (20, 50, '稀有 (20-50)'),
        (50, 100, '中等 (50-100)'),
        (100, 1000, '丰富 (>100)'),
    ]

    print("=" * 80)
    print("长尾分析：按训练集每类样本数分桶的准确率对比")
    print("=" * 80)

    # 表头
    header = f"{'分桶':<18} {'类数':<6} {'测试样本':<10}"
    for name in results_dict.keys():
        header += f" {name:<12}"
    print(header)
    print("-" * 80)

    for low, high, bname in buckets:
        mask = (train_counts >= low) & (train_counts < high)
        if np.sum(mask) == 0:
            continue
        num_classes = int(np.sum(mask))
        test_total = int(test_counts[mask].sum())
        row = f"{bname:<18} {num_classes:<6} {test_total:<10}"
        for name, per_class_acc in results_dict.items():
            bucket_test_total = test_counts[mask].sum()
            weighted_acc = np.sum(per_class_acc[mask] * test_counts[mask]) / bucket_test_total if bucket_test_total > 0 else 0
            row += f" {weighted_acc*100:<12.2f}"
        print(row)

    print("=" * 80)

    # 总体准确率
    print(f"\n总体准确率（加权）：")
    for name, per_class_acc in results_dict.items():
        total = test_counts.sum()
        weighted = np.sum(per_class_acc * test_counts) / total
        print(f"  {name}: {weighted*100:.2f}%")

    # 稀有类 vs 丰富类的差距
    print(f"\n稀有类(前50%类) vs 丰富类(后50%类) 差距：")
    median_count = np.median(train_counts)
    rare_mask = train_counts <= median_count
    rich_mask = train_counts > median_count
    for name, per_class_acc in results_dict.items():
        rare_acc = np.sum(per_class_acc[rare_mask] * test_counts[rare_mask]) / test_counts[rare_mask].sum()
        rich_acc = np.sum(per_class_acc[rich_mask] * test_counts[rich_mask]) / test_counts[rich_mask].sum()
        gap = rich_acc - rare_acc
        print(f"  {name}: 稀有类 {rare_acc*100:.2f}%, 丰富类 {rich_acc*100:.2f}%, 差距 {gap*100:+.2f}pp")

    print("=" * 80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='模型checkpoint路径，格式为 名称:路径 名称:路径 ...')
    parser.add_argument('--data_path', type=str, default='./ncaltech101_official_baidu/')
    parser.add_argument('--num_point', type=int, default=10240)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载数据
    (train_data, train_label), (test_data, test_label) = load_data(args.data_path, args.num_point)
    train_counts = get_class_counts(train_label)
    test_counts = get_class_counts(test_label)

    print(f"训练集：{len(train_label)} 样本，每类 {train_counts.min()}-{train_counts.max()} 样本")
    print(f"测试集：{len(test_label)} 样本，每类 {test_counts.min()}-{test_counts.max()} 样本")
    print(f"训练集类别分布：中位数 {np.median(train_counts):.0f}, 均值 {train_counts.mean():.1f}")

    # 加载多个模型并评估
    results_dict = {}
    for ckpt_spec in args.checkpoints:
        name, path = ckpt_spec.split(':')
        print(f"\n加载模型 {name}: {path}")
        checkpoint = torch.load(path, map_location=device)
        model = LRHyperNet(
            num_category=101, input_dim=4, embed_dim=64,
            feature_list=[64, 132, 268, 540],
            group_number=[2048, 1024, 512],
            num_classes=101,
            hyperbolic_type='lorentz',
            time_enc=True,
        )
        model.load_state_dict(checkpoint['net'], strict=False)
        per_class_acc, _ = evaluate_per_class(model, test_data, test_label, device)
        results_dict[name] = per_class_acc
        total_acc = np.sum(per_class_acc * test_counts) / test_counts.sum()
        print(f"  总体准确率: {total_acc*100:.2f}%")

    # 对比分析
    print_comparison(results_dict, train_counts, test_counts)

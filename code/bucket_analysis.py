"""
分桶分析：v1(双曲) vs baseline(SECNet) 按训练集每类样本数分桶的准确率对比
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


def main():
    data_path = './ncaltech101_official_baidu/'
    train_data, train_label = load_h5(f'{data_path}/train.h5')
    test_data, test_label = load_h5(f'{data_path}/test.h5')

    train_counts = np.array([np.sum(train_label == c) for c in range(101)])
    test_counts = np.array([np.sum(test_label == c) for c in range(101)])

    print(f"训练集: {len(train_label)} 样本, 每类 {train_counts.min()}-{train_counts.max()}")
    print(f"测试集: {len(test_label)} 样本, 每类 {test_counts.min()}-{test_counts.max()}")

    # 加载 v1 (LRHyperNet, Poincaré, time_enc=False)
    print("\n=== 加载 v1 (双曲 Poincaré) ===")
    v1_ckpt = torch.load('./log/lr_hypernet/lr_hyper_v1_b8_baidu/checkpoints/best_checkpoint.pt', map_location='cpu')
    v1_model = LRHyperNet(
        num_classes=101,
        feature_list=[64, 132, 268, 540],
        group_number=[2048, 1024, 512],
        remote_g=16,
        curvature_init=0.4,
        time_enc=False,
        hyperbolic_type='poincare',
    )
    v1_model.load_state_dict(v1_ckpt['net'], strict=True)
    v1_acc, _ = evaluate_per_class(v1_model, test_data, test_label)
    v1_overall = np.sum(v1_acc * test_counts) / test_counts.sum()
    print(f"v1 总体准确率: {v1_overall*100:.2f}%")

    # 加载 baseline (SECNet/FECNet)
    print("\n=== 加载 baseline (SECNet b32/150ep) ===")
    base_ckpt = torch.load('./log/classification/secnet_baidu_a100/checkpoints/best_checkpoint.pth', map_location='cpu')
    base_model = FECNet(num_classes=101)
    # SECNet checkpoint 可能有 'model.' 前缀
    state_dict = base_ckpt.get('net', base_ckpt.get('model', base_ckpt))
    if any(k.startswith('model.') for k in state_dict.keys()):
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    base_model.load_state_dict(state_dict, strict=False)
    base_acc, _ = evaluate_per_class(base_model, test_data, test_label)
    base_overall = np.sum(base_acc * test_counts) / test_counts.sum()
    print(f"baseline 总体准确率: {base_overall*100:.2f}%")

    # 分桶分析
    buckets = [
        (0, 20, '极稀有 (<20)'),
        (20, 50, '稀有 (20-50)'),
        (50, 100, '中等 (50-100)'),
        (100, 1000, '丰富 (>100)'),
    ]

    print("\n" + "=" * 70)
    print("分桶对比（按训练集每类样本数）")
    print("=" * 70)
    print(f"{'分桶':<18} {'类数':<6} {'测试样本':<10} {'baseline':<12} {'v1双曲':<12} {'差值':<10}")
    print("-" * 70)

    for low, high, name in buckets:
        mask = (train_counts >= low) & (train_counts < high)
        if mask.sum() == 0:
            continue
        n_cls = mask.sum()
        n_test = int(test_counts[mask].sum())
        base_bucket = np.sum(base_acc[mask] * test_counts[mask]) / n_test
        v1_bucket = np.sum(v1_acc[mask] * test_counts[mask]) / n_test
        diff = (v1_bucket - base_bucket) * 100
        print(f"{name:<18} {n_cls:<6} {n_test:<10} {base_bucket*100:<12.2f} {v1_bucket*100:<12.2f} {diff:+.2f}pp")

    print("=" * 70)
    print(f"\n总体: baseline {base_overall*100:.2f}%, v1 {v1_overall*100:.2f}%, 差值 {(v1_overall-base_overall)*100:+.2f}pp")

    # 稀有类 vs 丰富类
    median = np.median(train_counts)
    rare = train_counts <= median
    rich = train_counts > median
    base_rare = np.sum(base_acc[rare] * test_counts[rare]) / test_counts[rare].sum()
    base_rich = np.sum(base_acc[rich] * test_counts[rich]) / test_counts[rich].sum()
    v1_rare = np.sum(v1_acc[rare] * test_counts[rare]) / test_counts[rare].sum()
    v1_rich = np.sum(v1_acc[rich] * test_counts[rich]) / test_counts[rich].sum()

    print(f"\n稀有类(训练样本≤中位数{median:.0f}): baseline {base_rare*100:.2f}%, v1 {v1_rare*100:.2f}%, 差值 {(v1_rare-base_rare)*100:+.2f}pp")
    print(f"丰富类(训练样本>中位数{median:.0f}): baseline {base_rich*100:.2f}%, v1 {v1_rich*100:.2f}%, 差值 {(v1_rich-base_rich)*100:+.2f}pp")

    # 结论
    print("\n" + "=" * 70)
    print("结论判断:")
    rare_diff = (v1_rare - base_rare) * 100
    rich_diff = (v1_rich - base_rich) * 100
    if rare_diff > rich_diff + 1.0:
        print(f"  v1 在稀有类上的增益({rare_diff:+.2f}pp) > 丰富类({rich_diff:+.2f}pp)")
        print("  → 双曲翻盘假设成立：双曲在稀有类上表现更好")
    elif rare_diff < rich_diff - 1.0:
        print(f"  v1 在稀有类上的增益({rare_diff:+.2f}pp) < 丰富类({rich_diff:+.2f}pp)")
        print("  → 双曲翻盘假设不成立：双曲在丰富类上表现更好，稀有类叙事要换")
    else:
        print(f"  v1 在稀有类({rare_diff:+.2f}pp)和丰富类({rich_diff:+.2f}pp)上差异不大")
        print("  → 双曲效果均匀，可能只是正则化作用")
    print("=" * 70)


if __name__ == '__main__':
    main()

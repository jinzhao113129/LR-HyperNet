"""
模型评估脚本
加载训练好的 checkpoint，在测试集上评估，输出：
- overall accuracy
- per-class accuracy
- confusion matrix（保存为 png）
- 分类报告（precision, recall, f1）

用法:
  python evaluate.py --checkpoint path/to/best_checkpoint.pth --model secnet
  python evaluate.py --checkpoint path/to/best_checkpoint.pth --model hfecnet
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import provider_data
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'models'))


def parse_args():
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--model', type=str, default='secnet', choices=['secnet', 'hfecnet'],
                        help='model type')
    parser.add_argument('--data_path', type=str, default='./data/', help='path to data')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number')
    parser.add_argument('--num_category', type=int, default=101, help='number of categories')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--output_dir', type=str, default='./eval_results/', help='output directory')
    parser.add_argument('--curvature_init', type=float, default=0.4, help='hfecnet curvature init')
    return parser.parse_args()


def load_model(args, device):
    """加载模型和 checkpoint。"""
    if args.model == 'secnet':
        from models.FECNet import FECNet
        model = FECNet(num_classes=args.num_category)
    elif args.model == 'hfecnet':
        from hyperbolic_fecnet import HFECNet
        model = HFECNet(num_classes=args.num_category, curvature_init=args.curvature_init)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'net' in checkpoint:
        model.load_state_dict(checkpoint['net'])
        epoch = checkpoint.get('epoch', 'unknown')
        acc = checkpoint.get('acc', 'unknown')
        print(f"Loaded checkpoint from epoch {epoch}, train acc {acc}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw state dict")

    model = model.to(device)
    model.eval()
    return model


def evaluate(model, testloader, device):
    """在测试集上评估，返回预测和标签。"""
    all_preds = []
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for points, labels in testloader:
            points = points.to(device).float()
            points = points.permute(0, 2, 1)  # [B, 4, N]
            labels = labels.to(device).long()

            outputs = model(points)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_outputs.append(F.softmax(outputs, dim=1).cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_outputs = np.concatenate(all_outputs)
    return all_preds, all_labels, all_outputs


def plot_confusion_matrix(cm, output_path, num_classes=101):
    """绘制混淆矩阵并保存。"""
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=range(num_classes),
           yticklabels=range(num_classes),
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_per_class_accuracy(per_class_acc, output_path):
    """绘制 per-class accuracy 柱状图。"""
    fig, ax = plt.subplots(figsize=(20, 6))
    classes = range(len(per_class_acc))
    colors = ['red' if acc < 0.5 else 'orange' if acc < 0.7 else 'green' for acc in per_class_acc]
    ax.bar(classes, per_class_acc, color=colors, alpha=0.7)
    ax.axhline(y=np.mean(per_class_acc), color='blue', linestyle='--', label=f'Mean: {np.mean(per_class_acc):.3f}')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy')
    ax.legend()
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-class accuracy plot saved to {output_path}")


def main():
    args = parse_args()
    device = 'cpu' if args.use_cpu else 'cuda'
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== 模型评估 ===")
    print(f"模型: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"设备: {device}")
    print()

    # 加载数据
    print("加载测试数据...")
    test_data, test_label, test_mark = provider_data.load_h5_mark(args.data_path + "test.h5")
    test_label = np.squeeze(test_label)
    test_data = torch.from_numpy(test_data).reshape(-1, args.num_point, 4)
    test_label = torch.from_numpy(test_label.astype('int64'))
    dataset = torch.utils.data.TensorDataset(test_data, test_label)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"测试样本数: {len(dataset)}")

    # 加载模型
    print("\n加载模型...")
    model = load_model(args, device)

    # 评估
    print("\n开始评估...")
    preds, labels, outputs = evaluate(model, testloader, device)

    # Overall accuracy
    overall_acc = (preds == labels).mean()
    print(f"\n{'='*50}")
    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"{'='*50}")

    # Per-class accuracy
    per_class_acc = []
    for c in range(args.num_category):
        mask = labels == c
        if mask.sum() > 0:
            acc = (preds[mask] == c).mean()
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0.0)

    # 找出最好和最差的 5 个类
    sorted_classes = np.argsort(per_class_acc)
    print(f"\n最差的 5 个类:")
    for c in sorted_classes[:5]:
        print(f"  类 {c:3d}: acc={per_class_acc[c]:.4f} (样本数={(labels==c).sum()})")
    print(f"\n最好的 5 个类:")
    for c in sorted_classes[-5:][::-1]:
        print(f"  类 {c:3d}: acc={per_class_acc[c]:.4f} (样本数={(labels==c).sum()})")

    # 分类报告
    print(f"\n分类报告（前 20 类）:")
    report = classification_report(labels, preds, digits=4, output_dict=True, zero_division=0)
    for c in range(min(20, args.num_category)):
        key = str(c)
        if key in report:
            r = report[key]
            print(f"  类 {c:3d}: P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1-score']:.4f} support={r['support']}")

    # 混淆矩阵
    cm = confusion_matrix(labels, preds, labels=range(args.num_category))
    cm_path = os.path.join(args.output_dir, f'{args.model}_confusion_matrix.png')
    plot_confusion_matrix(cm, cm_path, args.num_category)

    # Per-class accuracy 图
    acc_path = os.path.join(args.output_dir, f'{args.model}_per_class_accuracy.png')
    plot_per_class_accuracy(per_class_acc, acc_path)

    # 保存详细结果
    results_path = os.path.join(args.output_dir, f'{args.model}_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)\n")
        f.write(f"Mean Per-Class Accuracy: {np.mean(per_class_acc):.4f}\n")
        f.write(f"\nPer-Class Accuracy:\n")
        for c in range(args.num_category):
            f.write(f"  Class {c:3d}: {per_class_acc[c]:.4f} (support={(labels==c).sum()})\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report(labels, preds, digits=4, zero_division=0))
    print(f"\n详细结果保存到: {results_path}")

    print(f"\n✅ 评估完成！")


if __name__ == '__main__':
    main()

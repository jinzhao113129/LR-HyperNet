"""
N-Caltech101 数据集分析脚本
分析训练集和测试集的统计信息，生成可视化图表
"""
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置
DATA_DIR = './data'
OUTPUT_DIR = './eval_results/dataset_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 中文字体设置（HPC 可能没有中文字体，用英文标签）
plt.rcParams['font.size'] = 10


def load_h5(path):
    """加载 H5 文件"""
    with h5py.File(path, 'r') as f:
        data = f['data'][:]      # [N, 8192, 4]
        label = f['label'][:]    # [N]
        mark = f['mark'][:]      # [N]
    return data, label, mark


def print_stats(name, data, label, mark):
    """打印数据集统计信息"""
    N = data.shape[0]
    num_points = data.shape[1]
    num_classes = len(np.unique(label))

    # 坐标统计
    x = data[:, :, 0]  # [N, 8192]
    y = data[:, :, 1]
    t = data[:, :, 2]
    p = data[:, :, 3]

    stats = []
    stats.append(f"=== {name} ===")
    stats.append(f"样本数: {N}")
    stats.append(f"点数/样本: {num_points}")
    stats.append(f"类别数: {num_classes}")
    stats.append(f"mark 范围: {mark.min()} - {mark.max()}")
    stats.append("")
    stats.append("--- 坐标范围 ---")
    stats.append(f"x: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}, std={x.std():.4f}")
    stats.append(f"y: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")
    stats.append(f"t: min={t.min():.6f}, max={t.max():.6f}, mean={t.mean():.6f}, std={t.std():.6f}")
    stats.append(f"p: min={p.min():.4f}, max={p.max():.4f}, mean={p.mean():.4f}")
    stats.append("")

    # 极性分布
    p_pos = (p > 0).sum()
    p_neg = (p < 0).sum()
    p_zero = (p == 0).sum()
    stats.append("--- 极性分布 ---")
    stats.append(f"p>0: {p_pos} ({p_pos/(N*num_points)*100:.2f}%)")
    stats.append(f"p<0: {p_neg} ({p_neg/(N*num_points)*100:.2f}%)")
    stats.append(f"p=0: {p_zero} ({p_zero/(N*num_points)*100:.2f}%)")
    stats.append("")

    # 每类样本数
    class_counts = np.bincount(label, minlength=num_classes)
    stats.append("--- 每类样本数 ---")
    stats.append(f"最少: {class_counts.min()} (class {class_counts.argmin()})")
    stats.append(f"最多: {class_counts.max()} (class {class_counts.argmax()})")
    stats.append(f"平均: {class_counts.mean():.1f}")
    stats.append(f"中位数: {np.median(class_counts):.1f}")
    stats.append("")

    text = '\n'.join(stats)
    print(text)
    return text, class_counts, x, y, t, p


def plot_class_distribution(train_counts, test_counts, output_path):
    """绘制类别分布直方图（前30类）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 训练集
    top30_train = train_counts[:30]
    axes[0].bar(range(30), top30_train, color='steelblue', alpha=0.8)
    axes[0].set_title('Train - Class Distribution (Top 30)')
    axes[0].set_xlabel('Class ID')
    axes[0].set_ylabel('Sample Count')
    axes[0].tick_params(axis='x', rotation=45)

    # 测试集
    top30_test = test_counts[:30]
    axes[1].bar(range(30), top30_test, color='coral', alpha=0.8)
    axes[1].set_title('Test - Class Distribution (Top 30)')
    axes[1].set_xlabel('Class ID')
    axes[1].set_ylabel('Sample Count')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path}")


def plot_long_tail(train_counts, test_counts, output_path):
    """绘制长尾分布图（降序）"""
    fig, ax = plt.subplots(figsize=(10, 5))

    train_sorted = np.sort(train_counts)[::-1]
    test_sorted = np.sort(test_counts)[::-1]

    ax.plot(range(len(train_sorted)), train_sorted, label='Train', color='steelblue', linewidth=2)
    ax.plot(range(len(test_sorted)), test_sorted, label='Test', color='coral', linewidth=2)
    ax.set_title('Long-tail Distribution (Sorted by Sample Count)')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Sample Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path}")


def plot_coord_distribution(x_train, y_train, t_train, output_path):
    """绘制坐标分布直方图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # x 坐标
    axes[0].hist(x_train.flatten(), bins=100, color='steelblue', alpha=0.8, density=True)
    axes[0].set_title('X Coordinate Distribution')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Density')

    # y 坐标
    axes[1].hist(y_train.flatten(), bins=100, color='coral', alpha=0.8, density=True)
    axes[1].set_title('Y Coordinate Distribution')
    axes[1].set_xlabel('Y')
    axes[1].set_ylabel('Density')

    # t 坐标
    axes[2].hist(t_train.flatten(), bins=100, color='seagreen', alpha=0.8, density=True)
    axes[2].set_title('T Coordinate Distribution')
    axes[2].set_xlabel('T')
    axes[2].set_ylabel('Density')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path}")


def plot_polarity_distribution(p_train, p_test, output_path):
    """绘制极性分布饼图"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 训练集
    p_pos_train = (p_train > 0).sum()
    p_neg_train = (p_train < 0).sum()
    p_zero_train = (p_train == 0).sum()
    sizes_train = [p_pos_train, p_neg_train, p_zero_train]
    labels_train = ['p>0', 'p<0', 'p=0']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    axes[0].pie(sizes_train, labels=labels_train, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Train - Polarity Distribution')

    # 测试集
    p_pos_test = (p_test > 0).sum()
    p_neg_test = (p_test < 0).sum()
    p_zero_test = (p_test == 0).sum()
    sizes_test = [p_pos_test, p_neg_test, p_zero_test]
    axes[1].pie(sizes_test, labels=labels_train, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Test - Polarity Distribution')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("N-Caltech101 数据集分析")
    print("=" * 60)

    # 加载数据
    print("\n加载训练集...")
    train_data, train_label, train_mark = load_h5(os.path.join(DATA_DIR, 'train.h5'))
    print(f"训练集 shape: data={train_data.shape}, label={train_label.shape}")

    print("\n加载测试集...")
    test_data, test_label, test_mark = load_h5(os.path.join(DATA_DIR, 'test.h5'))
    print(f"测试集 shape: data={test_data.shape}, label={test_label.shape}")

    # 统计
    print("\n")
    train_text, train_counts, x_train, y_train, t_train, p_train = print_stats(
        "训练集 (Train)", train_data, train_label, train_mark)
    print("\n")
    test_text, test_counts, x_test, y_test, t_test, p_test = print_stats(
        "测试集 (Test)", test_data, test_label, test_mark)

    # 保存统计文本
    stats_text = train_text + "\n\n" + test_text
    stats_path = os.path.join(OUTPUT_DIR, 'dataset_stats.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(stats_text)
    print(f"\n统计结果保存: {stats_path}")

    # 生成图表
    print("\n生成可视化图表...")
    plot_class_distribution(train_counts, test_counts,
                             os.path.join(OUTPUT_DIR, 'class_distribution.png'))
    plot_long_tail(train_counts, test_counts,
                    os.path.join(OUTPUT_DIR, 'long_tail_distribution.png'))
    plot_coord_distribution(x_train, y_train, t_train,
                             os.path.join(OUTPUT_DIR, 'coordinate_distribution.png'))
    plot_polarity_distribution(p_train, p_test,
                                os.path.join(OUTPUT_DIR, 'polarity_distribution.png'))

    print("\n" + "=" * 60)
    print("分析完成！所有结果保存在:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == '__main__':
    main()

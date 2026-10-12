"""百度单窗数据增强：在 provider_data 加载后按官方生成脚本的增强方式扩充训练集。

官方 generate_ncaltech101.py 的增强（test 无增强）：
  1. random_crop：随机裁剪窗口（时间上）
  2. reverse_T：时间反转

额外增强（Claude 建议，对小数据集 3596 是确定性收益）：
  3. polarity_flip：极性翻转（p → -p）
  4. time_jitter：时间抖动（±小量）
  5. point_dropout：随机丢弃 10% 点

设计原则：
- 增强只作用于训练集，test 保持原样（与官方一致）
- 每个样本独立增强，增强版本数可配（默认 ×3，3596→10788）
- 保持 (t,x,y,p) 布局、mark 语义（增强样本共享原 mark，投票逻辑不变）

用法（train_lr_hyper.py 加 --augment N 后内部调用）：
    from augment_data import augment_train_data
    data, label, mark = augment_train_data(data, label, mark, n_versions=3)
"""
import numpy as np


def random_crop(points: np.ndarray, rng: np.random.Generator,
                crop_ratio: float = 0.8) -> np.ndarray:
    """时间维随机裁剪：随机选一个起点，保留 crop_ratio 长度的事件。

    模拟官方 300ms 视频切 30ms 滑窗的思路（裁剪后时间戳需重新归一化）。

    Args:
        points: [N, 4] (t,x,y,p)，t ∈ [0,1]
        crop_ratio: 保留比例
        rng: 随机数生成器（由调用方传入，保证种子可控）
    Returns:
        [N', 4] 裁剪后的事件
    """
    n = len(points)
    keep = max(10, int(n * crop_ratio))
    # 按时间排序后随机切一段
    order = np.argsort(points[:, 0])
    sorted_pts = points[order]
    start = rng.integers(0, n - keep + 1)
    cropped = sorted_pts[start:start + keep].copy()
    # 重新归一化时间到 [0,1]
    t = cropped[:, 0]
    t_min, t_max = t.min(), t.max()
    if t_max > t_min:
        cropped[:, 0] = (t - t_min) / (t_max - t_min)
    else:
        cropped[:, 0] = 0.0
    return cropped


def reverse_t(points: np.ndarray) -> np.ndarray:
    """时间反转：t → 1-t（官方 reverse_T 增强）。"""
    out = points.copy()
    out[:, 0] = 1.0 - out[:, 0]
    return out


def polarity_flip(points: np.ndarray) -> np.ndarray:
    """极性翻转：p → -p。"""
    out = points.copy()
    out[:, 3] = -out[:, 3]
    return out


def time_jitter(points: np.ndarray, rng: np.random.Generator,
                jitter_std: float = 0.02) -> np.ndarray:
    """时间抖动：t += N(0, jitter_std)，clip 到 [0,1]。"""
    out = points.copy()
    out[:, 0] = np.clip(out[:, 0] + rng.normal(0, jitter_std, len(out)), 0, 1)
    return out


def point_dropout(points: np.ndarray, rng: np.random.Generator,
                  drop_ratio: float = 0.1) -> np.ndarray:
    """随机丢弃 10% 点。"""
    n = len(points)
    keep = int(n * (1 - drop_ratio))
    idx = rng.choice(n, keep, replace=False)
    return points[idx]


def augment_sample(points: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """单样本增强：随机组合 2-3 种增强。

    Args:
        points: [N, 4]
        rng: 随机数生成器
    Returns:
        [N', 4] 增强后的事件（可能点数不同，需重新 padding 到 10240）
    """
    aug = rng.choice(5, size=rng.integers(2, 4), replace=False)
    for a in aug:
        if a == 0:
            points = random_crop(points, rng=rng)
        elif a == 1:
            points = reverse_t(points)
        elif a == 2:
            points = polarity_flip(points)
        elif a == 3:
            points = time_jitter(points, rng=rng)
        else:
            points = point_dropout(points, rng=rng)
    return points


def augment_train_data(data: np.ndarray, label: np.ndarray, mark: np.ndarray,
                       n_versions: int = 3) -> tuple:
    """扩充训练数据：每个样本生成 n_versions 个增强版本。

    Args:
        data: [N, 10240, 4] float32
        label: [N, 1] 或 [N] int
        mark: [N] int（样本 ID，增强样本共享原 mark）
        n_versions: 每个样本的增强版本数（不含原样）
    Returns:
        (data_aug, label_aug, mark_aug)：[N*(1+n_versions), 10240, 4] 等
    """
    rng = np.random.default_rng(42)
    N = len(data)
    target_points = data.shape[1]
    versions = [data]  # 原样
    for _ in range(n_versions):
        aug_list = []
        for i in range(N):
            aug = augment_sample(data[i], rng)
            # 增强可能改变点数（crop/dropout），重采样回 target_points
            if aug.shape[0] != target_points:
                n = aug.shape[0]
                idx = rng.choice(n, target_points, replace=True) if n < target_points \
                    else rng.choice(n, target_points, replace=False)
                aug = aug[idx]
            aug_list.append(aug)
        versions.append(np.stack(aug_list))
    data_aug = np.concatenate(versions, axis=0)  # [N*(1+n), 10240, 4]
    label = np.squeeze(label)
    label_aug = np.tile(label, n_versions + 1)
    mark_aug = np.tile(mark, n_versions + 1)
    return data_aug, label_aug, mark_aug


if __name__ == "__main__":
    """快速自检"""
    rng = np.random.default_rng(0)
    dummy = rng.random((100, 4)).astype(np.float32)
    dummy[:, 0] = np.sort(dummy[:, 0])  # 时间有序
    dummy[:, 3] = np.where(dummy[:, 3] > 0.5, 1.0, -1.0)  # 极性 ±1

    for name, fn in [("random_crop", random_crop), ("reverse_t", reverse_t),
                     ("polarity_flip", polarity_flip), ("time_jitter", time_jitter),
                     ("point_dropout", point_dropout)]:
        # 含随机性的增强必须传 rng（确定性；不依赖全局 np.random）
        if name in ("random_crop", "time_jitter", "point_dropout"):
            out = fn(dummy, rng=rng)
        else:
            out = fn(dummy)
        assert out.shape[-1] == 4, f"{name}: shape {out.shape}"
        assert np.isfinite(out).all(), f"{name}: NaN/Inf"
        if name == "polarity_flip":
            assert set(np.unique(out[:, 3])) == set([-1.0, 1.0]), "极性翻转后 p 必须 ±1"
        if name == "time_jitter" or name == "random_crop":
            assert out[:, 0].min() >= 0 and out[:, 0].max() <= 1, "时间越界"
        print(f"  {name}: {tuple(out.shape)} ✓")

    # 端到端（rng=42 内部种子，可复现）
    data = rng.random((16, 10240, 4)).astype(np.float32)
    label = np.arange(16).reshape(-1, 1)
    mark = np.arange(16)
    d, l, m = augment_train_data(data, label, mark, n_versions=2)
    print(f"  端到端: {d.shape} / {l.shape} / {m.shape}")
    assert d.shape[0] == 16 * 3, f"期望 48，实际 {d.shape[0]}"
    assert l.shape[0] == 48 and m.shape[0] == 48, "label/mark 维度错误"
    print("  增强自检全部通过 ✓")

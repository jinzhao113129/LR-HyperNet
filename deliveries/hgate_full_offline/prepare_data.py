"""离线读取N-Caltech101原始bin，冻结划分后流式生成三窗口H5。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from common import dump_json, seed_all, sha256
from official_transforms import normaliztion, random_crop, reverse_T, shuffle_downsample

WINDOWS = [(35, 65), (135, 165), (235, 265)]


def read_bin(path: Path) -> np.ndarray:
    """按数据集作者40bit格式解码(t,x,y,p)，时间单位微秒。"""
    raw = np.fromfile(path, dtype=np.uint8)
    if len(raw) % 5:
        raise ValueError(f'不是完整5字节事件文件: {path}')
    raw = raw.reshape(-1, 5).astype(np.int64)
    times = ((raw[:, 2] & 127) << 16) | (raw[:, 3] << 8) | raw[:, 4]
    points = np.stack([times, raw[:, 0], raw[:, 1], raw[:, 2] >> 7], axis=-1)
    if len(points) == 0 or np.any(np.diff(points[:, 0]) < 0):
        raise ValueError(f'空事件或时间戳非单调: {path}')
    if np.any(points[:, 1] >= 240) or np.any(points[:, 2] >= 180):
        raise ValueError(f'坐标超出N-Caltech101范围，不静默解释为其他事件格式: {path}')
    return points


def record_list(root: Path, seed: int, split_file: Path | None) -> tuple:
    """稳定排序后按原记录80/20划分；输入划分须与文件全集一一对应。"""
    files = sorted(root.glob('*/*.bin'))
    files = [p for p in files if p.parent.name != 'BACKGROUND_Google']
    if not files:
        raise ValueError('raw-root必须直接包含101个类别目录及其中的bin文件')
    classes = sorted({p.parent.name for p in files})
    if len(classes) != 101:
        raise ValueError(f'找到{len(classes)}类，不是101类')
    names = [p.relative_to(root).as_posix() for p in files]
    if split_file:
        rows = json.loads(split_file.read_text(encoding='utf-8'))
        if len(rows) != len(names) or {r['path'] for r in rows} != set(names):
            raise ValueError('划分清单与原始文件全集不一致')
        mapping = {r['path']: r['split'] for r in rows}
        if len(mapping) != len(names) or not set(mapping.values()) <= {'train', 'test'}:
            raise ValueError('重复文件或非法split')
    else:
        order = np.random.RandomState(seed).permutation(len(files))
        train = set(order[:int(.8 * len(files))].tolist())
        mapping = {name: 'train' if i in train else 'test' for i, name in enumerate(names)}
    rows = [dict(id=i, path=name, label=classes.index(files[i].parent.name),
                 split=mapping[name], sha256=sha256(files[i])) for i, name in enumerate(names)]
    return classes, rows


def create_h5(path: Path, points: int) -> h5py.File:
    """新建可增长H5；拒绝覆盖，数据按窗口增量写入。"""
    source = h5py.File(path, 'x')
    source.create_dataset('data', (0, points, 4), maxshape=(None, points, 4),
                          chunks=(1, points, 4), dtype='f4')
    for name in ('label', 'mark', 'window', 'variant'):
        source.create_dataset(name, (0,), maxshape=(None,), chunks=True, dtype='i8')
    return source


def append_row(source: h5py.File, data: np.ndarray, row: dict, window: int, variant: int) -> None:
    """同时追加数据与原始记录/窗口/增强元数据。"""
    position = len(source['label'])
    values = dict(data=data, label=row['label'], mark=row['id'], window=window, variant=variant)
    for name, value in values.items():
        dataset = source[name]
        dataset.resize(position + 1, axis=0)
        dataset[position] = value


def window_events(points: np.ndarray, window: int) -> np.ndarray:
    """保留官方最近端点切窗和毫秒量化去重，再还原微秒。"""
    begin, end = WINDOWS[window]
    left = np.abs(points[:, 0] - begin * 1000).argmin()
    right = np.abs(points[:, 0] - end * 1000).argmin()
    subset = points[left:right].copy()
    subset[:, 0] //= 1000
    _, indices = np.unique(subset, axis=0, return_index=True)
    subset = subset[indices]
    subset[:, 0] *= 1000
    return subset.astype(float)


def make_variant(points: np.ndarray, variant: int, count: int) -> np.ndarray:
    """复用官方原始/裁剪/反转与有放回补点语义；不新增其他增强。"""
    if variant == 1:
        points = random_crop(points.copy(), 240, 180)
    elif variant == 2:
        points = reverse_T(points.copy())
    if len(points) == 0 or np.ptp(points[:, 0]) <= 0:
        raise ValueError('增强后空窗口或时间跨度为零')
    return normaliztion(shuffle_downsample(points, count), 240, 180, True)


def convert_record(raw: Path, row: dict, output: h5py.File, args: argparse.Namespace) -> list:
    """处理一条原始记录，返回每个未写入窗口的明确原因。"""
    points, skipped = read_bin(raw / row['path']), []
    for window in range(3):
        subset = window_events(points, window)
        variants = range(3) if row['split'] == 'train' else range(1)
        for variant in variants:
            seed_all(args.seed + row['id'] * 100 + window * 3 + variant)
            try:
                if len(subset) <= 100 or np.ptp(subset[:, 0]) <= 0:
                    raise ValueError('官方条件：去重后须大于100事件且时间跨度为正')
                result = make_variant(subset, variant, args.num_points)
                if not np.isfinite(result).all():
                    raise ValueError('归一化后出现非有限值')
                append_row(output, result, row, window, variant)
            except (ValueError, RuntimeError) as error:
                skipped.append(dict(record_id=row['id'], split=row['split'],
                                    window=window, variant=variant, reason=str(error)))
    return skipped


def main() -> None:
    """冻结原始文件清单并处理；失败目录保留，重跑必须选择新目录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split-file', type=Path)
    parser.add_argument('--seed', type=int, default=1234,
                        help='数据划分/预处理种子，不是模型训练seed42')
    parser.add_argument('--num-points', type=int, default=10240)
    args = parser.parse_args()
    classes, records = record_list(args.raw_root, args.seed, args.split_file)
    args.out.mkdir(parents=True, exist_ok=False)
    dump_json(args.out / 'split_manifest.json', records)
    skipped, files = [], {}
    for split in ('train', 'test'):
        path = args.out / f'{split}.h5'
        with create_h5(path, args.num_points) as output:
            for i, row in enumerate(r for r in records if r['split'] == split):
                skipped.extend(convert_record(args.raw_root, row, output, args))
                if i % 100 == 0:
                    print(f'{split}: records={i + 1}, rows={len(output["label"])}', flush=True)
            files[split] = dict(rows=len(output['label']))
        files[split].update(sha256=sha256(path), bytes=path.stat().st_size)
    meta = dict(num_points=args.num_points, classes=classes, records=records,
                seed=args.seed, windows_ms=WINDOWS, files=files, skipped=skipped,
                split_provenance='provided_manifest' if args.split_file else f'reconstructed_seed{args.seed}_not_official',
                mark_namespace='global', augmentation_seed_strategy='record_window_variant',
                transforms_sha256=sha256(Path(__file__).with_name('official_transforms.py')))
    dump_json(args.out / 'dataset_manifest.json', meta)
    print(json.dumps(dict(files=files, skipped=len(skipped)), indent=2), flush=True)


if __name__ == '__main__':
    main()

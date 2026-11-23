"""联网端封存搬运目录；离线端校验文件、大小和SHA256。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    """只用标准库校验，尚未安装PyTorch也能运行。"""
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def seal(root: Path) -> None:
    """把所有待搬运文件登记为相对路径，不记录密钥或系统目录。"""
    manifest = root / 'transfer_manifest.json'
    if manifest.exists():
        raise FileExistsError('封存清单已存在，请使用新的搬运目录')
    rows = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=sha256(p))
            for p in sorted(root.rglob('*')) if p.is_file()]
    manifest.write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'SEALED {len(rows)} files; bytes={sum(x["bytes"] for x in rows)}')


def verify(root: Path) -> None:
    """逐文件校验；缺失、截断或内容变化均拒绝进入安装。"""
    rows = json.loads((root / 'transfer_manifest.json').read_text(encoding='utf-8'))
    for row in rows:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f'非法或缺失路径: {row["path"]}')
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise ValueError(f'搬运校验失败: {row["path"]}')
    print(f'TRANSFER_OK {len(rows)} files')


def main() -> None:
    """输入搬运根目录及seal/verify动作。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['seal', 'verify'])
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    (seal if args.action == 'seal' else verify)(args.root)


if __name__ == '__main__':
    main()

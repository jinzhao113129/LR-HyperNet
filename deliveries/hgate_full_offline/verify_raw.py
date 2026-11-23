"""核对数据集作者发布的原始事件压缩包；不把图像Caltech101当事件数据。"""
import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    """验证Caltech101.zip的已发布MD5并报告本地字节数和SHA256。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    md5, sha = hashlib.md5(), hashlib.sha256()
    with args.archive.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest() != '66201824eabb0239c7ab992480b50ba3':
        raise ValueError('原始包MD5不匹配；不要继续解压/生成，先核对来源')
    print(json.dumps(dict(file=args.archive.name, bytes=args.archive.stat().st_size,
                         md5=md5.hexdigest(), sha256=sha.hexdigest(), verified=True)))


if __name__ == '__main__':
    main()

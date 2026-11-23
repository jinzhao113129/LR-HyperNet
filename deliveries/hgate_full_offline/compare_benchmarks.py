"""比较同前向统计batch16的单卡/四卡测速；不以显存占用代替吞吐。"""
import argparse
import json
from pathlib import Path


def main() -> None:
    """只在实测更快时建议四卡，峰值及口径同时报告。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('single', type=Path)
    parser.add_argument('four', type=Path)
    args = parser.parse_args()
    a, b = [json.loads(p.read_text()) for p in (args.single, args.four)]
    if (a['world_size'], a['micro_batch'], a['accumulation']) != (1, 16, 2):
        raise ValueError('单卡对照应batch16/累积2')
    if (b['world_size'], b['micro_batch'], b['accumulation']) != (4, 4, 2):
        raise ValueError('四卡应每卡batch4/累积2')
    speedup = b['samples_per_second'] / a['samples_per_second']
    print(json.dumps(dict(speedup=speedup, use_four_gpu=speedup > 1,
                         single=a, four=b, warning='短时测速不替代首轮实际时间'), indent=2))


if __name__ == '__main__':
    main()

"""固定几何选点的假 batch 验收：索引有效、无重复、聚合器可反传。"""

import torch

from local_remote_grouper_v2 import LocalRemoteGrouper, RemoteSelector


def check_treatment(name: str) -> None:
    """验证指定 treatment 的选择器与分组器生产路径。"""
    torch.manual_seed(42)
    selector = RemoteSelector(remote_g=16, chunk=32, channel=8,
                              g16_treatment=name, selector_novelty_weight=0.5)
    center = torch.randn(1, 3, 8)
    points = torch.randn(1, 64, 8)
    coords = torch.rand(1, 64, 3)
    local = torch.randint(0, 64, (1, 3, 24))
    indices = selector(center, points, local, all_xyz=coords,
                       center_xyz=coords[:, :3])
    assert indices.shape == (1, 3, 16)
    for stage_center in range(3):
        chosen = indices[0, stage_center]
        assert chosen.unique().numel() == 16
        assert not torch.isin(chosen, local[0, stage_center]).any()
    group = LocalRemoteGrouper(channel=8, groups=8, kneighbors=24,
                               remote_g=16, g16_treatment=name,
                               selector_novelty_weight=0.5)
    feature = torch.randn(1, 64, 8, requires_grad=True)
    _, output = group(torch.rand(1, 64, 4), feature)
    assert output.shape == (1, 8, 24, 20)
    output.square().mean().backward()
    assert feature.grad is not None
    assert torch.isfinite(feature.grad).all()
    assert feature.grad.abs().sum() > 0
    assert group.hyper_agg.out_proj.weight.grad is not None
    print(f"{name}: indices + forward + aggregator backward PASS")

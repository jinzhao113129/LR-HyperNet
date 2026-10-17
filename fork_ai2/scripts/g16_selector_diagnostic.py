"""Read-only synthetic checks for the current G=16 remote selector."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from models.local_remote_grouper import RemoteSelector


def cells(coords: torch.Tensor) -> list[int]:
    """Return the current selector's 4x4 cell IDs for (t, x, y) rows."""
    truncated = coords[:, :2].clamp(0.0, 1.0 - 1e-6)
    bins = (truncated * 4).long()
    return (bins[:, 0] * 4 + bins[:, 1]).tolist()


def main() -> None:
    """Check coordinate cells and local-neighbor exclusion on tiny inputs."""
    coords = torch.tensor([[0.5, 0.5, y] for y in (0.1, 0.3, 0.6, 0.9)])
    actual = cells(coords)
    expected_xy = ((coords[:, 1] * 4).long() * 4 + (coords[:, 2] * 4).long()).tolist()
    print(f"fixed t,x; varied y: current cells={actual}; intended x,y cells={expected_xy}")
    assert len(set(actual)) == 1 and len(set(expected_xy)) == 4

    selector = RemoteSelector(remote_g=16, chunk=16, selector_mode="spatial_uniform")
    features = torch.arange(32, dtype=torch.float32).reshape(1, 32, 1)
    centers = features[:, :1, :]
    local_idx = torch.arange(16).reshape(1, 1, 16)
    xyz = torch.zeros(1, 32, 3)
    torch.manual_seed(42)
    selected = selector(centers, features, local_idx, all_xyz=xyz)
    overlap = torch.isin(selected, local_idx).sum().item()
    print(f"N=32, chunk=16, K=16, G=16: selected={selected.tolist()}, local overlap={overlap}")
    assert selected.shape == (1, 1, 16)
    assert overlap > 0, "The regression witness no longer reproduces"


if __name__ == "__main__":
    main()

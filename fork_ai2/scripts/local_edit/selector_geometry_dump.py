"""Selector geometry dump：每个中心 + 最合适的 16 个远程点（+局部/随机对照）。

用途（学长 2026-10-27 的路线）：先把「点」找出来，再做 PCA / 欧氏与双曲空间
分析，看好的远程点和中心是什么几何关系，最后才谈学习。

做法：加载一个训练好的 checkpoint，用 forward hook 抓住指定 stage 的 grouper
输入 (xytp, points)，在该 stage 的特征空间里：
  - 中心 = FPS 点（与 grouper 完全一致）；
  - 最合适的 16 个 = 排除 24 个局部近邻后、特征距离最近的 16 个（= euc_near，
    等价 RP1 oracle 的选点规则）；
  - 局部 24 个 = 特征距离最近的 24 个（我们的 local 定义）；
  - 随机 16 个 = 候选池随机采样（对照）。
然后抽样保存特征与坐标，并打印余弦/距离分布与前几个主成分解释方差。

用法（gpu-server-3，cd ~/zhaojin/pi2）：
  PYTHONPATH=$PWD:$PWD/models:. ~/zhaojin/venv/bin/python \
    tools/selector_geometry_dump.py --log_dir secnet_pi2_fca_baidu --stage 0

输出：~/zhaojin/logs_run/geom_<log_dir>_stage<k>.npz
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path[:0] = [".", "tools", "models"]
import extract_preds  # noqa: E402  复用 checkpoint 解析 + 模型构建
from modules import furthest_point_sample, index_points, square_distance  # noqa: E402


def _pca_stats(x: np.ndarray, k: int = 5):
    """对 [N, D] 做中心化 SVD，返回前 k 个解释方差比例。"""
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    # 样本多、维度小：用经济型 SVD
    _, s, _ = np.linalg.svd(x, full_matrices=False)
    var = s ** 2
    ratio = var / max(var.sum(), 1e-12)
    return ratio[:k]


def _dist_stats(a: np.ndarray, b: np.ndarray):
    """逐对 cosine 与 L2 的均值/标准差。a,b: [N, D]。"""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    cos = (an * bn).sum(axis=1)
    l2 = np.linalg.norm(a - b, axis=1)
    return (float(cos.mean()), float(cos.std()), float(l2.mean()), float(l2.std()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="secnet_pi2_fca_baidu")
    p.add_argument("--ckpt", default="best", choices=["best", "last"])
    p.add_argument("--stage", type=int, default=0)
    p.add_argument("--max_samples", type=int, default=32)
    p.add_argument("--centers_per_sample", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_point", type=int, default=10240)
    p.add_argument("--num_category", type=int, default=101)
    p.add_argument("--data_path", default="./data/")
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_name = "last_checkpoint.pth" if args.ckpt == "last" else "best_checkpoint.pth"
    ckpt_path = os.path.join("log", "classification", args.log_dir,
                             "checkpoints", ckpt_name)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_args = ckpt["args"]
    model, _flags = extract_preds.build_model(dict(saved_args), args.num_category)
    key = "net_ema" if (args.use_ema and ckpt.get("net_ema")) else "net"
    model.load_state_dict(ckpt[key], strict=True)
    model.to(device).eval()
    if args.stage >= len(model.local_grouper_list):
        raise ValueError(f"stage={args.stage} 超出范围（模型有 "
                         f"{len(model.local_grouper_list)} 个 stage）")
    grouper = model.local_grouper_list[args.stage]

    captured = {}

    def _hook(_mod, inp, _out):
        captured["xytp"] = inp[0].detach()
        captured["points"] = inp[1].detach()

    handle = grouper.register_forward_hook(_hook)

    loader, total = extract_preds.load_test_loader(
        args.data_path, args.num_point, args.batch_size)
    K_LOCAL, G_REMOTE = 24, 16

    centers, locals_, remotes, rands = [], [], [], []
    cent_xyz, local_xyz, remote_xyz, rand_xyz = [], [], [], []
    seen = 0

    with torch.no_grad():
        for data, _label in loader:
            if seen >= args.max_samples:
                break
            x = data.permute(0, 2, 1).to(device)  # [B,4,N]（训练同口径）
            captured.clear()
            model(x)
            if "xytp" not in captured:
                raise RuntimeError("hook 没抓到 grouper 输入")
            xytp = captured["xytp"]
            points = captured["points"]
            B, N, _ = xytp.shape
            S = grouper.groups

            # 与 grouper 完全一致的 FPS（fps_parameter 缩放后取中心）
            if getattr(grouper, "use_scale_norm_fps", False):
                xyz = ((xytp - xytp.mean(1, keepdim=True)) /
                       (xytp.std(1, keepdim=True) + 1e-5))
            else:
                xyz = grouper.fps_parameter * xytp
            fps_idx = furthest_point_sample(xyz[:, :, 0:3], S).long()
            fps_idx, _ = torch.sort(fps_idx, dim=1)
            centroid = index_points(points, fps_idx)          # [B,S,C]

            # 最近 40 个 = 24 局部 + 16 远程（euc_near = 排除局部后的最近 16）
            dists = square_distance(centroid, points)          # [B,S,N]
            top = dists.topk(K_LOCAL + G_REMOTE, dim=-1, largest=False)[1]
            local_idx = top[..., :K_LOCAL]
            remote_idx = top[..., K_LOCAL:]
            # 随机对照：从全池随机，拒绝落在 top40 里的（碰撞率 <0.5%）
            rand_idx = torch.randint(0, N, (B, S, G_REMOTE), device=device)
            in_top = (rand_idx.unsqueeze(-1) == top.unsqueeze(-2)).any(-1)
            rand_idx = torch.where(in_top, remote_idx, rand_idx)

            # 每个样本抽 centers_per_sample 个中心，控制文件大小
            sel = torch.stack([
                torch.randperm(S, device=device)[:args.centers_per_sample]
                for _ in range(B)
            ])
            b_idx = torch.arange(B, device=device).unsqueeze(-1).expand_as(sel)

            def take(src, idx):
                return index_points(src, idx)[b_idx, sel]      # [B,Q,...,C]

            c_feat = centroid[b_idx, sel]                       # [B,Q,C]
            l_feat = take(points, local_idx)
            r_feat = take(points, remote_idx)
            n_feat = take(points, rand_idx)
            l_xyz = take(xytp[:, :, 0:3], local_idx)
            r_xyz = take(xytp[:, :, 0:3], remote_idx)
            n_xyz = take(xytp[:, :, 0:3], rand_idx)
            c_xyz = take(xytp[:, :, 0:3], fps_idx)

            centers.append(c_feat.half().cpu())
            locals_.append(l_feat.half().cpu())
            remotes.append(r_feat.half().cpu())
            rands.append(n_feat.half().cpu())
            cent_xyz.append(c_xyz.half().cpu())
            local_xyz.append(l_xyz.half().cpu())
            remote_xyz.append(r_xyz.half().cpu())
            rand_xyz.append(n_xyz.half().cpu())
            seen += B

    handle.remove()

    def cat(xs):
        return torch.cat(xs, dim=0).reshape(-1, *xs[0].shape[2:]).numpy()

    out = {
        "center": cat(centers),
        "local": cat(locals_),
        "remote": cat(remotes),
        "random": cat(rands),
        "center_xyz": cat(cent_xyz),
        "local_xyz": cat(local_xyz),
        "remote_xyz": cat(remote_xyz),
        "random_xyz": cat(rand_xyz),
        "stage": np.array([args.stage]),
        "log_dir": np.array([args.log_dir]),
    }
    out_path = args.out or os.path.expanduser(
        f"~/zhaojin/logs_run/geom_{args.log_dir}_stage{args.stage}.npz")
    np.savez_compressed(out_path, **out)

    # ---- 快速统计（详细分析可后续读 npz）----
    c = out["center"].astype(np.float32)
    r = out["remote"].astype(np.float32)
    rnd = out["random"].astype(np.float32)
    l = out["local"].astype(np.float32)
    print(f"saved: {out_path}")
    print(f"centers: {len(c)}  C={c.shape[1]}  K_local={l.shape[1]}  G={r.shape[1]}")
    for name, feats in (("remote(=euc_near)", r), ("local", l), ("random", rnd)):
        # 每个中心与它的全部 16/24 个邻居逐对统计
        g = feats.shape[1]
        f_all = feats.reshape(-1, feats.shape[-1])
        c_all = np.repeat(c, g, axis=0)
        cos_m, cos_s, l2_m, l2_s = _dist_stats(c_all, f_all)
        print(f"{name:20s} cos(center,x)={cos_m:+.4f}±{cos_s:.4f}  "
              f"L2={l2_m:.4f}±{l2_s:.4f}")
    for name, feats in (("remote-center", (r - c[:, None, :]).reshape(-1, r.shape[-1])),
                        ("random-center", (rnd - c[:, None, :]).reshape(-1, r.shape[-1]))):
        ratio = _pca_stats(feats)
        print(f"PCA({name}) 前5主成分解释方差: " +
              " ".join(f"{v:.3f}" for v in ratio))


if __name__ == "__main__":
    main()

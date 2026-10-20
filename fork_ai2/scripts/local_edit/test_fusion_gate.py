"""融合层双候选组件回归测试（HyperbolicGate + FcaGate）。

可独立复跑：cd fork_ai2/scripts/local_edit && python test_fusion_gate.py
覆盖 Codex 审查（codex_review_fusion_gate_20261018.md）修复后的关键断言：
  - 两个组件 forward / 跨层组合 / 互斥 / DCT 退化 / seq_len 防御 / AMP dtype
  - P1-1 修复：HyperbolicGate 初始 gate ≈0.48（不再是塌缩的 ~0.0002）
"""
import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
from FECNet import FECNet, HyperbolicGate, FcaGate, HyperbolicRadialRectify, PolarFeatureLayer
from local_remote_grouper import lorentz_exp_origin, lorentz_distance

N = 2048
x = torch.randn(1, 4, N)


def check(name, fn):
    try:
        fn()
        print('[PASS]', name)
    except Exception as e:
        print('[FAIL]', name, '->', type(e).__name__, e)
        raise


def test_forward():
    for nm, kw in [('fca_gate', dict(use_fca_gate=True)),
                   ('hgate', dict(use_hyperbolic_gate=True)),
                   ('base', dict())]:
        m = FECNet(num_classes=101, **kw); m.eval()
        with torch.no_grad():
            y = m(x)
        assert tuple(y.shape) == (1, 101), (nm, y.shape)
        assert torch.isfinite(y).all(), nm
        print('  %-8s out=%s params=%d finite=%s'
              % (nm, tuple(y.shape), sum(p.numel() for p in m.parameters()),
                 bool(torch.isfinite(y).all())))


def test_combo():
    m = FECNet(num_classes=101, use_fca_gate=True, use_spike_time_enc=True,
               use_rp1_selector=True); m.eval()
    with torch.no_grad():
        y = m(x)
    assert tuple(y.shape) == (1, 101)


def test_hgate_initial_gate_neutral():
    """P1-1 修复：初始 gate 应 ≈0.48（alpha=softplus(alpha_raw)≈0.01，bias=0），
    不再是塌缩的 ~0.0002。"""
    g = HyperbolicGate(channels=132)
    x_src = torch.randn(2, 132, N)
    c = g._curvature()
    h = g.layer_norm(g.in_proj(x_src.permute(0, 2, 1)))   # [B,S,d_h]
    h_hyp = lorentz_exp_origin(h, c)
    centroid = lorentz_exp_origin(h.mean(dim=1, keepdim=True), c)
    dist = lorentz_distance(centroid.expand_as(h_hyp), h_hyp, c)
    alpha = F.softplus(g.alpha_raw)
    gate = torch.sigmoid(g.bias - alpha * dist)
    m = gate.mean().item()
    print('  alpha=%.4f bias=%.2f gate_mean=%.4f (应≈0.48)' % (alpha.item(), g.bias.item(), m))
    assert 0.40 < m < 0.50, m


def test_hgate_gradient_nonzero():
    """Codex 复审 P0：softplus 参数化后 alpha_raw/curvature/in_proj/bias 均应有
    非零梯度（abs 在 0 点梯度为 0 会卡死 alpha 及整个双曲分支）。"""
    g = HyperbolicGate(channels=132)
    x_local = torch.randn(2, 132, 512)
    x_src = torch.randn(2, 132, 512)
    out = g(x_local, x_src)
    out.sum().backward()
    for name, p in [('alpha_raw', g.alpha_raw), ('curvature', g.curvature),
                    ('in_proj', g.in_proj.weight), ('bias', g.bias)]:
        assert p.grad is not None and p.grad.abs().sum().item() > 0, name
        print('  %-10s.grad 非零: %.4g' % (name, p.grad.abs().sum().item()))


def test_fca_num_freq1_degenerate():
    g = FcaGate(channels=132, seq_len=N, num_freq=1)
    assert g.basis[0].std().item() < 1e-6                    # 常数向量
    assert abs(g.basis[0].mean().item() - 1 / N ** 0.5) < 1e-4  # = 1/sqrt(N)


def test_mutual_exclusion():
    try:
        FECNet(num_classes=101, use_hyperbolic_gate=True, use_fca_gate=True)
        raise AssertionError('should have raised')
    except ValueError:
        pass


def test_fca_seq_len_guard():
    g = FcaGate(channels=132, seq_len=2048)
    try:
        g(torch.randn(2, 132, 2048), torch.randn(2, 132, 1024))
        raise AssertionError('should have raised')
    except ValueError:
        pass


def test_fca_amp_dtype():
    """P1-2 修复：AMP（autocast）下 basis 自动 cast 到 fp16，不 dtype mismatch。
    手动 fp16 输入不是真实 AMP 场景（真实 AMP 用 autocast，Linear 由框架 cast）；
    这里用 GPU autocast 验证完整 forward。"""
    if not torch.cuda.is_available():
        print('  skip fca amp dtype (no cuda)')
        return
    g = FcaGate(channels=132, seq_len=1024).cuda()
    xs = torch.randn(2, 132, 1024, device='cuda')
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        out = g(xs, xs)   # 不抛 dtype mismatch 即通过；输出 dtype 由 autocast 决定
    assert torch.isfinite(out).all()


def test_chr_forward():
    m = FECNet(num_classes=101, use_hyperbolic_radial=True); m.eval()
    with torch.no_grad():
        y = m(x)
    assert tuple(y.shape) == (1, 101)
    assert torch.isfinite(y).all()


def test_chr_gradient_nonzero():
    r = HyperbolicRadialRectify(channels=64)
    xr = torch.randn(2, 128, 64)
    r(xr).sum().backward()
    for name, p in [('alpha_raw', r.alpha_raw), ('curvature', r.curvature),
                    ('in_proj', r.in_proj.weight), ('complexity', r.complexity.weight),
                    ('out_proj', r.out_proj.weight)]:
        assert p.grad is not None and p.grad.abs().sum().item() > 0, name


def test_chr_residual_scale():
    r = HyperbolicRadialRectify(channels=64)
    xr = torch.randn(2, 128, 64)
    with torch.no_grad():
        residual = (r(xr) - xr).abs().mean().item()
    assert 1e-3 < residual < 10.0, residual


def test_chr_not_identity():
    """chr 的 exp/log 不能恒等：径向校正必须在 Lorentz 流形内（防回归成装饰）。"""
    from local_remote_grouper import lorentz_exp_origin, lorentz_log_origin
    r = HyperbolicRadialRectify(channels=64)
    xr = torch.randn(2, 128, 64)
    c = r._curvature()
    h = r.layer_norm(r.in_proj(xr))
    kappa = torch.sigmoid(r.complexity(xr)); alpha = F.softplus(r.alpha_raw)
    h_hyp = lorentz_exp_origin(h, c)
    x0 = h_hyp[..., 0:1]; x_rest = h_hyp[..., 1:]
    x0_new = x0 * (1.0 + alpha * kappa)
    scale = torch.sqrt((x0_new ** 2 - 1.0 / c).clamp_min(1e-6)
                       / (x0 ** 2 - 1.0 / c).clamp_min(1e-6))
    h_euc = lorentz_log_origin(torch.cat([x0_new, x_rest * scale], dim=-1), c)
    assert (h_euc - h).abs().max().item() > 1e-3


def test_polar_forward():
    m = FECNet(num_classes=101, use_polar_feature=True); m.eval()
    with torch.no_grad():
        y = m(x)
    assert tuple(y.shape) == (1, 101)
    assert torch.isfinite(y).all()


def test_polar_gradient_nonzero():
    p = PolarFeatureLayer(channels=64)
    xp = torch.randn(2, 128, 64)
    p(xp).sum().backward()
    for name, param in [('alpha_raw', p.alpha_raw), ('in_proj', p.in_proj.weight),
                        ('complexity', p.complexity.weight), ('out_proj', p.out_proj.weight)]:
        assert param.grad is not None and param.grad.abs().sum().item() > 0, name


def test_polar_chr_mutual_exclusion():
    try:
        FECNet(num_classes=101, use_hyperbolic_radial=True, use_polar_feature=True)
        raise AssertionError('should have raised')
    except ValueError:
        pass


if __name__ == '__main__':
    check('forward', test_forward)
    check('combo fca+spike+rp1', test_combo)
    check('hgate initial gate neutral', test_hgate_initial_gate_neutral)
    check('hgate gradient nonzero', test_hgate_gradient_nonzero)
    check('fca num_freq=1 degenerate', test_fca_num_freq1_degenerate)
    check('mutual exclusion', test_mutual_exclusion)
    check('fca seq_len guard', test_fca_seq_len_guard)
    check('fca amp dtype', test_fca_amp_dtype)
    check('chr forward', test_chr_forward)
    check('chr gradient nonzero', test_chr_gradient_nonzero)
    check('chr residual scale', test_chr_residual_scale)
    check('chr not identity', test_chr_not_identity)
    check('polar forward', test_polar_forward)
    check('polar gradient nonzero', test_polar_gradient_nonzero)
    check('polar/chr mutual exclusion', test_polar_chr_mutual_exclusion)
    print('=== all passed ===')

# Codex survey: stacking hyperbolic attention (2026-10-15)

Primary source: Alyoussef et al., *HexFormer: Hyperbolic Vision Transformer with Exponential Map Aggregation*, arXiv:2601.19849v1 (2026-01-27). Local PDF: `D:/论文/2601.19849_HexFormer_HyperbolicVisionTransformer.pdf` (duplicate: `2601.19849v1_HexFormer_双曲ViT.pdf`), SHA256 `BD6107A50D3AE7D406331D1A87BFCA05CA10C76F68916F5B84E0B4F3FC7F5BD3`. Status checked as arXiv preprint; acceptance/peer-review status not independently verified.

## Direct answers

### 1. Depth

HexFormer's reported stable configurations are:

| Variant | Hyperbolic attention blocks | Width / heads | Result status |
|---|---:|---|---|
| ViT-Tiny | 9 | 192 / 12 | Main result, with and without warmup |
| ViT-Small | 12 | 384 / 6 | Main result, with and without warmup |
| ViT-Base | 12 | 768 / 12 | Main result, with and without warmup |

Thus the demonstrated maximum is **12 stacked hyperbolic attention blocks**, not 8-12 as a range. The paper does not test beyond 12 and does not report a failure threshold.

The most relevant failure result is at **9 layers**: with centroid aggregation and ordinary AdamW, 4 of 5 Tiny-ViT seeds produced NaN/Inf and training collapsed. With exponential-map aggregation (ExpAgg), all 5 seeds completed. Therefore, the relevant stability boundary is not simply depth; it is **depth x aggregation/optimizer/numerical form**.

### 2. Numerical protection

HexFormer does not rely on a broad "clamp everything" strategy. Its main protections are structural:

1. **Squared Lorentz distance for scores**: attention uses squared Lorentzian distance rather than directly taking `arccosh` in the score path.
2. **ExpAgg**: values are mapped with `log_0`, weighted in the tangent space, then mapped back with `exp_0`. This avoids weighted Lorentz centroids, whose denominator can collapse to zero.
3. **Scale and learnable temperature**: scores use `softmax(-d^2 / (sqrt(d_k) * tau))`.
4. **LorentzFC validity handling**: the time component is generated with a sigmoid-bounded affine map plus an epsilon satisfying the numerical-validity condition; the space components are rescaled to satisfy the Lorentz norm.
5. **Lorentz batch normalization in each block**.
6. **Pre-layer normalization** and residual connections applied to space-like components.
7. **Riemannian AdamW** for the main hyperbolic runs.

The paper explicitly explains why centroid aggregation can fail: a Float32 Lorentz norm can subtract two large nearly equal squared values and become zero; epsilon only prevents the literal divide-by-zero and can leave `~1e15` backward updates, causing divergence. ExpAgg removes this cancellation site instead of masking it.

### 3. Curvature

Main experiments use **fixed shared curvature K = -1**. The appendix reports:

- fixed `K=-0.5` and `K=-2`: stable but lower accuracy;
- learnable curvature in Tiny-ViT: slight CIFAR-100 improvement and no reported stability issue;
- no curvature schedule and no per-layer curvature in the main HexFormer results;
- no curvature trajectory or drift statistics are reported.

Therefore HexFormer does **not** establish that independently learnable per-layer curvature is stable at 9-12 layers. It only supports fixed shared curvature in the main results and a preliminary learnable-curvature check at Tiny scale.

### 4. Fully hyperbolic versus hybrid head

HexFormer-Hybrid uses a Lorentz encoder and a Euclidean linear head. It is consistently the best variant on the hierarchical image benchmarks:

| Benchmark / scale | HexFormer | Hybrid | Hybrid gain |
|---|---:|---:|---:|
| CIFAR-10, Tiny | 93.42 | 93.53 | +0.11 |
| CIFAR-100, Tiny | 75.65 | 75.85 | +0.20 |
| Tiny-ImageNet, Tiny | 60.87 | 62.93 | +2.06 |
| CIFAR-100, Small, warmup=10 | 75.54 | 76.09 | +0.55 |
| CIFAR-100, Base, warmup=10 | 76.49 | 76.81 | +0.32 |

The paper attributes the gain to reduced overfitting in the final classifier. On weak/flat-hierarchy datasets (MNIST, Fashion-MNIST, SVHN), all variants are essentially tied and Euclidean ViT is slightly better.

## Implications for SECNet/new architecture

1. **Do not directly infer that our zero-projection attention can be stacked to 12 layers.** HexFormer uses LorentzFC projections for Q/K/V, Lorentz BN, pre-LN, bounded LorentzFC time coordinates, and ExpAgg. Our current 6-scalar distance attention is a different, more minimal mechanism.
2. **A 3 -> 6 -> 9 -> 12 staged depth pilot is the right test.** The literature supports 12 blocks only with full stabilization machinery; it does not support assuming the same for a bare distance-attention residual.
3. **Avoid weighted Lorentz centroids in multi-layer attention.** Use tangent-space weighted aggregation followed by exponential map, as in HexFormer, or another cancellation-free form.
4. **Prefer squared Lorentz distance over raw `arccosh` in score computation.** If raw distance is required, monitor and clamp the `arccosh` argument, but treat this as a last-resort mask rather than the primary fix.
5. **Keep curvature fixed initially.** Shared `K=-1` or our existing `c=0.4` is safer. Learnable per-layer curvature should be a separate ablation after depth stability is established.
6. **Hybrid head is the stronger default.** For the new architecture, a hyperbolic encoder with a Euclidean classification head is better supported than a fully Lorentz MLR head. The Lorentz head can remain an ablation, not the main path.
7. **Required diagnostics for a depth pilot:** per-layer activation radius/norm, Lorentz-inner-product validity, `arccosh` argument margin, attention entropy, gradient norm, NaN/Inf counters, and curvature value (if learnable). These should be logged before claiming stability.

## Cross-checks

- Mishne et al., *The Numerical Stability of Hyperbolic Representation Learning* (ICML 2023; arXiv:2211.00181), quantifies Float64/Float32 representation-radius limits and cancellation in Lorentz operations. It supports HexFormer's diagnosis.
- Bdeir & Landwehr, *Rethinking the Fully Hyperbolic Vision Transformer in Polar Coordinates* (arXiv:2609.32899), reports that ambient Lorentz distance and centroid operations fail at large radii and proposes polar forms; it reinforces that clipping is not the only or best remedy.
- Guo et al., *Clipped Hyperbolic Classifiers Are Super-Hyperbolic Classifiers* (CVPR 2022), shows feature clipping can prevent vanishing gradients in hybrid Euclidean-to-hyperbolic classifiers, but also identifies a capacity/stability trade-off.
- DeepHGCN (arXiv:2310.02027) demonstrates deep hyperbolic graph networks up to 64 layers only with hyperbolic residuals, weight alignment, and feature regularization; vanilla HGCN collapses with depth. This is a useful warning that depth gains require residual/regularization design.
- LResNet (KDD 2025) provides a stable Lorentz residual based on a weighted Lorentz centroid, but HexFormer's NaN report is specifically about attention centroid aggregation; the exact operation and context differ.

## Bottom line

The literature supports **12-layer hyperbolic attention as feasible**, but not as a naive stack. The safest design rule is:

> squared-distance scores + tangent-space ExpAgg + fixed shared curvature + Euclidean/hybrid head + staged depth diagnostics.

For SECNet, the immediate architectural recommendation is to treat the current 3-stage attention as the depth-3 baseline, then run a smoke-scale 6/9/12 pilot with the above diagnostics before making multi-layer hyperbolic attention a core architecture claim.

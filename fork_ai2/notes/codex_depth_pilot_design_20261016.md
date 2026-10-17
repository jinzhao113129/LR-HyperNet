# Codex depth pilot design: 3 -> 6 -> 9 -> 12 hyperbolic attention

Goal: determine whether the SECNet 6-scalar, zero-projection Lorentz
distance attention can be stacked to 6/9/12 blocks before it becomes part
of the new architecture.

This is a smoke/stability pilot, not an accuracy benchmark.

## 1. Definitions

### Current zero-projection block

Current implementation: `models/FECNet.py`, `HyperbolicAttention`.

```text
x: [B,S,C]
c = softplus(c_param) + 1e-4
x_hyp = lorentz_exp_origin(x, c)
d(i,j) = lorentz_distance(x_hyp_i, x_hyp_j, c)
w = softmax(-d / (abs(T)+1e-6))
out = w @ value
residual = x + 0.1 * out
```

It has two trainable scalars per stage: curvature and temperature.
It has no Q/K/V projection and no tangent-space aggregation.

For this pilot, define a reusable block:

```text
HyperbolicDepthBlock(
    channels,
    depth_index,
    score_mode in {"raw_distance", "squared_distance"},
    aggregation in {"ambient", "expagg"},
    curvature_mode in {"fixed", "learnable"},
    residual_scale=0.1,
    pre_norm=False,
)
```

### Aggregation modes

`ambient` reproduces the current mechanism:

```text
out = softmax(score) @ value
```

`expagg` reproduces the HexFormer stable mechanism:

```text
V_tangent = lorentz_log_origin(V)
u = sum_i w_i * V_tangent_i
out_hyp = lorentz_exp_origin(u)
out = out_hyp[..., 1:]  # if the block returns Euclidean channels
```

Source: HexFormer PDF p. 5, equations (3)-(5).

### Score modes

`raw_distance`:

```text
score = -lorentz_distance(q, k) / temperature
```

`squared_distance`:

```text
d2 = -2/K - 2 * lorentz_inner_product(q, k)
score = -d2 / (sqrt(C) * temperature)
```

Source: HexFormer PDF p. 5, equations (1)-(2).

Use squared distance in the primary stable configuration. Do not take
`arccosh` in the score path for this mode.

## 2. Pilot objective

Answer three questions:

1. Does ambient aggregation fail at depth 6/9/12?
2. Does tangent ExpAgg fix that failure?
3. Is fixed shared curvature sufficient, or does learnable curvature
   destabilize deeper stacks?

Do not ask this pilot for final N-Caltech accuracy.

## 3. Configuration matrix

### Mandatory runs

| id | depth | score | aggregation | curvature |
|---|---:|---|---|---|
| D3-A-raw-fixed | 3 | raw | ambient | fixed |
| D6-A-raw-fixed | 6 | raw | ambient | fixed |
| D9-A-raw-fixed | 9 | raw | ambient | fixed |
| D12-A-raw-fixed | 12 | raw | ambient | fixed |
| D9-A-raw-learn | 9 | raw | ambient | learnable |
| D3-B-sq-fixed | 3 | squared | expagg | fixed |
| D9-B-sq-fixed | 9 | squared | expagg | fixed |
| D12-B-sq-fixed | 12 | squared | expagg | fixed |
| D12-B-sq-learn | 12 | squared | expagg | learnable |

The first four are the staged failure scan.
The last four are the primary stabilization candidates.

### Optional/secondary runs

Run only if the mandatory matrix is inconclusive:

- D6-B-sq-fixed
- D9-B-raw-fixed
- D12-A-raw-learn
- D6/D9/D12 with pre-norm residual
- Euclidean score control at depth 12

### Fixed settings

```text
seed: 0, 37, 42 for every failure or marginal run
batch: 2
tokens S: 128
channels C: 64
residual_scale: 0.1
max_steps: 1000
optimizer: AdamW lr=1e-3, no warmup in first pass
```

Run a second optimizer pass with Riemannian AdamW only for the two final
candidates if available. Ordinary AdamW is the stricter stability test.

## 4. Input generation

### Synthetic planted Lorentz tree

Generate:

```text
S=128
depth levels=8
angle: uniform on sphere in R^C
radius: planted increasing from level 1 to level 8
c=0.4
```

Map tangent vectors to Lorentz with `exp_origin`.

This gives known radial order and a controlled stress test. It also lets the
diagnostics be validated: shuffling the planted radius should remove radial
signal.

### Real feature smoke

Use one `official_gen` test batch:

```text
B=2, N=1024
run FECNet stage 0 only and collect [B,S,C]
```

This checks the actual feature scale. It is descriptive only; no accuracy
claim is made from this one batch.

## 5. Seven diagnostics

All diagnostics are collected per block, per step.

### D1 activation radius and norm

```text
rho = ||x_hyp[..., 1:]||_2
u = sqrt(c) * rho
H = u / asinh(u)
```

Report per block:

```text
rho median, p95, p99, max
u median, p95, p99, max
H median, p95, max
fraction u > 0.84
fraction u > 1
```

Provisional stability gate:

```text
u_max <= 4 in float32
H_max <= 1.8
```

These values must be calibrated on the planted tree before use.

### D2 Lorentz-inner-product validity

```text
err = max | lorentz_inner_product(x, x) + 1/c |
rel_err = err / (1/c)
```

Report p95/p99/max.

Provisional gate:

```text
rel_err_p99 <= 1e-3
```

### D3 arccosh argument margin

For raw-distance blocks:

```text
arg = -c * lorentz_inner_product(q, k)
arg_min, p1, median
fraction arg < 1 + 1e-4
```

Gate:

```text
arg_min >= 1 + 1e-6
fraction_clamped <= 1e-5
```

For squared-distance blocks, report the same inner-product margin but do
not use it in the score.

### D4 attention entropy

```text
p = softmax(score)
H_att = -sum(p * log(p + eps)) / log(S)
```

Report mean, p5, p95.

Gate:

```text
mean entropy >= 0.05 for at least 90% of steps
```

Near-one-hot attention for many steps is a warning, not automatically a
failure.

### D5 gradient norm

```text
g_global = total gradient L2 norm
g_curv = curvature parameter gradient norm
g_temp = temperature gradient norm
```

Report median, p95, p99, max over a 100-step window.

Gate:

```text
all values finite
p95 <= 20 in the first pass
```

Compare against the D3 baseline instead of using an absolute threshold
when possible.

### D6 NaN/Inf counters

Count:

```text
non-finite input
non-finite output
non-finite loss
non-finite gradient
```

Hard gate:

```text
all counters are zero for every completed seed
```

Any NaN/Inf run fails.

### D7 curvature

For learnable blocks:

```text
c_effective
sign and norm of dc/dstep
```

For fixed blocks, confirm no gradient path changes `c`.

Gate:

```text
1e-4 <= c_effective <= 2.0
```

Curvature collapse below `1e-4` is a warning and should trigger the
floor-release comparison later.

## 6. Hook and output interface

### Preferred implementation

`ai2_g16/hyper_depth_profile_agent.py`:

```text
class DepthDiagnostics:
    attach(module)
    forward_pre_hook(module, args)
    forward_hook(module, args, output)
    step_summary()
```

Every block updates an internal statistics buffer. An aggregator collects
them after `loss.backward()`.

### Output

```text
log/ai2_g16/depth_pilot/<run_id>/steps.jsonl
log/ai2_g16/depth_pilot/<run_id>/summary.h5
log/ai2_g16/depth_pilot/<run_id>/verdict.json
```

JSONL schema:

```text
run_id, seed, step, layer, metric, value
```

Verdict schema:

```text
{
  "run_id": "...",
  "passed": false,
  "failed_gates": ["D6", "D3"],
  "seed_results": [...]
}
```

### Relation to `g16_profile_agent.py`

Write a new agent. `g16_profile_agent.py` is selector-domain-specific and
should remain focused on remote/local/centroid selection. The new depth
agent can reuse the HDF5 append/schema conventions, but it should not be
extended into a dual-purpose tool.

## 7. Pass/fail decision

```text
pass =
    all three seeds complete
    D6 counters zero
    D1-D5 gates pass
```

Marginal:

```text
one seed fails only D1 or D4
```

Marginal means investigate, not promote.

Decision ladder:

1. If D3-A-raw-fixed fails, fix the harness first.
2. If D6/D9/D12 ambient fail and D9/D12 ExpAgg pass, use ExpAgg.
3. If both aggregation modes fail at depth 12, do not claim depth 12.
4. If learnable curvature fails, keep fixed shared curvature.

## 8. Execution phases

1. Local CPU synthetic pilot: all nine mandatory runs.
2. Local 4060 one-step real-feature smoke for memory and hook sanity.
3. HPC one-GPU 100-step timing/memory smoke for the two final candidates.
4. Freeze thresholds in `verdict.json`.
5. Only then consider a small real N-Caltech training comparison.

No full N-Caltech depth training is authorized by this pilot alone.


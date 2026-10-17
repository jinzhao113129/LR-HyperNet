# Codex deep read: hyperbolic operating-point audit

Source: `D:\论文\2607.05268v3_双曲几何真的在工作吗_审计.pdf`

Paper: *Is the Geometry Doing the Work? An Operating-Point Audit of
Hierarchy in Hyperbolic Vision-Language Models*, arXiv 2607.05268v3.

This note is an executable adaptation for SECNet / LRHyperNet, not a
summary of the audited VLM papers.

## 0. Scope and important caveat

The audit targets radial/cone hierarchy in VLM embeddings. Our system is
not a hierarchy-learning VLM and does not have trained entailment cones.
Therefore we should not mechanically transplant all five diagnostics.

The reusable core is:

1. measure the effective geometry with `u = sqrt(c) * rho`;
2. use matched interventions, not raw accuracy only;
3. use decoupling probes to show a metric does not identify geometry;
4. use shuffle/permutation controls where ordering is claimed;
5. report the operating point, not only scalar curvature.

The paper explicitly says a small `u` does not prove hierarchy is absent
(PDF p. 8). The same caution applies in reverse: a large accuracy delta
does not prove nonlocal geometry is the cause.

## Q1: complete audit method

### 1.1 Geometry operating point

For a Lorentz representation `x = (x0, x_space)`, define:

```text
rho = ||x_space||_2
u = sqrt(c) * rho
H(u) = u / asinh(u)
```

Source: PDF pp. 4-5, Section 3.2.

Meaning:

- `H(u) ~= 1`: locally Euclidean.
- `u > 0.84`: exact local metric distortion exceeds 10%.
- `u > 1`: conservative "nonlocal hyperbolic" marker.
- The `sinh(u)/u` proxy is nearly identical below `u=0.37`.

This is the central audit quantity. Scalar `c` alone is not enough:
curvature can fall while `rho` grows and preserve `u`.

### 1.2 Curvature-floor release intervention

They train matched baseline and `clampOff` runs that differ only in the
lower curvature clamp: `0.1 -> 0.001`, same data, optimizer, batch order,
hyperparameters, and budget (PDF p. 7, Section 4.2).

This is directly reusable in our code:

```text
--curvature_floor 1e-4   # matched baseline
--curvature_floor 1e-6   # clampOff intervention
--freeze_curvature       # additional fixed-curvature control
```

The intervention answers whether learned curvature is doing anything or
whether norm/radius growth compensates.

### 1.3 Downstream decoupling probes

They evaluate retrieval, zero-shot classification, compositionality, and
hierarchy metrics under the curvature intervention. The conclusion is not
"small changes mean the geometry is bad"; it is "mixed, modest changes
while the operating point stays near-Euclidean mean these metrics are not
diagnostic of geometry" (PDF pp. 8-9, Section 4.3).

Our adapted probes:

- fixed-seed test accuracy;
- exact McNemar / paired difference;
- selected-remote-point overlap with Euclidean treatment;
- occupancy, XY distance, temporal gap, feature cosine;
- selection latency and memory.

### 1.4 Direct radial and angular diagnostics

For VLMs they use:

- directed parent-child norm ordering;
- 10,000-permutation pairing-shuffle null;
- `z >= +1.6` one-sided detection;
- cosine-only versus cosine-plus-norm taxonomy decomposition;
- `delta R_norm^2` beyond cosine.

Source: PDF pp. 5-6, 9-10, Sections 3.4 and 5.1.

For us, if we claim class-level radial structure:

- use the WordNet hierarchy of the 101 N-Caltech classes;
- compare class-centroid Lorentz spatial norm order against a shuffle null;
- report `z`, not raw consistency.

This should be separate from the G16 selector claim. A selection rule is
not a hierarchy claim.

### 1.5 Cone activity and traversal

Cone aperture and semantic traversal are VLM-specific and not applicable
to our current FECNet/LRHyperNet architecture. Do not import them without
an entailment objective.

### 1.6 Gradient attribution

They decompose `dL/d(log c)` by objective term and show entailment drives
curvature down (PDF pp. 15-18, Section 7.3).

Our equivalent:

```text
for each parameter named "curvature":
    g_c = parameter.grad
    log g_c_norm and sign
```

Also run:

- entailment/regularizer-off ablation equivalent to our curvature regularizer
  removal;
- fixed curvature versus learnable curvature;
- curvature floor release.

### 1.7 Planted controls and sensitivity calibration

They validate with planted hierarchy and report minimum detectable effects:

- directed test 80% power at about `delta_pair = 13.5pp`;
- taxonomy test 80% power around `delta_R^2 = 0.0089-0.0090`.

Source: PDF p. 21 and Appendix F.

We should do the same before interpreting a negative distribution-separation
result.

## Q2: how to implement `u = sqrt(c) * rho` for our event point cloud

### Which `c`

| Site | Current parameter | Current positivity rule |
|---|---|---|
| `RemoteSelector` `hyp_far` | fixed buffer `c` | not learnable |
| `HyperbolicRemoteAggregator` | `nn.Parameter(curvature)` | `abs(c) + 1e-4` |
| `HyperbolicAttention` | `nn.Parameter(curvature)` | `softplus(c) + 1e-4` |
| `HyperbolicPrototypeClassifier` | `nn.Parameter(curvature)` | `abs(c) + 1e-4` |

The audit should use the `c` that feeds the operation whose geometry is
being claimed. For G16 selector evidence, record both selector `c` and
aggregator `c`.

### Which `rho`

Use the Lorentz spatial norm of the embedding/feature after `exp_origin`,
not the raw `(t,x,y)` coordinate radius:

```text
rho = ||x_hyp[..., 1:]||_2
u = sqrt(c) * rho
H = u / asinh(u)
```

Record at these points:

1. selector candidates and centroid, after `lorentz_exp_origin`;
2. remote aggregator input `x_hyp`, anchor, and aggregated `agg_hyp`;
3. hyperbolic-attention query/key/value and output, if attention is on;
4. final embedding before classification.

### Minimum code changes

In `models/local_remote_grouper.py`:

```text
1. Add a helper:
   geometry_stats(x_hyp, c)
   -> rho, u, H, inner_product_error

2. In RemoteSelector._hyp_dist_chunk:
   cache last stats for centroid_h and points_h.

3. In HyperbolicRemoteAggregator._aggregate:
   cache last stats for x_hyp, anchor, agg_hyp.

4. In HyperbolicAttention.forward:
   cache last stats for query/key/value manifold points.
```

Do not add these to the default profiling sink permanently. Use an
opt-in `--geometry_audit` flag and an `ai2_g16/geometry_audit_agent.py`.

### Thresholds and report

Use the paper's thresholds exactly:

| Quantity | Report | Meaning |
|---|---|---|
| `u` median/p95/p99/max | yes | operating point |
| fraction `u > 0.84` | yes | samples above 10% distortion |
| fraction `u > 1` | yes | nonlocal marker |
| `H(u)` median/p95/max | yes | local distortion |
| scalar `c` trajectory | yes | curvature collapse |

Our stability target is not necessarily `u > 0.84`. A model can be useful
in the near-Euclidean regime. We report these values to prevent an
unsupported "the geometry is doing the work" claim.

### Curvature perturbation probe

For a fixed checkpoint, run inference with:

```text
c_eff in {0.5c, c, 2c}
```

Then compare:

- selected remote indices;
- prediction flips;
- `u` distribution;
- occupancy and feature-cosine profiles.

If selection and accuracy are nearly invariant to a factor-two curvature
change while `u` changes, the nonlocal curvature is not carrying the
measured signal.

## Q3: is profile-agent separation accepted evidence?

Short answer: useful, but not sufficient under this audit.

What `g16_profile_agent.py` currently provides:

- occupancy gain;
- spatial novelty;
- XY distance ratios;
- temporal gaps;
- feature cosine;
- clamp rates.

These establish that `hyp_far`, `euc_far`, and other treatments change
the selector policy. That is evidence of "geometry changes behavior",
but not yet evidence that nonlocal hyperbolic curvature is the operative
mechanism.

A reviewer can correctly say:

> The selected sets differ because the scoring function differs. That does
> not prove the radial curvature regime is active.

Add these before claiming mechanism:

1. `u/H` statistics at selector and aggregator.
2. Euclidean-versus-hyperbolic rank agreement:

```text
Kendall tau between Euclidean-distance ranks and Lorentz-distance ranks
```

   If `tau ~= 1` for most samples, the hyperbolic score is only a monotone
   transformation of Euclidean distance in the observed regime.

3. Curvature perturbation as above.
4. Matched fixed/learnable curvature and floor-release runs.
5. Separability test: report the distribution distance under a permutation
   null, not only a heatmap difference.

## Q4: does the 7.6pp 2x2 interaction survive this audit?

### What is strong

The 2x2 design is the right shape:

```text
coordinate scale x {unchanged, learned scaled}
grid geometry  x {Euclidean, hyperbolic}
```

The same coordinates, same FPS budget, and a large interaction delta are a
much stronger design than comparing two independently trained architectures.

### Weaknesses a reviewer can raise

1. It is one training seed.
2. The hyperbolic grid changes bin boundaries, bin area, and edge density;
   the effect may be a discrete grid/edge effect, not nonlocal curvature.
3. Scaling coordinates can push points outside `[0,1]`; clamp behavior may
   differ between grid families.
4. No `u` audit shows the hyperbolic grid operates above 0.84.
5. No matched Euclidean grid with the same effective radial bin density.
6. Test-set selection and multiple comparisons are not pre-registered.
7. The original baseline was a small/baidu protocol result, so the absolute
   delta must be reproduced on `official_gen` before use.

### Required upgrades

- Replicate on training seeds 7, 13, 42.
- Report paired delta and exact McNemar/permutation `p`; do not select
  best-on-test.
- Keep identical point count, optimizer, seed, augmentation, and parameter
  budget.
- Record center overlap, grid occupancy, clamp rate, and coordinate
  distribution for both grids.
- Run the `u` audit on both arms.
- Add a matched Euclidean-grid control whose radial/bin density is
  explicitly matched to the hyperbolic grid, if we claim geometry rather
  than coverage.

## Q5: five likely reviewer attacks and responses

### Attack 1

> Your accuracy gain is selection noise or regularization. You have no
> evidence that nonlocal hyperbolic geometry is active.

Response:

- report `u=sqrt(c)*rho` distributions and `H(u)`;
- show selection changes under controlled curvature perturbation;
- run fixed-versus-learnable and floor-release matched runs;
- do not claim curvature from accuracy alone.

### Attack 2

> Your 7.6pp interaction is one seed and one test split.

Response:

- train seeds 7/13/42;
- fixed eval seed 42;
- paired exact McNemar or permutation test;
- pre-register before seeing full test results.

### Attack 3

> The hyperbolic grid differs in bin geometry and density, not geometry.

Response:

- report occupancy and edge coverage;
- add a matched Euclidean radial-density control;
- show hyperbolic-distance sensitivity beyond rank-preserving Euclidean
  distance;
- isolate coordinate scaling in a 2x2 factorial analysis.

### Attack 4

> Your curvature collapses to the implementation floor.

Response:

- report curvature trajectories per stage;
- release the floor in a matched run;
- decompose `dL/d(log c)`;
- show whether accuracy/selection behavior changes when `c` changes.

### Attack 5

> Selection distribution separation is only angular/feature change.

Response:

- report hyperbolic-vs-Euclidean rank agreement;
- perform angular/radial decomposition when class hierarchy is used;
- use matched Euclidean treatment and permutation nulls;
- state the claim precisely: "the selector policy changes and remains
  coupled to curvature", not "a radial hierarchy has been learned".

## Immediate execution order

1. Write `geometry_audit_agent.py` that hooks the three hyperbolic sites
   and logs `u/H/c/inner-product`.
2. Run it on existing FECNet/LRHyperNet smoke checkpoints.
3. Add the matched curvature floor and perturbation probes.
4. Use the results only to calibrate thresholds; do not yet write a paper
   claim.
5. Re-run all strong claims on `official_gen`, three seeds, fixed eval seed.


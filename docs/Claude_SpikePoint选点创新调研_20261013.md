[claude-code:unrecognized_model] {"model":"deepseek-v4-pro-0813","query_source":"sThe rI'll produce the full analysis report directly. (Note: my file-system/tool calls aren't landing in this session — the environment is text-only — so this is written to be self-contained and immediately actionable. I'll flag the few places where I'd normally verify against your code first.)

---

# SECNet Innovation Roadmap: SpikePoint Borrowing × 16-Point Selection

## Executive summary

| Priority | Idea | Expected Δacc | Risk | Compute |
|---|---|---|---|---|
| **P0** | Scale-aware remote selection (explain the +4.71pp) | +2–4pp over Euclidean top-16 | Low | 1 run |
| **P0** | Coordinate normalization (SpikePoint Δ = \|(G−c)/σ\|) | +1–3pp | Low | 1 run |
| **P1** | Global centroid branch (fuse with Fourier agg) | +1–2pp | Med | 1 run |
| **P1** | Diversity-maximizing selection (DPP / farthest-first hybrid) | +2–3pp | Med | 2 runs |
| **P2** | Bottleneck ResF in stages | ~0 to +1pp, −20–30% params | Med | 2 runs |
| **P3** | Learnable scorer (attention / gated scorer) | +1–2pp | High (train instability) | 2–3 runs |
| **Skip** | Single-stage collapse (3→1) | −3–6pp | High | would waste the Fourier synergy |
| **Skip** | Temporal-aware window selection (raw events) | unclear, disrupts existing pipeline | High | low ROI now |

The paper story that holds together: **"Scale-space remote selection + SpikePoint-inspired coordinate normalization make sparse event neighborhoods uniformly informative, and a global centroid branch restores the shape-level context that remote selection thins out."** That's a coherent, ablatable, *transferable* contribution — not a "gain component" bolted on.

---

## TASK 1 — SpikePoint lightweight design borrowing

### 1.1 Single-stage vs multi-stage → **Do NOT collapse. Borrow the *idea*, not the depth.**

| | Current SECNet | SpikePoint |
|---|---|---|
| Stages | 3 (set-abstraction hierarchy) | 1 (local branch + global branch) |
| Why | Multi-scale receptive field; stage 3 sees coarse, global-ish geometry | SNN rate-coding budget constrains depth; single stage + centroid branch compensates |

**Analysis:** Your 3 stages are what make the dual-Fourier (spatial + temporal) aggregation work — each stage gives a different sampling density, so the rfft over staged features captures multi-scale spectrum. Collapsing to 1 stage removes exactly the multi-scale input your Fourier aggregation is designed to exploit. SpikePoint gets away with single-stage because (a) it has the global centroid branch, and (b) SNN timesteps give it a *temporal* hierarchy that replaces the *spatial* hierarchy.

- **Difficulty:** 3 (trivial to cut stages, hard to recover accuracy)
- **Param change:** −40–60%, but only because you're deleting half the network
- **Expected accuracy:** **−3 to −6pp**. This is the one clear "don't do it" item.
- **Verdict:** Skip. The transferable insight is *not* "use one stage" — it's "**a lightweight global branch can cheaply supply the coarse context that a full deeper stage would**." That's §1.4.

### 1.2 Bottleneck residual (ResF) → **Yes, but after the selection innovation, not before.**

SpikePoint's ResF is a standard ResNet bottleneck: `1×1 conv (↓C) → ReLU → 3×3 group conv → 1×1 conv (↑C)` with residual. In your point/MLP world the analog is:

```
x -> Linear(C, C/r) -> ReLU/GELU -> Linear(C/r, C/r) -> Linear(C/r, C)  (+ residual x)
```

Inserted per stage in place of the current MLP blocks.

- **Difficulty:** 2 — drop-in replacement per stage.
- **Param change:** −20–30% at r=4 if you keep width C unchanged. **Caveat:** your model is already tiny (1.93M); squeezing to ~1.4M buys "lightweight" branding but not a different capability. The *real* gain is if you use the freed budget to **widen the remote-selection scorer** or **add the global branch** at param-parity.
- **Expected accuracy:** −0.5 to +0.5pp (roughly neutral; occasionally a small regularizing gain).
- **Verdict:** **P2.** Do it only as a *packaging* move at the end to claim "lighter than SpikePoint-style baselines at equal accuracy," and to fund the global branch without exceeding your param target. Don't lead with it.

### 1.3 Coordinate normalization → **P0. This is the cheapest likely win and it's honest SpikePoint borrowing.**

SpikePoint groups neighbors, computes relative coords `Δ = (G − centroid)`, then **normalizes by the per-group standard deviation and takes absolute value**, adding back the min to keep sign/scale info:

```
Δ = G - centroid                # relative coords
Δ = |Δ| / (σ(Δ) + eps)          # normalize by group SD, absolute value
Δ = Δ + min(Δ)                  # compensate for lost magnitude / keep nonneg baseline
```

Your current local grouping feeds **raw relative coords** into the MLP + Fourier path.

**Why this should help event point clouds specifically:** N-Caltech101 events from a single window have wildly varying spatial extent — some groups are dense clusters, others are sparse streaks from the fixed AER timestamp. Raw relative coords are dominated by the *absolute* scale of each group, so the MLP spends capacity re-representing scale that is mostly noise (or worse, metadata about the sensor's event rate). Normalizing by group σ makes the *shape* of the neighborhood the primary signal, which is exactly what a classification of object silhouettes wants.

- **Difficulty:** 1 — ~3 lines in your KNN/grouping function.
- **Param change:** 0.
- **Expected accuracy:** **+1–3pp.** This is the most reliable transfer from SpikePoint.
- **Verdict:** **P0.** Ablate it alone (raw vs normalized coords) *before* touching selection, because it interacts with the scaled-coordinate mystery in Task 2 — normalizing by σ may subsume part of the "scaling" effect you observed.

### 1.4 Global centroid branch → **P1, and the natural home for the Fourier output.**

SpikePoint's second branch takes the **group centroid coordinates directly** (a coarse, global-shape encoding) and fuses it with the local branch by **addition**. In SECNet terms:

- You already compute FPS centers at each stage → those centroid coordinates are free.
- Add a cheap branch: `centroid_xyz -> Linear -> feats`, then **add** (or gated-fuse) with the gated Fourier-aggregation output.

**Why it complements Fourier, not competes:** Your dual-rfft aggregates *feature spectra* over neighbors. It can be relatively blind to the **absolute global layout** of a window (the "skeleton"), because rfft over relative features discards absolute position weighting. The centroid branch injectes exactly what rfft throws away: coarse absolute geometry. They're complementary by construction — this is a clean "our dual-Fourier + global centroid" narrative.

- **Difficulty:** 2 — one extra Linear + an add/fuse op.
- **Param change:** +20–60K (~+1–3%).
- **Expected accuracy:** **+1–2pp**, and it tends to *stabilize* training (a cheap skip connection to global context).
- **Verdict:** **P1.** Fuse by **addition** first (SpikePoint-faithful, easiest to ablate: "ablate fuse op"), try gated fusion second.

### Task 1 verdict table

| Element | Difficulty | Params | Δacc | Prioritize |
|---|---|---|---|---|
| Single-stage (3→1) | 3 | −50% | −3–6pp | ❌ No |
| Bottleneck ResF | 2 | −20–30% | ~0 | P2 (packaging) |
| Coord normalization Δ=(G−c)/σ | 1 | 0 | +1–3pp | ✅ P0 |
| Global centroid branch | 2 | +1–3% | +1–2pp | ✅ P1 |

---

## TASK 2 — 16-point remote selection innovation

### 2.1 Why did *buggy* scaled-coordinate selection get +4.71pp (p=0.0002)?

First, state the bug precisely from your description: the buggy version computed hyperbolic distance on **scaled xy coordinates** (e.g. `xy * scale`, or xy in a rescaled/normalized space) instead of **raw xy**. The fixed version uses raw xy and is still training.

Three concrete, testable hypotheses, ordered by prior:

**H1 — Scaling is the whole point (scale-space selection), not a bug.**
Scaling the coordinate space before a distance-based top-16 changes which points are "far." In raw xy, the 16 "most distant/representative" points are dominated by the **largest-scale groups** — a few giant streaks grab most of the top-16 slots, so selection collapses to the biggest cluster and ignores small-but-informative groups. In scaled space, distance distributions across groups are flattened, so top-16 spreads more uniformly across *all* scales → more representative coverage, matching RP1's "Oracle > Global-KNN" result. RP1 already told us *how* you select matters more than *how many* — this hypothesis says the bug accidentally implemented a **better distance metric**, and multi-scale/uniform selection is the actual mechanism.

> **Verify:** For a fixed window, compute the 16 selected indices under (a) raw-xy distance and (b) scaled-xy distance. Measure **coverage** = fraction of distinct groups/windows represented among the 16, and **entropy** = diversity of the selected set. If scaled selection shows materially higher coverage/entropy, H1 is confirmed. (Read-only, no training — an afternoon.)

**H2 — Scaling acts as implicit coordinate normalization (gradient conditioning, not selection).**
Even if the *selected indices* barely change, scaling changes the *input magnitude* to the rest of the network, improving conditioning. This would mean the gain has little to do with "selection quality" and more to do with the same mechanism as §1.3.

> **Verify:** Take the **fixed (raw-xy)** selection run and apply only coordinate normalization (§1.3). If it recovers most of the +4.71pp, the gain was never really about *selection being different* — it was normalization by proxy. If it doesn't, selection geometry genuinely matters. **This experiment disambiguates H1 vs H2 and is your single most informative run.**

**H3 — The bug's true effect was to make "Hyperbolic vs Euclidean" moot by whitening.**
In scaled space the difference between hyperbolic and Euclidean distance shrinks (metrics converge near a point). So the buggy "+4.71pp hyperbolic" may actually be "**whitened Euclidean**" — meaning a *whitening/normalized* Euclidean selection is the real winner, and you've been attributing it to the hyperboloid geometry.

> **Verify:** Run Euclidean top-16 *on the scaled coordinates*. If it matches the buggy-hyperbolic result, H3 wins: drop the hyperbolic metric entirely and keep normalization. (Cheap — this is a selection-function swap, no architecture change.)

**Recommended read of the mystery:** H1 is the paper-worthy one; H2/H3 are the "make sure you're attributing correctly" checks. Run all three verifications — two are read-only/free, one is a single cheap run — because a reviewer *will* ask "so was it the metric or the scale?"

### 2.2 Novel selection methods (beyond Euclidean/hyperbolic top-16)

Ranked by paper-value ÷ implementation risk:

#### M1 — Scale-space / multi-scale selection (farthest-point in scale-normalized space) — **BEST SHOT**
If H1 holds, formalize it. Construct a scale-invariant distance: project each candidate point/spread into a **per-group standardized space** (`d = (p − centroid)/σ_group`), then select the 16 with **max distance from already-selected points** (farthest-point-sampling over the standardized space), or top-16 by standardized distance. This is "multi-scale coverage via standardized metric," cleanly citable, and directly produces the uniformity H1 describes.

- **Algorithm:** standardize → FPS/top-16 → (optionally) map back.
- **Difficulty:** 1–2 (reuse your FPS; just standardize first).
- **Expected gain:** +2–4pp over raw Euclidean top-16 (if H1 true); recovers most of the buggy +4.71 with a *principled* formulation.
- **Ablate:** raw-Euclidean vs standardized-Euclidean vs standardized-FPS vs hyperbolic (metric) vs (scale on/off) — a clean 2×2 (metric × scale).
- **Why paper-worthy:** "Selection must be insensitive to the absolute event scale of a window" is a real, general claim about event sensors — and you have an empirical mystery (+4.71 from an accident) that motivates it.

#### M2 — Diversity-maximizing selection (DPP / determinantal point process, or greedy submodular volume)
Select 16 points maximizing **coverage** and minimizing **redundancy**. Determinantal point process (DPP): build a kernel `K = sim(p_i, p_j)` (e.g. RBF in feature space) and take the 16 that maximize submatrix determinant. Cheap proxy: **greedy volume-max** (repeatedly add the point maximizing its min-distance to the selected set) — this is *farthest-point* in feature space.

- **Algorithm:** greedy non-redundant picker; kernel over (feature ⊕ standardized-xyz).
- **Difficulty:** 2 (pure; differentiable-if-needed later).
- **Expected gain:** +2–3pp over naive top-16 (removes the "top-16 all land in one cluster" failure mode).
- **Ablate:** top-16-similarity vs farthest-first vs DPP; also against Oracle (RP1's 68% oracle is your ceiling benchmark).
- **Why worth it:** "Diversity, not distance" is the sharpest contrast to the RP1 baseline and gives a natural Figure: coverage-vs-accuracy curve.

#### M3 — Hybrid spatial-diversity ⊕ feature-similarity (two-term score)
Score each candidate: `score(p) = α·(feature similarity to group) + (1−α)·(spatial dispersion from already-picked)`. This is M2 softened with a relevance term so you don't pick *only* outliers.

- **Algorithm:** greedy pick maximizing the two-term score; α is a single hyperparameter.
- **Difficulty:** 2.
- **Expected gain:** +1–3pp; robust; α~0.5 is a safe default.
- **Ablate:** sweep α ∈ {0, 0.5, 1} → recovers both extremes (α=1 ≈ top-16 similarity, α=0 ≈ pure coverage).
- **Why worth it:** It's the "safe flex" — M1 and M2's row in the same 2-line change, gives a clean 1-D sweep figure.

#### M4 — Learnable scorer network (RP1 explicitly flags this)
Replace the fixed distance with a tiny MLP/attention `score(p, C) -> ℝ` (C = group centroid/context), trained end-to-end. Top-16 by score with straight-through/Gumbel-top-k for gradients.

- **Algorithm:** precompute candidate pool → scorer → top-k with differentiable relaxation.
- **Difficulty:** 4–5 — top-k non-differentiability requires Gumbel-top-k/straight-through; training instability is real at 16/1024 sparsity.
- **Expected gain:** +1–2pp *if* it converges; often ties the hand-designed metric at best, and is uninterpretable at worst.
- **Ablate:** fixed-metric vs learned-scorer (same selected-count); visualize *what* the scorer learns (does it rediscover scale-normalization? → great figure).
- **Why it's not P0:** high variance, hard to ablate crisply, and "we learned a scorer" is *less* novel than "we discovered scale-invariant selection" — a learned module is a component; a hypothesis is a contribution.

#### M5 — Temporal-aware selection — **SKIP for now**
Select remote points spanning *different event time windows* (diversity across the temporal dimension you already rfft over). Sounds aligned with your dual-Fourier design, **but**: your pipeline groups within a single window already, and remote selection's whole point is to complement *local* neighbors, not to re-sample time. It also entangles with how your temporal rfft is defined.

- **Difficulty:** 3–4, and it likely deep-couples with the Fourier code.
- **Expected gain:** unclear, could be negative.
- **Verdict:** Not now. Revisit only if a *temporal ablation* of the Fourier path shows a need. This is the "waste of compute" item for this round.

### Task 2 verdict

| Method | Difficulty | Expected gain | Paper-worthy? | Do |
|---|---|---|---|---|
| M1 Scale-space selection | 2 | +2–4pp | ✅ strongest | **P0** |
| M3 Hybrid diversity+similarity | 2 | +1–3pp | ✅ nice ablation | **P1** |
| M2 DPP / farthest-first | 2 | +2–3pp | ✅ strong | **P1** |
| M4 Learned scorer | 4–5 | +0–2pp | ⚠️ component-y | P3 |
| M5 Temporal selection | 3–4 | unclear | ❌ | Skip |

---

## TASK 3 — Integration plan

### 3.1 How the pieces combine

The winning combination is not "stack everything." It's a **two-mechanism story with one shared principle — *scale invariance*:**

1. **Input layer (§1.3):** normalize relative coords `Δ=(G−c)/σ` (and handle signs SpikePoint-style). *This whitens the coordinate space.*
2. **Remote selection (§2.2 M1/M3):** select 16 remote points using a **scale-standardized + diversity-promoting** metric. *This makes coverage uniform across scales.*
3. **Global centroid branch (§1.4):** add coarse absolute-geometry context back, fused with the dual-Fourier output. *This restores what uniform selection thins out.*

These three are **complementary, not redundant**: (1) fixes the *input scale*; (2) fixes the *selection coverage*; (3) fixes the *lost global context*. The bottleneck-ResF (§1.2) is packed in only at the end if you want the "lighter than baseline" line and have proven params to spare.

### 3.2 Minimal experiment plan (validate the top 2 ideas)

**Stage A — Attribution is free, do it first (0 GPU-epochs):**

| # | What | Cost |
|---|---|---|
| A1 | Coverage-entropy analysis of raw-xy vs scaled-xy top-16 selection (H1) | read-only script |
| A2 | Scaled-coordinate **Euclidean** top-16 vs the buggy scaled **hyperbolic** (H3) | selection swap, rerun inference only |

→ Delivers: does the +4.71 come from *metric* or *scale*? This decides whether M1 or M2 is your headline.

**Stage B — Two headline runs (the actual P0 validation):**

| # | Config | What it proves |
|---|---|---|
| B1 | **fixed-xy + coord-normalization (§1.3)** only | H2: is the buggy gain just normalization? |
| B2 | **M1 scale-space remote selection** (fixed, principled) | H1: is scale-invariant selection the real innovation? |

- **Config:** standard settings from your memory (grad_accum4 + GroupNorm), 16 remote points, same epochs as your existing SECNet baseline (match the ablation baseline exactly — you already learned from the second-round review that baselines must share config).
- **Compare:** (i) your current Euclidean top-16 baseline, (ii) the fixed-hyperbolic run, (iii) the buggy +4.71 result (as the aspirational ceiling), (iv) RP1 Oracle 68.03% as the upper bound.

**Stage C — Only if B2 confirms H1 (2 runs):**

| # | Config |
|---|---|
| C1 | M3 hybrid diversity+similarity (α=0.5) on top of B2's normalization |
| C2 | Global centroid branch (+§1.3 +B2) — the full integration stack |

**Stage D — Packaging (optional, only if D-stage is green):**

| # | Config |
|---|---|
| D1 | Bottleneck ResF to show param-parity / -30% params at equal accuracy |

Total committed GPU cost if things go well: **~4–6 single-window runs.** Everything else is read-only or a one-line selection swap.

### 3.3 What NOT to do (explicit waste-of-compute guardrails)

1. **Do not collapse to single-stage.** It directly undermines the dual-Fourier multi-scale aggregation that is your backbone. This is the highest-risk, lowest-reward idea on the list.
2. **Do not train a learned scorer (M4) before M1/M3 are settled.** It's high-variance and, worse, uninformative for the paper if it works ("a small network improved things" is not a contribution). Only attempt it if a reviewer later asks "why not learned?"
3. **Do not do temporal-aware selection (M5) this round.** It entangles with the Fourier temporal path and has no evidence-backed prior; it's how compute disappears.
4. **Do not change the architecture while chasing the +4.71 reproduction.** Fix *one* variable at a time. Two of the three attribution experiments (A1, A2) are free — run them before spending a single epoch, or you risk "fixing" the bug into something else and never learning why.
5. **Do not run ablations against a baseline that differs in any config from the target.** (You've already hit this in the second-round review; the memory note about "C/E/B silently changing hyper/rp1 flags" is exactly the trap to avoid.) Pin config first, then ablate.

---

## One-line bottom line

**Your innovation is "scale-invariant remote selection + SpikePoint-style coordinate normalization," with a global centroid branch to refill the blank. Prove it with two free attribution scripts, then two GPU runs (B1, B2). Everything else — bottleneck, DPP, learnable scorer — is downstream polish.**

*Caveats I couldn't verify without file access (would confirm before coding):* exact variable names/flag values in your `hyper`/`rp1` selection code, where `xy` is scaled in the buggy path, and whether `grad_accum4 + GroupNorm` is currently on for the fixed-hyperbolic run that's still training. If you can paste the relevant selection/grouping function, I'll turn M1 (§2.2) and §1.3 into exact diffs.

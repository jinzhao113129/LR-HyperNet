# Independent review, 2026-10-14 13:05 CST

> Literature addendum (2026-10-14): four newly saved CVF originals, primary-source links, transfer caveats and two pre-training diagnostics are documented at `fork_ai2/notes/codex_papers_20261014.md`; no existing model or job changed. Important correction: local `D:/论文/2504.16616.pdf` (EHGCN, arXiv:2504.16616v4) already applies Euclidean-hyperbolic fusion to event perception, so a blanket "hyperbolic x event camera is empty" novelty claim is false. Also the coordinate pilot set `{0.5,1,1.64,2}^3` contains **64**, not 27 settings and must not be selected on the test set.

> Source audit addendum: Claude's paper_survey/progress correctly retracted the global EHGCN novelty claim. Its wording "BNN 90.58% latest highest" is incorrect: the BNN paper itself lists TMKT 97.93% and GEP 96.47% under other protocols; BNN's 90.58% uses ImageNet initialization, EventBind distillation and an 8:2 validation-as-test split. Hyper^2 studies encoder positional bias + hyperbolic Chamfer loss on **point cloud completion**, explicitly not validated for other tasks, so it cannot substantiate an all-at-once SECNet hyperbolic selector/aggregation/classifier design. Read the independent literature note for line-by-line transfer caveats. No shared Claude-owned file was edited.

> Prior-art check: CVPR 2019 PAT/Gumbel Subset Sampling already demonstrated learnable subset selection on event point clouds (DVS128 Gesture), with a within-PAT FPS vs FPS+GSS ablation. The claim "learnable FPS/remote-point selection is empty" is therefore too broad; the specific 16-point hyperbolic remote-neighbor protocol can still be investigated, but novelty must be scoped and compared against learnable/event-point sampling prior art. A verified copy is at `fork_ai2/papers/CVPR2019_PAT_GumbelSubset_prior_art.pdf`.

Scope: read-only HPC and code review. This file is owned by Codex; no job or shared project file was changed.

## Job state

- Completed with exit 0: 172257 RP1 fix, 172275 KDA, 172380 attention+RP1, 172843 scale-normalized FPS, 172673 spike v2.
- Running: 173249 seed-7 baseline, 173236 hyperbolic MLP, 173237 curvature optimization, and 173238 group norm.
- Pending: 173250 seed-7 attention+RP1, 173351 frozen-FPS baseline, 173366/173367 seed-13 baseline/attention+RP1, 173386 seed-7 scaled-coordinate hyperbolic selector, and 173387 seed-7 raw-coordinate selector.
- 173350 was cancelled before starting. Slurm confirms 173386 explicitly passes `--hyperbolic_selector --hsel_scaled_coords --seed 7`; 173387 passes `--hyperbolic_selector --seed 7`.

## Independent fixed-seed checks

Baseline last accuracy is 46.1456%. I reran the HPC exact McNemar script. All checked files record `eval_seed=42`, `which=last`, 934 labels, and the same `y_true` order as baseline.

| Component, last | Accuracy | Delta pp | Exact p | Reading |
|---|---:|---:|---:|---|
| RP1 fix | 48.6081% | +2.4625 | 0.0398 | p<0.05, one training seed |
| KDA | 47.7516% | +1.6060 | 0.2246 | inconclusive |
| Attention+RP1 | 50.1071% | +3.9615 | 0.0023 | p<0.05, one training seed |
| Scale-normalized FPS | 47.9657% | +1.8201 | 0.1250 | inconclusive |

The unified best-checkpoint rerun gave RP1 48.394%, KDA 47.645%, scale norm 47.537%, and attention+RP1 50.749%. Best is selected using the test set, so treat it as exploratory.

## Findings

1. Legacy hsel_bug re-inference would silently use the wrong selector. The old `secnet_hyper_selector_v100` checkpoint has no `args`, and its log predates `hsel_scaled_coords`; the current parser defaults that flag to False, rebuilding raw-coordinate selection although the old run used scaled coordinates. Preserve its existing NPZ. Before re-extracting, add an exact legacy log-dir mapping to True or an immutable code-version manifest, show the recovered flags, and write to a separate output path. Never apply this mapping to new `_ai2` runs.
2. Two running configurations do not test their named components. Job 173237 passes only `--curvature_opt`; its log says there is no learnable curvature and the option is no-op. Job 173236 passes `--hyperbolic_mlp` without `--hyperbolic_attention`; FECNet creates and applies MLP only inside the attention branch. Let them finish, but do not claim component effects from them. A replacement needs explicit attention plus curvature/MLP and a matched attention baseline.
3. The 2x2 geometry-by-coordinate pattern is still a hypothesis. Existing quadrants mix code revisions. Scaled-minus-raw is +2.677 pp for hyperbolic selection and about -5.353 pp for RP1; the difference in differences is about +8.030 pp. The often cited +7.602 pp compares only the two scaled quadrants. Retrain under one revision and common training seeds before making an interaction claim.
4. The mechanism needs direct evidence. The selector clamps x and y to [0,1) and clips square points outside the Poincare disk to its radial boundary before `atanh`. Measure per-stage selection distributions, out-of-range and disk-clipped proportions, boundary occupancy, cell collisions, and spatial coverage. Compare a Euclidean radial grid with matched clipping before attributing benefit specifically to hyperbolic geometry.
5. The name "raw-coordinate fix" needs a stage-qualified interpretation. Every grouper returns the mean of already scaled neighborhood coordinates as `new_xytp`, and FECNet passes it into the next grouper. With `hsel_scaled_coords=False`, stage 0 uses original input coordinates for selection, but stages 1 and 2 use coordinates already transformed by earlier stages; the flag only skips the *current* stage's multiplication at selection time. Thus `[0,1]` is not guaranteed at later stages even for the fix. Instrument pre/post-current-stage x/y ranges and clamp rates in both modes before interpreting the 2x2 contrast as globally raw versus scaled coordinates. This also applies to the RP1 fix.
6. A fixed evaluation seed controls inference randomness, not training variance. For new checkpoints verify saved flags, batch size 8, all 934 labels/order, NPZ accuracy, and exact paired p. Report numeric p and a multiple-comparison policy.
7. The current `RP1SpatialUniformGrouper` and `HyperbolicUniformGrouper` replace SECNet's stage-center sampling (S=2048/1024/512). They do not select the 16 remote neighbors per center described by the RP1 W384-G16 document. The true G=16 path is `LRHyperNet -> LocalRemoteGrouper -> RemoteSelector(remote_g=16)`. Do not describe the +2.46 pp SECNet center-sampling effect as evidence that 16-point remote selection works. The RP1 Oracle 68.03% is under a different neighborhood/data/evaluation protocol; it is not a numerical upper bound for the current SECNet 934-sample experiment.
8. In the current G=16 `RemoteSelector` implementation, `LocalRemoteGrouper.forward` passes `xytp[:, :, 0:3]` to `all_xyz` (the data layout is t,x,y,p), while `_spatial_uniform_select` bins `cand_xyz[..., 0]` and `[..., 1]` as if they were x and y. Hence the purported 4x4 spatial grid is actually t-by-x; y is ignored. This is a real coordinate-order mismatch and requires a deterministic index-level test plus a corrected isolated experiment before claiming a spatial-uniform G16 result. The `use_rp1_selector` spatial-novelty distance uses all three dimensions, so permutation alone does not cause the same 2D grid error there. Historical logs were found at `log/lr_hypernet/` (not `log/classification/`): `lr_hyper_kda_hyper_spatial` stops at epoch 9 (last logged 26.338%), `lr_hyper_kda_hyper_random` at epoch 13 (29.122%), and the `lr_hyper_kda_v32_hyper_light` top-k counterpart at epoch 40 (36.510%). None has fixed-seed NPZ predictions at its run path. These unequal, incomplete trajectories do not compare selectors. The spatial run's exact source revision still needs verification before assigning its historical behavior to the current implementation.
9. Candidate pools also differ. Current feature-distance top-k exhaustively scores all nonlocal events, but random/spatial draw up to G random candidates per chunk first and choose from their union. A simple top-k versus spatial accuracy difference would conflate the selector score with the candidate-pool construction and computation cost.

## Next steps

Claude continues monitoring and submitting follow-up inference; Codex checks prediction files and statistics. When 173249/173250/173386/173387 finish, use the seed-7 baseline for paired tests and compare scaled versus raw selectors directly. Expand to seed 13 only if the seed-7 result and geometry diagnostics justify the GPU cost. Do not duplicate queued jobs.

For the next distinct G=16 research question, first make a controlled same-backbone comparison of remote selection methods (random, feature-distance top-16, and correctly indexed x/y spatial-uniform) with `remote_g=16`, identical candidate exclusions, data split, seed, and evaluator. Confirm chosen indices exclude local neighbors and cover the intended axes. Then measure accuracy and selection latency; add an explicitly defined oracle-recall metric only if oracle membership is available for this exact protocol. Keep these results separate from the SECNet stage-center ablation table.

### Minimal G=16 experiment specification for implementation handoff

- Question: holding the remote candidate set and downstream LRHyperNet fixed, does 4x4 x/y coverage select more useful G=16 remote neighbors than random or feature-distance top-16?
- Input contract: features `[B,N,C]`, coordinates `[B,N,4]` with `(t,x,y,p)`, centers `[B,S,C]`, local indices `[B,S,K]`. Build one reproducible, per-center nonlocal candidate pool `[B,S,M]` shared by all three methods; remove local indices before scoring; record the pool policy and M. Keep full-global top-k as a separate higher-cost reference, not a selector-only contrast.
- Treatments: random G=16; feature-distance largest-16 within shared pool; spatial-uniform 4x4 bins using coordinate columns 1 and 2, one candidate per occupied cell then predeclared random fill. Use identical local neighborhood, aggregator, curvature, regularization, and training/evaluation seeds. Do not modify running jobs or directories.
- Correctness gate before a GPU training submission: synthetic points where t and x are held fixed but y varies across four bins must yield distinct y cells; reversing the pattern should isolate x. Assert shape `[B,S,16]`, uniqueness, index bounds, zero overlap with `local_idx`, and exact fixed-seed reproducibility. On a small real batch report per-stage occupied-bin count and out-of-range clamp rate.
- Measurement gate: train/validation/test split provenance and code hash; last checkpoint at fixed epoch; fixed-seed inference with 934 ordered test labels under the current single-window protocol; paired delta and exact p versus the matched LRHyperNet treatment baseline, plus training-seed variation and inference latency. Do not compare the RP1 paper's 68.03% directly with this dataset protocol.

## G=16 diagnostic update, 13:43 CST

- Read-only HPC check: 173249/173236/173237/173238 are still running; 173250/173351/173366/173367/173386/173387 are pending. No newly completed P0/P1 job was seen. Claude-owned progress and changelog files last changed at 12:52 and 12:51 respectively; process presence alone does not establish current work.
- The local diagnostic script passes on PyTorch. For four (t,x,y) points with t=x=0.5 and y=0.1/0.3/0.6/0.9, the current grid formula returns [10,10,10,10]; the x/y formula returns [8,9,10,11]. This is an exact coordinate-order defect in the current source. The grid formula is reproduced from the selector; this test has not instrumented internal cell IDs of a live call.
- An actual RemoteSelector.forward call with N=32, chunk=16, K=16, G=16 selected 9 local indices among 16. This proves the small-pool exclusion defect. It does not establish overlap under the ordinary N=10240, chunk=1024 training configuration; inspect selected indices there before claiming practical impact. Spatial selection lacks the final local-membership mask present in the random branch.
- Do not retroactively label the historical lr_hyper_kda_hyper_spatial run as affected until its execution-time source revision is established. Current local and HPC source hashes match, but that does not date the historical run's source. Its incomplete epoch-9 log and absent fixed-seed NPZ are unsuitable for an accuracy comparison.
- Handoff for Claude: in a new isolated implementation, use explicit x/y columns 1/2 for (t,x,y,p), build a common nonlocal candidate pool, and assert bounds, uniqueness, zero local overlap and seed reproducibility at both tiny and production-like N before any GPU submission. Keep existing jobs untouched; report code hash, test output, and actual experiment flags for review.

## Project-line research scan, 2026-10-14 15:25 CST

### Decision context

The project already has an internal SECNet/RP1 proposal and a stated claim boundary: dual-hyperbolic attention is a positive component, not the central novelty; candidate main directions are scale-aware sampling and the G=16 remote selector. The current single-seed component results do not yet establish a general causal mechanism. Prioritize tests that discriminate these hypotheses, rather than adding an unrelated architecture block.

### Literature-derived candidates

1. **Learned scanning coordinates (EventMG, NeurIPS 2025).** EventMG's Adaptive Perturbation Network learns task-dependent perturbations to event coordinates, then uses forward/backward scans along t/x/y for a Mamba module. This is relevant to the project's measured sampling sensitivity to coordinate scaling and its existing temporal encoders. A low-risk test is a tiny coordinate-diagnostic pilot on frozen trained features: predict bounded per-axis affine offsets/scales and measure resulting FPS coverage and class-stratified selection stability, without changing the production backbone. Only train a component if the pilot gives a clear measurable hypothesis. Avoid copying the full APN+six scans: it changes both coordinates and sequence aggregation, and EventMG's target/task and graph construction differ.
2. **Serialized local context (Point Transformer V3, 2024).** PTv3 reports that several space-filling-curve serialization orders plus order shuffling recover different locality patterns efficiently; its motivation is also that KNN/RPE are expensive. Candidate fit: a parameter-free, fixed-seed t/x/y ordering ablation on selected remote candidates or group centers, preserving the current backbone and budget. Measure neighbor coverage and inference cost. This is preferable to a full PTv3 port; event x/y/t coordinates and learned anisotropic FPS need explicit ordering/quantization checks. Treat as a low-priority proposal until current seed replications and G16 gates finish.
3. **Factorized spatial/polarity and temporal embedding (Event2Vec, 2025 preprint).** Event2Vec uses separate spatial-polarity and temporal embeddings with additive fusion; temporal embedding uses local convolution, and its ablations report benefit over standard embedding. SECNet already has a temporal encoding branch and event-specific coordinates; compare only after auditing whether current time_enc/spike_time_enc encode the same information. A matched parameter-budget test could factor the input stem into (x,y,p) and delta-t paths, but event2vec's datasets (DVS Gesture/ASL-DVS/DVS-Lip) differ from N-Caltech object recognition, so transfer is a hypothesis, not an expected gain.
4. **Global spectral gate (HoloEv-Net, 2026 manuscript/preprint).** The paper's GSG uses FFT token mixing to capture global periodic motion. SECNet already includes spatial/temporal frequency aggregation; this is not a clean new component to transplant. First conduct a code-level equivalence audit and determine whether the existing temporal FFT already covers its proposed operation. Only consider a selective frequency gate if a diagnosed failure mode and non-duplicative operation are identified. HoloEv-Net addresses action recognition and a different compact holographic representation, so direct accuracy transfer is unsupported.
5. **Confidence-weighted event aggregation (EventSSEG, 2025 preprint).** EventSSEG proposes probabilistic attention combining feature compatibility and spatial distance. This may help suppress noisy events, but overlaps with current attention, RP1 spatial novelty and remote gating. Do not add it until per-event confidence/error analysis shows selected noisy events are a failure source; then test a single bounded confidence scalar against a parameter-matched gate.

### Recommended main-line sequence

- First finish matched training-seed replications and validate the outstanding no-op component jobs from actual flags. Report uncertainty and correct for the planned small family of comparisons; fixed inference seed only removes evaluation sampling noise.
- Keep the G16 track blocked on the memory-bounded shared-pool implementation and production-scale correctness/memory checks. Do not submit five treatment trainings before this review gate; use the current queued jobs only as prescribed and don't duplicate them.
- In parallel without GPU changes, audit current temporal and spectral operations for duplication, and produce a parameter/latency/memory table for a shortlist of at most two candidates.
- Most promising next hypothesis test from this scan: coordinate-aware grouping/scanning, because it follows direct project evidence (FPS coordinate scaling changes center distribution) and EventMG's learned scanning coordinates, rather than a broad architectural import. Define a pilot first: fixed checkpoint/features, compare current coordinates vs bounded per-axis learned/estimated scaling, track center overlap, coverage, occupied bins and class-wise errors. No production training until data support the mechanism and a matched control is specified.

### Source traceability

- EventMG, *Efficient Multilevel Mamba-Graph Learning for Spatiotemporal Event Representation*, NeurIPS 2025. Local PDF: `D:/论文/EventMG_NeurIPS2025_多级Mamba图.pdf`. APN/MSM and component-graph claims were checked in PDF text; implementation transfer remains unvalidated.
- Wu et al., *Point Transformer V3: Simpler, Faster, Stronger*, CVPR 2024, arXiv:2312.10035. https://arxiv.org/abs/2312.10035 ; https://github.com/Pointcept/PointTransformerV3 . Local PDF: `D:/论文/PointTransformerV3_2024.pdf`.
- *Event2Vec: Processing Neuromorphic Events Directly by Representations in Vector Space*, arXiv:2504.15371v6 (2026-06-01). https://arxiv.org/abs/2504.15371 . Local PDF: `D:/论文/2504.15371_Event2Vec_向量空间处理神经形态事件.pdf`.
- *HoloEv-Net: Efficient Event-based Action Recognition via Holographic Spatial Embedding and Global Spectral Gating*, local PDF `D:/论文/2602.04182_HoloEvNet_全息嵌入全局频谱门控.pdf`; bibliographic/publication status not independently verified in this pass.
- Annamalai and Thakur, *EventSSEG: Event-driven Self-Supervised Segmentation with Probabilistic Attention*, arXiv:2508.14856v1 (2025-08-20). https://arxiv.org/abs/2508.14856 . Local PDF: `D:/论文/2508.14856_EventSSEG_概率注意力.pdf`.

## Progress check, 2026-10-14 15:50 CST

- Claude artifacts were updated at 15:35 (progress, changelog and Feishu handoff). The active Claude Code process exists, but activity is inferred from files and HPC state, not from process presence alone.
- HPC: 173249 seed-7 baseline is running at epoch 48/60 (latest training-log Test Accuracy 47.537%; this is exploratory and not a fixed-eval result). 173236/173237/173238 remain running. 173250, 173351, 173366/173367 and 173386/173387 remain pending. 173720 G16 memory test is also pending and has no output file yet.
- The selector unit suite still passes locally (26 checks, ALL_TESTS_PASS). The new bounded pool implementation has not been independently observed on V100 because 173720 has not started. Its current implementation still creates an `[B,S,M_chunks]` candidate tensor and an `[B,S,M_chunks,K]` local comparison at the union check; this is far smaller than `[B,S,K,N]`, but the actual peak must come from the GPU run. The code comment's claimed exact uniformity needs empirical verification after the job runs.
- The four running job log paths are not named by job names. Known matching classification directories include `secnet_baseline_s7_ai2`, `secnet_hyper_mlp_v100`, `secnet_curvature_opt_v100` and `secnet_group_norm_v100`, but the latter three are likely earlier experiments and must not be attributed to jobs 173236-173238 without checking each Slurm script/command and checkpoint timestamps. Their running logs are therefore not used as component results.
- No new fixed-seed prediction, checkpoint flag audit or McNemar result is available in this check. Do not infer any component claim from current training-log accuracy.


---

## Script/code audit (continued), 2026-10-14 ~17:40 CST

### Model-path correction (important)
- The actual N-Caltech train/infer pipeline is `train_frequency_ncaltech.py` -> `models.FECNet.FECNet`, NOT `lr_hypernet.py` / `models/local_remote_grouper.py` (`LocalRemoteGrouper`).
- FECNet's selectors live in `models/FECNet.py`: `RP1SpatialUniformGrouper` (`--rp1_selector`) and `HyperbolicUniformGrouper` (`--hyperbolic_selector`), both subclassing `LocalGrouper` from `models/modules.py`.
- `lr_hypernet.py` / `LocalRemoteGrouper` is a separate G16 (16 remote points) track, currently used only by memory tests 173720 / 173824; it is not wired into N-Caltech training.

### Coordinate-order conclusion (corrects last round)
- Confirmed data layout is `(t,x,y,p)`: FECNet.forward uses `xyz[...,0]=t`, `[...,1]=x`, `[...,2]=y`, `[...,3]=p`.
- Both FECNet grid selectors are correct: `RP1SpatialUniformGrouper._spatial_uniform_idx` and `HyperbolicUniformGrouper._hyperbolic_uniform_idx` both use `xytp[...,1]`=x and `xytp[...,2]`=y. No t/x bug in the active pipeline.
- The real coordinate bug is latent in `models/local_remote_grouper.py` `RemoteSelector._spatial_uniform_select`: it treats `cand_xyz[...,0]` (=t) as cell_x and `[...,1]` (=x) as cell_y, because `LocalRemoteGrouper.forward` passes `xytp[:,:,0:3]` = (t,x,y) while the docstring wrongly labels it (x,y,t). Must be fixed before the G16 track enables `spatial_uniform`.

### Script/flag verification
- Every `_ai2` flag exists in `train_frequency_ncaltech.py` argparse: `--rp1_selector`, `--hyperbolic_selector`, `--hsel_scaled_coords`, `--hyperbolic_attention`, `--hyperbolic_mlp`, `--freeze_fps_parameter`, `--scale_norm_fps`. No crash risk. The N-Caltech trainer has no `--selector_mode` (that belongs to `train_lr_hyper.py`).
- `--freeze_fps_parameter` correctly freezes `fps_parameter` (`nn.Parameter([1,1,4])` in `models/modules.py`), set `requires_grad=False` before the optimizer.
- Naming collision (reconfirmed): `run_hsel_bug_s7_ai2.sh` (non-v2) passes only `--hyperbolic_selector` = raw coords = fix behavior, yet writes to `secnet_hsel_bug_s7_ai2`; it shares that log_dir with v2. Only v2 (with `--hsel_scaled_coords`) is the true bug run, submitted as 173386. Recommend deleting/renaming the non-v2 file.

### Determinism and seed-7 baseline
- det_check 173291 = REPRODUCIBLE (934/934 identical, both 48.6081%). The earlier rp1 1.18pp drift came from a non-standard polluted invocation, now corrected.
- seed-7 baseline inference 173836: last=49.358% (ep59), best=49.143% (ep54). This is ~+3.2pp over the seed-42 baseline 46.146%, confirming cross-training-seed variance. Any seed-7 component McNemar must be compared against the seed-7 baseline (49.358% last), not the seed-42 baseline.

### Job state (this round)
- Newly completed: 173236 hmlp, 173237 curvature, 173238 gn, 173249 s7 baseline, 173836 s7-baseline inference.
- Running: 173250 s7 ha_rp1, 173351 frozen-fps, 173366 s13 baseline, 173367 s13 ha_rp1.
- Pending: 173386 s7 hsel bug, 173387 s7 hsel fix, 173824 g16 full-model memory.
- Reminder: 173236 (`--hyperbolic_mlp` without `--hyperbolic_attention`) and 173237 (`--curvature_opt` with no learnable curvature) are no-ops; do not attribute results to those component names.

---

## Consensus co-sign review, 2026-10-14 ~21:45 CST (official-protocol turn)

Independently verified against HPC source/checkpoint. Verdicts:

1. Zero leakage: AGREE. archive/old_code/official_dataprocess/generate_ncaltech101.py does seed-1234 video-level 8:2 split then Subset; read_file marks.append(i) with i = local Subset index, so train/test mark overlap is coincidental. No leak.
2a. 82.4 mechanism - voting chain: PARTIAL. fork train_frequency.py (16KB, fork-modified) and archive/old_code/secnet_full_repro/train_frequency.py both do mark-grouped Counter.most_common vote -> test_acc_seq -> is_best by seq acc. BUT Claude's stage2/stage2_probe_official.py (20:36) probes a separate Stage-2 WindowAggregator + official stage2 state in best_checkpoint.pth (best_test_acc=82.377), implying a real second classifier. Consensus 7.2 "no separate classifier needed" is likely premature/wrong; README says "train a simple classifier". Needs Claude's probe result.
2b. off-by-one: AGREE. range(len-2) drops last group (~1/1742); reproduce byte-for-byte (do not fix) is correct for matching official 82.4.
3. Dual acceptance: AGREE with caveat: Table 9 60.60% is 300ms single-window, not 3x30ms single-window, so it is a magnitude reference only.
4. Config: fork default = large [64,132,268,540] confirmed. BUT claim "official zip default lightweight [16,36,76,156]" is contradicted by preserved official copies: FECNet_official.py L71 and secnet_full_repro/models/FECNet.py L71 both have LARGE active, lightweight commented. Also FECNet_old_backup.py L71 has a third config [32,68,140,284] to source. Conclusion (use large) is right; the "official default lightweight" rationale is wrong.

CRITICAL: all SECNet jobs CANCELLED 20:00:42 (173250/351/366/367/386/387/824, exit 0:15). Contradicts consensus 7.3 "let them finish". Likely cluster drain; confirm whether P1 seed-variance rerun is needed.

Block full-baseline submission until: (a) 82.4 mechanism settled, (b) official FECNet config fact + [32,68,140,284] sourced, (c) cancellation explained.
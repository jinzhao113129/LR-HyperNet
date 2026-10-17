# Codex survey supplement: 2026-10-16

Search date: 2026-10-16.

Tools: arXiv search page, OpenAlex, limited Semantic Scholar/arXiv API
attempts. arXiv API was rate-limited during this session; the decisive
queries below were repeated through the arXiv search interface.

Scope: four requested directions. Only yes/no and identifiable hits are
reported. No new PDF was copied to `fork_ai2/papers/` because none of the
new hits is useful enough to archive.

## 1. Hyperbolic event serialization

Queries:

```text
"hyperbolic" "event camera"
"hyperbolic" "event stream"
```

Result:

- One direct hit: EHGCN, arXiv 2504.16616, *Hierarchical
  Euclidean-Hyperbolic Fusion via Motion-Aware GCN for Hybrid Event Stream
  Perception*. This is already in the project survey.

Conclusion:

- No new event-stream serialization + hyperbolic-embedding classification
  work was found.
- EHGCN remains the only direct hyperbolic-event neighbor and is graph /
  detection oriented, not point-cloud serialization.
- The requested "event sequence serialization + hyperbolic" direction is a
  zero hit for classification.

## 2. Multiscale hyperbolic aggregation

Queries:

```text
"multiscale" "hyperbolic" "point cloud"
"hyperbolic" "hierarchical" "event"
```

Result:

- No exact new hit in the arXiv search.
- Existing known neighbors are general hyperbolic graph/vision models
  (HexFormer, HyperCore, HyperIPC), not an event point-cloud multiscale
  aggregation method.

Conclusion:

- No direct method to borrow as a drop-in multiscale hyperbolic event
  aggregator.
- The project can still describe its design as a novel combination, but
  the novelty claim must say "combination and mechanism", not "multiscale
  hyperbolic aggregation itself has no related work".

## 3. Hyperbolic contrastive learning for event cameras

Queries:

```text
"hyperbolic contrastive" "event"
"hyperbolic contrastive" "N-Caltech101"
```

Result:

- No direct hyperbolic contrastive event-camera hit.
- Closest but non-matching:
  - SpikeCLR, arXiv 2603.16338: contrastive self-supervised SNN learning,
    not hyperbolic.
  - HyperMVP, already archived: hyperbolic multiview point-cloud
    pretraining, not event cameras.

Conclusion:

- Hyperbolic contrastive learning on N-Caltech101 remains an open
  direction. It is not required for the current architecture decision.

## 4. N-Caltech101 2026 point-based SOTA recheck

Search: `"N-Caltech101"` sorted by recent arXiv submission.

Recent hits worth classifying:

| ID | Work | Relevant class |
|---|---|---|
| 2610.09873 | Bringing BNNs to Fast Event Processing | BNN + RGB pretraining + PBEV; already known protocol caveat, not a clean point-based from-scratch competitor |
| 2609.32537 | RIPE-MambaSpike | Spiking-Mamba hybrid, 0.870M parameters; SNN/state-space family |
| 2609.30119 | T-Backdoor | SNN adversarial evaluation |
| 2605.13887 | LSFormer | Spiking transformer with local attention |
| 2603.16338 | SpikeCLR | Few-shot self-supervised SNN |
| 2603.14285 | MorphSNN | Adaptive graph diffusion SNN |
| 2603.06228 | Spatially-sparse linear attention | Detection, not classification |

None of the newly found works changes the SECNet 82.4 reference for
point-based from-scratch classification. The higher reported numbers
still belong to SNN, RGB-pretrained, cross-modal, or augmented-protocol
families.

Conclusion:

- 82.4 remains the correct point-based from-scratch target.
- The protocol caveat around the BNN paper still applies.

## Bottom line

| Direction | Result |
|---|---|
| Hyperbolic event serialization | zero direct new hit; EHGCN remains the only neighbor |
| Multiscale hyperbolic event aggregation | zero direct new hit |
| Hyperbolic contrastive event | zero direct hit |
| 2026 N-Caltech point-based SOTA | 82.4 target unchanged |

No new original PDF was archived. If later work wants a deeper sweep, the
next best action is citation-search of EHGCN and a full-text Google
Scholar/OpenAlex query for "event camera hyperbolic classification", not
another broad arXiv keyword pass.


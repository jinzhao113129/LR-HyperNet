# Codex architecture survey addendum (2026-10-15)

Scope: project-specific follow-up to `arch_survey_20261015.md`, focused on the SECNet/new-architecture decision rather than generic literature collection.

## 1. Direct event-cloud backbone evidence

### EventMamba (`2405.06116`)
- Treats `(x,y,t,p)` as a point cloud, but unlike generic point-cloud networks it enforces strict chronological ordering and splits temporal features into implicit and explicit branches.
- Architecture:
  - LocalFE: FPS + KNN + residual MLP blocks
  - Temporal Aggregation (TAG): attention over the K temporally ordered neighbors
  - GlobalFE: Mamba block over centroids for explicit long-range temporal dependencies
- The most important design signal for SECNet is not "use Mamba"; it is that **time must be a first-class axis**, not just one coordinate among four.
- Results: strong point-based performance on multiple action recognition datasets with very small models.

### EVSTr (`2303.03856`)
- Voxel-set transformer for event streams.
- Two-stage encoder:
  - Multi-Scale Neighbor Embedding Layer (MNEL): KNN over voxel coordinates, jointly encoding relative position and semantic feature relations
  - Voxel Self-Attention Layer (VSAL): global interaction with absolute-relative positional encoding
- For long streams, it segments the event stream and applies temporal modeling over the sequence of voxel-set features.
- This is a good reference for a **local-to-global event voxel backbone**, but it does not directly solve our remote-point selection problem.

## 2. Local-global / remote-context evidence

### DSPoint (`2111.10332`)
- Channel-splits features into a local point branch and a global voxel branch.
- Local branch: point-wise convolution
- Global branch: voxel-wise attention
- Co-attention fusion with high-frequency coordinate communication.
- Important implication: a **dual-path local/global encoder** is well supported and does not require abandoning point-based processing.

### PointLearner (`2610.11342`)
- Point-focused attention uses:
  - local KNN branch for fine-grained structure
  - induced-point pooling branch for coarse global semantics
  - a single competitive softmax across both branches
- Context-scan state space serializes with Hilbert curves and uses bidirectional S6 for global context.
- This is the closest recent analogue to our target design: **local geometric detail + global sparse context**, with linear complexity.

### CloudMamba (`2511.07823`)
- Uses sequence expanding along X/Y/Z axes and sequence merging rather than concatenating Hilbert curves.
- Adds chained bidirectional Mamba and grouped selective state space (GS6) to reduce overfitting.
- Relevant if we pursue a serialization backbone; less directly relevant to G16 remote selection.

### Dilated Point Convolutions (`1907.12046`)
- Expands receptive field by taking `k*d` nearest neighbors and selecting every `d`-th neighbor.
- Sublinear overhead, no extra parameters.
- This is a simple, strong baseline for a **dilated remote-neighbor selector** if our current hyperbolic / spatial uniform / feature-distance selectors underperform.

### diffConv (`2111.14658`)
- Uses density-dilated ball query and masked attention to create an irregular, adaptive neighborhood.
- Motivated by uneven point density and noise sensitivity in KNN.
- Relevant to event clouds because event density is highly non-uniform.

### PointViG (`2407.00921`)
- Adaptive dilated graph convolution searches sparse neighbors in a dilated neighborhood using semantic correlation.
- Lightweight and avoids over-smoothing.
- Supports the idea that remote neighbor selection should be **task-aware**, not purely geometric.

## 3. Hyperbolic point-cloud evidence

### PointCHR (`2607.24052`)
- Curvature-aware hyperbolic rectification for point clouds.
- Uses:
  - Hyperbolic Semantic Transformation (HST)
  - Point-wise Curvature-Adaptive Perception (PCP)
  - Closed-Form Geodesic Dilation (CGD)
- Pushes high-curvature points toward the hyperbolic boundary while keeping low-curvature points near the origin.
- This is the closest hyperbolic-point-cloud analogue to our remote-selection problem, because it uses geometry to redistribute representational capacity.

### HyperIPC (`2409.15810`)
- Hyperbolic image-and-point-cloud contrastive learning.
- Uses hyperbolic distance and midpoint hierarchy in a Poincare ball.
- Useful as prior art for hyperbolic point-cloud representation, but not directly transferable to our supervised classification backbone.

### EHGCN (`2504.16616`)
- Euclidean-hyperbolic fusion for event streams.
- Local Euclidean GCN + global hyperbolic GCN.
- This is the closest event-camera paper to our direction, though its tasks and architecture differ from SECNet classification.
- Supports our hypothesis that event streams benefit from a dual-space representation: Euclidean for local continuity, hyperbolic for global hierarchy.

## 4. Architecture implications for SECNet

1. **Dual-path local/global is better supported than a single-path backbone.**
   - Local branch should preserve fine-grained spatiotemporal geometry.
   - Global branch should handle hierarchy and long-range context.

2. **Time should not be treated as a passive coordinate.**
   - EventMamba and EVSTr both explicitly model temporal order or segment-level dynamics.
   - A future SECNet architecture should consider either:
     - temporal aggregation within local neighborhoods, or
     - a separate temporal branch over serialized / segmented event clouds.

3. **Remote selection should be compared against a dilated-neighbor baseline.**
   - DPC and diffConv provide simple, effective ways to expand receptive fields.
   - If G16 hyperbolic selection does not outperform these baselines, the novelty claim weakens.

4. **Hyperbolic geometry is most defensible when used to allocate representational capacity.**
   - PointCHR's curvature-aware radial dilation is a strong design precedent.
   - Our G16 selector could be reframed as capacity allocation for sparse high-information events rather than merely "farthest neighbor selection."

5. **A plausible next architecture combines four ideas:**
   - local Euclidean KNN/MLP branch
   - global hyperbolic remote-selection branch
   - explicit temporal aggregation or serialization
   - hybrid Euclidean classification head

## 5. Most useful local PDFs from this pass

- `D:/论文/2405.06116_EventMamba_PointBased_Event_Classification.pdf`
- `D:/论文/2303.03856_EventVoxelSetTransformer.pdf`
- `D:/论文/2111.10332_DSPoint_DualScale_PointCloud.pdf`
- `D:/论文/2610.11342_PointFocusedAttention_ContextScanStateSpace.pdf`
- `D:/论文/2511.07823_CloudMamba_GroupedSelectiveStateSpaces.pdf`
- `D:/论文/1907.12046_Dilated_PointConvolutions.pdf`
- `D:/论文/2111.14658_diffConv_Irregular_PointCloud.pdf`
- `D:/论文/2407.00921_PointViG_Lightweight_GNN_PointCloud.pdf`
- `D:/论文/双曲方法论/2607.24052_PointCHR_2026双曲点云最新.pdf`
- `D:/论文/双曲方法论/2409.15810_HyperIPC_双曲图像点云对比.pdf`
- `D:/论文/2504.16616.pdf`

## Bottom line

The literature does not yet show a direct competitor for **hyperbolic remote-point selection in event-cloud classification**. The closest neighbors are:

1. EventMamba / EVSTr for event point-cloud backbones
2. DSPoint / PointLearner for local-global fusion
3. PointCHR for hyperbolic point-cloud capacity allocation
4. EHGCN for Euclidean-hyperbolic event fusion

Our strongest defensible direction is therefore not simply "hyperbolic event classification," but:

> a local-global event-cloud backbone where remote events are selected and aggregated using hyperbolic geometry, with explicit temporal modeling and a hybrid Euclidean classifier.

# Experiment protocol

Every number in the README comes from the protocol below. Each run is an
independent training from scratch; no run was tuned on test images.

## Design at a glance

| | |
| --- | --- |
| Training runs | 9 pose conditions × 5 methods = **45 runs**, 30,000 steps each, seed 42 |
| Pose audit | 9 conditions × 4 pose-optimizing methods = **36 saved pose sets**, no training |
| Aligned evaluation | GT condition × 5 methods, preview (20 views) and full (200 views), no training |
| Showcase renders | GT, T1, T5 and R5 conditions × 5 methods, 4 views, plus 4 orbit videos, no training |

## Shared training configuration

| Item | Setting |
| --- | --- |
| Dataset / scene | NeRF-Synthetic Lego |
| Split | 100 train, 100 val, 200 test images |
| Resolution | 800 × 800 |
| Backbone | Nerfstudio 1.1.5 `vanilla-nerf`, coarse + fine fields |
| Initialization | Every run trained from scratch |
| Seed | 42 (single seed) |
| Steps | 30,000 |
| Ray batch | 1,024 rays |
| Samples per ray | 64 coarse + 128 importance |
| Near / far | **Training: 2 / 6.** At inference Nerfstudio's `NearFarCollider` resets near to 0 (far stays 6); every reported test metric, including the historical full evaluation, uses this inference behavior |
| Background | White |
| Blender scale | `scale_factor = 1` |
| Scene optimizer | RAdam, lr 5e-4, eps 1e-8, no schedule |
| Loss | Coarse + fine RGB MSE, weight 1 each |
| Mixed precision | Off |
| Intrinsics | Fixed |
| Val / test cameras | Original GT poses; the training-camera correction table is never applied |

## Methods

| Method | Pose parameterization | Camera optimizer | Camera LR schedule | Warm-up steps |
| --- | --- | --- | --- | --- |
| Fixed | Input poses held fixed | — | — | — |
| Naive | 6-D SO(3) × R³ correction, δ = z | Adam, eps 1e-8 | 1e-3 → 1e-4, exponential decay | 0 |
| CamP-style | Preconditioned 6-D correction, δ = Pz | Adam, eps 1e-8 | 1e-3 → 1e-4, exponential decay | 0 |
| Naive + warm-up | 6-D SO(3) × R³ correction, δ = z | Adam, eps 1e-8 | 1e-8 → 1e-3 cosine ramp, then decay to 1e-4 | 2,500 |
| CamP-style + warm-up | Preconditioned 6-D correction, δ = Pz | Adam, eps 1e-8 | 1e-8 → 1e-3 cosine ramp, then decay to 1e-4 | 2,500 |

Implementation: [`poseguard/methods.py`](../poseguard/methods.py) (model and
Nerfstudio method registration) and [`poseguard/geometry.py`](../poseguard/geometry.py)
(pose correction, projection Jacobian, preconditioner).

## Pose conditions

| Condition | Translation noise | Rotation noise |
| --- | --- | --- |
| SV100-GT | 0 | 0° |
| SV100-T0.5 / T1 / T2 / T5 | 0.5% / 1% / 2% / 5% of mean camera radius | 0° |
| SV100-R0.5 / R1 / R2 / R5 | 0 | 0.5° / 1° / 2° / 5° |

Every condition is trained with all five methods (45 runs).

### Perturbation details

| Item | Setting |
| --- | --- |
| What is perturbed | Training-camera extrinsics only; images, intrinsics, val and test poses unchanged |
| Translation scale | Mean distance of GT training cameras to the trajectory center |
| Translation noise | Random unit direction per camera, length = ratio × mean radius |
| Rotation noise | Random world axis per camera, rotation of the given angle left-multiplied |
| Seeds | Translation generator 42, rotation generator 43 |
| Control | All methods reuse the same perturbed dataset |

### Pose-optimization details

| Item | Setting |
| --- | --- |
| Initial correction | All zeros |
| Update | World-frame translation added, axis-angle rotation left-multiplied |
| GT pose supervision, pose regularizer, fixed anchor | None |
| Intrinsics optimization, frequency coarse-to-fine | None |
| Warm-up curve | Nerfstudio cosine ramp: 1e-8 + (1e-3 − 1e-8) · sin(πs / 5000), 0 ≤ s < 2500 |
| CamP-style proxy points | 4,096 per camera; uniform pixels, uniform depth in [2, 6] along unit rays |
| Proxy sampling seed | Isolated CPU generator, seed 42 |
| Preconditioner | Projection Jacobian at the initial pose; Σ = JᵀJ / 4096; P = (Σ + 0.1 diag(Σ) + 1e-8 I)^(−1/2) |
| Numerics | Computed in CPU float64, stored as float32, fixed during training |

## Pose audit

For each of the 36 pose-optimizing runs, a Sim(3) transform is fitted between
the GT and the final optimized **training camera centers**, then the global
transform and the residual center / rotation errors are recorded.
Script: [`scripts/evaluation/audit_pose_alignment.py`](../scripts/evaluation/audit_pose_alignment.py).
Results: [`artifacts/analysis/pose_audit/`](../artifacts/analysis/pose_audit/).

## Aligned rendering evaluation

| Evaluation | Condition | Methods | Test views | Resolution | Raw metrics | Aligned metrics |
| --- | --- | --- | --- | --- | --- | --- |
| Preview | SV100-GT | all 5 | indices 0, 10, …, 190 (20) | 200 × 200 | re-rendered | re-rendered |
| Full | SV100-GT | all 5 | all 200 | 800 × 800 | reused from the original evaluation of the same checkpoint | re-rendered |

| Item | Setting |
| --- | --- |
| Model | Frozen step-29,999 checkpoint, no network update |
| Sim(3) fit data | GT and optimized training camera centers only; no test images |
| Direction | GT world → reconstruction world |
| Test cameras | Position and orientation transformed together; intrinsics unchanged; far bound scaled by the fitted scale (near stays 0 at inference) |
| Control | Fixed + GT with the identity transform |
| Metrics | Fine-network PSNR, SSIM, LPIPS |

Script: [`scripts/evaluation/evaluate_alignment.py`](../scripts/evaluation/evaluate_alignment.py).
Results: [`artifacts/analysis/alignment_full/`](../artifacts/analysis/alignment_full/).

## Showcase renders

The README figures and videos were rendered afterwards from the same
checkpoints, with the same inference bounds as the evaluations above.

| Item | Setting |
| --- | --- |
| Stills | Test views 0, 50, 100, 175 at 800 × 800; conditions GT, T1, T5, R5; all 5 methods; raw, plus Sim(3)-aligned for the 4 GT pose-optimizing runs |
| Error maps | Per-pixel ‖pred − gt‖₂ on unquantized RGB, turbo colormap, fixed range [0, 0.3] |
| Videos | All 200 test cameras in index order, 400 × 400, 30 fps: GT/Fixed, GT/Naive + warm-up (aligned), T5/Fixed, T5/Naive + warm-up |
| Identity check | Aligned stills match the historical per-view metrics of the full aligned evaluation exactly (16/16, max deviation 0 dB) |
| Raw check | Mean of the 4 rendered views vs. the 200-view metric (no per-view raw record exists); max gap 3.6 dB, which is expected for a 4-view sample |

Per-image metrics: [`artifacts/analysis/showcase_renders/metrics.csv`](../artifacts/analysis/showcase_renders/metrics.csv).
Provenance (run inventory, checkpoint hashes, server scripts):
[`artifacts/records/showcase/`](../artifacts/records/showcase/).

## Scope

Single scene (Lego), single seed (42). CamP-style is a controlled
re-implementation of the preconditioning idea, not the original CamP code. The
warm-up changes the whole camera learning-rate schedule, so the effect of the
early phase alone is not isolated. Full aligned rendering covers the GT
condition only. None of this supports claims about other scenes or seeds.

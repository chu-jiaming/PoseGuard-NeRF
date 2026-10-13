# Evaluation alignment diagnostic

This diagnostic reuses the final 29,999-step checkpoints; it never trains or
optimizes using test images. Current rendering scope is the GT initialization
condition, with fixed, naive, CamP-style, and both warm-up variants. A separate
CPU audit covers all 36 saved optimized pose sets.

The existing image metrics use cameras in the original coordinate system.
Joint camera/scene optimization can drift globally. To diagnose this, fit one
proper Sim(3) transform from GT training camera centers to optimized training
camera centers, minimizing center squared error with an SVD similarity fit:

`c_reconstruction = s Q c_GT + b`, `R_reconstruction = Q R_GT`.

The transform is fitted directly in this direction. It is not obtained by
inverting the reverse least-squares fit, which can differ slightly when local
pose residuals remain. All held-out cameras receive the same transform. Their
intrinsics are unchanged. Parser-transformed GT poses are used and camera/frame
ordering is checked. Near/far distances are scaled by s. The installed collider
resets inference near to zero, which stays zero after scaling. No network or
density rescaling is applied: the frozen network already represents its own
reconstruction coordinate system.

This is a diagnostic center-fit alignment, not an optimal photometric alignment.
Local position/orientation errors may remain; changes in image metrics do not
identify their cause or provide an additive decomposition of reconstruction error.

## Runs

```bash
bash scripts/run_alignment_diagnostic.sh --output results/alignment_gt_preview
bash scripts/run_alignment_diagnostic.sh --output results/alignment_gt_full --full
python scripts/audit_pose_alignment.py
```

Preview: test indices 0,10,...,190, 200x200 resolution. Both original and aligned
cameras are rendered from scratch. Do not compare these numbers directly to the
historical 800x800 metrics. The same fixed indices are chosen before observing
metrics. They are a diagnostic subset, not a replacement benchmark.

Full: all 200 test images at 800x800. Raw metrics reuse the recorded original
full-set evaluation for the identical checkpoint. Aligned metrics are rendered
anew using the existing model metric implementation. The fixed GT control must
reproduce original metrics within 0.01 dB PSNR / 0.0001 SSIM and LPIPS, otherwise
the job fails rather than presenting an unvalidated comparison.

Each run writes per-method alignment.json, result.json, progress.jsonl and images
for fixed test indices 0,50,100, plus summary.csv, report.md and state.json.
Preview also generates paired comparison panels. The manifest records source
paths and diagnostic code hashes; checkpoint paths/steps are checked at load.
Rerun the same command to skip completed methods. Source artifacts are read-only.

Geometry tests recover a known similarity, check projection invariance, reject
degenerate centers, and verify identity fitting. The paired fixed-GT preview
checks end-to-end identity rendering. Current evidence remains single-scene,
single-training-seed evidence, with no claims of cross-seed stability.

After a run completes, generate the interpretation and chart with:

```bash
python scripts/summarize_alignment.py results/alignment_gt_preview
python scripts/summarize_alignment.py results/alignment_gt_full
```

`results/alignment_pose_audit/source_fingerprints.json` records SHA-256 hashes
of the five source checkpoints, configs and historical metrics, GT train/test
metadata, model code, plus package versions. The camera rotation residual uses
the usual trace-based angle on stored float32 matrices; it has a small numerical
floor even for the identity control, so it should not be interpreted below that
precision. Image metrics for the identity control are checked separately.

## Completed evidence

Both preview and full GT runs completed successfully. Published [full results](artifacts/alignment_gt_full/report.md), [findings](artifacts/alignment_gt_full/findings.md), and [36-pose audit](artifacts/alignment_pose_audit/summary.csv) are available without local checkpoints. Fresh-machine setup: [REPRODUCING.md](REPRODUCING.md).

# Camera warm-up experiment

Run `bash scripts/run_warmup_experiments.sh` from this repository. Outputs go to
`results/poseopt_sv100_warmup2500`; existing no-warm-up results are in
`results/poseopt_sv100`. Both methods run the same nine SV100 conditions with
seed 42, 30,000 training steps, and diagnostics/checkpoints every 1,000 steps.
This is a single-seed paired experiment, not a cross-seed stability study.

Only the camera learning-rate schedule changes. Scene settings and pose
parameterization remain the same. The installed Nerfstudio 1.1.5 scheduler's
`cosine` ramp uses the following actual learning rate at step s:

- s < 2500: `1e-8 + (1e-3 - 1e-8) * sin(pi * s / (2 * 2500))`
- s >= 2500: exponential interpolation from `1e-3` at step 2500 to
  `1e-4` at step 30000.

This is Nerfstudio's native warm-up schedule, not an exact reproduction of the
CamP paper's multiplier schedule. Cameras are optimized from the first step;
there is no frozen-camera stage. The post-warm-up decay spans the remaining
27,500 steps. `diagnostics.jsonl` records `camera_lr` used by that iteration.

Monitor with `tmux attach -t poseguard-warmup2500` or inspect
`results/poseopt_sv100_warmup2500/pipeline.log` and `state.json`.
The pipeline writes `summary.csv`, `report.md`, metrics and comparison images.
Compare each method/case row against the same row in
`results/poseopt_sv100/summary.csv` for the no-warm-up control.

After interruption, run:

```bash
bash scripts/run_warmup_experiments.sh --resume --retry-failed
```

Resume restores both optimizer and scheduler states. A changed warm-up setting,
code, data or environment is rejected by the run fingerprint guard. Existing
runs created before this code change also cannot resume under the new fingerprint.
The shared lock prevents concurrent formal pipelines from competing for the GPU.

The general runner accepts `--camera-warmup-steps` (default 0). Use a separate
`--run-dir` for another warm-up duration. Warm-up must be nonnegative and shorter
than the training budget.

## Completed evidence

All 18 runs completed. See the [paired analysis snapshot](artifacts/warmup/report.md) and the subsequent [full GT alignment findings](artifacts/alignment_gt_full/findings.md). The paired report predates the alignment diagnostic. Setup and fresh-machine reproduction: [REPRODUCING.md](REPRODUCING.md).

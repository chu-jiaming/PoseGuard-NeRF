# Running the code

All commands run from the repository root. Python environments are managed with
[uv](https://docs.astral.sh/uv/) in the root `.venv`.

## 1. Re-analyze the published results (CPU only)

No GPU, dataset or checkpoint is needed.

```bash
uv sync --extra analysis
uv run python scripts/analysis/analyze_warmup_results.py --plots
```

This reads `artifacts/analysis/pose_optimization/` and `artifacts/analysis/warmup/`
and writes the paired CSV, aggregate JSON, report and plot to
`results/published_analysis/`. `--original` / `--warmup` select other directories
containing `summary.csv`; `--output` changes the destination. It only re-aggregates
existing metrics; nothing is trained or rendered.

The metric figures in `docs/assets/` can be rebuilt the same way:

```bash
uv run python scripts/showcase/make_figures.py
```

## 2. Training environment

Training and aligned rendering need Linux, an NVIDIA CUDA GPU, `bash`, `flock`,
`pgrep` and uv. The shell entry points use the repository `.venv` automatically.

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .venv/bin/python -r requirements-train.txt
uv pip check --python .venv/bin/python
uv run --no-sync python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

`requirements-train.txt` pins the key versions observed in the experiment
environment; it is not a full lock. The original runs used Python 3.8.10, and the
fresh-install path above has not been re-verified end to end on a new CUDA machine.
Nerfstudio extensions may need CUDA build tools.

> **Important:** in the training environment use `uv run --no-sync` (as below) or
> the bash entry points. A plain `uv run` syncs `.venv` to `uv.lock`, which only
> covers the CPU analysis path, and would remove PyTorch and Nerfstudio.

## 3. Data and output locations

Download NeRF-Synthetic and point `POSEGUARD_DATA` at the directory that contains `lego/`:

```text
nerf_synthetic/
└── lego/
    ├── train/  val/  test/
    └── transforms_{train,val,test}.json
```

```bash
export POSEGUARD_DATA=/absolute/path/to/nerf_synthetic
export POSEGUARD_RUN_ROOT=/absolute/path/to/experiment_workspace
```

| Variable | Purpose | Default |
| --- | --- | --- |
| `POSEGUARD_DATA` | Dataset root | `<repo>/data/nerf_synthetic` |
| `POSEGUARD_RUN_ROOT` | Parent of `results/`, `outputs/`, `logs/` | repository root |
| `POSEGUARD_BASELINE_SUMMARY` | Fixed-camera summary CSV | `<run root>/results/SV100_full_summary.csv` |
| `POSEGUARD_ORIGINAL_RUN` | Pose-optimization run without warm-up | `<run root>/results/poseopt_sv100` |
| `POSEGUARD_WARMUP_RUN` | Pose-optimization run with warm-up | `<run root>/results/poseopt_sv100_warmup2500` |

Relative paths resolve against the repository root; absolute paths are
recommended. If you pass `--run-dir` to a training entry point, set the matching
variable to the same location for later evaluation. Use a fresh run root for the
first fixed-camera baseline.

## 4. Generate perturbed data and train

Run in order; continue only after the previous step succeeds.

```bash
# 8 perturbed copies of the training split; refuses to overwrite existing ones.
uv run --no-sync python scripts/data/prepare_datasets.py

# Fixed cameras: GT + 8 perturbations, 9 runs.
bash scripts/training/run_full_sv100_pipeline.sh

# Check environment, data and the finished baseline before pose optimization.
bash scripts/training/run_pose_optimization_pipeline.sh --preflight
bash scripts/training/run_pose_optimization_pipeline.sh --dry-run

# Naive and CamP-style without warm-up, 18 runs.
bash scripts/training/run_pose_optimization_pipeline.sh

# Both methods with a 2,500-step camera learning-rate warm-up, 18 runs.
bash scripts/training/run_warmup_experiments.sh
```

`--preflight` and `--dry-run` also need the full environment, data and baseline.
Entry points share a file lock, so they never run concurrently. Progress is in
the run directory's `state.json` and `pipeline.log`. To resume after an
interruption or retry failed cases:

```bash
bash scripts/training/run_pose_optimization_pipeline.sh --resume --retry-failed
bash scripts/training/run_warmup_experiments.sh --resume --retry-failed
```

Resume with the same arguments, data and environment (including any
`--run-dir` / `--cases`). The fixed-camera entry point has no resume option and
refuses to reuse existing results.

## 5. Analyze new runs

```bash
uv run --no-sync python scripts/analysis/analyze_warmup_results.py \
  --original "$POSEGUARD_RUN_ROOT/results/poseopt_sv100" \
  --warmup "$POSEGUARD_RUN_ROOT/results/poseopt_sv100_warmup2500" \
  --output "$POSEGUARD_RUN_ROOT/results/warmup_analysis" \
  --plots
```

`--diagnostics` additionally plots early pose drift; it needs the full run
directories with the `diagnostics.jsonl` files referenced by `state.json`.

## 6. Pose audit and aligned rendering

These reuse the trained models and poses; nothing is retrained.

```bash
# Audit all 36 optimized pose sets (CPU + NumPy).
uv run --no-sync python scripts/evaluation/audit_pose_alignment.py

# GT condition, low-resolution preview.
bash scripts/evaluation/run_alignment_diagnostic.sh \
  --output "$POSEGUARD_RUN_ROOT/results/alignment_preview"
uv run --no-sync python scripts/analysis/summarize_alignment.py \
  "$POSEGUARD_RUN_ROOT/results/alignment_preview"

# GT condition, full test set.
bash scripts/evaluation/run_alignment_diagnostic.sh \
  --output "$POSEGUARD_RUN_ROOT/results/alignment_full" --full
uv run --no-sync python scripts/analysis/summarize_alignment.py \
  "$POSEGUARD_RUN_ROOT/results/alignment_full"
```

The audit accepts `--gt`, `--original`, `--warmup` and `--output`. Aligned
rendering reads the baseline and both pose-optimization runs from the variables
in step 3 and needs valid configs, checkpoints, metrics and the original data.
`artifacts/` has no model weights, so it cannot be resumed or re-rendered from
directly; server paths in historical records must map to real local files.

## 7. Showcase figures

The README renders and videos were produced on the training server into an
extracted `delivery_<date>/` directory (see [EXPERIMENTS.md](EXPERIMENTS.md#showcase-renders)).
With that directory available locally:

```bash
# Copy per-image metrics, slimmed curves, diagnostics and provenance into artifacts/.
uv run python scripts/showcase/import_delivery.py path/to/delivery_20261014

# Rebuild every figure in docs/assets/ (the GIF needs ffmpeg on PATH).
# --gt-transforms is the public NeRF-Synthetic lego/transforms_train.json; the T5
# perturbation is regenerated from it with seed 42 and checked against the archive.
uv run python scripts/showcase/make_figures.py --delivery path/to/delivery_20261014 \
  --gt-transforms path/to/nerf_synthetic/lego/transforms_train.json
```

## 8. Tests and help

```bash
# CPU tests: script portability, orchestration, Sim(3) alignment.
uv run python -m unittest discover -s tests -p "test_script_portability.py" -v
uv run python -m unittest discover -s tests -p "test_orchestration.py" -v
uv run python -m unittest discover -s tests -p "test_alignment.py" -v

# The full suite also needs PyTorch and Nerfstudio.
uv run --no-sync python -m unittest discover -s tests -v

# Entry-point arguments and environment variables.
bash scripts/training/run_pose_optimization_pipeline.sh --help
bash scripts/evaluation/run_alignment_diagnostic.sh --help
```

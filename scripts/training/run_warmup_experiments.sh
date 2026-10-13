#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
# Additional arguments support --resume, --retry-failed, --cases, and --dry-run.
exec bash "$ROOT/scripts/training/run_pose_optimization_pipeline.sh" \
  --run-dir "$ROOT/results/poseopt_sv100_warmup2500" \
  --steps 30000 --interval 1000 --camera-warmup-steps 2500 "$@"

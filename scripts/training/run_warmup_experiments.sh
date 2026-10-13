#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../runtime.sh"
WARMUP_RUN="$("${PYTHON[@]}" -c 'from scripts.runtime import WARMUP_RUN; print(WARMUP_RUN)')"
exec bash "$ROOT/scripts/training/run_pose_optimization_pipeline.sh" \
  --run-dir "$WARMUP_RUN" \
  --steps 30000 --interval 1000 --camera-warmup-steps 2500 "$@"

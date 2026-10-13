#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export NERFSTUDIO_METHOD_CONFIGS='poseguard-naive=poseguard.methods:naive,poseguard-camp=poseguard.methods:camp'
# Share the old fixed-pose pipeline's lock to prevent two formal jobs competing.
exec 9>.sv100_pipeline.lock
flock -n 9 || { echo 'Another PoseGuard pipeline is running'; exit 1; }
exec python scripts/training/pose_optimization_pipeline.py "$@"

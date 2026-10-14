#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../runtime.sh"
export NERFSTUDIO_METHOD_CONFIGS='poseguard-naive=poseguard.methods:naive,poseguard-camp=poseguard.methods:camp'
for arg in "$@"; do
    if [[ "$arg" == --help || "$arg" == -h ]]; then
        exec "${PYTHON[@]}" "$ROOT/scripts/training/pose_optimization_pipeline.py" --help
    fi
done
require_gpu_tools
lock_pipeline
exec "${PYTHON[@]}" "$ROOT/scripts/training/pose_optimization_pipeline.py" "$@"

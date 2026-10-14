#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../runtime.sh"
for arg in "$@"; do
    if [[ "$arg" == --help || "$arg" == -h ]]; then
        exec "${PYTHON[@]}" "$ROOT/scripts/evaluation/evaluate_alignment.py" --help
    fi
done
require_gpu_tools
lock_pipeline
exec "${PYTHON[@]}" "$ROOT/scripts/evaluation/evaluate_alignment.py" "$@"

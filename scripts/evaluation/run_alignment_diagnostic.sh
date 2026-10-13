#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec 9>.sv100_pipeline.lock
flock -n 9 || { echo 'Another pipeline is running'; exit 1; }
exec python scripts/evaluation/evaluate_alignment.py "$@"

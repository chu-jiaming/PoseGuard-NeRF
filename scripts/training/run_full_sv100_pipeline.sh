#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../runtime.sh"
for arg in "$@"; do
    if [[ "$arg" == --help || "$arg" == -h ]]; then
        exec "${PYTHON[@]}" "$ROOT/scripts/training/sv100_pipeline.py" --help
    fi
done
[[ $# == 0 ]] || { echo 'This entrypoint takes no arguments; configure paths through POSEGUARD_* variables (see --help).' >&2; exit 1; }
require_gpu_tools
lock_pipeline
RUN_ROOT="$("${PYTHON[@]}" -c 'from scripts.runtime import RUN_ROOT; print(RUN_ROOT)')"
mkdir -p "$RUN_ROOT/logs/gt" "$RUN_ROOT/logs/translation" "$RUN_ROOT/logs/rotation" "$RUN_ROOT/results" "$RUN_ROOT/outputs"
[[ ! -e "$RUN_ROOT/logs/poseguard_pipeline.log" ]] || { echo 'Pipeline log already exists; select a fresh POSEGUARD_RUN_ROOT.' >&2; exit 1; }
if pgrep -f '(^|/|[[:space:]])ns-train([[:space:]]|$)' >/dev/null; then
    echo 'An external ns-train process is active' >&2; exit 1
fi
exec > >(tee "$RUN_ROOT/logs/poseguard_pipeline.log") 2>&1
trap 'rc=$?; echo "FAILED shell exit=$rc line=$LINENO"; exit "$rc"' ERR
"${PYTHON[@]}" "$ROOT/scripts/training/sv100_pipeline.py"

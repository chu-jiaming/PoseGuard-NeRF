#!/usr/bin/env bash
# Shared Bash entrypoint setup. Python packages are managed in ROOT/.venv by uv.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v uv >/dev/null 2>&1 || { echo 'Missing uv; install uv and prepare the project .venv.' >&2; exit 1; }
[[ -x "$ROOT/.venv/bin/python" ]] || { echo "Missing project environment: $ROOT/.venv. Create it with uv and install the required dependencies." >&2; exit 1; }
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export PATH="$ROOT/.venv/bin:$PATH"
PYTHON=(uv run --no-project --python "$ROOT/.venv/bin/python" python)

require_gpu_tools() {
    [[ "$(uname -s)" == Linux ]] || { echo 'GPU entrypoints require Linux and CUDA; CPU analysis scripts can run separately.' >&2; exit 1; }
    local command_name
    for command_name in flock pgrep; do
        command -v "$command_name" >/dev/null 2>&1 || { echo "Required command unavailable: $command_name" >&2; exit 1; }
    done
    "${PYTHON[@]}" -c 'from scripts.runtime import require_training; require_training()'
}

lock_pipeline() {
    exec 9>"$ROOT/.sv100_pipeline.lock"
    flock -n 9 || { echo 'Another PoseGuard pipeline is running' >&2; exit 1; }
}

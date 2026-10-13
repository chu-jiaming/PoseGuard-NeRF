#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1
exec 9>.sv100_pipeline.lock
flock -n 9 || { echo 'FAILED: another formal pipeline owns the lock'; exit 1; }
mkdir -p logs/gt logs/translation logs/rotation results outputs
# Refuse to append to an earlier formal run or reuse any generated result.
[[ ! -e logs/poseguard_pipeline.log ]] || { echo 'FAILED: pipeline log already exists'; exit 1; }
exec > >(tee logs/poseguard_pipeline.log) 2>&1
trap 'rc=$?; echo "$(date -u --iso-8601=seconds) FAILED shell exit=$rc line=$LINENO"; exit "$rc"' ERR
printf '%s START full SV100 pipeline\n' "$(date -u --iso-8601=seconds)"
python - <<'PY'
import importlib.metadata
v = importlib.metadata.version('nerfstudio')
print(v)
if v != '1.1.5': raise SystemExit('FAILED: Nerfstudio must be 1.1.5')
PY
# No concurrent external ns-train process may exist at launch.
if pgrep -f '(^|/|[[:space:]])ns-train([[:space:]]|$)' >/dev/null; then
    echo 'FAILED: existing ns-train process'; exit 1
fi
python scripts/training/sv100_pipeline.py
printf '%s SUCCESS END full SV100 pipeline\n' "$(date -u --iso-8601=seconds)"

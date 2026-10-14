#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
DELIVERY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec /root/miniconda3/bin/python "$DELIVERY_DIR/scripts/gates.py" "$@"

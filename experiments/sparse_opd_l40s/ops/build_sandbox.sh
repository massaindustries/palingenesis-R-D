#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi
echo "[$(date -u --iso-8601=seconds)] docker build -t sparse-opd-executor $ROOT/sandbox"
if (( ! DRY_RUN )); then
  docker build -t sparse-opd-executor "$ROOT/sandbox"
fi


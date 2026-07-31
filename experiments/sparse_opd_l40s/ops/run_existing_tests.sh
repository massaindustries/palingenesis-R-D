#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/logs/tests_${STAMP}.log"
echo "[$(date -u --iso-8601=seconds)] original and sparse tests"
cd "$ROOT/repos/palingenesis"
set -o pipefail
"$ROOT/venvs/train/bin/pytest" -q 2>&1 | tee "$LOG"


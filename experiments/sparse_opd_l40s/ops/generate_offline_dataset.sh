#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
DRY_RUN=${1:-}
command=(
  "$ROOT/venvs/train/bin/python"
  "$ROOT/scripts/generate_offline_teacher_dataset.py"
)
echo "[$(date -u --iso-8601=seconds)] ${command[*]}"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  "$ROOT/ops/teacher_status.sh"
  "${command[@]}" 2>&1 | tee "$ROOT/logs/offline_dataset_$(date -u +%Y%m%dT%H%M%SZ).log"
fi

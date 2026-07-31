#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
if [[ $# -lt 1 ]]; then
  echo "usage: $0 CONDITION [--dry-run]" >&2
  exit 2
fi
CONDITION=$1
DRY_RUN=${2:-}
for seed in 0 1 2; do
  output="$ROOT/runs/sweep/${CONDITION}_seed${seed}"
  command=(
    "$ROOT/venvs/train/bin/pgs" distill
    --config "$ROOT/configs/sparse_opd/${CONDITION}.yaml"
    --train.seed "$seed"
    --train.output_dir "$output"
    --logging.run_name "${CONDITION}_seed${seed}"
  )
  echo "[$(date -u --iso-8601=seconds)] ${command[*]}"
  if [[ "$DRY_RUN" != "--dry-run" ]]; then
    (cd "$ROOT/repos/palingenesis" && "${command[@]}") \
      2>&1 | tee "$ROOT/logs/sweep_${CONDITION}_seed${seed}_$(date -u +%Y%m%dT%H%M%SZ).log"
  fi
done


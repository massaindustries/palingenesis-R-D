#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
SUITE=${1:-$(cat "$ROOT/runs/smoke/latest_suite_path.txt")}
DRY_RUN=${2:-}
DATASET="$ROOT/data/splits/smoke_test.jsonl"
EVAL_SUBDIR=${EVAL_SUBDIR:-evaluation}
cp "$ROOT/data/splits/manifest.json" "$SUITE/dataset_manifest_${EVAL_SUBDIR}.json"
base="$SUITE/base_student"
register=(
  "$ROOT/venvs/train/bin/python"
  "$ROOT/scripts/register_base_run.py"
  --output-dir "$base"
)
echo "[$(date -u --iso-8601=seconds)] ${register[*]}"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  "${register[@]}"
fi
for condition in base_student interval_1 interval_32 interval_128 final_only; do
  if [[ "$condition" == "base_student" ]]; then
    checkpoint=base
  else
    checkpoint="$SUITE/$condition/final"
  fi
  command=(
    "$ROOT/ops/evaluate_checkpoint.sh"
    "$checkpoint"
    "$DATASET"
    "$SUITE/$condition/$EVAL_SUBDIR"
  )
  echo "[$(date -u --iso-8601=seconds)] ${command[*]}"
  if [[ "$DRY_RUN" != "--dry-run" ]]; then
    "${command[@]}"
  fi
done

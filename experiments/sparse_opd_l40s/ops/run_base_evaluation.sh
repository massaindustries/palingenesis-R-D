#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
DATASET=${DATASET:-"$ROOT/data/splits/pilot_test.jsonl"}
DRY_RUN=${1:-}
SHA=$(git -C "$ROOT/repos/palingenesis" rev-parse --short HEAD)
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_base_student_seed0_${SHA}"
OUTPUT="$ROOT/runs/pilot/$RUN_ID"
register=(
  "$ROOT/venvs/train/bin/python"
  "$ROOT/scripts/register_base_run.py"
  --output-dir "$OUTPUT"
)
evaluate=(
  "$ROOT/ops/evaluate_checkpoint.sh"
  base
  "$DATASET"
  "$OUTPUT/evaluation"
)
echo "[$(date -u --iso-8601=seconds)] ${register[*]}"
echo "[$(date -u --iso-8601=seconds)] ${evaluate[*]}"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  "${register[@]}"
  printf '%s\n' "$OUTPUT" > "$ROOT/runs/pilot/latest_base_path.txt"
  "$ROOT/ops/capture_run_metadata.sh" "$OUTPUT" start
  "${evaluate[@]}"
  if [[ -s "$ROOT/data/offline/teacher_eval_pilot_test_v2.jsonl" ]]; then
    "$ROOT/ops/analyze_evaluations.sh" \
      "$OUTPUT/evaluation" \
      "$ROOT/data/offline/teacher_eval_pilot_test_v2.jsonl"
  fi
  "$ROOT/ops/capture_run_metadata.sh" "$OUTPUT" end
fi

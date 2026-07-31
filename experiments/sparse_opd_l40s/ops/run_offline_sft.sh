#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
STEPS=${STEPS:-300}
DRY_RUN=${1:-}
SHA=$(git -C "$ROOT/repos/palingenesis" rev-parse --short HEAD)
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_offline_sft_seed0_${SHA}"
OUTPUT="$ROOT/runs/pilot/$RUN_ID"
command=(
  "$ROOT/venvs/train/bin/python"
  "$ROOT/scripts/train_offline_sft.py"
  --output-dir "$OUTPUT"
  --steps "$STEPS"
)
echo "[$(date -u --iso-8601=seconds)] ${command[*]}"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  test -s "$ROOT/data/offline/teacher_sft_train_v2.jsonl"
  mkdir -p "$OUTPUT"
  printf '%s\n' "$OUTPUT" > "$ROOT/runs/pilot/latest_offline_sft_path.txt"
  "$ROOT/ops/capture_run_metadata.sh" "$OUTPUT" start
  "${command[@]}" 2>&1 | tee "$ROOT/logs/${RUN_ID}.log" "$OUTPUT/stdout_stderr.log"
  "$ROOT/ops/evaluate_checkpoint.sh" \
    "$OUTPUT/final" \
    "$ROOT/data/splits/pilot_test.jsonl" \
    "$OUTPUT/evaluation"
  if [[ -s "$ROOT/data/offline/teacher_eval_pilot_test_v2.jsonl" ]]; then
    "$ROOT/ops/analyze_evaluations.sh" \
      "$OUTPUT/evaluation" \
      "$ROOT/data/offline/teacher_eval_pilot_test_v2.jsonl"
  fi
  "$ROOT/ops/capture_run_metadata.sh" "$OUTPUT" end
fi

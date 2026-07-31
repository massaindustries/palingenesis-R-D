#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
TARGET=all
DRY_RUN=0
for argument in "$@"; do
  if [[ "$argument" == "--dry-run" ]]; then
    DRY_RUN=1
  elif [[ "$argument" == "--condition" ]]; then
    :
  elif [[ "$TARGET" == "all" && "$argument" != "--condition" ]]; then
    TARGET=$argument
  fi
done
if [[ "${1:-}" == "--condition" ]]; then TARGET=${2:?missing condition}; fi
STEPS=${PILOT_STEPS:-300}
SHA=$(git -C "$ROOT/repos/palingenesis" rev-parse --short HEAD)
echo "[$(date -u --iso-8601=seconds)] pilot start"
"$ROOT/ops/teacher_status.sh"
for condition in interval_1 interval_32 interval_64 interval_128 final_only; do
  if [[ "$TARGET" != "all" && "$TARGET" != "$condition" ]]; then
    continue
  fi
  config="$ROOT/configs/sparse_opd/${condition}.yaml"
  run_id="$(date -u +%Y%m%dT%H%M%SZ)_${condition}_seed0_${SHA}"
  output="$ROOT/runs/pilot/$run_id"
  command=(
    "$ROOT/venvs/train/bin/pgs" distill
    --config "$config"
    --train.output_dir "$output"
    --train.steps "$STEPS"
  )
  echo "[$(date -u --iso-8601=seconds)] ${command[*]}"
  if (( ! DRY_RUN )); then
    mkdir -p "$output"
    printf '%s\n' "$output" > "$ROOT/runs/pilot/latest_${condition}_path.txt"
    "$ROOT/ops/capture_run_metadata.sh" "$output" start
    log="$ROOT/logs/pilot_${condition}_$(date -u +%Y%m%dT%H%M%SZ).log"
    (cd "$ROOT/repos/palingenesis" && "${command[@]}") \
      2>&1 | tee "$log" "$output/stdout_stderr.log"
    "$ROOT/ops/evaluate_checkpoint.sh" \
      "$output/final" \
      "$ROOT/data/splits/pilot_test.jsonl" \
      "$output/evaluation"
    if [[ -s "$ROOT/data/offline/teacher_eval_pilot_test_v2.jsonl" ]]; then
      "$ROOT/ops/analyze_evaluations.sh" \
        "$output/evaluation" \
        "$ROOT/data/offline/teacher_eval_pilot_test_v2.jsonl"
    fi
    "$ROOT/ops/capture_run_metadata.sh" "$output" end
  fi
done
if [[ "$TARGET" == "all" || "$TARGET" == "base" ]]; then
  if (( DRY_RUN )); then
    "$ROOT/ops/run_base_evaluation.sh" --dry-run
  else
    "$ROOT/ops/run_base_evaluation.sh"
  fi
fi
if [[ "$TARGET" == "all" || "$TARGET" == "offline_sft" ]]; then
  if (( DRY_RUN )); then
    "$ROOT/ops/generate_offline_dataset.sh" --dry-run
    STEPS="$STEPS" "$ROOT/ops/run_offline_sft.sh" --dry-run
  else
    "$ROOT/ops/generate_offline_dataset.sh"
    STEPS="$STEPS" "$ROOT/ops/run_offline_sft.sh"
  fi
fi
echo "[$(date -u --iso-8601=seconds)] pilot target $TARGET complete"

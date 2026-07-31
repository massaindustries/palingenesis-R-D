#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi
STEPS=${LR_SWEEP_STEPS:-20}
SHA=$(git -C "$ROOT/repos/palingenesis" rev-parse --short HEAD)
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_lr_mini_sweep_interval32_seed0_${SHA}"
RUN_ROOT="$ROOT/runs/sweep/$RUN_ID"
mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT" > "$ROOT/runs/sweep/latest_lr_sweep_path.txt"
git -C "$ROOT/repos/palingenesis" rev-parse HEAD > "$RUN_ROOT/harness_sha.txt"
git -C "$ROOT/repos/palingenesis" diff > "$RUN_ROOT/harness.diff"
cp "$ROOT/data/splits/manifest.json" "$RUN_ROOT/dataset_manifest.json"

echo "[$(date -u --iso-8601=seconds)] LR mini-sweep start: interval32, $STEPS steps"
"$ROOT/ops/teacher_status.sh"
for lr in 2e-5 5e-5 1e-4; do
  label=${lr/e-/e-}
  output="$RUN_ROOT/lr_$label"
  command=(
    "$ROOT/venvs/train/bin/pgs" distill
    --config "$ROOT/configs/sparse_opd/interval_32.yaml"
    --data.prompts_path "$ROOT/data/splits/smoke_train.jsonl"
    --data.dev_prompts_path "$ROOT/data/splits/smoke_dev.jsonl"
    --train.output_dir "$output"
    --train.steps "$STEPS"
    --train.learning_rate "$lr"
    --train.warmup_steps 2
    --train.eval_every 0
    --train.save_steps "$STEPS"
    --sampling.batch_prompts 1
    --sampling.max_new_tokens 64
  )
  echo "[$(date -u --iso-8601=seconds)] ${command[*]}"
  if (( ! DRY_RUN )); then
    mkdir -p "$output"
    log="$ROOT/logs/lr_sweep_${label}_$(date -u +%Y%m%dT%H%M%SZ).log"
    (cd "$ROOT/repos/palingenesis" && "${command[@]}") 2>&1 | tee "$log"
  fi
done
if (( ! DRY_RUN )); then
  "$ROOT/venvs/train/bin/python" "$ROOT/scripts/summarize_lr_sweep.py" "$RUN_ROOT"
fi
echo "[$(date -u --iso-8601=seconds)] LR mini-sweep complete"

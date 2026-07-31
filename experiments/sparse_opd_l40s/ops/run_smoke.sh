#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi
source "$ROOT/env/common.env"
echo "[$(date -u --iso-8601=seconds)] sparse OPD smoke gates"
HARNESS_SHA=$(git -C "$ROOT/repos/palingenesis" rev-parse --short HEAD)
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_smoke_suite_seed0_${HARNESS_SHA}"
RUN_ROOT="$ROOT/runs/smoke/$RUN_ID"
mkdir -p "$RUN_ROOT"
printf '%s\n' "$RUN_ROOT" > "$ROOT/runs/smoke/latest_suite_path.txt"
git -C "$ROOT/repos/palingenesis" rev-parse HEAD > "$RUN_ROOT/harness_sha.txt"
git -C "$ROOT/repos/palingenesis" diff > "$RUN_ROOT/harness.diff"
cp "$ROOT/data/splits/manifest.json" "$RUN_ROOT/dataset_manifest.json"
nvidia-smi > "$RUN_ROOT/nvidia_smi_start.txt"
commands=(
  "$ROOT/ops/teacher_status.sh"
  "$ROOT/venvs/train/bin/python $ROOT/scripts/check_tokenizer_compatibility.py"
  "$ROOT/ops/run_existing_tests.sh"
)
for command in "${commands[@]}"; do
  echo "+ $command"
  if (( ! DRY_RUN )); then bash -lc "$command"; fi
done

for condition in interval_1 interval_32 interval_128 final_only; do
  config="$ROOT/configs/sparse_opd/${condition}.yaml"
  output="$RUN_ROOT/${condition}"
  log="$ROOT/logs/smoke_${condition}_$(date -u +%Y%m%dT%H%M%SZ).log"
  command=(
    "$ROOT/venvs/train/bin/pgs" distill
    --config "$config"
    --data.prompts_path "$ROOT/data/splits/smoke_train.jsonl"
    --data.dev_prompts_path "$ROOT/data/splits/smoke_dev.jsonl"
    --train.output_dir "$output"
    --train.steps 20
    --train.warmup_steps 2
    --train.eval_every 0
    --train.save_steps 10
    --sampling.batch_prompts 1
    --sampling.max_new_tokens 64
  )
  echo "[$(date -u --iso-8601=seconds)] ${command[*]}"
  if (( ! DRY_RUN )); then
    mkdir -p "$output"
    (cd "$ROOT/repos/palingenesis" && "${command[@]}") 2>&1 | tee "$log"
  fi
done
nvidia-smi > "$RUN_ROOT/nvidia_smi_end.txt"
echo "[$(date -u --iso-8601=seconds)] smoke complete"

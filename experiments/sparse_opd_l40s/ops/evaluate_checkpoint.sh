#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
if [[ $# -lt 3 ]]; then
  echo "usage: $0 CHECKPOINT_OR_BASE DATASET_JSONL OUTPUT_DIR [--dry-run]" >&2
  exit 2
fi
CHECKPOINT=$1
DATASET=$2
OUTPUT=$3
DRY_RUN=${4:-}
mkdir -p "$OUTPUT"
command=(
  "$ROOT/venvs/train/bin/python"
  "$ROOT/scripts/evaluate_mbpp.py"
  --dataset "$DATASET"
  --output-dir "$OUTPUT"
)
if [[ "$CHECKPOINT" != "base" ]]; then
  command+=(--checkpoint "$CHECKPOINT")
fi
echo "[$(date -u --iso-8601=seconds)] ${command[*]}"
if [[ "$DRY_RUN" != "--dry-run" ]]; then
  "${command[@]}" 2>&1 | tee \
    "$ROOT/logs/evaluate_$(date -u +%Y%m%dT%H%M%SZ).log" \
    "$OUTPUT/evaluation_stdout_stderr.log"
fi

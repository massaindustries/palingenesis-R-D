#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
if [[ $# -lt 1 ]]; then
  echo "usage: $0 EVALUATION_DIR [TEACHER_COMPLETIONS_JSONL] [--dry-run]" >&2
  exit 2
fi
EVALUATION_DIR=$1
TEACHER_COMPLETIONS=${2:-}
DRY_RUN=${3:-}
command=(
  "$ROOT/venvs/train/bin/python"
  "$ROOT/scripts/analyze_evaluations.py"
  "$EVALUATION_DIR"
)
if [[ -n "$TEACHER_COMPLETIONS" && "$TEACHER_COMPLETIONS" != "--dry-run" ]]; then
  command+=(--teacher-completions "$TEACHER_COMPLETIONS")
fi
echo "[$(date -u --iso-8601=seconds)] ${command[*]}"
if [[ "$DRY_RUN" != "--dry-run" && "$TEACHER_COMPLETIONS" != "--dry-run" ]]; then
  "${command[@]}"
fi

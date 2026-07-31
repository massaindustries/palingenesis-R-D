#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
PID_FILE="$ROOT/logs/teacher.pid"
SESSION=sparse-opd-teacher
if tmux has-session -t "$SESSION" 2>/dev/null \
  && [[ -f "$PID_FILE" ]] \
  && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "teacher process: running pid=$(cat "$PID_FILE")"
else
  echo "teacher process: stopped"
  exit 1
fi
"$ROOT/venvs/train/bin/python" "$ROOT/ops/teacher_healthcheck.py"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv

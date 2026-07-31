#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
PID_FILE="$ROOT/logs/teacher.pid"
SESSION=sparse-opd-teacher
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[$(date -u --iso-8601=seconds)] stopping teacher tmux session"
  tmux kill-session -t "$SESSION"
fi
if [[ ! -f "$PID_FILE" ]]; then
  echo "teacher is not recorded as running"
  exit 0
fi
pid=$(cat "$PID_FILE")
if kill -0 "$pid" 2>/dev/null; then
  echo "[$(date -u --iso-8601=seconds)] stopping teacher pid=$pid"
  kill "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "teacher did not stop after 30 seconds; sending SIGKILL"
    kill -KILL "$pid"
  fi
fi
rm -f "$PID_FILE"
echo "teacher stopped"

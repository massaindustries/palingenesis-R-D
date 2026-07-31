#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
source "$ROOT/env/common.env"
source "$ROOT/env/gpu_map.env"
export PATH="$ROOT/venvs/teacher/bin:$PATH"
PID_FILE="$ROOT/logs/teacher.pid"
SESSION=sparse-opd-teacher
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/logs/teacher_${STAMP}.log"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "teacher already running in tmux session $SESSION"
  exit 0
fi
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "teacher already running pid=$(cat "$PID_FILE")"
  exit 0
fi
command=(
  "$ROOT/venvs/teacher/bin/python" -m sglang.launch_server
  --model-path "$ROOT/models/Qwen3-Coder-Next-FP8"
  --tp-size 2
  --host 127.0.0.1
  --port 30000
  --context-length 8192
  --mem-fraction-static 0.90
  --chunked-prefill-size 4096
  --max-running-requests 32
  --max-mamba-cache-size 32
  --enable-metrics
  --enable-deterministic-inference
  --enable-p2p-check
)
echo "[$(date -u --iso-8601=seconds)] CUDA_VISIBLE_DEVICES=$TEACHER_GPUS ${command[*]}"
if (( DRY_RUN )); then exit 0; fi
printf -v command_line '%q ' "${command[@]}"
printf -v gpu_value '%q' "$TEACHER_GPUS"
printf -v log_value '%q' "$LOG"
tmux new-session -d -s "$SESSION" \
  "exec env CUDA_VISIBLE_DEVICES=$gpu_value $command_line > $log_value 2>&1"
pid=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
echo "$pid" > "$PID_FILE"
echo "$LOG" > "$ROOT/logs/teacher.current_log"
echo "teacher starting pid=$pid log=$LOG"

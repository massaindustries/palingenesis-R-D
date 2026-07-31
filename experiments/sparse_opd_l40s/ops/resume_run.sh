#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
if [[ $# -lt 2 ]]; then
  echo "usage: $0 CONFIG CHECKPOINT [extra pgs args...]" >&2
  exit 2
fi
CONFIG=$1
CHECKPOINT=$2
shift 2
echo "[$(date -u --iso-8601=seconds)] resume config=$CONFIG checkpoint=$CHECKPOINT"
cd "$ROOT/repos/palingenesis"
exec "$ROOT/venvs/train/bin/pgs" distill \
  --config "$CONFIG" \
  --train.resume_from "$CHECKPOINT" \
  "$@"


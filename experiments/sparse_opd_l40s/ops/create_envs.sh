#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
UV=${UV_BIN:-/root/.local/bin/uv}
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi
run() {
  echo "+ $*"
  if (( ! DRY_RUN )); then "$@"; fi
}
echo "[$(date -u --iso-8601=seconds)] create environments start"
run "$UV" python install 3.11
run "$UV" venv --python 3.11 "$ROOT/venvs/train"
run "$UV" pip install --python "$ROOT/venvs/train/bin/python" -e "$ROOT/repos/palingenesis[dev]" peft httpx aiohttp pytest-timeout huggingface_hub matplotlib pandas
run "$UV" venv --python 3.11 "$ROOT/venvs/teacher"
run "$UV" pip install --python "$ROOT/venvs/teacher/bin/python" "torch==2.9.1+cu128" --index https://download.pytorch.org/whl/cu128
run "$UV" pip install --python "$ROOT/venvs/teacher/bin/python" "sglang==0.5.8" huggingface_hub httpx aiohttp --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
if (( ! DRY_RUN )); then
  "$UV" pip freeze --python "$ROOT/venvs/train/bin/python" > "$ROOT/reports/pip_freeze_trainer.txt"
  "$UV" pip freeze --python "$ROOT/venvs/teacher/bin/python" > "$ROOT/reports/pip_freeze_teacher.txt"
fi
echo "[$(date -u --iso-8601=seconds)] create environments complete"


#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
source "$ROOT/env/common.env"
export HF_HUB_ENABLE_HF_TRANSFER=1
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi
download() {
  local repo=$1
  local revision=$2
  local name=$3
  echo "[$(date -u --iso-8601=seconds)] download $repo@$revision"
  if (( DRY_RUN )); then return; fi
  local snapshot
  snapshot=$("$ROOT/venvs/teacher/bin/hf" download "$repo" --revision "$revision" --cache-dir "$HF_HOME")
  ln -sfn "$snapshot" "$ROOT/models/$name"
}
download Qwen/Qwen3-Coder-Next-FP8 da6e2ed27304dd39abadd9c82ef50e8de67bdd4c Qwen3-Coder-Next-FP8
download Qwen/Qwen3-4B 1cfa9a7208912126459214e8b04321603b3df60c Qwen3-4B
echo "[$(date -u --iso-8601=seconds)] model downloads complete"


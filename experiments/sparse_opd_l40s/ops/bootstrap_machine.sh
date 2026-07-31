#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$ROOT/logs/bootstrap_${STAMP}.log"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; fi
mkdir -p "$ROOT"/{env,models,cache/{huggingface,torch,datasets},repos,venvs,data/{raw,processed,splits},runs/{smoke,pilot,sweep},results,reports,logs,ops}
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -u --iso-8601=seconds)] bootstrap start dry_run=$DRY_RUN"

commands=(
  "apt-get update"
  "DEBIAN_FRONTEND=noninteractive apt-get install -y git git-lfs curl wget jq tmux htop nvtop build-essential python3.11 python3.11-dev python3.11-venv docker-ce docker-compose-plugin"
  "git lfs install"
)
for command in "${commands[@]}"; do
  echo "+ $command"
  if (( ! DRY_RUN )); then bash -lc "$command"; fi
done

nvidia-smi
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv
df -h / /opt /dev/shm
free -h
echo "[$(date -u --iso-8601=seconds)] bootstrap complete"


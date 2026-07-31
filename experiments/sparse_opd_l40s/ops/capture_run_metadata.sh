#!/usr/bin/env bash
set -euo pipefail

ROOT=${SPARSE_OPD_ROOT:-/opt/sparse-opd}
if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_DIR start|end" >&2
  exit 2
fi
RUN_DIR=$1
STAGE=$2
mkdir -p "$RUN_DIR"
case "$STAGE" in
  start)
    date -u --iso-8601=seconds > "$RUN_DIR/start_timestamp.txt"
    git -C "$ROOT/repos/palingenesis" rev-parse HEAD > "$RUN_DIR/harness_sha.txt"
    git -C "$ROOT/repos/palingenesis" diff > "$RUN_DIR/harness.diff"
    cp "$ROOT/data/splits/manifest.json" "$RUN_DIR/dataset_manifest.json"
    cp "$ROOT/reports/pip_freeze_teacher.txt" "$RUN_DIR/pip_freeze_teacher.txt"
    cp "$ROOT/reports/pip_freeze_trainer.txt" "$RUN_DIR/pip_freeze_trainer.txt"
    tar \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      -czf "$RUN_DIR/experiment_source.tar.gz" \
      -C "$ROOT" ops scripts configs sandbox
    (
      cd "$ROOT"
      find ops scripts configs sandbox -type f \
        ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
        | sort -z | xargs -0 sha256sum
    ) > "$RUN_DIR/experiment_files_sha256.txt"
    cat > "$RUN_DIR/model_revisions.json" <<'JSON'
{
  "teacher": {
    "model": "Qwen/Qwen3-Coder-Next-FP8",
    "revision": "da6e2ed27304dd39abadd9c82ef50e8de67bdd4c"
  },
  "student": {
    "model": "Qwen/Qwen3-4B",
    "revision": "1cfa9a7208912126459214e8b04321603b3df60c"
  }
}
JSON
    nvidia-smi > "$RUN_DIR/nvidia_smi_start.txt"
    nvidia-smi topo -m > "$RUN_DIR/gpu_topology.txt"
    ;;
  end)
    if [[ ! -s "$RUN_DIR/experiment_source.tar.gz" ]]; then
      tar \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        -czf "$RUN_DIR/experiment_source.tar.gz" \
        -C "$ROOT" ops scripts configs sandbox
      (
        cd "$ROOT"
        find ops scripts configs sandbox -type f \
          ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
          | sort -z | xargs -0 sha256sum
      ) > "$RUN_DIR/experiment_files_sha256.txt"
    fi
    date -u --iso-8601=seconds > "$RUN_DIR/end_timestamp.txt"
    nvidia-smi > "$RUN_DIR/nvidia_smi_end.txt"
    ;;
  *)
    echo "stage must be start or end" >&2
    exit 2
    ;;
esac
echo "[$(date -u --iso-8601=seconds)] captured $STAGE metadata in $RUN_DIR"

# Sparse On-Policy Distillation on 4×L40S

This directory contains the reproducible experiment bundle, validation
evidence, and final report for sparse anchor reverse-KL distillation of
Qwen3-4B from Qwen3-Coder-Next-FP8.

## Result

All quality figures use the same 257-task MBPP prompt-v2 test set and locked
Docker sandbox.

| Condition | Steps | Pass@1 | Wilson 95% CI | Test pass | Estimated GPU-hours |
|---|---:|---:|---:|---:|---:|
| Base Qwen3-4B | 0 | 60.31% | 54.22%–66.10% | 66.49% | 0.00 |
| Offline teacher SFT | 300 | 60.70% | 54.61%–66.47% | 65.85% | 0.84 |
| Sparse RKL, K=64, interval=32 | 300 | **61.09%** | 55.01%–66.85% | 64.82% | 13.53 |

The sparse pilot has the highest point estimate, but the observed gain over
the base model is not statistically resolved: +0.78 percentage points, paired
bootstrap 95% CI −3.50 to +5.06 points, McNemar p=0.864. The current
single-seed evidence therefore does not establish a quality improvement or a
better quality/compute trade-off over offline SFT.

Start with:

- [`reports/final_report.html`](reports/final_report.html) for the portable
  technical report with seven charts.
- [`reports/final_report.md`](reports/final_report.md) for the text version.
- [`reports/result_validation.json`](reports/result_validation.json) for the
  combined data-quality audit.
- [`results/summary.json`](results/summary.json) and
  [`results/pareto_quality_cost.csv`](results/pareto_quality_cost.csv) for
  machine-readable results.

The HTML artifact passed schema validation and structural packaging. Visual
browser verification is marked `structural_only` because Chromium was not
installed on the experiment host.

## What is included

- `configs/`: the frozen LR=1e-4 experiment matrix.
- `ops/`: machine setup, model/service management, execution, metadata,
  validation, collection, and report entry points.
- `scripts/`: dataset preparation, evaluation, paired statistics, offline
  corpus generation, and offline-SFT training.
- `sandbox/`: the locked code-execution sandbox.
- `reports/` and `results/`: final evidence and machine-readable summaries.
- `PLAN.md` and `STATUS.md`: original requirements and completion record.

Model weights, virtual environments, generated datasets, run logs, and
checkpoints are intentionally excluded.

## Reproduce

The shell wrappers use `SPARSE_OPD_ROOT`, defaulting to `/opt/sparse-opd`.
The frozen YAML and Python defaults record the official `/opt/sparse-opd`
layout; using another root requires rewriting those absolute paths. Clone this
repository at `$SPARSE_OPD_ROOT/repos/palingenesis`, then copy the experiment
bundle into the expected workspace layout:

```bash
export SPARSE_OPD_ROOT=/opt/sparse-opd
bundle="$SPARSE_OPD_ROOT/repos/palingenesis/experiments/sparse_opd_l40s"
mkdir -p \
  "$SPARSE_OPD_ROOT/configs/sparse_opd" \
  "$SPARSE_OPD_ROOT/ops" \
  "$SPARSE_OPD_ROOT/scripts" \
  "$SPARSE_OPD_ROOT/sandbox"
rsync -a "$bundle/configs/" "$SPARSE_OPD_ROOT/configs/sparse_opd/"
rsync -a "$bundle/ops/" "$SPARSE_OPD_ROOT/ops/"
rsync -a "$bundle/scripts/" "$SPARSE_OPD_ROOT/scripts/"
rsync -a "$bundle/sandbox/" "$SPARSE_OPD_ROOT/sandbox/"
```

Provision and verify the host:

```bash
"$SPARSE_OPD_ROOT/ops/bootstrap_machine.sh"
"$SPARSE_OPD_ROOT/ops/create_envs.sh"
"$SPARSE_OPD_ROOT/ops/download_models.sh"
"$SPARSE_OPD_ROOT/ops/start_teacher.sh"
"$SPARSE_OPD_ROOT/ops/run_existing_tests.sh"
```

Prepare prompt-v2 MBPP data and the immutable offline teacher corpus:

```bash
"$SPARSE_OPD_ROOT/venvs/train/bin/python" \
  "$SPARSE_OPD_ROOT/scripts/prepare_mbpp.py"
"$SPARSE_OPD_ROOT/ops/generate_offline_dataset.sh"
```

Run the three official conditions:

```bash
"$SPARSE_OPD_ROOT/ops/run_base_evaluation.sh"
PILOT_STEPS=300 \
  "$SPARSE_OPD_ROOT/ops/run_pilot.sh" --condition interval_32
STEPS=300 "$SPARSE_OPD_ROOT/ops/run_offline_sft.sh"
```

Collect, validate, and render:

```bash
"$SPARSE_OPD_ROOT/venvs/train/bin/python" \
  "$SPARSE_OPD_ROOT/ops/collect_results.py"
"$SPARSE_OPD_ROOT/venvs/train/bin/python" \
  "$SPARSE_OPD_ROOT/ops/validate_results.py" RUN_DIR...
"$SPARSE_OPD_ROOT/venvs/train/bin/python" \
  "$SPARSE_OPD_ROOT/ops/make_report.py"
```

Resume an interrupted sparse run:

```bash
"$SPARSE_OPD_ROOT/ops/resume_run.sh" \
  "$SPARSE_OPD_ROOT/configs/sparse_opd/interval_32.yaml" \
  RUN_DIR/step_250 \
  --train.steps 300
```

## Provenance

- Experiment harness commit used by official runs: `7d6f8ac`.
- Teacher revision:
  `da6e2ed27304dd39abadd9c82ef50e8de67bdd4c`.
- Student revision:
  `1cfa9a7208912126459214e8b04321603b3df60c`.
- MBPP source revision:
  `4bb6404fdc6cacfda99d4ac4205087b89d32030c`.
- Hardware map: teacher GPUs 0–1, trainer GPU 2, rollout GPU 3.
- Final test result: 277 passed, 5 non-fatal warnings.

Every official run captured resolved configuration, dataset/model revisions,
source archive and checksums, package freezes, GPU topology/snapshots,
timestamps, logs, metrics, and resumable checkpoints.

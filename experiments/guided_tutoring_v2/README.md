# Guided Tutoring v2

This experiment tests whether a frozen large teacher can guide an on-policy
student by inserting real continuation tokens into the student's trajectory.
It compares 8, 32, and 128 autonomous student-token intervals with a fixed
8-token teacher guidance group.

The preregistration in [`preregistration.md`](preregistration.md) was frozen
before any v2 model-quality result was generated.

Engineering evidence showed that MBPP completions usually terminate before an
interval-128 intervention. The preregistered deviations document the bounded
retry accounting and the move to BigCodeBench-Long for the primary tutoring
task. MBPP+ remains a secondary short-code generalization evaluation.

## Primary data

The primary dataset is the pinned BigCodeBench-Long96-Stdlib subset described
in [`protocol_deviation_002.md`](protocol_deviation_002.md). Build and validate
it in two stages so that only canonical solutions executable in the locked
sandbox enter the split:

```bash
python experiments/guided_tutoring_v2/validate_bcb_subset.py \
  --output /tmp/bcb_canonical_validation.jsonl

python experiments/guided_tutoring_v2/prepare_bcb_long.py \
  --canonical-validation /tmp/bcb_canonical_validation.jsonl \
  --output-dir experiments/guided_tutoring_v2/data/bcb_long96
```

Generated data is ignored by Git. The manifest pins the source revision,
records every output hash, and verifies zero task-ID and prompt-hash overlap.
All probe-inspected tasks are quarantined into train.

## Secondary MBPP+ data

```bash
python experiments/guided_tutoring_v2/prepare_dataset.py \
  --splits-dir /path/to/prompt-v2/splits \
  --output-dir experiments/guided_tutoring_v2/data
```

The command downloads the pinned EvalPlus MBPP+ revision, joins it only to the
already held-out MBPP test IDs, and writes a validation manifest. Augmented
MBPP+ tests are never copied into train or dev rows.

## Causal and training validation

The paired causal probe applies matched and shuffled-label updates from the
same model, optimizer state, and exact trajectories:

```bash
python experiments/guided_tutoring_v2/paired_causal_probe.py \
  --config experiments/guided_tutoring_v2/configs/guided_i32_g8_matched_probe.yaml \
  --output-dir /tmp/paired_causal_i32 \
  --trials 20

python experiments/guided_tutoring_v2/analyze_paired_causal.py \
  --metrics /tmp/paired_causal_i32/paired_causal_metrics.jsonl \
  --output /tmp/paired_causal_i32/analysis.json
```

The smoke comparator samples independently from the same student policy. For
a functional intervention check without sampling ambiguity, run the
train-only greedy counterfactual:

```bash
python experiments/guided_tutoring_v2/deterministic_guidance_probe.py \
  --dataset experiments/guided_tutoring_v2/data/bcb_long96/train_evaluation.jsonl \
  --output-dir /tmp/deterministic_guidance_probe \
  --limit 20
```

The frozen scientific budget is in
[`scientific_budget.md`](scientific_budget.md). After all nine runs finish,
validate their telemetry and perform dev-only checkpoint selection with:

```bash
python experiments/guided_tutoring_v2/summarize_training.py \
  --run-root /path/to/scientific_run \
  --output /path/to/scientific_run/training_summary.json

python experiments/guided_tutoring_v2/evaluate_checkpoints.py \
  --run-root /path/to/scientific_run \
  --dataset experiments/guided_tutoring_v2/data/bcb_long96/dev_evaluation.jsonl \
  --base-evaluation /path/to/dev_base_evaluation \
  --output-root /path/to/dev_evaluations
```

The test split must remain unopened until `selection.json` has fixed one
checkpoint for each interval and seed. The selector applies the preregistered
quality-collapse and truncation kill switches before ranking eligible
checkpoints.

After selection, evaluate the frozen conditions once and create the paired
analysis:

```bash
python experiments/guided_tutoring_v2/evaluate_selected.py \
  --selection /path/to/dev_evaluations/selection.json \
  --dataset experiments/guided_tutoring_v2/data/bcb_long96/test_evaluation.jsonl \
  --output-root /path/to/test_evaluations \
  --offline-checkpoint /path/to/offline_sft/final

python experiments/guided_tutoring_v2/analyze_test.py \
  --evaluation-root /path/to/test_evaluations \
  --output /path/to/test_evaluations/analysis.json
```

The completed experiment, including negative results and limitations, is
summarized in [`report.md`](report.md).

# Protocol deviation 002 — primary tutoring dataset

Recorded: 2026-07-31, before any Guided Tutoring v2 held-out quality result.

## Engineering evidence

MBPP+ remains a valid executable short-code benchmark, but it is not a valid
primary dataset for comparing 8/32/128-token interventions. A bounded
`guided_i128_g8` smoke exhausted 64 additional sampled prompts inside one
microstep without finding a completion that reached the first intervention.

Candidate probes were therefore run before defining new held-out splits:

- LiveCodeBench 2025 had adequate length, but only a 3.13-point public-test
  teacher/student gap at 384 tokens and severe teacher truncation.
- BigCodeBench tasks with stdlib-only dependencies and canonical solutions of
  at least 96 Qwen tokens had 84.4% observed student coverage and 100% teacher
  coverage at token 128 on a 32-task probe.
- On the same probe, the frozen teacher passed 43.75% and the base student
  34.38% in the locked unittest container.

## New primary dataset

The primary tutoring task is now **BigCodeBench-Long96-Stdlib**:

- source: `bigcode/bigcodebench`
- dataset revision: `b74c0d0bf70d2c0bc459be537895cca163007f1a`
- source split: `v0.1.4`
- Qwen-tokenized canonical solution length: at least 96
- dependencies: Python standard library only
- canonical solution must pass in the pinned locked evaluator

Of 142 length/library candidates, 121 canonical solutions pass the evaluator.
They are divided into 60 train, 21 dev, and 40 test tasks.

All task IDs inspected in any BigCodeBench model probe are quarantined into
the training split. Remaining assignments are deterministic by task-ID
SHA-256. Tests are never present in trainer/teacher prompt files.

## Metrics

Primary quality becomes student-alone pass@1 on the 40 untouched
BigCodeBench-Long96-Stdlib test tasks, with exact task-paired comparisons.
MBPP+ remains a secondary generalization benchmark and cannot select a
checkpoint. The frozen teacher is evaluated on the same 40-task denominator.

The small denominator is reported with exact paired tests and confidence
intervals; multiple training seeds are required before declaring an interval
superior.

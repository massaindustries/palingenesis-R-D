# Guided Tutoring v2 — preregistered experiment

Status: frozen before generating any v2 model-quality result.

## Question

Does verified, token-level intervention by Qwen3-Coder-Next-FP8 improve the
student's executable-code quality, and how does the intervention interval
affect quality, teacher compute, and autonomous generation?

## Dataset

Primary evaluation uses EvalPlus MBPP+ revision
`b2d74c91837c3f2a20c1299ae98133cbe7cfa077`, restricted to task IDs in the
existing leakage-checked MBPP sanitized test partition. This yields 224
held-out tasks. The 120 train and 43 dev task IDs remain disjoint from this
test set.

- Training feedback may use only the original public MBPP assertions from the
  120-task train split.
- Model selection may use the 43-task dev split.
- MBPP+ augmented tests for the 224-task test split are never exposed to
  training, teacher prompts, checkpoint selection, or hyperparameter sweeps.
- HumanEval+ revision
  `d32357cf319e50e9c8d8dab5ea876c72b0fd321b` is an optional secondary OOD
  evaluation and cannot control model selection.

## Guided intervention

For each prompt, the current student policy generates `interval` autonomous
tokens. The frozen teacher then generates an 8-token guidance group from the
same prompt and mixed trajectory. The teacher tokens are appended to the
trajectory, after which the student resumes. This repeats until a stop token or
the fixed maximum completion length.

Conditions:

- `guided_i8_g8`: 8 autonomous student tokens, then 8 teacher tokens.
- `guided_i32_g8`: 32 autonomous student tokens, then 8 teacher tokens.
- `guided_i128_g8`: 128 autonomous student tokens, then 8 teacher tokens.

The student loss is teacher-forced cross entropy only on teacher-inserted
tokens. Every optimizer update uses four on-policy micro-batches from one
immutable policy version, followed by exact adapter synchronization to the
rollout replica.

## Controls

- Untrained base student.
- Existing fixed-corpus offline teacher SFT.
- Frozen teacher generation on the exact primary test denominator.
- A short `guided_i32_g8_shuffled` causal control: teacher groups are permuted
  across prompts before loss calculation while preserving token counts and
  optimizer settings.
- Student-only rollouts paired with every guided rollout during smoke/probe
  evaluation.

## Teacher-following gates

A guided run is invalid unless all are true:

1. Teacher request success rate is at least 99%; no open circuit breaker.
2. Policy staleness is exactly zero and policy versions are contiguous.
3. At least 95% of teacher groups are non-empty and token provenance is logged.
4. Matched teacher-token NLL decreases from pre-update to post-update in more
   than 60% of optimizer steps and has a negative mean delta.
5. On the causal probe, matched teacher labels outperform shuffled labels in
   teacher-token NLL improvement.
6. Guided and student-only trajectories are graded with the same sandbox and
   public tests; paired outcomes and teacher intervention counts are retained.

## Quality metrics

Primary:

- Student-alone MBPP+ pass@1 on 224 held-out tasks, greedy decoding, one sample
  per task.

Secondary:

- Original MBPP public-test pass@1 on the same 224 tasks.
- Individual-test pass rate.
- Syntax/runtime success.
- Completion-token mean, median, p90, and maximum-length rate.
- Teacher-reference pass@1 on the same denominator.
- Guided-trajectory versus paired student-only pass rate on train/dev prompts.
- Teacher-token NLL before and after each update.
- Teacher-token top-1 agreement before and after each update.
- Teacher tokens, requests, wall-clock, GPU-hours, student tokens, and total
  estimated GPU-hours.

Uncertainty:

- Wilson intervals for individual condition pass rates.
- Task-paired bootstrap intervals and exact McNemar tests for condition
  differences.
- Seed-level mean, standard deviation, and range once replicated.

## Selection and stopping

- Engineering smoke: 5 updates per interval.
- Causal probe: 20 matched and 20 shuffled interval-32 updates.
- Scientific pilot: equal optimizer-step and student-rollout budgets for
  intervals 8/32/128. The step count is chosen once from smoke throughput
  before inspecting held-out quality and is then frozen.
- Checkpoint selection uses dev executable quality, with teacher-token NLL as a
  diagnostic only. The final checkpoint is not automatically preferred.
- After seed 0, all three conditions are replicated with the same additional
  seeds if resources permit; otherwise the result remains explicitly
  exploratory and no interval is declared superior.

## Kill switches

- Any train/dev/test task-ID or prompt-hash overlap.
- Any MBPP+ augmented test appearing in a training or teacher prompt.
- Non-finite loss/gradient, stale rollout, missing teacher provenance, or
  adapter hash mismatch.
- More than 5% teacher request failure.
- More than 30% relative held-out quality collapse versus base.
- Maximum-length rate more than 10 percentage points above base without a
  compensating statistically credible quality gain.

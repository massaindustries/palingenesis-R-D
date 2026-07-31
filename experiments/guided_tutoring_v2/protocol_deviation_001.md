# Protocol deviation 001 — no-intervention microbatches at interval 128

Recorded: 2026-07-31, before any Guided Tutoring v2 held-out quality result.

## Observation

The engineering smoke for `guided_i128_g8` stopped before its first optimizer
update because two consecutive four-prompt microbatches produced only
completions shorter than 128 tokens. No teacher group existed and taking a
zero-gradient update would have falsely counted as tutoring.

## Amendment

Each of the four microbatches in an optimizer step may draw up to 15 additional
four-prompt batches until at least one trajectory reaches the configured
teacher intervention. This bound is identical in all guided configs.

Every rejected rollout and every student token generated before termination is
included in the run metrics:

- `rejected_no_teacher_rollouts`
- `rejected_no_teacher_student_tokens`
- `empty_microstep_retries`
- `teacher_intervention_coverage`

Only trajectories with at least one teacher group contribute to the guided CE
loss. Scientific budget comparisons must include rejected rollout compute and
must report coverage; interval 128 cannot be described as equivalent-cost if
it needs more attempts.

## Rationale

This preserves the meaning of an interval: the teacher does not intervene
before token 128 and no artificial minimum generation length is imposed. It
also prevents silent selection-cost omission. No MBPP+ held-out result was
generated or inspected before making this engineering amendment.

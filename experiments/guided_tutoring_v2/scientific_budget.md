# Frozen scientific pilot budget

Frozen: 2026-07-31, after engineering/causal probes and before opening the
BigCodeBench-Long96-Stdlib dev or test quality results.

## Conditions

- `guided_i8_g8`
- `guided_i32_g8`
- `guided_i128_g8`

Each update draws 4 prompts in each of 4 microbatches: 16 attempted on-policy
student rollouts. The completion cap is 512 mixed-trajectory tokens. Teacher
groups are always 8 tokens; only the autonomous student interval changes.

## Optimization

- 20 optimizer updates per run
- LoRA rank 32, alpha 64
- constant learning rate `2e-5`, two-step linear warmup
- BF16, gradient norm clip 1.0
- exact rollout-adapter synchronization after every update
- checkpoint at steps 5, 10, 15, and final step 20

Paired student-only trajectories are disabled in scientific training because
the engineering smokes already graded 227 pairs and the autonomous checkpoint
evaluation is the scientific outcome.

## Replication and selection

- Seeds: 0, 1, 2 for all three intervals.
- Dev checkpoint selection: highest executable pass@1 on the 21-task dev set;
  ties prefer the earlier checkpoint, then lower maximum-length rate.
- Final reporting: selected checkpoint evaluated once on the untouched
  40-task test set.
- Base student, frozen teacher, and existing offline-SFT student are evaluated
  on the same 40 tasks.

No interval may be declared superior from seed 0 alone. Test results do not
change the step count, learning rate, checkpoint grid, seed count, or dataset.

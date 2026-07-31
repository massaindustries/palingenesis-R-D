# Protocol deviation 003 — causal probe compute

Recorded: 2026-07-31, before any Guided Tutoring v2 held-out quality result.

The three engineering smokes already graded every guided trajectory against a
paired student-only trajectory: 80 interval-8 pairs, 80 interval-32 pairs, and
67 interval-128 pairs.

The 20-step matched-versus-shuffled causal probe therefore omits the additional
paired rollout and uses one four-prompt microbatch per update. Both arms use
identical prompt, optimizer, teacher, and rollout budgets; only
`shuffle_teacher_groups` differs. This reduces duplicated execution while
preserving the causal teacher-token NLL test:

- matched arm optimizes the actual teacher groups;
- shuffled arm optimizes cross-prompt teacher groups;
- both arms measure pre/post NLL on the actual matched teacher tokens.

This probe is diagnostic only. It cannot select a final interval or supply a
held-out quality result.

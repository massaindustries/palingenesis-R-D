# Protocol deviation 004: zero-based checkpoint cadence

Detected: 2026-07-31, after opening the dev baselines but before producing any
scientific-run checkpoint or inspecting any trained checkpoint quality.

The OPD loop indexes optimizer updates from zero. Its inherited checkpoint
condition used the raw loop index, so a requested `save_steps: 5` would have
saved `step_5` after six completed optimizer updates. The first scientific run
was stopped during training, before its first checkpoint, and is excluded.

The cadence now uses `completed_updates = step + 1` and saves intermediate
checkpoints after exactly 5, 10, and 15 completed updates. The final checkpoint
is still saved after exactly 20 completed updates. No model-quality result
from the partial run was evaluated or used to choose this correction.

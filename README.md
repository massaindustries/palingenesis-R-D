# palingenesis

<p align="center">
  <img src="./assets/pgs.png" alt="palingenesis" width="100%">
</p>

**Fire-and-forget LLM fine-tuning with state-of-the-art defaults.**

Papers distilled into one command. Every optimization applied automatically.

```bash
git clone https://github.com/your-org/palingenesis.git && cd palingenesis
uv pip install -e ".[train]"
./run.sh configs/quickstart.yaml
```

Short CLI alias: `pgs`

```bash
pgs train --config configs/quickstart.yaml
pgs autopilot --model Qwen/Qwen3.5-4B --dataset your_data.jsonl
```

---

## What you get

- **4B full fine-tune in 15 GB**: no LoRA compromise needed (RTX 4090 compatible)
- **DEFT loss**: parameter-free token weighting; the original paper (arxiv:2602.11424) reports math-reasoning gains, not independently reproduced
- **Hyperball optimizer**: 20-30% convergence speedup via norm constraints (arxiv:2606.16899, single paper)
- **Power-decay scheduler**: theoretically optimal (better than cosine at all scales)
- **Best-model tracking**: automatically saves the checkpoint with lowest eval loss
- **Auto-resume**: crash and re-run, picks up from last valid checkpoint
- **Multi-node SLURM**: sharded DCP checkpoints, zero extra memory

## Hardware

| GPU | Model | Config |
|-----|-------|--------|
| RTX 4090 (24 GB) | Qwen3.5-4B full ft | `configs/qwen35_4b/a100_40gb.yaml` |
| A100-80GB | Qwen3.5-4B, batch=4 | `configs/qwen35_4b/a100_80gb.yaml` |
| 8× A100 | Qwen3.5-35B MoE | `configs/qwen35_35b_moe/a100_80gb_multigpu.yaml` |
| H100 | Qwen3.5-4B, FP8 | `configs/qwen35_4b/h100_80gb.yaml` |

## Data preparation (one config, closed loop)

Score, filter, and select your data with the *same* config you train with — the scoring model is `model.name_or_path`, the raw data is `data.dataset`, and the `preprocess:` section controls selection. Output is parquet plus a provenance manifest:

```bash
pgs prepare --config configs/qwen35_4b/a100_80gb.yaml            # score → filter → parquet
pgs train   --config configs/qwen35_4b/a100_80gb.yaml \
            --preprocess.enabled true                            # trains on the prepared data
```

Strategies: `optimal` (research-backed J-shaped difficulty mix), `curriculum` (easy→hard, ordering preserved during training), `balanced`, `flow`, and more.

## Training dynamics you can actually see

wandb + trackio, wired for real investigation: loss/ppl, grad norm, spike/clip counters, gradient noise scale, output entropy, and the generalization gap (`eval/gap`) — all on a single `train/global_step` axis. Crash-resume continues the *same* wandb run (run id persisted next to your checkpoints), and a tracker outage can never kill training.

## Agentic data support

Native support for reasoning traces with `reasoning_content`, `tool_calls`, and tool responses. ShareGPT, Alpaca, and OpenAI formats auto-normalized. Tool-call validation against declared schemas.

```yaml
data:
  include_observations: true  # ECHO: train on tool outputs (world model)
  turn_scaling: progressive   # Later turns weighted more
```

## Autopilot

Zero-config mode. Profiles your GPU, sweeps LR, trains to completion:

```bash
pgs autopilot --model Qwen/Qwen3.5-4B --dataset your_data.jsonl
```

## On-policy distillation

Shrink a teacher into a student by scoring the student's **own samples** — full-distribution reverse KL, no train/inference mismatch. Works across mismatched chat templates (e.g. ChatML student ← Llama-3-template teacher) as long as the pair shares a base vocabulary; the token bridge maps end-of-turn tokens so the teacher also supervises *when to stop*.

```bash
pgs distill-score --config configs/distill_opd.yaml --out data/prompts_scored.jsonl  # annotate pool with teacher answers
pgs distill       --config configs/distill_opd.yaml
```

`distill-score` marks every pool row with the teacher's own answer so you can filter before training — pure KL faithfully distills the teacher's *errors* too, making its accuracy a hard ceiling. Works on multiple-choice pools (`data.format: mcqa`) and generic chat prompts (`data.format: messages`, see `configs/distill_chat.yaml`); custom tasks implement the three-method `PromptSource` protocol. See [docs/on_policy_distillation.md](docs/on_policy_distillation.md).

## Multi-GPU / Multi-Node

```bash
./scripts/train_multi_gpu.sh configs/qwen35_4b/a100_80gb_multigpu.yaml
sbatch scripts/train_slurm.sh configs/qwen35_35b_moe/a100_80gb_multigpu.yaml
```

## Documentation

```bash
pip install mkdocs-material
mkdocs serve  # → http://localhost:8000
```

## Tests

```bash
pytest tests/
```

---

*A new form emerging from what came before.*

## What I Did

I added and validated **Guided Tutoring v2**, an on-policy tutoring mode in
which the student writes an autonomous span and the frozen teacher then inserts
a real eight-token continuation into the same trajectory. I tested autonomous
student intervals of **8, 32, and 128 tokens**, always with an eight-token
teacher group. The full protocol, deviations, scripts, and limitations are in
[`experiments/guided_tutoring_v2`](experiments/guided_tutoring_v2/README.md),
with the final scientific results in the
[`experimental report`](experiments/guided_tutoring_v2/report.md).

### Additions

- Added real teacher-token generation to the SGLang backend; tutoring no longer
  treats a sparse next-token top-K distribution as a program correction.
- Added mixed student/teacher trajectory construction for intervals 8, 32,
  and 128, with loss applied only to the inserted teacher tokens.
- Added exact teacher-to-student token mapping, shared-vocabulary validation,
  stop-token handling, and rejection of incompatible tokenizer pairs.
- Added per-group provenance: prompt hash, insertion position, token IDs, and
  SHA-256 digest are retained for every teacher intervention.
- Added bounded retry and explicit compute accounting for interval-128
  completions that terminate before the first teacher intervention.
- Added teacher request success, intervention coverage, group nonempty rate,
  pre/post teacher-token NLL, policy staleness, rejected rollout, gradient,
  token, wall-clock, and allocated GPU-hour telemetry.
- Fixed checkpoint cadence so `step_5`, `step_10`, and `step_15` mean exactly
  5, 10, and 15 completed optimizer updates rather than zero-based loop indices.
- Added locked executable graders, deterministic intervention probes,
  matched-vs-shuffled causal controls, dev checkpoint selection, kill switches,
  task-paired bootstrap intervals, and exact McNemar analysis.
- Added reproducible data preparation and validation for the pinned
  BigCodeBench subset, plus secondary MBPP+ and LiveCodeBench probe tooling.

### Dataset and evaluation controls

MBPP+ was rejected as the primary interval benchmark because most completions
ended before token 128. The selected benchmark is
**BigCodeBench-Long96-Stdlib**, built from pinned BigCodeBench revision
`b74c0d0bf70d2c0bc459be537895cca163007f1a`:

- 142 tasks passed the length and standard-library screen;
- 121 canonical solutions passed the locked evaluator;
- 60 train, 21 dev, and 40 untouched test tasks;
- zero task-ID or prompt-hash overlap between splits;
- all previously inspected tasks quarantined into train;
- hidden tests excluded from trainer and teacher prompt files.

Code was graded in a pinned Python 3.11.9 container with no network, read-only
root filesystem, non-root execution, dropped capabilities, and a ten-second
per-task timeout. Checkpoints were selected only on dev; the test split was
opened once after all nine selections were frozen.

### Tests and benchmarks

The original sparse-OPD logs showed why the earlier gain was small: teacher
and student already agreed on the top-1 token at **92.17%** of anchors, leaving
only **0.409 disagreements per rollout**. The old method contacted the teacher,
but did not ask it to generate a correction.

Three paired engineering smokes verified that real teacher groups were inserted
and executable:

| Student interval | Paired trajectories | Teacher groups | Teacher tokens | Guided pass@1 | Student-only pass@1 | Fixes | Regressions |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 80 | 1,231 | 9,721 | 53.75% | 47.50% | 11 | 6 |
| 32 | 80 | 417 | 3,296 | 47.50% | 38.75% | 13 | 6 |
| 128 | 67 | 82 | 640 | 35.82% | 35.82% | 4 | 4 |

A 20-trial exact-trajectory causal control compared matched teacher groups with
cross-prompt shuffled groups. Matched training improved true teacher-token NLL
more in **20/20 trials**: mean matched delta `-0.04180`, shuffled true-target
delta `-0.02603`, advantage `0.01577`, paired bootstrap 95% CI
`[0.01121, 0.02071]`, diagnostic sign-test `p=1.91e-6`. This demonstrates
learning of the teacher's actual targets rather than a generic effect from
adding arbitrary labels or gradient magnitude.

A deterministic greedy, train-only functional probe used the same student
decoding with and without teacher insertions:

| Student interval | Student-only | Guided | Fixes | Regressions | Teacher tokens |
|---:|---:|---:|---:|---:|---:|
| 8 | 9/20 | 9/20 | 2 | 2 | 2,501 |
| 32 | 9/20 | **11/20** | 2 | 0 | 836 |
| 128 | 9/20 | 8/20 | 0 | 1 | 187 |

The frozen scientific grid contained three seeds per interval, 20 optimizer
updates per run, and checkpoints after updates 5, 10, 15, and 20. All **9/9
runs** passed telemetry validation, all **180/180 updates** reduced NLL on the
inserted teacher tokens, teacher request success was **100%**, group nonempty
rate was **100%**, circuit breakers never opened, and policy staleness was zero.

| Student interval | Seeds | Teacher tokens | Teacher groups | Mean intervention coverage | Mean NLL delta | Allocated GPU-hours |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3 | 135,873 | 17,167 | 100.00% | -0.02874 | 7.21 |
| 32 | 3 | 49,031 | 6,216 | 100.00% | -0.03708 | 6.55 |
| 128 | 3 | 9,524 | 1,214 | 93.13% | -0.04435 | 5.77 |

All 36 scientific checkpoints were graded on dev. The selected pass counts
were `9/9/9` for interval 8, `7/8/8` for interval 32, and `8/8/9` for interval
128. One unselected interval-32 final checkpoint tripped the preregistered
maximum-length kill switch; no selected checkpoint tripped a kill switch.

Final autonomous results on the untouched 40-task test split:

| Condition | Seed pass counts | Mean pass@1 | Delta vs base |
|---|---:|---:|---:|
| Base Qwen3-4B | 11/40 | 27.50% | — |
| Existing offline SFT | 11/40 | 27.50% | 0.00 pp |
| Frozen Qwen3-Coder-Next-FP8 teacher | 14/40 | 35.00% | +7.50 pp |
| Guided interval 8 | 10, 11, 11 | 26.67% | -0.83 pp |
| Guided interval 32 | **13, 13, 11** | **30.83%** | **+3.33 pp** |
| Guided interval 128 | 11, 9, 13 | 27.50% | 0.00 pp |

#### Why interval 32 performed best

For interval `k`, let `F_k` be the number of base failures fixed and `R_k` the
number of base successes regressed across the three seeds and 40 test tasks:

```text
delta_k = (F_k - R_k) / (3 * 40).
```

| Interval | Fixes / 120 | Regressions / 120 | Net | Delta vs base | Observed teacher share | Groups / guided rollout | Mean completion tokens | Max-length / 120 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 6 | 7 | -1 | -0.83 pp | 49.35% | 17.42 | 288.29 | 7 |
| 32 | 5 | 1 | +4 | **+3.33 pp** | 18.46% | 5.48 | 214.85 | 0 |
| 128 | 3 | 3 | 0 | 0.00 pp | 4.32% | 1.35 | 252.40 | 3 |

Interval 32 therefore won mainly by avoiding collateral regressions, not by
creating the most fixes: interval 8 made one additional fix but six additional
regressions. Its observed advantage over interval 8 was
`(4 - (-1)) / 120 = 4.17 pp`; its advantage over interval 128 was
`(4 - 0) / 120 = 3.33 pp`.

With eight teacher tokens per intervention, the ideal teacher share is
`8 / (k + 8)`: 50.00%, 20.00%, and 5.88% for intervals 8, 32, and 128. The
logs closely follow that geometry. Interval 8 lets the teacher occupy roughly
half the trajectory and creates about 17 continuation seams per rollout; its
selected models are longer and have seven truncations. Interval 128 provides
only about 1.35 groups per rollout and 92.22% intervention coverage. Interval
32 supplies about five or six groups with 100% coverage while leaving roughly
four fifths of the trajectory autonomous.

The deterministic train-only probe supports the same mechanism without dev
checkpoint selection: interval 32 moves 9/20 student-only passes to 11/20 with
two fixes and no regressions; interval 8 has two fixes and two regressions, and
interval 128 has no fixes and one regression. The selected interval-32 targets
also have the largest weighted teacher-token NLL reduction (`-0.06145`), though
cross-interval NLLs score different token positions and are not directly
exchangeable.

This remains a **best observed point, not a proven optimum**. The task-cluster
bootstrap 95% interval for interval 32 versus base is `[-1.67, +10.00] pp`, and
versus interval 8 it is `[-4.17, +11.67] pp`; both include zero. All interval-32
checkpoints were selected at step 5, while the interval-8 and interval-128
selections average 16.67 and 18.33 updates. At the matched step-5 dev checkpoint
interval 32 is still best (36.51% versus 31.75% and 30.16%), but the untouched
test did not evaluate every matched checkpoint. The positive interval-32 events
are concentrated on two tasks, and seeds 0 and 1 have identical pass/fail
vectors.

#### Compute economics and potential savings

Allocated GPU-hours mean four reserved GPUs—two teacher, one training, and one
rollout GPU—multiplied by run wall-clock hours. They measure reserved capacity,
not utilization or electricity. The complete 20-update scientific grid actually
consumed:

| Interval | Three-seed allocated GPU-hours | Teacher tokens | Teacher requests | Test delta vs base |
|---:|---:|---:|---:|---:|
| 8 | 7.2144 | 135,873 | 6,087 | -0.83 pp |
| 32 | 6.5503 | 49,031 | 2,199 | **+3.33 pp** |
| 128 | 5.7722 | 9,524 | 468 | 0.00 pp |

On the actually executed full grid, interval 32 dominates interval 8 in this
pilot: it used **0.6641 fewer allocated GPU-hours (9.20%)** and **86,842 fewer
teacher tokens (63.91%)**, while improving rather than reducing pass@1.
Interval 128 is cheaper than interval 32 by 0.7782 allocated GPU-hours and uses
far fewer teacher tokens, but it produced no mean quality gain; this is a
quality/cost tradeoff rather than a savings claim.

The checkpoint choices show a larger *potential* saving for a future validated
early-stop recipe:

| Interval | Selected steps by seed | Allocated GPU-hours through selected checkpoints | Teacher tokens through selected checkpoints |
|---:|---:|---:|---:|
| 8 | 15, 20, 15 | 5.8974 | 110,577 |
| 32 | 5, 5, 5 | **1.4625** | 10,357 |
| 128 | 15, 20, 20 | 5.2213 | 8,609 |

Stopping all interval-32 runs at step 5 would reduce their allocation from
6.5503 to 1.4625 GPU-hours (**77.67%**) and teacher tokens from 49,031 to 10,357
(**78.88%**). Relative to the selected interval-8 checkpoints, it would use
**75.20% fewer GPU-hours** and **90.63% fewer teacher tokens**. These are
retrospective savings, not savings realized by this experiment: the later
checkpoints still had to be trained and evaluated to discover that step 5 won.
A follow-up must preregister or independently validate the step-5 stopping rule
before treating these savings as operational.

If `p_GPU` is the blended price per reserved GPU-hour, the measured full-grid
cost is `6.5503 * p_GPU` for interval 32 and the saving versus interval 8 is
`0.6641 * p_GPU`. A validated step-5 recipe would cost `1.4625 * p_GPU` for
three seeds, or approximately `0.4875 * p_GPU` per seed, saving
`5.0879 * p_GPU` versus running all three interval-32 seeds to step 20. No
currency total or energy estimate is asserted because neither an hourly price
nor GPU energy counters were recorded. After training, evaluation and
deployment are student-only, so the final checkpoint incurs no online teacher
inference cost.

Finally, the complete repository test suite passed: **300 tests passed**. The
new diagnostic and its tests pass Ruff; whitespace and secret scans are clean.

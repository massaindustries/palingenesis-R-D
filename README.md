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

Interval 32 was the best observed tradeoff, outperforming base/offline by 3.33
percentage points on average. Across the 120 seed-task outcomes it made five
fixes and one regression, so `(5 - 1) / 120 = +3.33 pp`. Interval 8 made six
fixes but seven regressions, and interval 128 made three of each: the advantage
at interval 32 came mainly from avoiding collateral regressions, not from
creating the largest number of fixes. Two seeds fixed two base failures without
a regression, while the third had one fix and one regression. The individual
paired McNemar value for each +5-point seed was `p=0.5`, and the paired
bootstrap intervals include zero, so this pilot is **evidence of real teacher
following but not yet a statistically conclusive autonomous quality gain**.
Interval 8 over-intervened; interval 128 under-intervened.

Finally, the complete repository test suite passed: **299 tests passed**. The
new diagnostic and its tests pass Ruff; whitespace and secret scans are clean.

# Sparse On-Policy Distillation on 4×L40S

Generated: 2026-07-31T00:06:58.927029+00:00

## Technical summary

At the completed single-seed scope, **Interval 32** has the highest observed 257-task pass@1 (**61.09%**, Wilson 95% CI 55.01%–66.85%). Paired checkpoint comparisons, not training KL, determine whether this is a real gain.

## 1. Hardware and software

- 4× NVIDIA L40S 46,068 MiB: GPU 0–1 teacher, GPU 2 trainer, GPU 3 rollout.
- Ubuntu 22.04.5, driver 570.211.01, CUDA toolkit 12.8.
- Trainer: PyTorch 2.8.0+cu128. Teacher: SGLang 0.5.8 and PyTorch 2.9.1+cu128.
- All distinct GPU pairs are PHB and CUDA peer access is unavailable.

## 2. Implemented architecture

Qwen3-Coder-Next-FP8 runs tensor-parallel across GPUs 0–1. Qwen3-4B uses BF16 LoRA (r=32, α=64) on GPU 2, while a separate BF16 no-grad rollout replica runs on GPU 3. Four rollout micro-batches share one immutable policy version before one optimizer update. CPU-staged adapter synchronization is hash-verified after every update.

## 3. Full KL versus sparse anchor RKL

The original full KL materializes both full-vocabulary distributions at every completion position. Sparse anchor RKL scores selected positions only and preserves explicit probability on teacher top-K ∪ student top-K ∪ stop IDs; all other vocabulary mass forms one residual category. Interval 1 is dense in positions but remains a coarse top-K loss.

## 4. Tokenizer compatibility

Compatibility passed vocabulary, special-token, deterministic-ID, multilingual, Unicode, and code probes. The shared base vocabulary has 151,669 IDs and stop ID 151,645.

## 5. Teacher service and benchmark

The 60-case batch/prompt/top-K matrix completed, followed by 100/100 deterministic stable requests with zero failures. Peak teacher VRAM was 44,175 MiB/GPU.
On prompt-v2, fixed greedy teacher generations score 71.21% pass@1 and 74.48% individual-test pass rate over 257 tasks, with 0 sandbox timeouts.

## 6. Dataset, sandbox, and prompt audit

Prompt-v2 preserves MBPP sanitized partitions (120 train, 43 dev, 257 test), exposes the exact callable signature, and withholds implementation and tests. Independent checks recompute file/prompt hashes, reject ID/prompt/reference overlap, and reject assertion leakage. The original prompt-v1 artifacts are archived because hidden function names confounded quality (teacher entry-point compliance rose from 11.3% to 96.5% after correction).

Generated code runs only in a network-disabled, read-only Docker sandbox with dropped capabilities, no-new-privileges, resource/time limits, and per-assert accounting.

## 7. Learning-rate selection

The predeclared 20-step interval-32 stability rule selected `1.0e-04`. Every candidate was finite and zero-staleness; an independent 32-task paired quality gate then verified that the selected LR did not trigger the >30% collapse kill switch.

## 8. Test and smoke evidence

The complete harness suite passed 277 tests. A real 10-step gate verified finite forward/backward/update, zero staleness, stable memory, checkpointing, and exact next-step resume. Four 20-step conditions completed without OOM or zombie processes.

| Prompt-v2 smoke condition | Pass@1 (32 tasks) | Wilson 95% CI | Test pass | Anchors |
|---|---:|---:|---:|---:|
| Base student | 68.75% | 51.43%–82.05% | 71.13% | 0 |
| Interval 1 | 68.75% | 51.43%–82.05% | 69.07% | 4,844 |
| Interval 32 | 65.62% | 48.31%–79.59% | 68.04% | 151 |
| Interval 128 | 65.62% | 48.31%–79.59% | 68.04% | 80 |
| Final only | 62.50% | 45.25%–77.07% | 63.92% | 80 |

## 9. Pilot and full-denominator quality

| Condition | Steps | Pass@1 | Wilson 95% CI | Test pass | Syntax | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| Base student | 0 | 60.31% | 54.22%–66.10% | 66.49% | 100.00% | 99.61% |
| Offline teacher SFT | 300 | 60.70% | 54.61%–66.47% | 65.85% | 98.05% | 98.05% |
| Interval 32 | 300 | 61.09% | 55.01%–66.85% | 64.82% | 93.39% | 93.00% |

## 10. Teacher compute, wall-clock, and GPU-hours

| Condition | Anchors | Teacher requests | Teacher GPU-h | Run wall s | Total GPU-h estimate |
|---|---:|---:|---:|---:|---:|
| Base student | 0 | 0 | 0.0000 | 0.0 | 0.0000 |
| Offline teacher SFT | 0 | 0 | 0.0000 | 1514.3 | 0.8413 |
| Interval 1 | 4,844 | 80 | 0.0518 | 464.8 | 0.3100 |
| Interval 32 | 24,123 | 1,200 | 0.5874 | 23291.3 | 13.5271 |
| Interval 128 | 80 | 80 | 0.0148 | 364.8 | 0.2174 |
| Final only | 80 | 80 | 0.0123 | 329.0 | 0.1950 |

Teacher GPU-hours are two teacher GPUs multiplied by measured client-side request time. NVML does not expose total-energy counters on this host, so no energy estimate is reported.

## 11. Pareto quality/cost

The machine-readable full-denominator frontier is `results/pareto_quality_cost.csv`. Rows with a 32-task denominator are excluded from that frontier.

## 12. Training diagnostics and profiling

Per-step artifacts include KL, both residual masses, gradient norm, LR, entropy, completion length, teacher/student agreement, policy version, sync latency, rollout/top-K/teacher/forward/backward time, VRAM, anchors, scored tokens, and requests. The HTML report renders the required seven quality/cost/diagnostic charts.

## 13. Autonomy and diversity

Evaluation summaries report normalized AST uniqueness, exact duplicate rate, solution length, and prompt-matched normalized token edit distance to the fixed teacher corpus. These are descriptive; executable tests remain the primary quality outcome.

## 14. Anomalies and limitations

- The initial scientific comparison uses seed 0 only.
- The 32-task interval estimates have wide uncertainty and are not mixed with 257-task results.
- Sparse RKL is a coarse distribution, not full-vocabulary KL.
- Client teacher timing includes queueing and retry overhead.
- Tiny negative residuals from floating-point rounding are interpreted as zero.
- Prompt-v1 results are retained only as a diagnosed measurement failure.
- CPU-staged adapter transport is specific to this PHB/no-P2P topology.

## 15. Recommendation and next experiment

Do not claim a quality or quality/compute improvement from this single seed. Use **Interval 32** only as the confirmatory sparse candidate because it has the highest point estimate and did not trigger the quality-collapse gate; offline SFT remains far cheaper. Next, repeat the sparse candidate, dense interval-1, and offline SFT with seeds 0/1/2; then test top-K 64 vs 128 and anchor window 1 vs 4.

## 16. Reproducibility

- Initial harness SHA: `097df4a85a828b2df95d7ab230a7803f121db00e`.
- Sparse runtime commit: `fccdb9a099e0f1c865ca796edb0dd07afef4a1b7`.
- Pilot harness commit: `7d6f8acef28b24d6f0380a914ff9af5d627860be`.
- Teacher revision: `da6e2ed27304dd39abadd9c82ef50e8de67bdd4c`.
- Student revision: `1cfa9a7208912126459214e8b04321603b3df60c`.
- Every official run contains resolved config, dataset/model revisions, source tarball and checksums, pip freezes, GPU topology/snapshots, timestamps, logs, metrics, and checkpoints.

Resume an interrupted checkpoint:

```bash
SPARSE_OPD_ROOT=/opt/sparse-opd /opt/sparse-opd/ops/resume_run.sh <config> <checkpoint>
```

Rerun the selected pilot:

```bash
SPARSE_OPD_ROOT=/opt/sparse-opd PILOT_STEPS=300 /opt/sparse-opd/ops/run_pilot.sh --condition interval_32
```

## 17. Further questions

- Does the apparent frequency optimum persist at equal teacher-anchor budget?
- Is any quality change mediated by completion length rather than anchor placement?
- Can NCCL adapter broadcast reduce sync time without introducing staleness?

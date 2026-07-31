# Sparse OPD status

Last updated: 2026-07-30

## Preflight

- [x] OS, kernel, RAM, disk, `/dev/shm`, GPU inventory recorded
- [x] Four NVIDIA L40S visible and idle
- [x] GPU topology inspected; all pairs are PHB
- [x] GPU map written (`teacher=0,1`, `train=2`, `rollout=3`)
- [x] Required host packages and uv installed
- [x] Minimal CUDA test on every GPU
- [x] CUDA peer-to-peer runtime matrix (all distinct pairs report disabled)

## Harness and environments

- [x] Palingenesis `odp` checked out; experiment branch at `7d6f8ac`
- [x] Original test baseline captured (260 pass, 1 upstream FSDP failure)
- [x] Trainer environment installed
- [x] Teacher SGLang 0.5.8 environment installed and CUDA import tested

## Models and compatibility

- [x] Teacher FP8 checkpoint downloaded and verified
- [x] Student BF16 checkpoint downloaded and verified
- [x] Tokenizer compatibility gate passed

## Implementation and execution

- [x] Sparse anchor reverse-KL implementation and unit tests
- [x] Dedicated rollout worker and adapter synchronization
- [x] Dataset and sandbox
- [x] Smoke gates and four 20-step smoke runs
- [x] Prompt-v1 quality confound diagnosed and archived
- [x] Prompt-v2 dataset integrity/leakage validation
- [x] Corrected base and teacher evaluation on all 257 test tasks
- [x] Offline-SFT real one-step/checkpoint probe
- [x] Three-point LR mini-sweep and independent quality-collapse gate
- [x] Pilot (interval-32, 300 steps)
- [x] Offline-SFT baseline (300 steps)
- [x] Metrics, paired comparisons, plots, and final report

Current validated results:

- Teacher benchmark: 60/60 matrix cases, 100/100 stable requests, 0 failures,
  peak 44,175 MiB/GPU.
- Full harness suite: 277 passed, 5 non-fatal warnings.
- Real 10-step interval-32 gate: 10/10 steps, staleness 0, finite metrics,
  successful checkpoint and exact next-step resume.
- Four-condition smoke: 4 × 20 optimizer steps complete, zero staleness,
  finite metrics, no OOM, and all invariant checks passed.
- Prompt-v1 revealed an entry-point confound (teacher defined the tested name in
  only 11.3% of tasks); the original split/corpora are preserved under
  `data/archive/prompt_v1_20260730T152800Z/`.
- Prompt-v2 exposes only the callable signature and withholds tests/solution;
  dataset validation found zero issues across 120 train, 43 dev, 257 test.
- Prompt-v2 base: pass@1 60.31%; teacher: pass@1 71.21%; both evaluated with the
  same locked Docker sandbox, with zero timeouts.
- Offline-SFT probe: one optimizer update, finite loss 0.9048, nonzero gradient
  norm 3.7031, and final adapter/trainer state saved.
- LR mini-sweep selected `1e-4` by a predeclared stability score; prompt-v2
  quality gate was 21/32 versus base 22/32 (no >30% collapse).
- Full-scale pilot probe: 26.25 s/update, 715 generated tokens, 30 anchors,
  staleness 0, peak trainer/rollout VRAM 13,349/8,016 MiB.
- Official 300-step interval-32 pilot: pass@1 61.09% (157/257, Wilson 95% CI
  55.01%–66.85%), test-pass 64.82%, zero timeouts, staleness 0, and 13.53
  estimated total GPU-hours.
- Official 300-step offline-SFT baseline: pass@1 60.70% (156/257, Wilson 95%
  CI 54.61%–66.47%), test-pass 65.85%, zero timeouts, and 0.84 estimated total
  GPU-hours.
- Base student: pass@1 60.31% (155/257). The sparse pilot's +0.78 percentage
  point change is not statistically resolved (paired bootstrap 95% CI −3.50
  to +5.06 points; McNemar p=0.864).
- Combined validation of base, sparse pilot, and offline-SFT: Ready to share,
  zero issues.
- Portable report: `reports/final_report.html`; validation and packaging
  passed, with structural-only rendering verification because Chromium is not
  installed.

Open issue: publication of the completed experiment branch remains in
progress.

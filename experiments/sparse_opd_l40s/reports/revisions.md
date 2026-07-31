# Revisions

## Harness

- Upstream: `https://github.com/mii-llm/palingenesis.git`
- Upstream branch: `odp`
- Initial SHA: `097df4a85a828b2df95d7ab230a7803f121db00e`
- Local branch: `experiment/sparse-opd-l40s`

Original test baseline on the unmodified SHA:

- 260 passed
- 1 failed: `tests/test_distributed.py::test_chunked_loss_fsdp_consistency`
- Failure: mixed local `torch.Tensor` and `DTensor` passed to embedding under
  FSDP2 with the repository-pinned PyTorch 2.8.0.
- Log: `logs/original_tests_097df4a.log`

This was fixed without weakening or skipping the test by entering through the
root FSDP wrapper before the backbone-only path.

Functional commits:

| SHA | Phase |
|---|---|
| `d22a58eed6c75211a804ae29590ca5a59fe4747b` | Root-FSDP consistency test fix |
| `fccdb9a099e0f1c865ca796edb0dd07afef4a1b7` | Sparse anchor RKL, SGLang backend, rollout replica, LoRA sync/resume, metrics and tests |
| `7d6f8acef28b24d6f0380a914ff9af5d627860be` | Experiment provenance and final-evaluation metrics |

## Models

| Role | Repository | Revision | Size reported by Hub |
|---|---|---|---:|
| Teacher | `Qwen/Qwen3-Coder-Next-FP8` | `da6e2ed27304dd39abadd9c82ef50e8de67bdd4c` | 74.89 GiB |
| Student | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | 7.51 GiB |

## Runtime

- Trainer: PyTorch 2.8.0+cu128 (repository pin)
- Teacher: SGLang 0.5.8, PyTorch 2.9.1+cu128
- CUDA runtime: 12.8
- CUDA toolkit/compiler: 12.8, nvcc V12.8.93

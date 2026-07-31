# Machine inventory

Captured: 2026-07-30 13:22 UTC

## Operating system

- Ubuntu 22.04.5 LTS (Jammy)
- Kernel: `5.15.0-139-generic`
- Architecture: `x86_64`

## Compute

- 4 × NVIDIA L40S
- Reported framebuffer per GPU: 46,068 MiB
- NVIDIA driver: 570.211.01
- Driver-supported CUDA: 12.8
- GPU processes at preflight: none
- Compute mode: Default
- Persistence mode: On
- MIG: not applicable

PCI bus IDs:

| Index | PCI bus ID |
|---:|---|
| 0 | `00000000:05:00.0` |
| 1 | `00000000:06:00.0` |
| 2 | `00000000:07:00.0` |
| 3 | `00000000:08:00.0` |

`nvidia-smi topo -m` reports `PHB` for every distinct pair, with all GPUs
on NUMA node 0 and CPU affinity 0-31. Because no pair is topologically
preferable, GPUs 0 and 1 were selected deterministically for tensor-parallel
teacher inference. GPU 2 is training and GPU 3 is rollout.

## Host resources

- RAM: 125 GiB total, 123 GiB available at preflight
- Swap: none
- Root filesystem: 1.9 TiB total, 1.9 TiB available at preflight
- `/dev/shm`: 63 GiB
- Experiment root: `/opt/sparse-opd` (symlink to `/root/finetuneMII`)

## Installed baseline tools

- Docker 29.6.2
- Docker Compose v5.3.1
- uv 0.12.0
- Git LFS 3.0.2
- Python system package 3.11.0rc1

The Ubuntu Jammy repository exposes a prerelease Python 3.11 package. The
project environments therefore use a stable uv-managed CPython 3.11 instead
of the system interpreter; this is an environment-only compatibility
variation, not an experimental model/loss/precision fallback.

## Raw preflight commands

The following commands are also captured by `ops/bootstrap_machine.sh` on
every rerun:

```bash
nvidia-smi
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,memory.total,driver_version,pci.bus_id --format=csv
df -h / /opt /dev/shm
free -h
```


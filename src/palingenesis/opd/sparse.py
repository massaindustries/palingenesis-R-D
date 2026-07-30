"""Sparse tutoring anchors and coarse-grained reverse KL.

The sparse loss preserves exact probabilities on the union of teacher top-k,
student top-k, and required stop tokens. Everything outside that explicit
support is represented by one residual category. This is intentionally not a
full-vocabulary KL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Interval = int | Literal["final"]


@dataclass(frozen=True, slots=True)
class SparseRKLStats:
    mean_kl: float
    mean_student_residual: float
    mean_teacher_residual: float
    clipped_probabilities: int


def select_anchor_positions(
    completion_ids: list[int] | tuple[int, ...],
    interval_tokens: Interval,
    *,
    stop_ids: tuple[int, ...] = (),
    always_include_final_anchor: bool = True,
    include_eos_anchor: bool = True,
    anchor_window_tokens: int = 1,
    max_anchors_per_sequence: int = 64,
) -> list[int]:
    """Return deterministic, zero-based completion positions to supervise.

    Periodic anchors are placed at the end of each one-based interval: for an
    interval of 8 they are positions 7, 15, 23, ... . A window greater than one
    adds the immediately preceding positions for every selected anchor.
    """
    length = len(completion_ids)
    if length == 0:
        return []
    if interval_tokens != "final":
        if not isinstance(interval_tokens, int) or isinstance(interval_tokens, bool):
            raise TypeError("interval_tokens must be a positive integer or 'final'")
        if interval_tokens <= 0:
            raise ValueError("interval_tokens must be positive")
        anchors = set(range(interval_tokens - 1, length, interval_tokens))
    else:
        anchors = set()

    if always_include_final_anchor or interval_tokens == "final":
        anchors.add(length - 1)
    if include_eos_anchor and stop_ids:
        stop_set = set(stop_ids)
        anchors.update(i for i, token_id in enumerate(completion_ids) if token_id in stop_set)

    if anchor_window_tokens <= 0:
        raise ValueError("anchor_window_tokens must be positive")
    expanded = {
        position
        for anchor in anchors
        for position in range(max(0, anchor - anchor_window_tokens + 1), anchor + 1)
    }
    ordered = sorted(expanded)
    if max_anchors_per_sequence <= 0:
        raise ValueError("max_anchors_per_sequence must be positive")
    if len(ordered) <= max_anchors_per_sequence:
        return ordered

    # Preserve the final anchor and keep earlier anchors deterministically.
    return ordered[: max_anchors_per_sequence - 1] + [ordered[-1]]


def sparse_anchor_rkl(
    student_logits: torch.Tensor,
    explicit_token_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, SparseRKLStats]:
    """Compute coarse-grained reverse KL with one residual bucket.

    Args:
        student_logits: ``[anchors, vocab]`` differentiable student logits.
        explicit_token_ids: ``[anchors, support]`` IDs forming the selected
            union. Padding is not supported; callers should batch equal support
            sizes or call this per anchor.
        teacher_logprobs: exact teacher log probabilities for the same IDs.
        epsilon: controlled lower clamp for explicit and residual masses.
    """
    if student_logits.ndim != 2:
        raise ValueError("student_logits must have shape [anchors, vocab]")
    if explicit_token_ids.ndim != 2 or teacher_logprobs.ndim != 2:
        raise ValueError("explicit_token_ids and teacher_logprobs must be rank 2")
    if explicit_token_ids.shape != teacher_logprobs.shape:
        raise ValueError("explicit token IDs and teacher logprobs must have identical shapes")
    if explicit_token_ids.shape[0] != student_logits.shape[0]:
        raise ValueError("anchor batch dimension mismatch")
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be in (0, 0.5)")
    if explicit_token_ids.numel() and (
        explicit_token_ids.min() < 0 or explicit_token_ids.max() >= student_logits.shape[-1]
    ):
        raise ValueError("explicit token ID outside student vocabulary")
    if not torch.isfinite(teacher_logprobs).all():
        raise ValueError("teacher logprobs must all be finite")

    student_logp_full = torch.log_softmax(student_logits.float(), dim=-1)
    student_logp = student_logp_full.gather(-1, explicit_token_ids)
    student_p = student_logp.exp()
    teacher_p = teacher_logprobs.float().exp()

    student_residual_raw = 1.0 - student_p.sum(-1)
    teacher_residual_raw = 1.0 - teacher_p.sum(-1)
    clipped = (
        (student_p < epsilon).sum()
        + (teacher_p < epsilon).sum()
        + (student_residual_raw < epsilon).sum()
        + (teacher_residual_raw < epsilon).sum()
    )

    student_p_safe = student_p.clamp_min(epsilon)
    teacher_p_safe = teacher_p.clamp_min(epsilon)
    student_residual = student_residual_raw.clamp(min=epsilon, max=1.0)
    teacher_residual = teacher_residual_raw.clamp(min=epsilon, max=1.0)

    explicit_kl = student_p_safe * (student_p_safe.log() - teacher_p_safe.log())
    residual_kl = student_residual * (student_residual.log() - teacher_residual.log())
    per_anchor = explicit_kl.sum(-1) + residual_kl
    if not torch.isfinite(per_anchor).all():
        raise FloatingPointError("sparse anchor reverse KL is non-finite")

    stats = SparseRKLStats(
        mean_kl=per_anchor.detach().mean().item(),
        mean_student_residual=student_residual_raw.detach().mean().item(),
        mean_teacher_residual=teacher_residual_raw.detach().mean().item(),
        clipped_probabilities=int(clipped.detach().item()),
    )
    return per_anchor.mean(), stats


"""Interleaved student/teacher trajectory construction and target controls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

BatchGenerator = Callable[[list[int], list[list[int]], int], list[list[int]]]


@dataclass(slots=True)
class TeacherGroup:
    start: int
    end: int
    finish_reason: str = "unknown"


@dataclass(slots=True)
class GuidedTrajectory:
    completion: list[int] = field(default_factory=list)
    teacher_mask: list[bool] = field(default_factory=list)
    teacher_groups: list[TeacherGroup] = field(default_factory=list)
    student_tokens: int = 0
    teacher_tokens: int = 0
    stopped: bool = False
    shuffled_targets: dict[int, int] = field(default_factory=dict)

    def append_student(self, tokens: list[int]) -> None:
        self.completion.extend(tokens)
        self.teacher_mask.extend([False] * len(tokens))
        self.student_tokens += len(tokens)

    def append_teacher(self, tokens: list[int], finish_reason: str) -> None:
        start = len(self.completion)
        self.completion.extend(tokens)
        self.teacher_mask.extend([True] * len(tokens))
        self.teacher_tokens += len(tokens)
        self.teacher_groups.append(
            TeacherGroup(start=start, end=len(self.completion), finish_reason=finish_reason)
        )

    @property
    def teacher_positions(self) -> list[int]:
        return [index for index, is_teacher in enumerate(self.teacher_mask) if is_teacher]


def _clean_segment(tokens: list[int], stop_ids: set[int], budget: int) -> tuple[list[int], bool]:
    output: list[int] = []
    stopped = False
    for token_id in tokens[:budget]:
        output.append(int(token_id))
        if token_id in stop_ids:
            stopped = True
            break
    return output, stopped


def build_guided_trajectories(
    *,
    count: int,
    max_completion_tokens: list[int],
    interval_tokens: int,
    guidance_group_tokens: int,
    stop_ids: tuple[int, ...],
    student_generate: BatchGenerator,
    teacher_generate: BatchGenerator,
) -> list[GuidedTrajectory]:
    """Alternate batched student and teacher continuations until each budget ends.

    Generator arguments are `(active_indices, current_completions, max_tokens)`.
    Teacher output must already be mapped into the student vocabulary.
    """
    if count <= 0 or len(max_completion_tokens) != count:
        raise ValueError("count and max_completion_tokens disagree")
    if interval_tokens <= 0 or guidance_group_tokens <= 0:
        raise ValueError("student interval and teacher group must be positive")
    if any(limit <= 0 for limit in max_completion_tokens):
        raise ValueError("completion budgets must be positive")

    stops = set(stop_ids)
    trajectories = [GuidedTrajectory() for _ in range(count)]
    active = list(range(count))
    max_rounds = max(max_completion_tokens) + 1
    rounds = 0
    while active:
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError("guided rollout made no progress")

        current = [trajectories[index].completion for index in active]
        raw_student = student_generate(active, current, interval_tokens)
        if len(raw_student) != len(active):
            raise ValueError("student generator batch-size mismatch")
        after_student: list[int] = []
        for index, raw in zip(active, raw_student):
            trajectory = trajectories[index]
            remaining = max_completion_tokens[index] - len(trajectory.completion)
            segment, stopped = _clean_segment(raw, stops, min(interval_tokens, remaining))
            trajectory.append_student(segment)
            trajectory.stopped = stopped
            if not stopped and len(trajectory.completion) < max_completion_tokens[index]:
                after_student.append(index)
        active = after_student
        if not active:
            break

        current = [trajectories[index].completion for index in active]
        raw_teacher = teacher_generate(active, current, guidance_group_tokens)
        if len(raw_teacher) != len(active):
            raise ValueError("teacher generator batch-size mismatch")
        after_teacher: list[int] = []
        for index, raw in zip(active, raw_teacher):
            trajectory = trajectories[index]
            finish_reason = getattr(raw, "finish_reason", "unknown")
            raw_tokens = list(getattr(raw, "token_ids", raw))
            remaining = max_completion_tokens[index] - len(trajectory.completion)
            segment, stopped = _clean_segment(
                raw_tokens, stops, min(guidance_group_tokens, remaining)
            )
            trajectory.append_teacher(segment, str(finish_reason))
            trajectory.stopped = stopped
            if not segment and not stopped:
                # The next student segment can still make progress.
                pass
            if not stopped and len(trajectory.completion) < max_completion_tokens[index]:
                after_teacher.append(index)
        active = after_teacher

    for trajectory in trajectories:
        if len(trajectory.completion) != len(trajectory.teacher_mask):
            raise AssertionError("token provenance length mismatch")
    return trajectories


def assign_shuffled_group_targets(trajectories: list[GuidedTrajectory]) -> None:
    """Move teacher groups across trajectories while preserving target count.

    Full groups are exchanged intact. A rare stop-truncated group receives the
    same-length prefix of a donor group from another trajectory.
    """
    groups: list[tuple[int, TeacherGroup, list[int]]] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        trajectory.shuffled_targets.clear()
        for group in trajectory.teacher_groups:
            tokens = trajectory.completion[group.start : group.end]
            if tokens:
                groups.append((trajectory_index, group, tokens))

    if len({trajectory_index for trajectory_index, _, _ in groups}) < 2:
        raise ValueError("cannot shuffle teacher groups across fewer than two trajectories")
    for group_index, (recipient_index, recipient_group, recipient_tokens) in enumerate(groups):
        donor_tokens = None
        for offset in range(1, len(groups) + 1):
            donor_index, _, candidate = groups[(group_index + offset) % len(groups)]
            if donor_index != recipient_index and len(candidate) >= len(recipient_tokens):
                donor_tokens = candidate[: len(recipient_tokens)]
                break
        if donor_tokens is None:
            raise ValueError(
                "could not find a cross-trajectory donor long enough for a teacher group"
            )
        trajectories[recipient_index].shuffled_targets.update(
            {
                position: token_id
                for position, token_id in zip(
                    range(recipient_group.start, recipient_group.end), donor_tokens
                )
            }
        )

    expected = sum(trajectory.teacher_tokens for trajectory in trajectories)
    actual = sum(len(trajectory.shuffled_targets) for trajectory in trajectories)
    if actual != expected:
        raise ValueError(
            f"shuffle did not preserve teacher-token count: {actual} != {expected}"
        )

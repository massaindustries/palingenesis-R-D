from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from palingenesis.opd.config import OPDConfig, OPDConfigError
from palingenesis.opd.guided import (
    GuidedTrajectory,
    TeacherGroup,
    assign_shuffled_group_targets,
    build_guided_trajectories,
)
from palingenesis.opd.token_bridge import TokenBridge, TokenBridgeError
from palingenesis.opd.trainer import OPDTrainer, intermediate_checkpoint_name


def test_intermediate_checkpoint_names_count_completed_updates() -> None:
    names = [
        intermediate_checkpoint_name(step, save_steps=5, total_steps=20)
        for step in range(20)
    ]
    assert [(step, name) for step, name in enumerate(names) if name] == [
        (4, "step_5"),
        (9, "step_10"),
        (14, "step_15"),
    ]


def test_guided_rollout_alternates_and_tracks_exact_provenance() -> None:
    student_calls = []
    teacher_calls = []

    def student_generate(indices, completions, limit):
        student_calls.append((indices, [list(c) for c in completions], limit))
        return [[10 + len(c) + offset for offset in range(limit)] for c in completions]

    def teacher_generate(indices, completions, limit):
        teacher_calls.append((indices, [list(c) for c in completions], limit))
        return [
            SimpleNamespace(token_ids=tuple([90 + len(c)] * limit), finish_reason="length")
            for c in completions
        ]

    [trajectory] = build_guided_trajectories(
        count=1,
        max_completion_tokens=[14],
        interval_tokens=4,
        guidance_group_tokens=2,
        stop_ids=(99,),
        student_generate=student_generate,
        teacher_generate=teacher_generate,
    )

    assert trajectory.student_tokens == 10
    assert trajectory.teacher_tokens == 4
    assert trajectory.teacher_positions == [4, 5, 10, 11]
    assert [(group.start, group.end) for group in trajectory.teacher_groups] == [
        (4, 6),
        (10, 12),
    ]
    assert student_calls[1][1][0][:6] == trajectory.completion[:6]
    assert teacher_calls[1][1][0][:10] == trajectory.completion[:10]


def test_guided_rollout_teacher_stop_terminates_sequence() -> None:
    def student_generate(indices, completions, limit):
        return [[1] * limit for _ in indices]

    def teacher_generate(indices, completions, limit):
        return [SimpleNamespace(token_ids=(2, 99, 3), finish_reason="stop") for _ in indices]

    [trajectory] = build_guided_trajectories(
        count=1,
        max_completion_tokens=[20],
        interval_tokens=4,
        guidance_group_tokens=3,
        stop_ids=(99,),
        student_generate=student_generate,
        teacher_generate=teacher_generate,
    )
    assert trajectory.completion == [1, 1, 1, 1, 2, 99]
    assert trajectory.teacher_mask == [False, False, False, False, True, True]
    assert trajectory.stopped


def test_shuffled_targets_preserve_groups_but_cross_prompts() -> None:
    def student_generate(indices, completions, limit):
        return [[1] * limit for _ in indices]

    def teacher_generate(indices, completions, limit):
        return [
            SimpleNamespace(
                token_ids=tuple([20 + index] * limit), finish_reason="length"
            )
            for index in indices
        ]

    trajectories = build_guided_trajectories(
        count=3,
        max_completion_tokens=[6, 6, 6],
        interval_tokens=4,
        guidance_group_tokens=2,
        stop_ids=(99,),
        student_generate=student_generate,
        teacher_generate=teacher_generate,
    )
    assign_shuffled_group_targets(trajectories)
    assert [list(t.shuffled_targets.values()) for t in trajectories] == [
        [21, 21],
        [22, 22],
        [20, 20],
    ]
    assert sum(len(t.shuffled_targets) for t in trajectories) == 6


def test_shuffled_targets_support_stop_truncated_group() -> None:
    first = GuidedTrajectory(
        completion=[1, 10],
        teacher_mask=[False, True],
        teacher_groups=[TeacherGroup(1, 2, "stop")],
        student_tokens=1,
        teacher_tokens=1,
    )
    second = GuidedTrajectory(
        completion=[2, 20, 21],
        teacher_mask=[False, True, True],
        teacher_groups=[TeacherGroup(1, 3, "length")],
        student_tokens=1,
        teacher_tokens=2,
    )
    third = GuidedTrajectory(
        completion=[3, 30, 31],
        teacher_mask=[False, True, True],
        teacher_groups=[TeacherGroup(1, 3, "length")],
        student_tokens=1,
        teacher_tokens=2,
    )
    assign_shuffled_group_targets([first, second, third])
    assert first.shuffled_targets == {1: 20}
    assert len(first.shuffled_targets) == first.teacher_tokens


def test_token_bridge_maps_teacher_stop_back_to_student() -> None:
    bridge = TokenBridge(shared_vocab_size=100, swap={101: 99}, stop_ids=(101,))
    assert bridge.to_student([5, 99]) == [5, 101]
    with pytest.raises(TokenBridgeError, match="outside shared"):
        bridge.to_student([100])


class _TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 8)

    def forward(self, input_ids=None, attention_mask=None):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _TinyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyBackbone()
        self.lm_head = torch.nn.Linear(8, 16, bias=False)


def bare_guided_trainer(shuffled: bool = False) -> OPDTrainer:
    trainer = OPDTrainer.__new__(OPDTrainer)
    trainer.config = OPDConfig()
    trainer.config.train.loss_fn = "guided_ce"
    trainer.config.tutoring.mode = "guided_tokens"
    trainer.config.tutoring.shuffle_teacher_groups = shuffled
    trainer.student = _TinyLM()
    trainer.device = "cpu"
    trainer.s_pad = 0
    return trainer


def trajectory_with_two_teacher_tokens() -> GuidedTrajectory:
    return GuidedTrajectory(
        completion=[3, 4, 5],
        teacher_mask=[False, True, True],
        teacher_groups=[TeacherGroup(1, 3, "length")],
        student_tokens=1,
        teacher_tokens=2,
    )


def test_guided_ce_update_reduces_the_exact_teacher_token_nll() -> None:
    torch.manual_seed(7)
    trainer = bare_guided_trainer()
    rollout = ({"s_prompt": [1, 2]}, trajectory_with_two_teacher_tokens())
    optimizer = torch.optim.SGD(trainer.student.parameters(), lr=0.2)

    loss, count, before = trainer._guided_loss_on_chunk([rollout])
    assert count == 2
    optimizer.zero_grad()
    (loss / count).backward()
    optimizer.step()
    after = trainer._measure_guided_rollouts([rollout])

    assert after["teacher_token_nll"] < before["teacher_token_nll"]


def test_shuffled_control_optimizes_other_targets_but_measures_teacher() -> None:
    trainer = bare_guided_trainer(shuffled=True)
    trajectory = trajectory_with_two_teacher_tokens()
    trajectory.shuffled_targets = {1: 8, 2: 9}
    _, count, stats = trainer._guided_loss_on_chunk(
        [({"s_prompt": [1, 2]}, trajectory)]
    )
    assert count == 2
    assert stats["training_target_nll"] != stats["teacher_token_nll"]


def test_guided_config_requires_generation_backend_and_matching_mode() -> None:
    config = OPDConfig()
    config.model.student = "student"
    config.model.teacher = "teacher"
    config.data.format = "messages"
    config.train.loss_fn = "guided_ce"
    with pytest.raises(OPDConfigError, match="guided_tokens"):
        config.validate()

    config.tutoring.mode = "guided_tokens"
    with pytest.raises(OPDConfigError, match="teacher_backend"):
        config.validate()

    config.model.teacher_backend = "sglang"
    assert config.validate() == []

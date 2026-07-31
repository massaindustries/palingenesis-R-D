#!/usr/bin/env python3
"""Exact paired counterfactual matched-vs-shuffled optimizer probe."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import torch

from palingenesis.opd.config import OPDConfig
from palingenesis.opd.guided import assign_shuffled_group_targets
from palingenesis.opd.trainer import OPDTrainer


def adapter_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def restore_adapter(model, state: dict[str, torch.Tensor]) -> None:
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if parameters.keys() != state.keys():
        raise RuntimeError("adapter snapshot keys changed")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(parameter.device))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    config = OPDConfig.from_yaml(args.config)
    config.train.output_dir = str(args.output_dir)
    config.train.steps = args.trials
    config.train.gradient_accumulation_steps = 1
    config.tutoring.paired_student_only = False
    config.tutoring.shuffle_teacher_groups = False
    config.logging.use_wandb = False
    config.validate()
    trainer = OPDTrainer(config)
    records = []

    for trial in range(args.trials):
        trainer.current_step = trial
        trainer.current_microstep = 0
        for group in trainer.opt.param_groups:
            group["lr"] = trainer._lr_at(trial)
        pre_adapter = adapter_snapshot(trainer.student)
        pre_optimizer = copy.deepcopy(trainer.opt.state_dict())

        # Matched branch: generate once and apply the real teacher targets.
        trainer.opt.zero_grad(set_to_none=True)
        matched_result = trainer._train_microstep(gradient_scale=1.0)
        if matched_result is None:
            raise RuntimeError("paired causal trial has no teacher intervention")
        rollouts = matched_result["_guided_rollouts"]
        matched_grad = torch.nn.utils.clip_grad_norm_(
            trainer.student.parameters(), config.train.max_grad_norm
        )
        trainer.opt.step()
        matched_post = trainer._measure_guided_rollouts(rollouts)
        matched_adapter = adapter_snapshot(trainer.student)
        matched_optimizer = copy.deepcopy(trainer.opt.state_dict())

        # Shuffled branch: restore the exact pre-update state and reuse the
        # exact same trajectories, changing only cross-prompt target groups.
        restore_adapter(trainer.student, pre_adapter)
        trainer.opt.load_state_dict(pre_optimizer)
        trainer.opt.zero_grad(set_to_none=True)
        config.tutoring.shuffle_teacher_groups = True
        trajectories = [trajectory for _, trajectory in rollouts]
        assign_shuffled_group_targets(trajectories)
        shuffled_loss, token_count, shuffled_pre = trainer._guided_loss_on_chunk(
            rollouts
        )
        if token_count != matched_result["teacher_inserted_tokens"]:
            raise RuntimeError("counterfactual target count changed")
        (shuffled_loss / token_count).backward()
        shuffled_grad = torch.nn.utils.clip_grad_norm_(
            trainer.student.parameters(), config.train.max_grad_norm
        )
        trainer.opt.step()
        shuffled_post = trainer._measure_guided_rollouts(rollouts)

        matched_pre = matched_result["teacher_token_nll_pre"]
        if abs(shuffled_pre["teacher_token_nll"] - matched_pre) > 1e-6:
            raise RuntimeError("paired branches do not share the same pre-update NLL")
        record = {
            "trial": trial,
            "policy_version": trainer.policy_version,
            "teacher_tokens": token_count,
            "trajectory_hashes": [
                __import__("hashlib")
                .sha256(json.dumps(trajectory.completion).encode())
                .hexdigest()
                for trajectory in trajectories
            ],
            "matched_teacher_nll_pre": matched_pre,
            "matched_teacher_nll_post": matched_post["teacher_token_nll"],
            "matched_teacher_nll_delta": matched_post["teacher_token_nll"]
            - matched_pre,
            "shuffled_teacher_nll_pre": shuffled_pre["teacher_token_nll"],
            "shuffled_teacher_nll_post": shuffled_post["teacher_token_nll"],
            "shuffled_teacher_nll_delta": shuffled_post["teacher_token_nll"]
            - shuffled_pre["teacher_token_nll"],
            "shuffled_training_nll_pre": shuffled_pre["training_target_nll"],
            "shuffled_training_nll_post": shuffled_post["training_target_nll"],
            "shuffled_training_nll_delta": shuffled_post["training_target_nll"]
            - shuffled_pre["training_target_nll"],
            "matched_grad_norm": float(matched_grad),
            "shuffled_grad_norm": float(shuffled_grad),
            "lr": trainer.opt.param_groups[0]["lr"],
        }
        if not all(
            math.isfinite(value)
            for key, value in record.items()
            if isinstance(value, float)
        ):
            raise FloatingPointError("non-finite paired causal metric")
        records.append(record)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "paired_causal_metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

        # Advance the on-policy sequence along the matched branch.
        restore_adapter(trainer.student, matched_adapter)
        trainer.opt.load_state_dict(matched_optimizer)
        config.tutoring.shuffle_teacher_groups = False
        trainer.policy_version += 1
        trainer.last_sync_metrics = trainer.rollout_worker.sync_from(
            trainer.student, trainer.policy_version
        )

    matched_deltas = [row["matched_teacher_nll_delta"] for row in records]
    shuffled_deltas = [row["shuffled_teacher_nll_delta"] for row in records]
    differences = [
        shuffled - matched
        for matched, shuffled in zip(matched_deltas, shuffled_deltas)
    ]
    summary = {
        "status": "pass",
        "trials": len(records),
        "same_trajectory_counterfactual": True,
        "matched_delta_mean": sum(matched_deltas) / len(records),
        "shuffled_delta_mean": sum(shuffled_deltas) / len(records),
        "shuffled_minus_matched_delta_mean": sum(differences) / len(records),
        "matched_better_trials": sum(value > 0 for value in differences),
        "matched_nll_decreased_trials": sum(value < 0 for value in matched_deltas),
        "shuffled_nll_decreased_trials": sum(value < 0 for value in shuffled_deltas),
        "shuffled_training_nll_decreased_trials": sum(
            row["shuffled_training_nll_delta"] < 0 for row in records
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

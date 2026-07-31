#!/usr/bin/env python3
"""Diagnose why one guided-tutoring interval outperforms another."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter
from pathlib import Path

INTERVALS = (8, 32, 128)
SEEDS = (0, 1, 2)
T_CRITICAL_95_DF2 = 4.302652729911275
ALLOCATED_GPUS = 4


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(probability * len(ordered))))
    return ordered[index]


def task_cluster_bootstrap(
    task_deltas: list[float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    """Resample tasks while preserving all seed outcomes within each task."""
    rng = random.Random(seed)
    count = len(task_deltas)
    draws = [
        statistics.mean(task_deltas[rng.randrange(count)] for _ in range(count))
        for _ in range(samples)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def seed_t_interval(seed_deltas: list[float]) -> list[float]:
    """Student-t interval across three independently trained seed estimates."""
    center = statistics.mean(seed_deltas)
    standard_error = statistics.stdev(seed_deltas) / len(seed_deltas) ** 0.5
    margin = T_CRITICAL_95_DF2 * standard_error
    return [center - margin, center + margin]


def checkpoint_updates(name: str) -> int:
    return 20 if name == "final" else int(name.removeprefix("step_"))


def weighted_mean(rows: list[dict], key: str, weight: str) -> float:
    denominator = sum(float(row[weight]) for row in rows)
    return sum(float(row[key]) * float(row[weight]) for row in rows) / denominator


def relative_savings(candidate: float, comparator: float) -> float:
    return 1 - candidate / comparator


def summarize_selected_training(
    *,
    run_dir: Path,
    checkpoint_name: str,
) -> dict:
    updates = checkpoint_updates(checkpoint_name)
    metric_rows = [
        row
        for row in read_jsonl(run_dir / "metrics.jsonl")
        if "teacher_token_nll_delta" in row
    ][:updates]
    if len(metric_rows) != updates:
        raise ValueError(f"{run_dir}: expected {updates} metric rows")
    provenance = [
        row
        for row in read_jsonl(run_dir / "teacher_provenance.jsonl")
        if int(row["step"]) < updates
    ]
    teacher_tokens = sum(int(row["teacher_inserted_tokens"]) for row in metric_rows)
    trajectory_tokens = sum(int(row["trajectory_tokens"]) for row in metric_rows)
    groups = sum(int(row["teacher_guidance_groups"]) for row in metric_rows)
    guided_rollouts = sum(int(row["guided_rollout_count"]) for row in metric_rows)
    wall_clock_seconds = float(metric_rows[-1]["run/wall_clock_seconds"])
    task_counts = Counter(str(row["task_id"]) for row in provenance)
    return {
        "selected_checkpoint": checkpoint_name,
        "updates": updates,
        "teacher_tokens": teacher_tokens,
        "trajectory_tokens": trajectory_tokens,
        "teacher_trajectory_fraction": teacher_tokens / trajectory_tokens,
        "teacher_groups": groups,
        "teacher_requests": sum(int(row["teacher_requests"]) for row in metric_rows),
        "teacher_wall_clock_seconds": sum(
            float(row["teacher_wall_clock_ms"]) for row in metric_rows
        )
        / 1_000,
        "guided_rollouts": guided_rollouts,
        "groups_per_guided_rollout": groups / guided_rollouts,
        "intervention_coverage_mean": statistics.mean(
            float(row["teacher_intervention_coverage"]) for row in metric_rows
        ),
        "gradient_norm_mean": statistics.mean(
            float(row["gradient_norm"]) for row in metric_rows
        ),
        "teacher_token_nll_pre_weighted": weighted_mean(
            metric_rows, "teacher_token_nll_pre", "teacher_inserted_tokens"
        ),
        "teacher_token_nll_post_weighted": weighted_mean(
            metric_rows, "teacher_token_nll_post", "teacher_inserted_tokens"
        ),
        "teacher_token_nll_delta_weighted": weighted_mean(
            metric_rows, "teacher_token_nll_delta", "teacher_inserted_tokens"
        ),
        "teacher_token_top1_pre_weighted": weighted_mean(
            metric_rows, "teacher_token_top1_pre", "teacher_inserted_tokens"
        ),
        "teacher_token_top1_post_weighted": weighted_mean(
            metric_rows, "teacher_token_top1_post", "teacher_inserted_tokens"
        ),
        "accepted_rollouts": len(provenance),
        "unique_training_tasks": len(task_counts),
        "mean_repeats_per_training_task": statistics.mean(task_counts.values()),
        "wall_clock_seconds": wall_clock_seconds,
        # The experiment reserved two teacher, one train, and one rollout GPU.
        "estimated_allocated_gpu_hours": ALLOCATED_GPUS
        * wall_clock_seconds
        / 3_600,
    }


def ordered_records(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    if len(rows) != 40:
        raise ValueError(f"{path}: expected 40 rows, found {len(rows)}")
    ids = [str(row["id"]) for row in rows]
    hashes = [str(row["prompt_hash"]) for row in rows]
    if len(set(ids)) != len(ids) or len(set(hashes)) != len(hashes):
        raise ValueError(f"{path}: duplicate task ID or prompt hash")
    return rows


def summarize_interval_test(
    *,
    interval: int,
    records_by_condition: dict[str, list[dict]],
    base: list[dict],
    teacher: list[dict],
    bootstrap_samples: int,
) -> tuple[dict, list[float]]:
    base_outcomes = [bool(row["sandbox"]["test_pass"]) for row in base]
    base_rate = statistics.mean(base_outcomes)
    seed_rows = [records_by_condition[f"i{interval}_seed{seed}"] for seed in SEEDS]
    seed_outcomes = [
        [bool(row["sandbox"]["test_pass"]) for row in rows]
        for rows in seed_rows
    ]
    fixes_total = 0
    regressions_total = 0
    fixed_tasks: Counter[str] = Counter()
    regressed_tasks: Counter[str] = Counter()
    fixed_teacher_passes = 0
    seed_passes = []
    seed_deltas = []
    for outcomes in seed_outcomes:
        seed_passes.append(sum(outcomes))
        seed_deltas.append(statistics.mean(outcomes) - base_rate)
        for index, (candidate, baseline) in enumerate(zip(outcomes, base_outcomes)):
            task_id = str(base[index]["id"])
            if candidate and not baseline:
                fixes_total += 1
                fixed_tasks[task_id] += 1
                fixed_teacher_passes += int(
                    bool(teacher[index]["sandbox"]["test_pass"])
                )
            elif baseline and not candidate:
                regressions_total += 1
                regressed_tasks[task_id] += 1

    task_deltas = [
        statistics.mean(int(seed[index]) for seed in seed_outcomes)
        - int(base_outcomes[index])
        for index in range(len(base))
    ]
    completions = [row for rows in seed_rows for row in rows]
    pairwise_seed_agreement = {}
    for left, right in ((0, 1), (0, 2), (1, 2)):
        pairwise_seed_agreement[f"seed{left}_seed{right}"] = {
            "identical_completions": sum(
                a["completion"] == b["completion"]
                for a, b in zip(seed_rows[left], seed_rows[right])
            ),
            "identical_outcomes": sum(
                bool(a["sandbox"]["test_pass"])
                == bool(b["sandbox"]["test_pass"])
                for a, b in zip(seed_rows[left], seed_rows[right])
            ),
        }
    return (
        {
            "tasks": len(base),
            "seeds": len(SEEDS),
            "seed_passes": seed_passes,
            "mean_pass_at_1": statistics.mean(seed_passes) / len(base),
            "mean_delta_vs_base": statistics.mean(seed_deltas),
            "seed_sample_std_pass_at_1": statistics.stdev(
                passes / len(base) for passes in seed_passes
            ),
            "seed_t_ci95_delta_vs_base": seed_t_interval(seed_deltas),
            "task_cluster_bootstrap_ci95_delta_vs_base": task_cluster_bootstrap(
                task_deltas,
                samples=bootstrap_samples,
                seed=20260731 + interval,
            ),
            "fixes_over_seed_tasks": fixes_total,
            "regressions_over_seed_tasks": regressions_total,
            "net_pass_events_over_seed_tasks": fixes_total - regressions_total,
            "fix_rate_over_seed_tasks": fixes_total / (len(base) * len(SEEDS)),
            "regression_rate_over_seed_tasks": regressions_total
            / (len(base) * len(SEEDS)),
            "fixed_task_seed_counts": dict(sorted(fixed_tasks.items())),
            "regressed_task_seed_counts": dict(sorted(regressed_tasks.items())),
            "teacher_passes_among_fix_events": fixed_teacher_passes,
            "max_length_count": sum(
                row["finish_reason"] == "length" for row in completions
            ),
            "syntax_failure_count": sum(
                not bool(row["sandbox"]["syntax_success"]) for row in completions
            ),
            "mean_completion_tokens": statistics.mean(
                int(row["completion_tokens"]) for row in completions
            ),
            "pairwise_seed_agreement": pairwise_seed_agreement,
        },
        task_deltas,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--deterministic-probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=200_000)
    args = parser.parse_args()

    selection = read_json(args.selection)
    if selection.get("status") != "pass" or len(selection["selections"]) != 9:
        raise ValueError("diagnostic requires nine valid dev selections")
    selected = {
        (int(row["interval"]), int(row["seed"])): row
        for row in selection["selections"]
    }

    condition_names = ["base_student", "frozen_teacher", "offline_sft"] + [
        f"i{interval}_seed{seed}" for interval in INTERVALS for seed in SEEDS
    ]
    records_by_condition = {
        name: ordered_records(args.evaluation_root / name / "records.jsonl")
        for name in condition_names
    }
    reference = [
        (row["id"], row["prompt_hash"])
        for row in records_by_condition["base_student"]
    ]
    for name, rows in records_by_condition.items():
        signature = [(row["id"], row["prompt_hash"]) for row in rows]
        if signature != reference:
            raise ValueError(f"task order/hash mismatch for {name}")

    training = {}
    dev_checkpoint_curves = {}
    for interval in INTERVALS:
        seed_summaries = []
        full_grid_seed_summaries = []
        interval_selections = [
            selected[(interval, seed)] for seed in SEEDS
        ]
        for seed in SEEDS:
            row = selected[(interval, seed)]
            seed_summaries.append(
                summarize_selected_training(
                    run_dir=args.run_root / f"i{interval}_seed{seed}",
                    checkpoint_name=str(row["selected_checkpoint_name"]),
                )
            )
            full_grid_seed_summaries.append(
                summarize_selected_training(
                    run_dir=args.run_root / f"i{interval}_seed{seed}",
                    checkpoint_name="final",
                )
            )
        selected_gpu_hours = sum(
            row["estimated_allocated_gpu_hours"] for row in seed_summaries
        )
        full_grid_gpu_hours = sum(
            row["estimated_allocated_gpu_hours"]
            for row in full_grid_seed_summaries
        )
        selected_teacher_tokens = sum(row["teacher_tokens"] for row in seed_summaries)
        full_grid_teacher_tokens = sum(
            row["teacher_tokens"] for row in full_grid_seed_summaries
        )
        training[str(interval)] = {
            "seeds": seed_summaries,
            "full_grid_seeds": full_grid_seed_summaries,
            "mean_selected_updates": statistics.mean(
                row["updates"] for row in seed_summaries
            ),
            "mean_teacher_trajectory_fraction": statistics.mean(
                row["teacher_trajectory_fraction"] for row in seed_summaries
            ),
            "mean_groups_per_guided_rollout": statistics.mean(
                row["groups_per_guided_rollout"] for row in seed_summaries
            ),
            "mean_intervention_coverage": statistics.mean(
                row["intervention_coverage_mean"] for row in seed_summaries
            ),
            "mean_cumulative_teacher_tokens": statistics.mean(
                row["teacher_tokens"] for row in seed_summaries
            ),
            "teacher_token_nll_pre_weighted": sum(
                row["teacher_token_nll_pre_weighted"] * row["teacher_tokens"]
                for row in seed_summaries
            )
            / sum(row["teacher_tokens"] for row in seed_summaries),
            "teacher_token_nll_delta_weighted": sum(
                row["teacher_token_nll_delta_weighted"] * row["teacher_tokens"]
                for row in seed_summaries
            )
            / sum(row["teacher_tokens"] for row in seed_summaries),
            "selected_total_allocated_gpu_hours": selected_gpu_hours,
            "full_grid_total_allocated_gpu_hours": full_grid_gpu_hours,
            "selected_total_teacher_tokens": selected_teacher_tokens,
            "full_grid_total_teacher_tokens": full_grid_teacher_tokens,
            "selected_total_teacher_requests": sum(
                row["teacher_requests"] for row in seed_summaries
            ),
            "full_grid_total_teacher_requests": sum(
                row["teacher_requests"] for row in full_grid_seed_summaries
            ),
            "retrospective_early_stop_gpu_hour_savings_fraction": relative_savings(
                selected_gpu_hours, full_grid_gpu_hours
            ),
            "retrospective_early_stop_teacher_token_savings_fraction": relative_savings(
                selected_teacher_tokens, full_grid_teacher_tokens
            ),
        }
        dev_checkpoint_curves[str(interval)] = {}
        for checkpoint_name in ("step_5", "step_10", "step_15", "final"):
            candidates = [
                next(
                    candidate
                    for candidate in row["candidates"]
                    if candidate["checkpoint_name"] == checkpoint_name
                )
                for row in interval_selections
            ]
            dev_checkpoint_curves[str(interval)][checkpoint_name] = {
                "seed_passes": [
                    round(float(candidate["pass_at_1"]) * 21)
                    for candidate in candidates
                ],
                "mean_pass_at_1": statistics.mean(
                    float(candidate["pass_at_1"]) for candidate in candidates
                ),
                "mean_max_length_rate": statistics.mean(
                    float(candidate["max_length_rate"])
                    for candidate in candidates
                ),
                "mean_completion_tokens": statistics.mean(
                    float(candidate["mean_completion_tokens"])
                    for candidate in candidates
                ),
            }

    economics = {
        "allocated_gpu_hour_definition": (
            "four reserved GPUs multiplied by run wall-clock hours"
        ),
        "currency_cost_formula": (
            "allocated GPU-hours multiplied by the blended reserved-GPU hourly price"
        ),
        "comparisons": {},
    }
    for basis in ("full_grid", "selected"):
        candidate_gpu_hours = training["32"][
            f"{basis}_total_allocated_gpu_hours"
        ]
        candidate_teacher_tokens = training["32"][f"{basis}_total_teacher_tokens"]
        for comparator in (8, 128):
            comparator_gpu_hours = training[str(comparator)][
                f"{basis}_total_allocated_gpu_hours"
            ]
            comparator_teacher_tokens = training[str(comparator)][
                f"{basis}_total_teacher_tokens"
            ]
            economics["comparisons"][f"32_vs_{comparator}_{basis}"] = {
                "allocated_gpu_hours_32": candidate_gpu_hours,
                "allocated_gpu_hours_comparator": comparator_gpu_hours,
                "allocated_gpu_hours_saved": comparator_gpu_hours
                - candidate_gpu_hours,
                "allocated_gpu_hour_savings_fraction": relative_savings(
                    candidate_gpu_hours, comparator_gpu_hours
                ),
                "teacher_tokens_32": candidate_teacher_tokens,
                "teacher_tokens_comparator": comparator_teacher_tokens,
                "teacher_tokens_saved": comparator_teacher_tokens
                - candidate_teacher_tokens,
                "teacher_token_savings_fraction": relative_savings(
                    candidate_teacher_tokens, comparator_teacher_tokens
                ),
            }

    base = records_by_condition["base_student"]
    teacher = records_by_condition["frozen_teacher"]
    test = {}
    task_deltas = {}
    for interval in INTERVALS:
        test[str(interval)], task_deltas[interval] = summarize_interval_test(
            interval=interval,
            records_by_condition=records_by_condition,
            base=base,
            teacher=teacher,
            bootstrap_samples=args.bootstrap_samples,
        )

    pairwise = {}
    for left, right in ((32, 8), (32, 128)):
        deltas = [
            left_delta - right_delta
            for left_delta, right_delta in zip(task_deltas[left], task_deltas[right])
        ]
        seed_deltas = [
            (
                test[str(left)]["seed_passes"][seed]
                - test[str(right)]["seed_passes"][seed]
            )
            / len(base)
            for seed in SEEDS
        ]
        pairwise[f"{left}_vs_{right}"] = {
            "mean_pass_at_1_difference": statistics.mean(seed_deltas),
            "seed_differences": seed_deltas,
            "seed_t_ci95": seed_t_interval(seed_deltas),
            "task_cluster_bootstrap_ci95": task_cluster_bootstrap(
                deltas,
                samples=args.bootstrap_samples,
                seed=20260731 + left + right,
            ),
        }

    report = {
        "status": "pass",
        "source_qa": {
            "conditions": len(condition_names),
            "tasks_per_condition": len(reference),
            "unique_task_ids": len({row[0] for row in reference}),
            "unique_prompt_hashes": len({row[1] for row in reference}),
            "task_order_and_hashes_match": True,
        },
        "base_pass_at_1": statistics.mean(
            bool(row["sandbox"]["test_pass"]) for row in base
        ),
        "training_at_selected_checkpoints": training,
        "economics": economics,
        "dev_checkpoint_curves": dev_checkpoint_curves,
        "test_by_interval": test,
        "pairwise_interval_differences": pairwise,
        "deterministic_functional_probe": read_json(args.deterministic_probe),
        "bootstrap_samples": args.bootstrap_samples,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

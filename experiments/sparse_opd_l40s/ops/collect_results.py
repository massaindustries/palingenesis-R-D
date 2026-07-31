#!/usr/bin/env python3
"""Collect training and sandbox-evaluation artifacts into condition summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def wilson_interval(
    success_rate: float | None, count: int | None
) -> tuple[float | None, float | None]:
    if success_rate is None or not count:
        return None, None
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (success_rate + z * z / (2 * count)) / denominator
    margin = (
        z
        * math.sqrt(
            success_rate * (1 - success_rate) / count + z * z / (4 * count * count)
        )
        / denominator
    )
    return center - margin, center + margin


def infer_condition(run_dir: Path, config: dict) -> tuple[str, str]:
    experiment = config.get("experiment", {})
    condition = experiment.get("condition", "")
    leaf = run_dir.name
    if not condition:
        mappings = {
            "interval_1": "dense_anchor_rkl_k64_interval1",
            "interval_8": "sparse_anchor_rkl_k64_interval8",
            "interval_32": "sparse_anchor_rkl_k64_interval32",
            "interval_64": "sparse_anchor_rkl_k64_interval64",
            "interval_128": "sparse_anchor_rkl_k64_interval128",
            "final_only": "final_only_rkl",
        }
        condition = mappings.get(
            leaf, "offline_teacher_sft" if "offline" in leaf else leaf
        )
    interval = config.get("tutoring", {}).get("interval_tokens", "")
    return condition, str(interval)


def collect(run_dir: Path) -> dict:
    config = read_json(run_dir / "config_resolved.json")
    metrics = read_metrics(run_dir / "metrics.jsonl")
    train_rows = [row for row in metrics if "gradient_norm" in row]
    condition, interval = infer_condition(run_dir, config)
    evaluation_paths = [
        run_dir / "evaluation_v2" / "evaluation_summary.json",
        run_dir / "evaluation" / "evaluation_summary.json",
        run_dir / "evaluation_summary.json",
    ]
    evaluation = next(
        (read_json(path) for path in evaluation_paths if path.is_file()),
        {},
    )
    teacher_seconds = sum(
        float(row.get("teacher_wall_clock_ms", 0.0)) / 1000 for row in train_rows
    )
    run_seconds = max(
        (float(row.get("run/wall_clock_seconds", 0.0)) for row in train_rows),
        default=0.0,
    )
    anchors = sum(int(row.get("teacher_anchor_positions", 0)) for row in train_rows)
    requests = sum(int(row.get("teacher_requests", 0)) for row in train_rows)
    student_tokens = sum(
        int(row.get("student_generated_tokens", row.get("student_training_tokens", 0)))
        for row in train_rows
    )
    pass_at_1 = evaluation.get("pass_at_1")
    pass_ci_low, pass_ci_high = wilson_interval(pass_at_1, evaluation.get("tasks"))
    teacher_gpu_hours = teacher_seconds * 2 / 3600
    timing_fields = {
        "rollout_generation_seconds": "rollout/generation_ms",
        "student_topk_seconds": "rollout/student_topk_ms",
        "student_forward_seconds": "student/scoring_and_forward_ms",
        "backward_seconds": "student/backward_ms",
        "adapter_sync_seconds": "rollout/sync_ms",
    }
    return {
        "run_dir": str(run_dir),
        "condition": condition,
        "interval_tokens": interval,
        "seed": config.get("train", {}).get("seed", 0),
        "optimizer_steps": len(train_rows),
        "student_tokens": student_tokens,
        "teacher_requests": requests,
        "teacher_anchor_positions": anchors,
        "teacher_scored_tokens": sum(
            int(row.get("teacher_scored_tokens", 0)) for row in train_rows
        ),
        "teacher_wall_clock_seconds": teacher_seconds,
        "teacher_gpu_hours": teacher_gpu_hours,
        "run_wall_clock_seconds": run_seconds,
        **{
            output: sum(float(row.get(source, 0.0)) for row in train_rows) / 1000
            for output, source in timing_fields.items()
        },
        "sandbox_evaluation_seconds": evaluation.get(
            "generation_and_evaluation_seconds"
        ),
        "student_train_gpu_hours": run_seconds / 3600,
        "rollout_gpu_hours": run_seconds / 3600,
        "total_gpu_hours_estimate": run_seconds * 2 / 3600 + teacher_gpu_hours,
        "mean_kl": (
            sum(float(row.get("kl", 0.0)) for row in train_rows) / len(train_rows)
            if train_rows and any("kl" in row for row in train_rows)
            else None
        ),
        "max_residual_student_mass": max(
            (float(row.get("residual_mass", 0.0)) for row in train_rows),
            default=None,
        ),
        "max_policy_staleness": max(
            (int(row.get("rollout/staleness", 0)) for row in train_rows),
            default=None,
        ),
        "peak_student_vram_mib": max(
            (float(row.get("student/peak_vram_mib", 0.0)) for row in train_rows),
            default=None,
        ),
        "peak_rollout_vram_mib": max(
            (float(row.get("rollout/peak_vram_mib", 0.0)) for row in train_rows),
            default=None,
        ),
        "mean_gradient_norm": (
            sum(float(row.get("gradient_norm", 0.0)) for row in train_rows)
            / len(train_rows)
            if train_rows
            else None
        ),
        "mean_completion_length": (
            sum(float(row.get("completion_len", 0.0)) for row in train_rows)
            / len(train_rows)
            if train_rows
            else None
        ),
        "mean_output_entropy": (
            sum(float(row.get("output_entropy", 0.0)) for row in train_rows)
            / len(train_rows)
            if train_rows
            else None
        ),
        "mean_teacher_residual_mass": (
            sum(float(row.get("teacher_residual_mass", 0.0)) for row in train_rows)
            / len(train_rows)
            if train_rows
            else None
        ),
        "mean_teacher_student_agreement": (
            sum(float(row.get("teacher_student_agreement", 0.0)) for row in train_rows)
            / len(train_rows)
            if train_rows
            else None
        ),
        "mean_adapter_sync_ms": (
            sum(float(row.get("rollout/sync_ms", 0.0)) for row in train_rows)
            / len(train_rows)
            if train_rows
            else None
        ),
        "policy_version_mismatch_count": sum(
            int(row.get("rollout/staleness", 0) != 0) for row in train_rows
        ),
        "evaluation_tasks": evaluation.get("tasks"),
        "evaluation_dataset": evaluation.get("dataset"),
        "pass_at_1_ci95_low": pass_ci_low,
        "pass_at_1_ci95_high": pass_ci_high,
        **{
            key: evaluation.get(key)
            for key in (
                "pass_at_1",
                "test_pass_rate",
                "syntax_success_rate",
                "runtime_success_rate",
                "task_completion_rate",
                "mean_tests_passed_per_task",
                "normalized_ast_uniqueness",
                "exact_duplicate_rate",
                "mean_solution_length_chars",
                "mean_solution_length_lines",
                "normalized_teacher_token_edit_distance",
                "teacher_edit_distance_matches",
            )
        },
        "pass_at_1_per_teacher_gpu_hour": (
            pass_at_1 / teacher_gpu_hours
            if pass_at_1 is not None and teacher_gpu_hours > 0
            else None
        ),
        "pass_at_1_per_million_anchors": (
            pass_at_1 / (anchors / 1_000_000)
            if pass_at_1 is not None and anchors > 0
            else None
        ),
    }


def pareto(rows: list[dict]) -> list[dict]:
    max_tasks = max(
        (int(row.get("evaluation_tasks") or 0) for row in rows),
        default=0,
    )
    candidates = [
        row
        for row in rows
        if row.get("pass_at_1") is not None
        and row.get("teacher_gpu_hours") is not None
        and int(row.get("evaluation_tasks") or 0) == max_tasks
    ]
    output = []
    for row in candidates:
        dominated = any(
            other["pass_at_1"] >= row["pass_at_1"]
            and other["teacher_gpu_hours"] <= row["teacher_gpu_hours"]
            and (
                other["pass_at_1"] > row["pass_at_1"]
                or other["teacher_gpu_hours"] < row["teacher_gpu_hours"]
            )
            for other in candidates
        )
        if not dominated:
            output.append(row)
    return sorted(output, key=lambda row: (row["teacher_gpu_hours"], -row["pass_at_1"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="/opt/sparse-opd/runs")
    parser.add_argument("--results-dir", default="/opt/sparse-opd/results")
    args = parser.parse_args()
    run_root = Path(args.runs_root)
    results = Path(args.results_dir)
    run_dirs = sorted({path.parent for path in run_root.rglob("config_resolved.json")})
    rows = [collect(path) for path in run_dirs]
    rows.sort(key=lambda row: (row["condition"], row["seed"], row["run_dir"]))
    frontier = pareto(rows)
    results.mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_csv(results / "summary.csv", rows)
    write_csv(results / "pareto_quality_cost.csv", frontier)
    print(
        json.dumps(
            {
                "runs": len(rows),
                "evaluated_runs": sum(row["pass_at_1"] is not None for row in rows),
                "pareto_runs": len(frontier),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

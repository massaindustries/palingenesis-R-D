#!/usr/bin/env python3
"""Summarize and validate the frozen guided-tutoring training runs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

EXPECTED_CHECKPOINTS = ("step_5", "step_10", "step_15", "final")


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def summarize(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.jsonl"
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    step_rows = [row for row in rows if "teacher_token_nll_delta" in row]
    checkpoints = [name for name in EXPECTED_CHECKPOINTS if (run_dir / name).is_dir()]
    summary = {
        "run_dir": str(run_dir.resolve()),
        "optimizer_updates": len(step_rows),
        "steps": [int(row["step"]) for row in step_rows],
        "checkpoints": checkpoints,
        "teacher_token_nll_delta_mean": mean(step_rows, "teacher_token_nll_delta"),
        "teacher_token_nll_decreased_updates": sum(
            float(row["teacher_token_nll_delta"]) < 0 for row in step_rows
        ),
        "teacher_inserted_tokens": sum(
            int(row["teacher_inserted_tokens"]) for row in step_rows
        ),
        "teacher_guidance_groups": sum(
            int(row["teacher_guidance_groups"]) for row in step_rows
        ),
        "teacher_requests": sum(int(row["teacher_requests"]) for row in step_rows),
        "teacher_wall_clock_seconds": sum(
            float(row["teacher_wall_clock_ms"]) for row in step_rows
        )
        / 1_000,
        "student_generated_tokens": sum(
            int(row["student_generated_tokens"]) for row in step_rows
        ),
        "trajectory_tokens": sum(int(row["trajectory_tokens"]) for row in step_rows),
        "teacher_intervention_coverage_mean": mean(
            step_rows, "teacher_intervention_coverage"
        ),
        "teacher_intervention_coverage_first": float(
            step_rows[0]["teacher_intervention_coverage"]
        ),
        "teacher_intervention_coverage_last": float(
            step_rows[-1]["teacher_intervention_coverage"]
        ),
        "teacher_request_success_rate_min": min(
            float(row["teacher_request_success_rate"]) for row in step_rows
        ),
        "teacher_circuit_breaker_open_max": max(
            float(row["teacher_circuit_breaker_open"]) for row in step_rows
        ),
        "teacher_group_nonempty_rate_min": min(
            float(row["teacher_group_nonempty_rate"]) for row in step_rows
        ),
        "rollout_staleness_max": max(
            int(row["rollout/staleness"]) for row in step_rows
        ),
        "empty_microstep_retries": sum(
            int(row["empty_microstep_retries"]) for row in step_rows
        ),
        "rejected_no_teacher_rollouts": sum(
            int(row["rejected_no_teacher_rollouts"]) for row in step_rows
        ),
        "gradient_norm_mean": mean(step_rows, "gradient_norm"),
        "gradient_norm_max": max(float(row["gradient_norm"]) for row in step_rows),
        "completion_len_mean": mean(step_rows, "completion_len"),
        "completion_len_first": float(step_rows[0]["completion_len"]),
        "completion_len_last": float(step_rows[-1]["completion_len"]),
        "wall_clock_seconds": max(
            float(row["run/wall_clock_seconds"]) for row in step_rows
        ),
    }
    # Two teacher GPUs plus one train and one rollout GPU are reserved for the
    # duration of each run. This is allocation time, not a utilization claim.
    summary["estimated_allocated_gpu_hours"] = (
        4 * summary["wall_clock_seconds"] / 3_600
    )
    summary["teacher_token_nll_decreased_fraction"] = (
        summary["teacher_token_nll_decreased_updates"]
        / summary["optimizer_updates"]
    )
    summary["status"] = (
        "pass"
        if summary["optimizer_updates"] == 20
        and summary["steps"] == list(range(20))
        and checkpoints == list(EXPECTED_CHECKPOINTS)
        and summary["teacher_request_success_rate_min"] >= 0.99
        and summary["teacher_circuit_breaker_open_max"] == 0.0
        and summary["teacher_group_nonempty_rate_min"] >= 0.95
        and summary["rollout_staleness_max"] == 0
        and summary["teacher_token_nll_decreased_fraction"] > 0.60
        and summary["teacher_token_nll_delta_mean"] < 0
        else "fail"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summaries = [
        summarize(run_dir)
        for run_dir in sorted(args.run_root.glob("i*_seed*"))
        if (run_dir / "metrics.jsonl").is_file()
    ]
    report = {
        "run_root": str(args.run_root.resolve()),
        "runs": summaries,
        "runs_found": len(summaries),
        "runs_passed": sum(row["status"] == "pass" for row in summaries),
        "status": (
            "pass"
            if len(summaries) == 9 and all(row["status"] == "pass" for row in summaries)
            else "incomplete"
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

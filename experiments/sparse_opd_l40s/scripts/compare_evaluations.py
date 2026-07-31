#!/usr/bin/env python3
"""Paired quality comparison for checkpoints evaluated on identical MBPP tasks."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def read_records(evaluation_dir: Path) -> dict[str, dict]:
    return {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (evaluation_dir / "evaluation_records.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        )
    }


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))]


def mcnemar_exact(gains: int, losses: int) -> float:
    discordant = gains + losses
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    base_dir = Path(args.base)
    candidate_dir = Path(args.candidate)
    base = read_records(base_dir)
    candidate = read_records(candidate_dir)
    if base.keys() != candidate.keys():
        raise ValueError("evaluation task IDs differ")
    ordered_ids = sorted(base)
    for task_id in ordered_ids:
        if base[task_id]["prompt_hash"] != candidate[task_id]["prompt_hash"]:
            raise ValueError(f"prompt hash differs for task {task_id}")
    paired = [
        (
            int(bool(base[task_id]["sandbox"]["test_pass"])),
            int(bool(candidate[task_id]["sandbox"]["test_pass"])),
        )
        for task_id in ordered_ids
    ]
    differences = [right - left for left, right in paired]
    gains = sum(left == 0 and right == 1 for left, right in paired)
    losses = sum(left == 1 and right == 0 for left, right in paired)
    ties_pass = sum(left == 1 and right == 1 for left, right in paired)
    ties_fail = sum(left == 0 and right == 0 for left, right in paired)
    rng = random.Random(args.seed)
    bootstrap = [
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(args.bootstrap_samples)
    ]
    base_rate = sum(left for left, _ in paired) / len(paired)
    candidate_rate = sum(right for _, right in paired) / len(paired)
    payload = {
        "base_evaluation": str(base_dir),
        "candidate_evaluation": str(candidate_dir),
        "tasks": len(paired),
        "base_pass_at_1": base_rate,
        "candidate_pass_at_1": candidate_rate,
        "absolute_pass_at_1_change": candidate_rate - base_rate,
        "relative_pass_at_1_change": (
            (candidate_rate - base_rate) / base_rate if base_rate else None
        ),
        "paired_bootstrap_ci95": [
            quantile(bootstrap, 0.025),
            quantile(bootstrap, 0.975),
        ],
        "mcnemar": {
            "candidate_only_passes": gains,
            "base_only_passes": losses,
            "both_pass": ties_pass,
            "both_fail": ties_fail,
            "two_sided_exact_p": mcnemar_exact(gains, losses),
        },
        "quality_collapse_kill_switch_triggered": (
            base_rate > 0 and candidate_rate < 0.70 * base_rate
        ),
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.seed,
            "unit": "task-level paired resampling",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

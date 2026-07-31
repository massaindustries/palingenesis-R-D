#!/usr/bin/env python3
"""Statistical analysis for exact paired causal-probe records."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    args = parser.parse_args()

    rows = [
        json.loads(line) for line in args.metrics.read_text().splitlines() if line
    ]
    advantages = [
        row["shuffled_teacher_nll_delta"] - row["matched_teacher_nll_delta"]
        for row in rows
    ]
    rng = random.Random(0)
    bootstrap = sorted(
        statistics.mean(rng.choice(advantages) for _ in advantages)
        for _ in range(args.bootstrap_samples)
    )
    lower = bootstrap[int(0.025 * len(bootstrap))]
    upper = bootstrap[int(0.975 * len(bootstrap)) - 1]
    positive = sum(value > 0 for value in advantages)
    negative = sum(value < 0 for value in advantages)
    non_ties = positive + negative
    extreme = max(positive, negative)
    sign_tail = sum(
        math.comb(non_ties, count)
        for count in range(extreme, non_ties + 1)
    ) / (2**non_ties)
    summary = {
        "status": "pass",
        "trials": len(rows),
        "matched_better_trials": positive,
        "shuffled_better_trials": negative,
        "ties": len(rows) - non_ties,
        "shuffled_minus_matched_delta_mean": statistics.mean(advantages),
        "paired_bootstrap_ci95": [lower, upper],
        "exact_two_sided_sign_test_p": min(1.0, 2 * sign_tail),
        "matched_delta_mean": statistics.mean(
            row["matched_teacher_nll_delta"] for row in rows
        ),
        "shuffled_delta_mean": statistics.mean(
            row["shuffled_teacher_nll_delta"] for row in rows
        ),
        "matched_grad_norm_mean": statistics.mean(
            row["matched_grad_norm"] for row in rows
        ),
        "shuffled_grad_norm_mean": statistics.mean(
            row["shuffled_grad_norm"] for row in rows
        ),
    }
    summary["shuffled_to_matched_grad_norm_ratio"] = (
        summary["shuffled_grad_norm_mean"] / summary["matched_grad_norm_mean"]
    )
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

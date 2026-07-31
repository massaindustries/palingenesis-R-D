#!/usr/bin/env python3
"""Task-paired and seed-level analysis for the frozen test evaluation."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from pathlib import Path

INTERVAL_PATTERN = re.compile(r"^i(8|32|128)_seed([012])$")


def exact_mcnemar(fixes: int, regressions: int) -> float:
    discordant = fixes + regressions
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, count)
        for count in range(min(fixes, regressions) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def paired_bootstrap(
    candidate: list[bool],
    baseline: list[bool],
    *,
    samples: int,
    seed: int = 0,
) -> list[float]:
    rng = random.Random(seed)
    count = len(candidate)
    draws = sorted(
        statistics.mean(
            int(candidate[index]) - int(baseline[index])
            for index in (rng.randrange(count) for _ in range(count))
        )
        for _ in range(samples)
    )
    return [draws[int(0.025 * samples)], draws[int(0.975 * samples)]]


def load_outcomes(path: Path) -> tuple[list[str], list[bool]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return (
        [row["id"] for row in rows],
        [bool(row["sandbox"]["test_pass"]) for row in rows],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()

    base_ids, base = load_outcomes(
        args.evaluation_root / "base_student" / "records.jsonl"
    )
    base_summary = json.loads(
        (args.evaluation_root / "base_student" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    base_rate = statistics.mean(base)
    base_max_length_rate = float(base_summary["max_length_rate"])
    conditions = {}
    interval_rates: dict[int, list[float]] = {8: [], 32: [], 128: []}
    for directory in sorted(args.evaluation_root.iterdir()):
        records = directory / "records.jsonl"
        if not records.is_file():
            continue
        task_ids, outcomes = load_outcomes(records)
        if task_ids != base_ids:
            raise ValueError(f"task order mismatch for {directory.name}")
        fixes = sum(candidate and not baseline for candidate, baseline in zip(outcomes, base))
        regressions = sum(
            baseline and not candidate for candidate, baseline in zip(outcomes, base)
        )
        rate = statistics.mean(outcomes)
        condition_summary = json.loads(
            (directory / "summary.json").read_text(encoding="utf-8")
        )
        paired_p = exact_mcnemar(fixes, regressions)
        relative_quality_change = (
            (rate - base_rate) / base_rate if base_rate else 0.0
        )
        max_length_rate = float(condition_summary["max_length_rate"])
        credible_gain = rate > base_rate and paired_p < 0.05
        kill_switch_reasons = []
        if relative_quality_change < -0.30:
            kill_switch_reasons.append("relative_quality_collapse_gt_30pct")
        if (
            max_length_rate - base_max_length_rate > 0.10
            and not credible_gain
        ):
            kill_switch_reasons.append(
                "max_length_increase_gt_10pp_without_credible_gain"
            )
        conditions[directory.name] = {
            "tasks": len(outcomes),
            "pass_at_1": rate,
            "passes": sum(outcomes),
            "delta_vs_base": rate - base_rate,
            "relative_quality_change_vs_base": relative_quality_change,
            "fixes_vs_base": fixes,
            "regressions_vs_base": regressions,
            "exact_mcnemar_p_vs_base": paired_p,
            "paired_bootstrap_ci95_delta_vs_base": paired_bootstrap(
                outcomes,
                base,
                samples=args.bootstrap_samples,
            ),
            "max_length_rate": max_length_rate,
            "max_length_rate_increase_vs_base": (
                max_length_rate - base_max_length_rate
            ),
            "kill_switch_reasons": kill_switch_reasons,
            "passes_kill_switches": not kill_switch_reasons,
        }
        match = INTERVAL_PATTERN.fullmatch(directory.name)
        if match:
            interval_rates[int(match.group(1))].append(rate)

    seed_summary = {}
    for interval, rates in interval_rates.items():
        if len(rates) != 3:
            raise ValueError(f"interval {interval} has {len(rates)} seeds, expected 3")
        seed_summary[str(interval)] = {
            "seeds": len(rates),
            "mean_pass_at_1": statistics.mean(rates),
            "sample_std_pass_at_1": statistics.stdev(rates),
            "min_pass_at_1": min(rates),
            "max_pass_at_1": max(rates),
        }

    selected_kill_switch_failures = {
        name: row["kill_switch_reasons"]
        for name, row in conditions.items()
        if INTERVAL_PATTERN.fullmatch(name) and row["kill_switch_reasons"]
    }
    report = {
        "status": "pass" if not selected_kill_switch_failures else "fail",
        "evaluation_root": str(args.evaluation_root.resolve()),
        "bootstrap_samples": args.bootstrap_samples,
        "conditions": conditions,
        "interval_seed_summary": seed_summary,
        "selected_kill_switch_failures": selected_kill_switch_failures,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

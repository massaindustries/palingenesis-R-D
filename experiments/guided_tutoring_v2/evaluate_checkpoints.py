#!/usr/bin/env python3
"""Evaluate the frozen checkpoint grid on dev and select without test leakage."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

CHECKPOINTS = ("step_5", "step_10", "step_15", "final")
RUN_PATTERN = re.compile(r"^i(8|32|128)_seed([012])$")


def exact_mcnemar(fixes: int, regressions: int) -> float:
    discordant = fixes + regressions
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, count)
        for count in range(min(fixes, regressions) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def outcomes(path: Path) -> tuple[list[str], list[bool]]:
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
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-evaluation", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    evaluator = Path(__file__).with_name("evaluate_bcb.py")
    base_summary = json.loads(
        (args.base_evaluation / "summary.json").read_text(encoding="utf-8")
    )
    base_ids, base_outcomes = outcomes(args.base_evaluation / "records.jsonl")
    base_rate = float(base_summary["pass_at_1"])
    base_max_length_rate = float(base_summary["max_length_rate"])
    selections = []
    for run_dir in sorted(args.run_root.iterdir()):
        match = RUN_PATTERN.fullmatch(run_dir.name)
        if match is None:
            continue
        interval, seed = map(int, match.groups())
        candidates = []
        for order, checkpoint_name in enumerate(CHECKPOINTS):
            checkpoint = run_dir / checkpoint_name
            if not checkpoint.is_dir():
                raise FileNotFoundError(checkpoint)
            output_dir = args.output_root / run_dir.name / checkpoint_name
            summary_path = output_dir / "summary.json"
            if args.force or not summary_path.is_file():
                output_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        sys.executable,
                        str(evaluator),
                        "--kind",
                        "student",
                        "--dataset",
                        str(args.dataset),
                        "--output-dir",
                        str(output_dir),
                        "--checkpoint",
                        str(checkpoint),
                        "--device",
                        args.device,
                    ],
                    check=True,
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            task_ids, candidate_outcomes = outcomes(output_dir / "records.jsonl")
            if task_ids != base_ids:
                raise ValueError(f"task order mismatch for {output_dir}")
            fixes = sum(
                candidate and not base
                for candidate, base in zip(candidate_outcomes, base_outcomes)
            )
            regressions = sum(
                base and not candidate
                for candidate, base in zip(candidate_outcomes, base_outcomes)
            )
            paired_p = exact_mcnemar(fixes, regressions)
            credible_gain = (
                float(summary["pass_at_1"]) > base_rate and paired_p < 0.05
            )
            relative_quality_change = (
                (float(summary["pass_at_1"]) - base_rate) / base_rate
                if base_rate
                else 0.0
            )
            excessive_truncation = (
                float(summary["max_length_rate"]) - base_max_length_rate > 0.10
            )
            kill_switch_reasons = []
            if relative_quality_change < -0.30:
                kill_switch_reasons.append("relative_quality_collapse_gt_30pct")
            if excessive_truncation and not credible_gain:
                kill_switch_reasons.append(
                    "max_length_increase_gt_10pp_without_credible_gain"
                )
            candidates.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "checkpoint": str(checkpoint.resolve()),
                    "selection_order": order,
                    "fixes_vs_base": fixes,
                    "regressions_vs_base": regressions,
                    "exact_mcnemar_p_vs_base": paired_p,
                    "relative_quality_change_vs_base": relative_quality_change,
                    "max_length_rate_increase_vs_base": (
                        float(summary["max_length_rate"]) - base_max_length_rate
                    ),
                    "kill_switch_reasons": kill_switch_reasons,
                    "eligible": not kill_switch_reasons,
                    **summary,
                }
            )

        eligible = [row for row in candidates if row["eligible"]]
        if not eligible:
            raise RuntimeError(f"all checkpoints tripped kill switches for {run_dir}")
        # Higher pass@1 wins. A tie prefers the earlier registered checkpoint,
        # then lower truncation, exactly as frozen in scientific_budget.md.
        selected = min(
            eligible,
            key=lambda row: (
                -float(row["pass_at_1"]),
                int(row["selection_order"]),
                float(row["max_length_rate"]),
            ),
        )
        selections.append(
            {
                "interval": interval,
                "seed": seed,
                "selected_checkpoint_name": selected["checkpoint_name"],
                "selected_checkpoint": selected["checkpoint"],
                "dev_pass_at_1": selected["pass_at_1"],
                "dev_max_length_rate": selected["max_length_rate"],
                "candidates": candidates,
            }
        )

    report = {
        "status": "pass" if len(selections) == 9 else "incomplete",
        "run_root": str(args.run_root.resolve()),
        "dataset": str(args.dataset.resolve()),
        "base_evaluation": str(args.base_evaluation.resolve()),
        "runs_selected": len(selections),
        "selections": selections,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output_root / "selection.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

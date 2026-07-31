#!/usr/bin/env python3
"""Validate and rank the pre-pilot learning-rate mini-sweep."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[max(0, index)]


def summarize(run_dir: Path) -> dict:
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]
    steps = [row for row in rows if "kl" in row and "gradient_norm" in row]
    final = next(
        (row for row in reversed(rows) if row.get("event") == "final_eval"), {}
    )
    config = json.loads((run_dir / "config_resolved.json").read_text())
    finite = all(
        math.isfinite(float(row[key]))
        for row in steps
        for key in ("kl", "gradient_norm", "residual_mass")
    )
    gradients = [abs(float(row["gradient_norm"])) for row in steps]
    losses = [float(row["kl"]) for row in steps]
    residuals = [max(0.0, float(row["residual_mass"])) for row in steps]
    tail = losses[len(losses) // 2 :]
    result = {
        "run_dir": str(run_dir),
        "learning_rate": float(config["train"]["learning_rate"]),
        "completed_steps": len(steps),
        "expected_steps": int(config["train"]["steps"]),
        "finite": finite,
        "max_staleness": max(
            (int(row.get("rollout/staleness", 0)) for row in steps), default=0
        ),
        "mean_residual_mass": statistics.fmean(residuals),
        "p95_gradient_norm": percentile(gradients, 0.95),
        "tail_kl_stddev": statistics.pstdev(tail) if len(tail) > 1 else 0.0,
        "final_dev_kl": final.get("dev_kl"),
        "final_dev_completion_length": final.get("dev_len"),
    }
    result["eligible"] = (
        result["completed_steps"] == result["expected_steps"]
        and result["finite"]
        and result["max_staleness"] == 0
        and result["mean_residual_mass"] <= 0.20
        and result["p95_gradient_norm"] <= 100.0
        and result["final_dev_kl"] is not None
    )
    # Predeclared stability score: gradient excursions dominate, with smaller
    # penalties for noisy KL and approximation residual. Lower is more stable.
    result["stability_score"] = (
        result["p95_gradient_norm"]
        + 2.0 * result["tail_kl_stddev"]
        + 10.0 * result["mean_residual_mass"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite")
    args = parser.parse_args()
    suite = Path(args.suite)
    results = [
        summarize(path)
        for path in sorted(suite.glob("lr_*"))
        if (path / "metrics.jsonl").is_file()
    ]
    if not results:
        raise SystemExit("no completed LR runs found")
    eligible = [result for result in results if result["eligible"]]
    if not eligible:
        raise SystemExit("no LR candidate passed the stability gates")
    selected = min(
        eligible,
        key=lambda result: (result["stability_score"], result["learning_rate"]),
    )
    payload = {
        "selection_rule": (
            "Among complete, finite, zero-staleness runs with mean residual <=0.20, "
            "p95 gradient norm <=100, and a final dev evaluation, minimize "
            "p95_gradient_norm + 2*tail_kl_stddev + 10*mean_residual_mass."
        ),
        "selected_learning_rate": selected["learning_rate"],
        "runs": results,
    }
    (suite / "lr_sweep_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (suite / "selected_lr.txt").write_text(f"{selected['learning_rate']:.12g}\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

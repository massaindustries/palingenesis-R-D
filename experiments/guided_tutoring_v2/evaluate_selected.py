#!/usr/bin/env python3
"""Evaluate only dev-selected checkpoints and frozen baselines on test."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def evaluate(
    *,
    evaluator: Path,
    dataset: Path,
    output_dir: Path,
    kind: str,
    checkpoint: str = "",
    device: str,
    force: bool,
) -> dict:
    summary_path = output_dir / "summary.json"
    if force or not summary_path.is_file():
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(evaluator),
            "--kind",
            kind,
            "--dataset",
            str(dataset),
            "--output-dir",
            str(output_dir),
            "--device",
            device,
        ]
        if checkpoint:
            command.extend(["--checkpoint", checkpoint])
        subprocess.run(command, check=True)
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--offline-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("status") != "pass" or len(selection.get("selections", [])) != 9:
        raise ValueError("test evaluation requires nine completed dev selections")

    evaluator = Path(__file__).with_name("evaluate_bcb.py")
    conditions = [
        ("base_student", "student", ""),
        ("frozen_teacher", "teacher", ""),
        ("offline_sft", "student", str(args.offline_checkpoint)),
    ]
    conditions.extend(
        (
            f"i{row['interval']}_seed{row['seed']}",
            "student",
            row["selected_checkpoint"],
        )
        for row in selection["selections"]
    )

    summaries = {}
    for name, kind, checkpoint in conditions:
        summaries[name] = evaluate(
            evaluator=evaluator,
            dataset=args.dataset,
            output_dir=args.output_root / name,
            kind=kind,
            checkpoint=checkpoint,
            device=args.device,
            force=args.force,
        )

    report = {
        "status": "pass",
        "selection": str(args.selection.resolve()),
        "dataset": str(args.dataset.resolve()),
        "conditions": summaries,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output_root / "summary.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Grade guided and paired student-only trajectories on public MBPP tests."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from importlib import import_module
from pathlib import Path

from transformers import AutoTokenizer


def extract_code(text: str) -> str:
    blocks = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def split_tests(tests: list[str]) -> tuple[list[str], list[str]]:
    imports = [
        test for test in tests if test.lstrip().startswith(("import ", "from "))
    ]
    return imports, [test for test in tests if test not in imports]


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("no provenance records to grade")
    guided_pass = sum(row["guided"]["test_pass"] for row in records)
    paired_pass = sum(row["student_only"]["test_pass"] for row in records)
    fixes = sum(
        row["guided"]["test_pass"] and not row["student_only"]["test_pass"]
        for row in records
    )
    regressions = sum(
        row["student_only"]["test_pass"] and not row["guided"]["test_pass"]
        for row in records
    )
    teacher_groups = sum(row["teacher_groups"] for row in records)
    teacher_tokens = sum(row["teacher_tokens"] for row in records)
    total = len(records)
    return {
        "status": "pass",
        "paired_trajectories": total,
        "guided_public_pass_rate": guided_pass / total,
        "student_only_public_pass_rate": paired_pass / total,
        "paired_pass_rate_difference": (guided_pass - paired_pass) / total,
        "guided_fixes": fixes,
        "guided_regressions": regressions,
        "both_pass": sum(
            row["guided"]["test_pass"] and row["student_only"]["test_pass"]
            for row in records
        ),
        "both_fail": sum(
            not row["guided"]["test_pass"] and not row["student_only"]["test_pass"]
            for row in records
        ),
        "token_sequences_changed": sum(row["token_sequences_changed"] for row in records),
        "teacher_groups": teacher_groups,
        "teacher_tokens": teacher_tokens,
        "mean_teacher_tokens_per_trajectory": teacher_tokens / total,
        "mean_teacher_group_size": teacher_tokens / max(1, teacher_groups),
        "all_sandbox_timings_finite": all(
            math.isfinite(float(side["elapsed_seconds"]))
            for row in records
            for side in (row["guided"], row["student_only"])
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", default="/opt/sparse-opd/models/Qwen3-4B"
    )
    parser.add_argument("--sandbox-dir", type=Path, default=Path("/opt/sparse-opd/sandbox"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.sandbox_dir))
    run_sandbox = import_module("run_task").run
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tasks = {
        str(row["id"]): row
        for row in (
            json.loads(line)
            for line in args.train_data.read_text().splitlines()
            if line.strip()
        )
    }
    provenance = [
        json.loads(line)
        for line in args.provenance.read_text().splitlines()
        if line.strip()
    ]
    records = []
    for index, row in enumerate(provenance):
        task_id = str(row["task_id"])
        if task_id not in tasks:
            raise ValueError(f"provenance task {task_id} is absent from training data")
        task = tasks[task_id]
        if row["prompt_hash"] != task["prompt_hash"]:
            raise ValueError(f"prompt hash mismatch for task {task_id}")
        paired_ids = row["paired_student_only_token_ids"]
        if not paired_ids:
            raise ValueError("paired student-only completion is missing")
        imports, tests = split_tests(task["public_feedback_tests"])
        sides = {}
        for name, token_ids in (
            ("guided", row["completion_token_ids"]),
            ("student_only", paired_ids),
        ):
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            code = extract_code(text)
            sides[name] = run_sandbox(
                {
                    "code": code,
                    "imports": imports,
                    "tests": tests,
                    "timeout": 10,
                }
            )
        records.append(
            {
                "record_index": index,
                "step": row["step"],
                "microstep": row.get("microstep"),
                "rollout_index": row["rollout_index"],
                "task_id": task_id,
                "prompt_hash": row["prompt_hash"],
                "teacher_groups": len(row["teacher_groups"]),
                "teacher_tokens": row["teacher_tokens"],
                "token_sequences_changed": (
                    row["completion_token_ids"] != paired_ids
                ),
                **sides,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "paired_grades.jsonl"
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = summarize(records)
    summary.update(
        {
            "provenance": str(args.provenance.resolve()),
            "train_data": str(args.train_data.resolve()),
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

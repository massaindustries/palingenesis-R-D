#!/usr/bin/env python3
"""Grade immutable teacher generations in the same isolated MBPP sandbox."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from importlib import import_module
from pathlib import Path

ROOT = Path("/opt/sparse-opd")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "sandbox"))
extract_code = import_module("analyze_evaluations").extract_code
run_sandbox = import_module("run_task").run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--completions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = {
        row["prompt_hash"]: row
        for row in (
            json.loads(line)
            for line in Path(args.dataset).read_text().splitlines()
            if line.strip()
        )
    }
    completions = [
        json.loads(line)
        for line in Path(args.completions).read_text().splitlines()
        if line.strip()
    ]
    records_path = output / "evaluation_records.jsonl"
    records = []
    started = time.perf_counter()
    for index, generated in enumerate(completions, start=1):
        row = source[generated["prompt_hash"]]
        code = extract_code(generated["teacher_completion"])
        imports = [
            test
            for test in row["tests"]
            if test.lstrip().startswith(("import ", "from "))
        ]
        tests = [test for test in row["tests"] if test not in imports]
        sandbox = run_sandbox(
            {
                "code": code,
                "imports": imports,
                "tests": tests,
                "timeout": 10,
            }
        )
        records.append(
            {
                "id": row["id"],
                "entry_point": row["entry_point"],
                "prompt_hash": row["prompt_hash"],
                "reference_hash": row["reference_hash"],
                "completion": generated["teacher_completion"],
                "code": code,
                "completion_tokens": None,
                "sandbox": sandbox,
            }
        )
        if index % 16 == 0 or index == len(completions):
            print(f"graded {index}/{len(completions)}", flush=True)
    records_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    count = len(records)
    total_tests = sum(row["sandbox"]["tests_total"] for row in records)
    passed_tests = sum(row["sandbox"]["tests_passed"] for row in records)
    pass_at_1 = sum(row["sandbox"]["test_pass"] for row in records) / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (pass_at_1 + z * z / (2 * count)) / denominator
    margin = (
        z
        * math.sqrt(pass_at_1 * (1 - pass_at_1) / count + z * z / (4 * count * count))
        / denominator
    )
    summary = {
        "tasks": count,
        "pass_at_1": pass_at_1,
        "pass_at_1_ci95_low": center - margin,
        "pass_at_1_ci95_high": center + margin,
        "test_pass_rate": passed_tests / max(1, total_tests),
        "syntax_success_rate": sum(row["sandbox"]["syntax_success"] for row in records)
        / count,
        "runtime_success_rate": sum(
            row["sandbox"]["runtime_success"] for row in records
        )
        / count,
        "task_completion_rate": sum(bool(row["code"].strip()) for row in records)
        / count,
        "mean_tests_passed_per_task": passed_tests / count,
        "sandbox_timeouts": sum(row["sandbox"]["timeout"] for row in records),
        "generation_and_evaluation_seconds": time.perf_counter() - started,
        "checkpoint": "teacher_fixed_greedy_generations",
        "dataset": args.dataset,
        "completions": args.completions,
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

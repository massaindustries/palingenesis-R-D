#!/usr/bin/env python3
"""Validate every canonical solution in the proposed BigCodeBench-Long subset."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
from grade_bcb_probe import (
    BCB_DATASET,
    BCB_REVISION,
    BCB_SPLIT,
    run_sandbox,
)
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-canonical-tokens", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image", default="guided-v2-bcb-executor")
    parser.add_argument(
        "--tokenizer", default="/opt/sparse-opd/models/Qwen3-4B"
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    stdlib = set(sys.stdlib_module_names)
    selected = []
    for row in load_dataset(
        BCB_DATASET, revision=BCB_REVISION, split=BCB_SPLIT
    ):
        libraries = set(ast.literal_eval(row["libs"]))
        tokens = len(
            tokenizer.encode(row["canonical_solution"], add_special_tokens=False)
        )
        if libraries <= stdlib and tokens >= args.minimum_canonical_tokens:
            selected.append((dict(row), tokens))

    def validate(item):
        row, tokens = item
        result = run_sandbox(
            row["code_prompt"] + row["canonical_solution"],
            row["test"],
            args.image,
        )
        return {
            "task_id": row["task_id"],
            "canonical_solution_tokens": tokens,
            "libraries": sorted(ast.literal_eval(row["libs"])),
            "result": result,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(validate, selected))
    passed_ids = [
        record["task_id"] for record in records if record["result"]["test_pass"]
    ]
    failed_ids = [
        record["task_id"] for record in records if not record["result"]["test_pass"]
    ]
    summary = {
        "status": "pass" if not failed_ids else "fail",
        "dataset": BCB_DATASET,
        "revision": BCB_REVISION,
        "split": BCB_SPLIT,
        "minimum_canonical_tokens": args.minimum_canonical_tokens,
        "stdlib_only": True,
        "selected_tasks": len(records),
        "canonical_passes": len(passed_ids),
        "canonical_failures": len(failed_ids),
        "canonical_pass_rate": len(passed_ids) / len(records),
        "passed_task_ids": passed_ids,
        "failed_task_ids": failed_ids,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

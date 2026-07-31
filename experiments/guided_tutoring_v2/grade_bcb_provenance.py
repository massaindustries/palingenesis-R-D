#!/usr/bin/env python3
"""Grade guided and paired student-only BCB trajectories."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from grade_bcb_probe import extract_code, run_sandbox
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--evaluation-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", default="/opt/sparse-opd/models/Qwen3-4B"
    )
    parser.add_argument("--image", default="guided-v2-bcb-executor")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    tasks = {
        str(row["id"]): row
        for row in (
            json.loads(line)
            for line in args.evaluation_data.read_text().splitlines()
            if line
        )
    }
    provenance = [
        json.loads(line)
        for line in args.provenance.read_text().splitlines()
        if line.strip()
    ]
    def grade(item):
        index, row = item
        task_id = str(row["task_id"])
        task = tasks[task_id]
        if row["prompt_hash"] != task["prompt_hash"]:
            raise ValueError(f"prompt hash mismatch for {task_id}")
        paired_ids = row["paired_student_only_token_ids"]
        if not paired_ids:
            raise ValueError("paired student-only completion is missing")
        sides = {}
        for name, token_ids in (
            ("guided", row["completion_token_ids"]),
            ("student_only", paired_ids),
        ):
            completion = tokenizer.decode(token_ids, skip_special_tokens=True)
            sides[name] = run_sandbox(
                extract_code(completion), task["test"], args.image
            )
        return {
            "record_index": index,
            "step": row["step"],
            "microstep": row.get("microstep"),
            "rollout_index": row["rollout_index"],
            "task_id": task_id,
            "prompt_hash": row["prompt_hash"],
            "teacher_groups": len(row["teacher_groups"]),
            "teacher_tokens": row["teacher_tokens"],
            "token_sequences_changed": row["completion_token_ids"] != paired_ids,
            **sides,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(grade, enumerate(provenance)))

    count = len(records)
    guided_pass = sum(row["guided"]["test_pass"] for row in records)
    student_pass = sum(row["student_only"]["test_pass"] for row in records)
    summary = {
        "status": "pass",
        "paired_trajectories": count,
        "guided_pass_rate": guided_pass / count,
        "student_only_pass_rate": student_pass / count,
        "paired_pass_rate_difference": (guided_pass - student_pass) / count,
        "guided_fixes": sum(
            row["guided"]["test_pass"] and not row["student_only"]["test_pass"]
            for row in records
        ),
        "guided_regressions": sum(
            row["student_only"]["test_pass"] and not row["guided"]["test_pass"]
            for row in records
        ),
        "token_sequences_changed": sum(
            row["token_sequences_changed"] for row in records
        ),
        "teacher_groups": sum(row["teacher_groups"] for row in records),
        "teacher_tokens": sum(row["teacher_tokens"] for row in records),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Grade paired student/teacher BigCodeBench probe completions."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from datasets import load_dataset

BCB_DATASET = "bigcode/bigcodebench"
BCB_REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
BCB_SPLIT = "v0.1.4"


def extract_code(text: str) -> str:
    blocks = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return (max(blocks, key=len) if blocks else text).strip()


def run_sandbox(code: str, test: str, image: str) -> dict:
    command = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "128",
        "--memory",
        "512m",
        "--cpus",
        "1",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        image,
    ]
    completed = subprocess.run(
        command,
        input=json.dumps({"code": code, "test": test, "timeout": 10}),
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr[:4096])
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="guided-v2-bcb-executor")
    args = parser.parse_args()

    dataset = {
        row["task_id"]: row
        for row in load_dataset(
            BCB_DATASET, revision=BCB_REVISION, split=BCB_SPLIT
        )
    }
    sides = {}
    for name, path in (("student", args.student), ("teacher", args.teacher)):
        sides[name] = {
            row["task_id"]: row
            for row in (
                json.loads(line) for line in path.read_text().splitlines() if line
            )
        }
    if sides["student"].keys() != sides["teacher"].keys():
        raise ValueError("student and teacher probe IDs differ")

    records = []
    for task_id in sides["student"]:
        record = {"task_id": task_id}
        for name in ("student", "teacher"):
            record[name] = run_sandbox(
                extract_code(sides[name][task_id]["completion"]),
                dataset[task_id]["test"],
                args.image,
            )
        records.append(record)

    count = len(records)
    student_pass = sum(row["student"]["test_pass"] for row in records)
    teacher_pass = sum(row["teacher"]["test_pass"] for row in records)
    summary = {
        "status": "pass",
        "dataset": BCB_DATASET,
        "revision": BCB_REVISION,
        "split": BCB_SPLIT,
        "tasks": count,
        "student_pass_rate": student_pass / count,
        "teacher_pass_rate": teacher_pass / count,
        "teacher_minus_student": (teacher_pass - student_pass) / count,
        "teacher_fixes": sum(
            row["teacher"]["test_pass"] and not row["student"]["test_pass"]
            for row in records
        ),
        "teacher_regressions": sum(
            row["student"]["test_pass"] and not row["teacher"]["test_pass"]
            for row in records
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

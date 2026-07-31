#!/usr/bin/env python3
"""Grade paired student/teacher LCB probe completions in a locked container."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

LCB_DATASET = "livecodebench/code_generation_lite"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
LCB_FILE = "test6.jsonl"


def extract_code(text: str) -> str:
    blocks = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return (max(blocks, key=len) if blocks else text).strip()


def run_sandbox(code: str, tests: list[dict], image: str) -> dict:
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
        input=json.dumps({"code": code, "tests": tests, "timeout": 10}),
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
    parser.add_argument("--image", default="guided-v2-lcb-executor")
    args = parser.parse_args()

    dataset_path = hf_hub_download(
        LCB_DATASET,
        LCB_FILE,
        repo_type="dataset",
        revision=LCB_REVISION,
    )
    dataset = {
        row["question_id"]: row
        for row in (
            json.loads(line) for line in Path(dataset_path).read_text().splitlines()
        )
    }
    sides = {}
    for name, path in (("student", args.student), ("teacher", args.teacher)):
        sides[name] = {
            row["question_id"]: row
            for row in (
                json.loads(line) for line in path.read_text().splitlines() if line
            )
        }
    if sides["student"].keys() != sides["teacher"].keys():
        raise ValueError("student and teacher probe IDs differ")

    records = []
    for question_id in sorted(
        sides["student"],
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    ):
        task = dataset[question_id]
        tests = json.loads(task["public_test_cases"])
        result = {"question_id": question_id}
        for name in ("student", "teacher"):
            completion = sides[name][question_id]["completion"]
            result[name] = run_sandbox(extract_code(completion), tests, args.image)
        records.append(result)

    count = len(records)
    student_pass = sum(record["student"]["test_pass"] for record in records)
    teacher_pass = sum(record["teacher"]["test_pass"] for record in records)
    summary = {
        "status": "pass",
        "dataset": LCB_DATASET,
        "revision": LCB_REVISION,
        "file": LCB_FILE,
        "tasks": count,
        "student_public_pass_rate": student_pass / count,
        "teacher_public_pass_rate": teacher_pass / count,
        "teacher_minus_student": (teacher_pass - student_pass) / count,
        "teacher_fixes": sum(
            record["teacher"]["test_pass"] and not record["student"]["test_pass"]
            for record in records
        ),
        "teacher_regressions": sum(
            record["student"]["test_pass"] and not record["teacher"]["test_pass"]
            for record in records
        ),
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

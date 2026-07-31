#!/usr/bin/env python3
"""Prepare leakage-controlled BigCodeBench-Long96-Stdlib splits."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET = "bigcode/bigcodebench"
REVISION = "b74c0d0bf70d2c0bc459be537895cca163007f1a"
SOURCE_SPLIT = "v0.1.4"
MINIMUM_CANONICAL_TOKENS = 96
EXPECTED_CANDIDATES = 142
EXPECTED_COMPATIBLE = 121
SPLIT_COUNTS = {"train": 60, "dev": 21, "test": 40}
SYSTEM_MESSAGE = (
    "You are a Python programming assistant. Return only a complete Python "
    "solution in one fenced code block. Do not explain the solution."
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(content, encoding="utf-8")
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "bytes": len(content.encode("utf-8")),
        "sha256": sha256_text(content),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-validation", type=Path, required=True)
    parser.add_argument("--probe-records", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tokenizer", default="/opt/sparse-opd/models/Qwen3-4B"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_bytes = args.canonical_validation.read_bytes()
    validation = json.loads(validation_bytes)
    required_validation = {
        "dataset": DATASET,
        "revision": REVISION,
        "split": SOURCE_SPLIT,
        "minimum_canonical_tokens": MINIMUM_CANONICAL_TOKENS,
        "selected_tasks": EXPECTED_CANDIDATES,
        "canonical_passes": EXPECTED_COMPATIBLE,
        "stdlib_only": True,
    }
    for key, expected in required_validation.items():
        if validation.get(key) != expected:
            raise ValueError(
                f"canonical validation {key}={validation.get(key)!r}, "
                f"expected {expected!r}"
            )
    compatible_ids = set(validation["passed_task_ids"])
    if len(compatible_ids) != EXPECTED_COMPATIBLE:
        raise ValueError("canonical-compatible task IDs are not unique")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    stdlib = set(sys.stdlib_module_names)
    source = {
        row["task_id"]: dict(row)
        for row in load_dataset(DATASET, revision=REVISION, split=SOURCE_SPLIT)
    }
    compatible = {}
    for task_id in compatible_ids:
        row = source[task_id]
        libraries = set(ast.literal_eval(row["libs"]))
        tokens = len(
            tokenizer.encode(row["canonical_solution"], add_special_tokens=False)
        )
        if not libraries <= stdlib or tokens < MINIMUM_CANONICAL_TOKENS:
            raise ValueError(f"compatible task {task_id} violates subset criteria")
        compatible[task_id] = (row, tokens, sorted(libraries))

    probe_ids: set[str] = set()
    probe_files = []
    for path in args.probe_records:
        content = path.read_bytes()
        rows = read_jsonl(path)
        ids = {str(row["task_id"]) for row in rows}
        probe_ids |= ids & compatible_ids
        probe_files.append(
            {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(content).hexdigest(),
                "rows": len(rows),
            }
        )
    if not probe_ids:
        raise ValueError("at least one compatible probed task must be quarantined")
    if len(probe_ids) > SPLIT_COUNTS["train"]:
        raise ValueError("probed task quarantine exceeds training split capacity")

    remaining = sorted(
        compatible_ids - probe_ids,
        key=lambda task_id: sha256_text(task_id),
    )
    train_fill = SPLIT_COUNTS["train"] - len(probe_ids)
    assignments = {
        "train": sorted(probe_ids, key=lambda task_id: sha256_text(task_id))
        + remaining[:train_fill],
        "dev": remaining[train_fill : train_fill + SPLIT_COUNTS["dev"]],
        "test": remaining[train_fill + SPLIT_COUNTS["dev"] :],
    }
    if {split: len(ids) for split, ids in assignments.items()} != SPLIT_COUNTS:
        raise ValueError("split counts do not match the frozen allocation")
    id_sets = {split: set(ids) for split, ids in assignments.items()}
    if (
        id_sets["train"] & id_sets["dev"]
        or id_sets["train"] & id_sets["test"]
        or id_sets["dev"] & id_sets["test"]
    ):
        raise ValueError("task-ID overlap across splits")
    if probe_ids - id_sets["train"]:
        raise ValueError("a probed task escaped the training quarantine")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    prompt_hashes = {}
    for split, task_ids in assignments.items():
        prompts = []
        evaluations = []
        for task_id in task_ids:
            row, canonical_tokens, libraries = compatible[task_id]
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": row["instruct_prompt"]},
            ]
            prompt_hash = sha256_text(
                json.dumps(messages, ensure_ascii=False, sort_keys=True)
            )
            prompt_hashes.setdefault(prompt_hash, []).append(task_id)
            base = {
                "id": task_id,
                "entry_point": row["entry_point"],
                "messages": messages,
                "prompt_hash": prompt_hash,
                "source": "bigcodebench_long96_stdlib",
                "split": split,
                "canonical_solution_tokens": canonical_tokens,
                "libraries": libraries,
            }
            prompts.append(base)
            evaluations.append(
                {
                    **base,
                    "code_prompt": row["code_prompt"],
                    "canonical_solution": row["canonical_solution"],
                    "test": row["test"],
                }
            )
        outputs[f"{split}_prompts"] = write_jsonl(
            args.output_dir / f"{split}_prompts.jsonl", prompts
        )
        outputs[f"{split}_evaluation"] = write_jsonl(
            args.output_dir / f"{split}_evaluation.jsonl", evaluations
        )
    duplicates = {
        prompt_hash: ids for prompt_hash, ids in prompt_hashes.items() if len(ids) > 1
    }
    if duplicates:
        raise ValueError(f"duplicate prompt hashes: {duplicates}")

    manifest = {
        "status": "pass",
        "dataset": DATASET,
        "revision": REVISION,
        "source_split": SOURCE_SPLIT,
        "subset": {
            "name": "BigCodeBench-Long96-Stdlib",
            "minimum_canonical_tokens": MINIMUM_CANONICAL_TOKENS,
            "stdlib_only": True,
            "candidate_tasks": EXPECTED_CANDIDATES,
            "canonical_compatible_tasks": EXPECTED_COMPATIBLE,
        },
        "canonical_validation": {
            "path": str(args.canonical_validation.resolve()),
            "sha256": hashlib.sha256(validation_bytes).hexdigest(),
        },
        "probe_quarantine": {
            "task_ids": sorted(probe_ids),
            "count": len(probe_ids),
            "all_in_train": True,
            "source_files": probe_files,
        },
        "splits": {split: len(ids) for split, ids in assignments.items()},
        "task_id_overlap": 0,
        "prompt_hash_overlap": 0,
        "tests_present_in_prompt_files": False,
        "outputs": outputs,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

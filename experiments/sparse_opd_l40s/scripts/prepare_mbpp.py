#!/usr/bin/env python3
"""Prepare deterministic, leakage-checked MBPP sanitized experiment splits."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

ROOT = Path("/opt/sparse-opd")
SOURCE = "google-research-datasets/mbpp"
CONFIG = "sanitized"
SYSTEM = (
    "You are a Python programming assistant. Return only a complete Python "
    "solution in one fenced code block. Do not explain the solution."
)
PROMPT_TEMPLATE_VERSION = "mbpp_function_signature_v2"


def digest(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def entry_point(code: str) -> str:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    raise ValueError("reference solution has no top-level function")


def function_signature(code: str) -> tuple[str, str]:
    """Return the public callable interface without exposing its implementation."""
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            return node.name, f"{prefix} {node.name}({ast.unparse(node.args)}):"
    raise ValueError("reference solution has no top-level function")


def convert(row: dict, split: str) -> dict:
    original_prompt = row["prompt"].strip()
    name, signature = function_signature(row["code"])
    prompt = (
        f"{original_prompt}\n\n"
        "Implement the callable with exactly this required signature:\n"
        f"```python\n{signature}\n```"
    )
    tests = [*row["test_imports"], *row["test_list"]]
    return {
        "id": str(row["task_id"]),
        "prompt": prompt,
        "original_prompt": original_prompt,
        "entry_point": name,
        "tests": tests,
        "split": split,
        "source": "mbpp_sanitized",
        "prompt_hash": digest(prompt),
        "reference_hash": digest(row["code"]),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    }


def unique(rows: list[dict]) -> list[dict]:
    by_prompt = {}
    reference_hashes = set()
    for row in rows:
        if row["prompt_hash"] in by_prompt or row["reference_hash"] in reference_hashes:
            continue
        by_prompt[row["prompt_hash"]] = row
        reference_hashes.add(row["reference_hash"])
    return list(by_prompt.values())


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    content = path.read_bytes()
    return {
        "path": str(path),
        "rows": len(rows),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def main() -> None:
    dataset = load_dataset(SOURCE, CONFIG)
    converted = {
        "train": unique([convert(row, "train") for row in dataset["train"]]),
        "dev": unique([convert(row, "dev") for row in dataset["validation"]]),
        "test": unique([convert(row, "test") for row in dataset["test"]]),
    }
    original_counts = {name: len(rows) for name, rows in converted.items()}
    # Preserve the official test split whenever sanitized MBPP repeats a task
    # across published partitions. Dev has priority over train.
    test_keys = {
        (row["prompt_hash"], row["reference_hash"]) for row in converted["test"]
    }
    test_prompts = {prompt for prompt, _ in test_keys}
    test_references = {reference for _, reference in test_keys}
    converted["dev"] = [
        row
        for row in converted["dev"]
        if row["prompt_hash"] not in test_prompts
        and row["reference_hash"] not in test_references
    ]
    dev_prompts = {row["prompt_hash"] for row in converted["dev"]}
    dev_references = {row["reference_hash"] for row in converted["dev"]}
    converted["train"] = [
        row
        for row in converted["train"]
        if row["prompt_hash"] not in test_prompts | dev_prompts
        and row["reference_hash"] not in test_references | dev_references
    ]
    train_hashes = {row["prompt_hash"] for row in converted["train"]}
    dev_hashes = {row["prompt_hash"] for row in converted["dev"]}
    test_hashes = {row["prompt_hash"] for row in converted["test"]}
    if (
        train_hashes & dev_hashes
        or train_hashes & test_hashes
        or dev_hashes & test_hashes
    ):
        raise RuntimeError("prompt leakage detected across official splits")

    for rows in converted.values():
        rows.sort(key=lambda row: (row["prompt_hash"], row["id"]))

    split_dir = ROOT / "data/splits"
    files = {}
    files["smoke_train"] = write_jsonl(
        split_dir / "smoke_train.jsonl", converted["train"][:64]
    )
    files["smoke_dev"] = write_jsonl(
        split_dir / "smoke_dev.jsonl", converted["dev"][:32]
    )
    files["smoke_test"] = write_jsonl(
        split_dir / "smoke_test.jsonl", converted["test"][:32]
    )
    files["pilot_train"] = write_jsonl(
        split_dir / "pilot_train.jsonl", converted["train"][:700]
    )
    files["pilot_dev"] = write_jsonl(split_dir / "pilot_dev.jsonl", converted["dev"])
    files["pilot_test"] = write_jsonl(split_dir / "pilot_test.jsonl", converted["test"])

    revision = HfApi().dataset_info(SOURCE).sha
    manifest = {
        "source": SOURCE,
        "config": CONFIG,
        "revision": revision,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_template": (
            "Original sanitized MBPP task plus the exact top-level callable signature "
            "derived from the reference AST; no implementation or tests are exposed."
        ),
        "system_message": SYSTEM,
        "selection": "lexicographic prompt SHA-256; seed-independent deterministic ordering",
        "leakage_check": "pass",
        "official_counts_after_dedup": {
            name: len(rows) for name, rows in converted.items()
        },
        "original_official_counts": original_counts,
        "cross_split_rows_removed": {
            name: original_counts[name] - len(converted[name]) for name in converted
        },
        "files": files,
    }
    manifest_path = ROOT / "data/splits/manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

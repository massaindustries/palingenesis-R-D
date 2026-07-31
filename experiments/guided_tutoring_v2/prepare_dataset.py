#!/usr/bin/env python3
"""Build and validate the leakage-controlled Guided Tutoring v2 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset

MBPPPLUS_DATASET = "evalplus/mbppplus"
MBPPPLUS_REVISION = "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"
EXPECTED_COUNTS = {"train": 120, "dev": 43, "test": 257}
EXPECTED_PLUS_INTERSECTION = 224


def canonical_id(value: object) -> str:
    """Return an integer-like MBPP task ID without a dataset prefix."""
    text = str(value)
    if "/" in text:
        text = text.rsplit("/", 1)[-1]
    return str(int(text))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(row)
    return rows


def encode_jsonl(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in rows
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    content = encode_jsonl(rows)
    path.write_bytes(content)
    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "bytes": len(content),
        "sha256": sha256_bytes(content),
    }


def require_unique(rows: list[dict[str, Any]], split: str) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    prompt_hashes: set[str] = set()
    for row in rows:
        task_id = canonical_id(row["id"])
        prompt_hash = str(row["prompt_hash"])
        if task_id in by_id:
            raise ValueError(f"duplicate task ID {task_id} in {split}")
        if prompt_hash in prompt_hashes:
            raise ValueError(f"duplicate prompt hash {prompt_hash} in {split}")
        if row.get("split") != split:
            raise ValueError(
                f"task {task_id}: row split {row.get('split')!r} != {split!r}"
            )
        by_id[task_id] = row
        prompt_hashes.add(prompt_hash)
    return by_id


def assert_disjoint(
    split_maps: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, int]]:
    overlaps: dict[str, dict[str, int]] = {}
    names = list(split_maps)
    for index, left in enumerate(names):
        overlaps[left] = {}
        for right in names[index + 1 :]:
            id_overlap = set(split_maps[left]) & set(split_maps[right])
            left_hashes = {row["prompt_hash"] for row in split_maps[left].values()}
            right_hashes = {row["prompt_hash"] for row in split_maps[right].values()}
            hash_overlap = left_hashes & right_hashes
            overlaps[left][right] = len(id_overlap) + len(hash_overlap)
            if id_overlap or hash_overlap:
                raise ValueError(
                    f"{left}/{right} leakage: IDs={sorted(id_overlap)}, "
                    f"prompt hashes={sorted(hash_overlap)}"
                )
    return overlaps


def sanitized_learning_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep original public tests but make their feedback-only status explicit."""
    allowed = {
        "entry_point",
        "id",
        "messages",
        "original_prompt",
        "prompt",
        "prompt_hash",
        "reference_hash",
        "source",
        "split",
    }
    clean = {key: row[key] for key in allowed if key in row}
    clean["public_feedback_tests"] = list(row.get("tests", []))
    return clean


def plus_eval_row(
    source_row: dict[str, Any], plus_row: dict[str, Any]
) -> dict[str, Any]:
    return {
        "entry_point": source_row["entry_point"],
        "id": canonical_id(source_row["id"]),
        "messages": source_row["messages"],
        "original_prompt": source_row["original_prompt"],
        "prompt": source_row["prompt"],
        "prompt_hash": source_row["prompt_hash"],
        "reference_hash": source_row["reference_hash"],
        "source": source_row["source"],
        "split": "test",
        "public_tests": list(source_row.get("tests", [])),
        "evalplus": {
            "dataset": MBPPPLUS_DATASET,
            "revision": MBPPPLUS_REVISION,
            "source_file": plus_row["source_file"],
            "test_imports": list(plus_row["test_imports"]),
            "test_list": list(plus_row["test_list"]),
            "test": plus_row["test"],
            "canonical_solution_sha256": sha256_text(plus_row["code"]),
        },
    }


def contains_augmented_test(
    learning_rows: list[dict[str, Any]], plus_rows: Iterable[dict[str, Any]]
) -> bool:
    visible = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) for row in learning_rows
    )
    for row in plus_rows:
        augmented = str(row.get("test", ""))
        if augmented and augmented in visible:
            return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir",
        type=Path,
        required=True,
        help="Directory containing pilot_train/dev/test.jsonl prompt-v2 files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_files = {
        split: args.splits_dir / f"pilot_{split}.jsonl"
        for split in ("train", "dev", "test")
    }
    source_rows = {split: load_jsonl(path) for split, path in source_files.items()}
    counts = {split: len(rows) for split, rows in source_rows.items()}
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"unexpected source counts: {counts} != {EXPECTED_COUNTS}")

    split_maps = {
        split: require_unique(rows, split) for split, rows in source_rows.items()
    }
    overlaps = assert_disjoint(split_maps)

    downloaded = load_dataset(
        MBPPPLUS_DATASET, revision=MBPPPLUS_REVISION, split="test"
    )
    plus_by_id = {canonical_id(row["task_id"]): dict(row) for row in downloaded}
    selected_ids = sorted(set(split_maps["test"]) & set(plus_by_id), key=int)
    if len(selected_ids) != EXPECTED_PLUS_INTERSECTION:
        raise ValueError(
            f"unexpected MBPP+ held-out intersection: {len(selected_ids)} "
            f"!= {EXPECTED_PLUS_INTERSECTION}"
        )

    learning_rows = {
        split: [sanitized_learning_row(row) for row in source_rows[split]]
        for split in ("train", "dev")
    }
    selected_plus = [plus_by_id[task_id] for task_id in selected_ids]
    if contains_augmented_test(
        learning_rows["train"] + learning_rows["dev"], selected_plus
    ):
        raise ValueError("an MBPP+ augmented test leaked into train/dev content")

    eval_rows = [
        plus_eval_row(split_maps["test"][task_id], plus_by_id[task_id])
        for task_id in selected_ids
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_meta = {
        f"{split}.jsonl": write_jsonl(
            args.output_dir / f"{split}.jsonl", learning_rows[split]
        )
        for split in ("train", "dev")
    }
    output_meta["mbppplus_test.jsonl"] = write_jsonl(
        args.output_dir / "mbppplus_test.jsonl", eval_rows
    )

    source_meta = {
        split: {
            "path": str(path.resolve()),
            "rows": len(source_rows[split]),
            "sha256": sha256_bytes(path.read_bytes()),
        }
        for split, path in source_files.items()
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "source": {
            "dataset": "google-research-datasets/mbpp",
            "config": "sanitized",
            "prompt_template_version": "mbpp_function_signature_v2",
            "files": source_meta,
        },
        "evalplus": {
            "dataset": MBPPPLUS_DATASET,
            "revision": MBPPPLUS_REVISION,
            "downloaded_rows": len(downloaded),
            "heldout_intersection_rows": len(eval_rows),
        },
        "leakage": {
            "id_and_prompt_hash_pairwise_overlap_counts": overlaps,
            "augmented_tests_visible_to_train_or_dev": False,
            "plus_test_ids_in_train": len(set(plus_by_id) & set(split_maps["train"])),
            "plus_test_ids_in_dev": len(set(plus_by_id) & set(split_maps["dev"])),
            "plus_test_ids_in_heldout_test": len(eval_rows),
        },
        "outputs": output_meta,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialize one deterministic teacher solution per training prompt."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
from transformers import AutoTokenizer


def render_ids(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    try:
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    return tokenizer.encode(text, add_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="/opt/sparse-opd/data/splits/pilot_train.jsonl"
    )
    parser.add_argument(
        "--output", default="/opt/sparse-opd/data/offline/teacher_sft_train_v2.jsonl"
    )
    parser.add_argument(
        "--manifest", default="/opt/sparse-opd/data/offline/manifest_v2.json"
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--teacher", default="/opt/sparse-opd/models/Qwen3-Coder-Next-FP8"
    )
    parser.add_argument(
        "--teacher-revision", default="da6e2ed27304dd39abadd9c82ef50e8de67bdd4c"
    )
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    source_rows = [
        json.loads(line)
        for line in Path(args.input).read_text().splitlines()
        if line.strip()
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    complete = {}
    if output.is_file():
        complete = {
            row["id"]: row
            for row in (
                json.loads(line)
                for line in output.read_text().splitlines()
                if line.strip()
            )
        }
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    pending = [row for row in source_rows if row["id"] not in complete]
    with httpx.Client(timeout=600) as client:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            response = client.post(
                f"{args.endpoint.rstrip('/')}/generate",
                json={
                    "input_ids": [
                        render_ids(tokenizer, row["messages"]) for row in batch
                    ],
                    "sampling_params": {
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": 0,
                    },
                },
            )
            response.raise_for_status()
            bodies = response.json()
            if not isinstance(bodies, list):
                bodies = [bodies]
            if len(bodies) != len(batch):
                raise RuntimeError("teacher offline-generation batch mismatch")
            for row, body in zip(batch, bodies):
                completion = body["text"]
                record = {
                    **row,
                    "messages": [
                        *row["messages"],
                        {"role": "assistant", "content": completion},
                    ],
                    "teacher_completion": completion,
                    "teacher_completion_sha256": hashlib.sha256(
                        completion.encode("utf-8")
                    ).hexdigest(),
                    "teacher_revision": args.teacher_revision,
                }
                complete[row["id"]] = record
                with output.open("a") as handle:
                    handle.write(
                        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                    )
            print(
                f"generated {min(offset + len(batch), len(pending))}/{len(pending)}",
                flush=True,
            )

    ordered = [complete[row["id"]] for row in source_rows]
    canonical = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered
    )
    output.write_text(canonical)
    manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source": args.input,
        "source_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        "output": args.output,
        "output_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "rows": len(ordered),
        "teacher": args.teacher,
        "teacher_revision": args.teacher_revision,
        "endpoint": args.endpoint,
        "generation": {
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

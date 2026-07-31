#!/usr/bin/env python3
"""Generate and sandbox-grade BigCodeBench evaluation rows."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from grade_bcb_probe import extract_code, run_sandbox
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from palingenesis.opd.teacher_backend import SGLangTeacherBackend


def render(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )


def wilson(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    rate = successes / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count))
        / denominator
    )
    return [center - margin, center + margin]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("student", "teacher"), required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--base-model", default="/opt/sparse-opd/models/Qwen3-4B")
    parser.add_argument(
        "--teacher-model", default="/opt/sparse-opd/models/Qwen3-Coder-Next-FP8"
    )
    parser.add_argument("--teacher-endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sandbox-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--image", default="guided-v2-bcb-executor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.dataset.read_text().splitlines()
        if line.strip()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.teacher_model if args.kind == "teacher" else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    started = time.time()
    generations = []

    if args.kind == "teacher":
        backend = SGLangTeacherBackend(
            args.teacher_endpoint, timeout_seconds=300, max_retries=2
        )
        for offset in range(0, len(rows), 32):
            batch = rows[offset : offset + 32]
            prefixes = [
                tuple(
                    tokenizer.encode(
                        render(tokenizer, row["messages"]),
                        add_special_tokens=False,
                    )
                )
                for row in batch
            ]
            generated = backend.generate_tokens(prefixes, args.max_new_tokens)
            generations.extend(
                {
                    "id": row["id"],
                    "token_ids": list(result.token_ids),
                    "completion": tokenizer.decode(
                        result.token_ids, skip_special_tokens=True
                    ),
                    "finish_reason": result.finish_reason,
                }
                for row, result in zip(batch, generated)
            )
    else:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16, local_files_only=True
        )
        if args.checkpoint:
            model = PeftModel.from_pretrained(
                model,
                args.checkpoint,
                is_trainable=False,
                torch_device="cpu",
            )
        model = model.to(args.device).eval().requires_grad_(False)
        for offset in range(0, len(rows), args.batch_size):
            batch = rows[offset : offset + args.batch_size]
            prompts = [render(tokenizer, row["messages"]) for row in batch]
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True
            ).to(args.device)
            with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16
            ):
                output = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            width = encoded["input_ids"].shape[1]
            for row, token_ids in zip(batch, output[:, width:].tolist()):
                if tokenizer.eos_token_id in token_ids:
                    token_ids = token_ids[
                        : token_ids.index(tokenizer.eos_token_id) + 1
                    ]
                generations.append(
                    {
                        "id": row["id"],
                        "token_ids": token_ids,
                        "completion": tokenizer.decode(
                            token_ids, skip_special_tokens=True
                        ),
                        "finish_reason": (
                            "stop"
                            if token_ids
                            and token_ids[-1] == tokenizer.eos_token_id
                            else "length"
                        ),
                    }
                )

    generation_by_id = {record["id"]: record for record in generations}

    def grade(row):
        generation = generation_by_id[row["id"]]
        sandbox = run_sandbox(
            extract_code(generation["completion"]), row["test"], args.image
        )
        return {
            "id": row["id"],
            "prompt_hash": row["prompt_hash"],
            "completion_tokens": len(generation["token_ids"]),
            "finish_reason": generation["finish_reason"],
            "completion": generation["completion"],
            "sandbox": sandbox,
        }

    with ThreadPoolExecutor(max_workers=args.sandbox_workers) as pool:
        records = list(pool.map(grade, rows))
    passes = sum(record["sandbox"]["test_pass"] for record in records)
    lengths = [record["completion_tokens"] for record in records]
    summary = {
        "status": "pass",
        "kind": args.kind,
        "checkpoint": args.checkpoint or (
            "base_student" if args.kind == "student" else "frozen_teacher"
        ),
        "dataset": str(args.dataset.resolve()),
        "tasks": len(records),
        "pass_at_1": passes / len(records),
        "pass_at_1_ci95": wilson(passes, len(records)),
        "syntax_success_rate": statistics.mean(
            record["sandbox"]["syntax_success"] for record in records
        ),
        "runtime_success_rate": statistics.mean(
            record["sandbox"]["runtime_success"] for record in records
        ),
        "mean_completion_tokens": statistics.mean(lengths),
        "median_completion_tokens": statistics.median(lengths),
        "max_length_rate": sum(
            length >= args.max_new_tokens for length in lengths
        )
        / len(lengths),
        "max_new_tokens": args.max_new_tokens,
        "elapsed_seconds": time.time() - started,
    }
    (args.output_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

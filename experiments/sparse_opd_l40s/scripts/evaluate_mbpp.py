#!/usr/bin/env python3
"""Generate MBPP solutions from a base/LoRA checkpoint and grade them safely."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from importlib import import_module
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/opt/sparse-opd")
sys.path.insert(0, str(ROOT / "sandbox"))
run_sandbox = import_module("run_task").run


def extract_code(text: str) -> str:
    blocks = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def render(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )


def load_model(base_model: str, checkpoint: str, device: str):
    tokenizer_path = (
        checkpoint
        if checkpoint and Path(checkpoint, "tokenizer_config.json").is_file()
        else base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint, is_trainable=False)
    model = model.to(device).eval().requires_grad_(False)
    return tokenizer, model


def summarize(records: list[dict], elapsed: float) -> dict:
    count = len(records)
    if not count:
        raise ValueError("no evaluation records")
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
    return {
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
        "mean_solution_tokens": sum(row["completion_tokens"] for row in records)
        / count,
        "sandbox_timeouts": sum(row["sandbox"]["timeout"] for row in records),
        "generation_and_evaluation_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--base-model", default="/opt/sparse-opd/models/Qwen3-4B")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "evaluation_records.jsonl"
    summary_path = output_dir / "evaluation_summary.json"
    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    complete = {}
    if records_path.is_file():
        complete = {
            row["id"]: row
            for row in (
                json.loads(line)
                for line in records_path.read_text().splitlines()
                if line.strip()
            )
        }

    started = time.perf_counter()
    tokenizer, model = load_model(args.base_model, args.checkpoint, args.device)
    pending = [row for row in rows if row["id"] not in complete]
    for offset in range(0, len(pending), args.batch_size):
        batch = pending[offset : offset + args.batch_size]
        prompts = [render(tokenizer, row["messages"]) for row in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(args.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        for index, row in enumerate(batch):
            completion_ids = generated[index, prompt_width:].tolist()
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            code = extract_code(text)
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
            record = {
                "id": row["id"],
                "entry_point": row["entry_point"],
                "prompt_hash": row["prompt_hash"],
                "reference_hash": row["reference_hash"],
                "completion": text,
                "code": code,
                "completion_tokens": len(completion_ids),
                "sandbox": sandbox,
            }
            if not all(
                math.isfinite(float(sandbox.get(key, 0.0)))
                for key in ("elapsed_seconds",)
            ):
                raise FloatingPointError("non-finite sandbox timing")
            complete[row["id"]] = record
            with records_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"evaluated {min(offset + len(batch), len(pending))}/{len(pending)}",
            flush=True,
        )

    ordered = [complete[row["id"]] for row in rows]
    summary = summarize(ordered, time.perf_counter() - started)
    summary.update(
        {
            "checkpoint": args.checkpoint or "base_student",
            "base_model": args.base_model,
            "dataset": args.dataset,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

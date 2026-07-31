#!/usr/bin/env python3
"""Greedy functional counterfactual for teacher-token interventions."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from grade_bcb_probe import extract_code, run_sandbox
from transformers import AutoModelForCausalLM, AutoTokenizer

from palingenesis.opd.guided import build_guided_trajectories
from palingenesis.opd.teacher_backend import (
    SGLangTeacherBackend,
    TeacherGeneration,
)
from palingenesis.opd.token_bridge import TokenBridge, check_compatible


def render(tokenizer, messages) -> list[int]:
    try:
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    ids = tokenizer.encode(text, add_special_tokens=False)
    if tokenizer.bos_token_id is not None and (
        not ids or ids[0] != tokenizer.bos_token_id
    ):
        ids.insert(0, tokenizer.bos_token_id)
    return ids


@torch.inference_mode()
def greedy_generate(
    model,
    prompts: list[list[int]],
    max_new_tokens: int,
    *,
    device: str,
    pad_token_id: int,
    stop_ids: tuple[int, ...],
) -> list[list[int]]:
    width = max(len(prompt) for prompt in prompts)
    ids = torch.full(
        (len(prompts), width),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros_like(ids)
    for row, prompt in enumerate(prompts):
        values = torch.tensor(prompt, dtype=torch.long, device=device)
        ids[row, width - len(prompt) :] = values
        mask[row, width - len(prompt) :] = 1
    with torch.autocast("cuda", dtype=torch.bfloat16):
        generated = model.generate(
            ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=list(stop_ids),
            pad_token_id=pad_token_id,
        )
    output = []
    for tokens in generated[:, width:].tolist():
        cleaned = []
        for token_id in tokens:
            cleaned.append(token_id)
            if token_id in stop_ids:
                break
        output.append(cleaned)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--intervals", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--guidance-group-tokens", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--student", default="/opt/sparse-opd/models/Qwen3-4B")
    parser.add_argument(
        "--teacher", default="/opt/sparse-opd/models/Qwen3-Coder-Next-FP8"
    )
    parser.add_argument("--teacher-endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sandbox-workers", type=int, default=4)
    parser.add_argument("--image", default="guided-v2-bcb-executor")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student, local_files_only=True
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        args.teacher, local_files_only=True
    )
    bridge = TokenBridge.from_tokenizers(
        student_tokenizer,
        teacher_tokenizer,
    )
    check_compatible(student_tokenizer, teacher_tokenizer, bridge)
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=torch.bfloat16, local_files_only=True
    )
    model = model.to(args.device).eval().requires_grad_(False)
    teacher = SGLangTeacherBackend(
        args.teacher_endpoint, timeout_seconds=300, max_retries=2
    )
    student_prompts = [
        render(student_tokenizer, row["messages"]) for row in rows
    ]
    teacher_prompts = [
        render(teacher_tokenizer, row["messages"]) for row in rows
    ]
    started = time.time()

    plain = []
    for offset in range(0, len(rows), args.batch_size):
        plain.extend(
            greedy_generate(
                model,
                student_prompts[offset : offset + args.batch_size],
                args.max_new_tokens,
                device=args.device,
                pad_token_id=student_tokenizer.pad_token_id,
                stop_ids=bridge.stop_ids,
            )
        )

    summaries = {}
    for interval in args.intervals:

        def student_generate(indices, completions, limit):
            prompts = [
                student_prompts[index] + completion
                for index, completion in zip(indices, completions)
            ]
            generated = []
            for offset in range(0, len(prompts), args.batch_size):
                generated.extend(
                    greedy_generate(
                        model,
                        prompts[offset : offset + args.batch_size],
                        limit,
                        device=args.device,
                        pad_token_id=student_tokenizer.pad_token_id,
                        stop_ids=bridge.stop_ids,
                    )
                )
            return generated

        def teacher_generate(indices, completions, limit):
            prefixes = [
                tuple(
                    teacher_prompts[index] + bridge.to_teacher(completion)
                )
                for index, completion in zip(indices, completions)
            ]
            return [
                TeacherGeneration(
                    token_ids=tuple(
                        bridge.to_student(list(generation.token_ids))
                    ),
                    finish_reason=generation.finish_reason,
                )
                for generation in teacher.generate_tokens(prefixes, limit)
            ]

        trajectories = build_guided_trajectories(
            count=len(rows),
            max_completion_tokens=[args.max_new_tokens] * len(rows),
            interval_tokens=interval,
            guidance_group_tokens=args.guidance_group_tokens,
            stop_ids=bridge.stop_ids,
            student_generate=student_generate,
            teacher_generate=teacher_generate,
        )

        def grade(index: int) -> dict:
            guided_completion = student_tokenizer.decode(
                trajectories[index].completion, skip_special_tokens=True
            )
            plain_completion = student_tokenizer.decode(
                plain[index], skip_special_tokens=True
            )
            return {
                "id": rows[index]["id"],
                "prompt_hash": rows[index]["prompt_hash"],
                "interval": interval,
                "teacher_groups": len(trajectories[index].teacher_groups),
                "teacher_tokens": trajectories[index].teacher_tokens,
                "guided_tokens": len(trajectories[index].completion),
                "student_only_tokens": len(plain[index]),
                "guided": run_sandbox(
                    extract_code(guided_completion),
                    rows[index]["test"],
                    args.image,
                ),
                "student_only": run_sandbox(
                    extract_code(plain_completion),
                    rows[index]["test"],
                    args.image,
                ),
            }

        with ThreadPoolExecutor(max_workers=args.sandbox_workers) as pool:
            records = list(pool.map(grade, range(len(rows))))
        guided_passes = sum(row["guided"]["test_pass"] for row in records)
        plain_passes = sum(row["student_only"]["test_pass"] for row in records)
        summary = {
            "tasks": len(records),
            "interval": interval,
            "guidance_group_tokens": args.guidance_group_tokens,
            "guided_pass_at_1": guided_passes / len(records),
            "student_only_pass_at_1": plain_passes / len(records),
            "paired_difference": (guided_passes - plain_passes) / len(records),
            "guided_fixes": sum(
                row["guided"]["test_pass"]
                and not row["student_only"]["test_pass"]
                for row in records
            ),
            "guided_regressions": sum(
                row["student_only"]["test_pass"]
                and not row["guided"]["test_pass"]
                for row in records
            ),
            "teacher_groups": sum(row["teacher_groups"] for row in records),
            "teacher_tokens": sum(row["teacher_tokens"] for row in records),
            "teacher_intervention_coverage": sum(
                row["teacher_tokens"] > 0 for row in records
            )
            / len(records),
        }
        interval_dir = args.output_dir / f"i{interval}"
        interval_dir.mkdir(parents=True, exist_ok=True)
        (interval_dir / "records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
            encoding="utf-8",
        )
        (interval_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries[str(interval)] = summary

    report = {
        "status": "pass",
        "dataset": str(args.dataset.resolve()),
        "tasks": len(rows),
        "deterministic_student_decoding": True,
        "intervals": summaries,
        "teacher_requests": teacher.request_count,
        "teacher_successful_requests": teacher.successful_request_count,
        "teacher_failures": teacher.failure_count,
        "elapsed_seconds": time.time() - started,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output_dir / "summary.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Required SGLang teacher throughput/logprob benchmark."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from transformers import AutoTokenizer


def gpu_snapshot() -> list[dict]:
    text = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in text.splitlines():
        index, used, total, utilization, power = [
            part.strip() for part in line.split(",")
        ]
        rows.append(
            {
                "index": int(index),
                "memory_used_mib": float(used),
                "memory_total_mib": float(total),
                "utilization_percent": float(utilization),
                "power_watts": float(power),
            }
        )
    return rows


def post(
    client, endpoint: str, input_ids, max_new_tokens: int, top_k: int
) -> tuple[object, float]:
    started = time.perf_counter()
    response = client.post(
        f"{endpoint}/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "top_logprobs_num": top_k,
            "token_ids_logprob": [0, 1, 2, 151645],
            "return_text_in_logprobs": False,
        },
    )
    response.raise_for_status()
    return response.json(), time.perf_counter() - started


def validate(body, expected_batch: int) -> tuple[int, list[float]]:
    rows = body if isinstance(body, list) else [body]
    if len(rows) != expected_batch:
        raise RuntimeError(f"batch mismatch: {len(rows)} != {expected_batch}")
    tokens = 0
    values = []
    for row in rows:
        meta = row["meta_info"]
        tokens += int(meta["completion_tokens"])
        values.extend(
            float(pair[0]) for step in meta["output_top_logprobs"] for pair in step
        )
        values.extend(
            float(pair[0])
            for step in meta["output_token_ids_logprobs"]
            for pair in step
        )
    if not values or not all(math.isfinite(value) for value in values):
        raise FloatingPointError("teacher returned non-finite logprob")
    return tokens, values


def markdown(report: dict) -> str:
    lines = [
        "# Teacher benchmark",
        "",
        f"- Captured: {report['captured_at']}",
        f"- Endpoint: `{report['endpoint']}`",
        f"- Stability requests: {report['stability']['requests']}",
        f"- Stability failures: {report['stability']['failures']}",
        f"- Peak teacher VRAM: {report['peak_teacher_vram_mib']:.0f} MiB",
        "",
        "## Scoring matrix",
        "",
        "| Batch | Prompt tokens | Top-k | TTFT proxy ms | Requests/s | Tokens/s |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["scoring"]:
        lines.append(
            f"| {row['batch_size']} | {row['prompt_tokens']} | {row['top_logprobs']} | "
            f"{row['ttft_proxy_ms']:.2f} | {row['requests_per_second']:.3f} | "
            f"{row['tokens_per_second']:.3f} |"
        )
    lines.extend(
        ["", "## Decode", "", "```json", json.dumps(report["decode"], indent=2), "```"]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--model", default="/opt/sparse-opd/models/Qwen3-Coder-Next-FP8"
    )
    parser.add_argument("--stability-requests", type=int, default=100)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--json-out", default="/opt/sparse-opd/reports/teacher_benchmark.json"
    )
    parser.add_argument(
        "--md-out", default="/opt/sparse-opd/reports/teacher_benchmark.md"
    )
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    seed_ids = tokenizer.encode(
        "def solve(value):\n    return value\n", add_special_tokens=False
    )
    batch_sizes = [1, 4] if args.quick else [1, 4, 8, 16, 32]
    prompt_lengths = [256] if args.quick else [256, 512, 1024, 2048]
    top_ks = [32] if args.quick else [32, 64, 128]
    report = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "scoring": [],
        "decode": {},
        "stability": {"requests": args.stability_requests, "failures": 0},
    }
    peak = 0.0
    with httpx.Client(timeout=180) as client:
        for batch_size in batch_sizes:
            for prompt_length in prompt_lengths:
                prompt = (seed_ids * (prompt_length // len(seed_ids) + 1))[
                    :prompt_length
                ]
                for top_k in top_ks:
                    body, elapsed = post(
                        client,
                        endpoint,
                        [prompt] * batch_size if batch_size > 1 else prompt,
                        1,
                        top_k,
                    )
                    tokens, _ = validate(body, batch_size)
                    report["scoring"].append(
                        {
                            "batch_size": batch_size,
                            "prompt_tokens": prompt_length,
                            "top_logprobs": top_k,
                            "ttft_proxy_ms": elapsed * 1000,
                            "requests_per_second": batch_size / elapsed,
                            "tokens_per_second": tokens / elapsed,
                        }
                    )
                    peak = max(
                        peak, *(row["memory_used_mib"] for row in gpu_snapshot()[:2])
                    )

        decode_prompt = (seed_ids * 32)[:256]
        body, elapsed = post(client, endpoint, decode_prompt, 64, 64)
        tokens, _ = validate(body, 1)
        report["decode"] = {
            "prompt_tokens": 256,
            "completion_tokens": tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens / elapsed,
        }

        stable_prompt = (seed_ids * 32)[:256]
        reference = None
        for _ in range(args.stability_requests):
            try:
                body, _ = post(client, endpoint, stable_prompt, 1, 64)
                _, values = validate(body, 1)
                current = values[:64]
                if reference is None:
                    reference = current
                elif max(abs(a - b) for a, b in zip(reference, current)) > 1e-4:
                    raise RuntimeError("deterministic logprobs changed materially")
            except (httpx.HTTPError, KeyError, RuntimeError, TypeError, ValueError):
                report["stability"]["failures"] += 1
    report["peak_teacher_vram_mib"] = peak
    if report["stability"]["failures"] / max(1, args.stability_requests) > 0.05:
        raise RuntimeError("teacher failure rate exceeded 5% kill switch")
    Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    Path(args.md_out).write_text(markdown(report))
    print(
        json.dumps(
            {
                "scoring_cases": len(report["scoring"]),
                "stability": report["stability"],
                "peak_teacher_vram_mib": peak,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

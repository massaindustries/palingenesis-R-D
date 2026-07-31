#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    endpoint = args.endpoint.rstrip("/")
    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(f"{endpoint}/health")
        health.raise_for_status()
        response = client.post(
            f"{endpoint}/generate",
            json={
                "input_ids": [9707, 1879],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "top_logprobs_num": 32,
                "token_ids_logprob": [0, 1, 2],
                "return_text_in_logprobs": False,
            },
        )
        response.raise_for_status()
        body = response.json()
    meta = body["meta_info"]
    top = meta["output_top_logprobs"][0]
    requested = meta["output_token_ids_logprobs"][0]
    values = [float(item[0]) for item in top + requested]
    result = {
        "healthy": bool(values) and all(math.isfinite(value) for value in values),
        "top_logprobs": len(top),
        "requested_logprobs": len(requested),
        "model": meta.get("model_name"),
    }
    print(json.dumps(result, indent=2))
    return 0 if result["healthy"] else 1


if __name__ == "__main__":
    sys.exit(main())

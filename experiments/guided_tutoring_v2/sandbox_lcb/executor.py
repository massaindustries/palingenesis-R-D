#!/usr/bin/env python3
"""Container-side exact-output executor for LiveCodeBench stdin tasks."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OUTPUT_LIMIT = 16_384


def clipped(value: str) -> str:
    return value.encode("utf-8", errors="replace")[:OUTPUT_LIMIT].decode(
        "utf-8", errors="replace"
    )


def normalized(value: str) -> list[str]:
    return value.strip().split()


def main() -> int:
    payload = json.load(sys.stdin)
    code = str(payload["code"])
    tests = list(payload.get("tests", []))
    timeout = min(float(payload.get("timeout", 10)), 10.0)
    result = {
        "syntax_success": False,
        "runtime_success": False,
        "test_pass": False,
        "timeout": False,
        "exception_type": None,
        "stderr": "",
        "tests_passed": 0,
        "tests_total": len(tests),
        "elapsed_seconds": 0.0,
    }
    started = time.perf_counter()
    try:
        compile(code, "<candidate>", "exec")
        result["syntax_success"] = True
    except SyntaxError as error:
        result["exception_type"] = "SyntaxError"
        result["stderr"] = clipped(str(error))
        result["elapsed_seconds"] = time.perf_counter() - started
        print(json.dumps(result))
        return 0

    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        candidate = Path(directory) / "candidate.py"
        candidate.write_text(code, encoding="utf-8")
        all_ran = True
        for index, test in enumerate(tests):
            try:
                completed = subprocess.run(
                    [sys.executable, "-I", str(candidate)],
                    input=str(test["input"]),
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=timeout,
                    cwd=directory,
                    env={
                        "PATH": "/usr/local/bin:/usr/bin:/bin",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                )
            except subprocess.TimeoutExpired:
                result["timeout"] = True
                result["exception_type"] = f"Timeout[{index}]"
                all_ran = False
                break
            if completed.returncode:
                result["exception_type"] = f"RuntimeError[{index}]"
                result["stderr"] = clipped(completed.stderr)
                all_ran = False
                break
            if normalized(completed.stdout) == normalized(str(test["output"])):
                result["tests_passed"] += 1
        result["runtime_success"] = all_ran
    result["test_pass"] = (
        result["runtime_success"] and result["tests_passed"] == len(tests)
    )
    result["elapsed_seconds"] = time.perf_counter() - started
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

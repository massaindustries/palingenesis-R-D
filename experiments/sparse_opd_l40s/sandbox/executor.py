#!/usr/bin/env python3
"""Container-side untrusted Python executor. Reads one JSON payload on stdin."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OUTPUT_LIMIT = 16_384


def clipped(value: str) -> str:
    encoded = value.encode("utf-8", errors="replace")[:OUTPUT_LIMIT]
    return encoded.decode("utf-8", errors="replace")


def main() -> int:
    payload = json.load(sys.stdin)
    code = str(payload["code"])
    tests = [str(test) for test in payload.get("tests", [])]
    imports = [str(item) for item in payload.get("imports", [])]
    timeout = min(float(payload.get("timeout", 10.0)), 10.0)
    result = {
        "syntax_success": False,
        "runtime_success": False,
        "test_pass": False,
        "timeout": False,
        "exception_type": None,
        "stdout": "",
        "stderr": "",
        "elapsed_seconds": 0.0,
        "tests_passed": 0,
        "tests_total": len(tests),
    }
    started = time.perf_counter()
    try:
        compile(code, "<candidate>", "exec")
        result["syntax_success"] = True
    except SyntaxError as error:
        result["exception_type"] = type(error).__name__
        result["stderr"] = clipped(str(error))
        result["elapsed_seconds"] = time.perf_counter() - started
        print(json.dumps(result))
        return 0

    base_script = "\n".join(
        [
            code,
            "",
            *imports,
        ]
    )
    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        candidate = Path(directory) / "candidate.py"
        candidate.write_text(base_script)
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(candidate)],
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
            result["runtime_success"] = completed.returncode == 0
            result["stdout"] = clipped(completed.stdout)
            result["stderr"] = clipped(completed.stderr)
            if completed.returncode:
                last_line = completed.stderr.strip().splitlines()[-1:] or [
                    "runtime_error"
                ]
                result["exception_type"] = last_line[0].split(":", 1)[0]
        except subprocess.TimeoutExpired as error:
            result["timeout"] = True
            result["exception_type"] = "TimeoutExpired"
            result["stdout"] = clipped(error.stdout or "")
            result["stderr"] = clipped(error.stderr or "")
        if result["runtime_success"]:
            for index, test in enumerate(tests):
                candidate.write_text(f"{base_script}\n\n{test}")
                try:
                    test_result = subprocess.run(
                        [sys.executable, "-I", str(candidate)],
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
                    if test_result.returncode == 0:
                        result["tests_passed"] += 1
                    elif not result["stderr"]:
                        result["stderr"] = clipped(test_result.stderr)
                        result["exception_type"] = f"TestFailure[{index}]"
                except subprocess.TimeoutExpired:
                    result["timeout"] = True
                    if not result["exception_type"]:
                        result["exception_type"] = f"TestTimeout[{index}]"
            result["test_pass"] = result["tests_passed"] == result["tests_total"]
    result["elapsed_seconds"] = time.perf_counter() - started
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

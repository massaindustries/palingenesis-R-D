#!/usr/bin/env python3
"""Container-side unittest executor for stdlib-only BigCodeBench tasks."""

from __future__ import annotations

import json
import re
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


def main() -> int:
    payload = json.load(sys.stdin)
    code = str(payload["code"])
    tests = str(payload["test"])
    timeout = min(float(payload.get("timeout", 10)), 10.0)
    tests_total = len(re.findall(r"(?m)^\s+def test_", tests))
    result = {
        "syntax_success": False,
        "runtime_success": False,
        "test_pass": False,
        "timeout": False,
        "exception_type": None,
        "stderr": "",
        "tests_total": tests_total,
        "elapsed_seconds": 0.0,
    }
    started = time.perf_counter()
    script = "\n".join(
        [
            code,
            "",
            tests,
            "",
            "if __name__ == '__main__':",
            "    unittest.main(verbosity=0)",
        ]
    )
    try:
        compile(script, "<candidate>", "exec")
        result["syntax_success"] = True
    except SyntaxError as error:
        result["exception_type"] = "SyntaxError"
        result["stderr"] = clipped(str(error))
        result["elapsed_seconds"] = time.perf_counter() - started
        print(json.dumps(result))
        return 0

    with tempfile.TemporaryDirectory(dir="/tmp") as directory:
        candidate = Path(directory) / "candidate.py"
        candidate.write_text(script, encoding="utf-8")
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
            result["test_pass"] = completed.returncode == 0
            result["stderr"] = clipped(completed.stderr)
            if completed.returncode:
                result["exception_type"] = "UnitTestFailure"
        except subprocess.TimeoutExpired:
            result["timeout"] = True
            result["exception_type"] = "TimeoutExpired"
    result["elapsed_seconds"] = time.perf_counter() - started
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

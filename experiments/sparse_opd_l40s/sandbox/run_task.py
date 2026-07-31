#!/usr/bin/env python3
"""Host-side wrapper for the locked-down Docker code sandbox."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(payload: dict, image: str = "sparse-opd-executor") -> dict:
    command = [
        "docker",
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "128",
        "--memory",
        "512m",
        "--cpus",
        "1",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        image,
    ]
    try:
        completed = subprocess.run(
            command,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "syntax_success": False,
            "runtime_success": False,
            "test_pass": False,
            "timeout": True,
            "exception_type": "ContainerTimeout",
            "stdout": "",
            "stderr": "",
        }
    if completed.returncode:
        raise RuntimeError(
            f"sandbox container failed with code {completed.returncode}: "
            f"{completed.stderr[:4096]}"
        )
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", help="JSON file containing code/tests/imports")
    parser.add_argument("--image", default="sparse-opd-executor")
    args = parser.parse_args()
    result = run(json.loads(Path(args.payload).read_text()), image=args.image)
    print(json.dumps(result, indent=2))
    return 0 if result["test_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Quantify the MBPP callable-name confound before and after prompt-v2."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


def extract_code(text: str) -> str:
    blocks = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```", text, re.DOTALL | re.IGNORECASE
    )
    return (max(blocks, key=len) if blocks else text).strip()


def audit(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    syntax = entry_point = 0
    for row in rows:
        try:
            tree = ast.parse(extract_code(row["teacher_completion"]))
        except SyntaxError:
            continue
        syntax += 1
        names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        entry_point += row["entry_point"] in names
    return {
        "path": str(path),
        "tasks": len(rows),
        "syntax_success_rate": syntax / len(rows),
        "required_entry_point_defined": entry_point,
        "required_entry_point_defined_rate": entry_point / len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-v1",
        default=(
            "/opt/sparse-opd/data/archive/prompt_v1_20260730T152800Z/"
            "offline/teacher_eval_pilot_test.jsonl"
        ),
    )
    parser.add_argument(
        "--prompt-v2",
        default="/opt/sparse-opd/data/offline/teacher_eval_pilot_test_v2.jsonl",
    )
    parser.add_argument(
        "--json-out",
        default="/opt/sparse-opd/reports/prompt_template_audit.json",
    )
    parser.add_argument(
        "--md-out",
        default="/opt/sparse-opd/reports/prompt_template_audit.md",
    )
    args = parser.parse_args()
    v1 = audit(Path(args.prompt_v1))
    v2 = audit(Path(args.prompt_v2))
    payload = {
        "question": (
            "Does exposing the callable signature remove function-name guessing "
            "without exposing tests or implementation?"
        ),
        "prompt_v1": v1,
        "prompt_v2": v2,
        "absolute_entry_point_compliance_change": (
            v2["required_entry_point_defined_rate"]
            - v1["required_entry_point_defined_rate"]
        ),
        "interpretation": (
            "Prompt-v1 executable quality is confounded by a hidden interface. "
            "Prompt-v2 is the official pilot template."
        ),
    }
    Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n")
    markdown = [
        "# Prompt-template audit",
        "",
        "| Version | Tasks | Syntax success | Required entry point defined |",
        "|---|---:|---:|---:|",
        (
            f"| Prompt-v1 | {v1['tasks']} | {v1['syntax_success_rate']:.2%} | "
            f"{v1['required_entry_point_defined_rate']:.2%} |"
        ),
        (
            f"| Prompt-v2 | {v2['tasks']} | {v2['syntax_success_rate']:.2%} | "
            f"{v2['required_entry_point_defined_rate']:.2%} |"
        ),
        "",
        payload["interpretation"],
    ]
    Path(args.md_out).write_text("\n".join(markdown) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

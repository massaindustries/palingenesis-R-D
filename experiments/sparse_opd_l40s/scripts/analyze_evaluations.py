#!/usr/bin/env python3
"""Add deterministic autonomy/diversity metrics to an MBPP evaluation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import tokenize
from pathlib import Path


def extract_code(text: str) -> str:
    blocks = re.findall(
        r"```(?:python|py)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE
    )
    return (max(blocks, key=len) if blocks else text).strip()


def lexical_tokens(code: str) -> list[str]:
    try:
        ignored = {
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NEWLINE,
            tokenize.NL,
        }
        return [
            item.string
            for item in tokenize.tokenize(io.BytesIO(code.encode()).readline)
            if item.type not in ignored
        ]
    except (IndentationError, SyntaxError, tokenize.TokenError):
        return re.findall(r"\w+|[^\w\s]", code)


def levenshtein(left: list[str], right: list[str]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_item in enumerate(right, start=1):
        current = [row]
        for column, left_item in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def teacher_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return {
        row["prompt_hash"]: extract_code(row["teacher_completion"])
        for row in (
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_dir")
    parser.add_argument("--teacher-completions")
    args = parser.parse_args()
    evaluation_dir = Path(args.evaluation_dir)
    records = [
        json.loads(line)
        for line in (evaluation_dir / "evaluation_records.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    teachers = teacher_map(
        Path(args.teacher_completions) if args.teacher_completions else None
    )

    exact_hashes: set[str] = set()
    ast_hashes: set[str] = set()
    parseable = 0
    lengths_chars: list[int] = []
    lengths_lines: list[int] = []
    edit_distances: list[float] = []
    for row in records:
        code = row["code"].strip()
        exact_hashes.add(hashlib.sha256(code.encode()).hexdigest())
        lengths_chars.append(len(code))
        lengths_lines.append(len(code.splitlines()))
        try:
            normalized = ast.dump(
                ast.parse(code), annotate_fields=True, include_attributes=False
            )
        except SyntaxError:
            pass
        else:
            parseable += 1
            ast_hashes.add(hashlib.sha256(normalized.encode()).hexdigest())
        teacher = teachers.get(row["prompt_hash"])
        if teacher is not None:
            student_tokens = lexical_tokens(code)
            teacher_tokens = lexical_tokens(teacher)
            denominator = max(1, len(student_tokens), len(teacher_tokens))
            edit_distances.append(
                levenshtein(student_tokens, teacher_tokens) / denominator
            )

    count = len(records)
    metrics = {
        "tasks": count,
        "normalized_ast_uniqueness": len(ast_hashes) / max(1, parseable),
        "normalized_ast_unique_count": len(ast_hashes),
        "ast_parseable_tasks": parseable,
        "exact_duplicate_rate": 1.0 - len(exact_hashes) / max(1, count),
        "exact_unique_count": len(exact_hashes),
        "mean_solution_length_chars": sum(lengths_chars) / max(1, count),
        "mean_solution_length_lines": sum(lengths_lines) / max(1, count),
        "normalized_teacher_token_edit_distance": (
            sum(edit_distances) / len(edit_distances) if edit_distances else None
        ),
        "teacher_edit_distance_matches": len(edit_distances),
        "definitions": {
            "normalized_ast_uniqueness": (
                "Unique SHA-256 hashes of ast.dump(include_attributes=False) divided by "
                "the number of syntactically parseable solutions."
            ),
            "exact_duplicate_rate": "One minus unique stripped-code SHA-256 hashes divided by tasks.",
            "normalized_teacher_token_edit_distance": (
                "Token-level Levenshtein distance divided by the longer token sequence, "
                "matched by prompt_hash."
            ),
        },
    }
    (evaluation_dir / "autonomy_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    summary_path = evaluation_dir / "evaluation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary.update(
        {key: value for key, value in metrics.items() if key != "definitions"}
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

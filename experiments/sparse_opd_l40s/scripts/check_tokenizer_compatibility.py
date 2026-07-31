#!/usr/bin/env python3
"""Strict tokenizer compatibility gate for sparse token-level distillation."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

PROBES = [
    "Hello world",
    "L'intelligenza artificiale",
    "def fibonacci(n: int) -> int:\n    return n",
    "é à è ò ù",
    'JSON: {"value": 42}',
    "tabs,\tspaces, newline\nand triple backticks\n```python\nprint('ok')\n```",
    "Unicode: Ελληνικά, русский, 中文, العربية, 👩🏽‍💻",
]


def compare_tokenizers(
    student, teacher, sample_size: int = 500, seed: int = 20260730
) -> dict:
    failures: list[dict] = []
    student_vocab = int(student.vocab_size)
    teacher_vocab = int(teacher.vocab_size)
    if student_vocab != teacher_vocab:
        failures.append(
            {
                "check": "vocab_size",
                "student": student_vocab,
                "teacher": teacher_vocab,
            }
        )

    shared = min(student_vocab, teacher_vocab)
    rng = random.Random(seed)
    token_ids = sorted(rng.sample(range(shared), min(sample_size, shared)))
    token_mismatches = []
    for token_id in token_ids:
        student_token = student.convert_ids_to_tokens(token_id)
        teacher_token = teacher.convert_ids_to_tokens(token_id)
        if student_token != teacher_token:
            token_mismatches.append(
                {
                    "id": token_id,
                    "student": student_token,
                    "teacher": teacher_token,
                }
            )
    if token_mismatches:
        failures.append(
            {
                "check": "deterministic_token_id_sample",
                "mismatch_count": len(token_mismatches),
                "examples": token_mismatches[:20],
            }
        )

    probe_results = []
    for text in PROBES:
        student_ids = student.encode(text, add_special_tokens=False)
        teacher_ids = teacher.encode(text, add_special_tokens=False)
        student_decoded = student.decode(student_ids, skip_special_tokens=False)
        teacher_decoded = teacher.decode(teacher_ids, skip_special_tokens=False)
        passed = student_ids == teacher_ids and student_decoded == teacher_decoded
        probe_results.append(
            {
                "text": text,
                "passed": passed,
                "student_ids": student_ids,
                "teacher_ids": teacher_ids,
                "student_decoded": student_decoded,
                "teacher_decoded": teacher_decoded,
            }
        )
        if not passed:
            failures.append({"check": "probe", "text": text})

    special_names = (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    )
    special_results = {}
    for name in special_names:
        student_id = getattr(student, name, None)
        teacher_id = getattr(teacher, name, None)
        special_results[name] = {"student": student_id, "teacher": teacher_id}
        if student_id != teacher_id:
            failures.append(
                {
                    "check": name,
                    "student": student_id,
                    "teacher": teacher_id,
                }
            )

    student_special = student.get_added_vocab()
    teacher_special = teacher.get_added_vocab()
    added_vocab_equal = student_special == teacher_special
    if not added_vocab_equal:
        failures.append(
            {
                "check": "added_vocab",
                "student_only": sorted(set(student_special) - set(teacher_special)),
                "teacher_only": sorted(set(teacher_special) - set(student_special)),
            }
        )

    return {
        "compatible": not failures,
        "seed": seed,
        "sample_size": len(token_ids),
        "student_vocab_size": student_vocab,
        "teacher_vocab_size": teacher_vocab,
        "token_id_mismatch_count": len(token_mismatches),
        "special_tokens": special_results,
        "added_vocab_equal": added_vocab_equal,
        "probes": probe_results,
        "failures": failures,
    }


def render_markdown(report: dict) -> str:
    status = "PASS" if report["compatible"] else "FAIL"
    rows = [
        "# Tokenizer compatibility",
        "",
        f"- Status: **{status}**",
        f"- Student: `{report['student']}`",
        f"- Teacher: `{report['teacher']}`",
        f"- Student vocab size: {report['student_vocab_size']}",
        f"- Teacher vocab size: {report['teacher_vocab_size']}",
        f"- Deterministic token IDs checked: {report['sample_size']}",
        f"- Token-ID mismatches: {report['token_id_mismatch_count']}",
        f"- Added vocab equal: {report['added_vocab_equal']}",
        "",
        "## Probes",
        "",
        "| Probe | Result |",
        "|---|---|",
    ]
    for probe in report["probes"]:
        compact = probe["text"].replace("\n", "\\n").replace("|", "\\|")
        rows.append(f"| `{compact}` | {'PASS' if probe['passed'] else 'FAIL'} |")
    rows.extend(["", "## Failures", ""])
    if report["failures"]:
        rows.append("```json")
        rows.append(json.dumps(report["failures"], indent=2, ensure_ascii=False))
        rows.append("```")
    else:
        rows.append("None.")
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="/opt/sparse-opd/models/Qwen3-4B")
    parser.add_argument(
        "--teacher", default="/opt/sparse-opd/models/Qwen3-Coder-Next-FP8"
    )
    parser.add_argument(
        "--json-out", default="/opt/sparse-opd/reports/tokenizer_compatibility.json"
    )
    parser.add_argument(
        "--md-out", default="/opt/sparse-opd/reports/tokenizer_compatibility.md"
    )
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()

    student = AutoTokenizer.from_pretrained(args.student)
    teacher = AutoTokenizer.from_pretrained(args.teacher)
    report = compare_tokenizers(student, teacher, sample_size=args.sample_size)
    report.update(
        {
            "student": args.student,
            "teacher": args.teacher,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    md_path.write_text(render_markdown(report))
    print(
        json.dumps(
            {
                "compatible": report["compatible"],
                "json_report": str(json_path),
                "markdown_report": str(md_path),
            }
        )
    )
    return 0 if report["compatible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

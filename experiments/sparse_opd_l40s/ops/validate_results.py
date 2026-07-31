#!/usr/bin/env python3
"""Data-quality and analytical-readiness checks for Sparse OPD artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path


def issue(severity: str, check: str, evidence: str, impact: str) -> dict:
    return {
        "severity": severity,
        "check": check,
        "evidence": evidence,
        "impact": impact,
    }


def metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def increasing_streak(values: list[float]) -> int:
    longest = current = 1 if values else 0
    for left, right in pairwise(values):
        current = current + 1 if right > left else 1
        longest = max(longest, current)
    return longest


def validate_run(run: Path) -> tuple[dict, list[dict]]:
    problems = []
    config_path = run / "config_resolved.json"
    if not config_path.is_file():
        return {"run": str(run), "ready": False}, [
            issue(
                "Critical",
                "config",
                "config_resolved.json missing",
                "run is not reproducible",
            )
        ]
    config = json.loads(config_path.read_text())
    rows = [row for row in metrics(run / "metrics.jsonl") if "gradient_norm" in row]
    expected = int(
        config.get("offline_runtime", {}).get(
            "steps",
            config.get("train", {}).get("steps", 0),
        )
    )
    steps = [int(row["step"]) for row in rows]
    if len(steps) != len(set(steps)):
        problems.append(
            issue(
                "Critical",
                "optimizer-step uniqueness",
                f"{len(steps) - len(set(steps))} duplicate steps",
                "training aggregates would double count work",
            )
        )
    if expected and len(rows) != expected:
        problems.append(
            issue(
                "High",
                "optimizer-step completeness",
                f"expected {expected}, observed {len(rows)}",
                "run is incomplete or metrics are missing",
            )
        )
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                problems.append(
                    issue(
                        "Critical",
                        "numeric validity",
                        f"step {row['step']} field {key} is {value}",
                        "non-finite evidence cannot support comparison",
                    )
                )
    max_residual = max(
        (float(row.get("residual_mass", 0.0)) for row in rows), default=0.0
    )
    if max_residual > 0.20:
        problems.append(
            issue(
                "Critical",
                "residual-mass kill switch",
                f"maximum student residual mass {max_residual:.6f}",
                "sparse support is too small for a trustworthy KL estimate",
            )
        )
    staleness = max((int(row.get("rollout/staleness", 0)) for row in rows), default=0)
    if staleness != 0:
        problems.append(
            issue(
                "Critical",
                "policy staleness",
                f"maximum staleness {staleness}",
                "rollouts are not strictly on-policy",
            )
        )
    policy_versions = [
        int(row["rollout/policy_version"])
        for row in rows
        if "rollout/policy_version" in row
    ]
    if policy_versions and any(
        right != left + 1 for left, right in pairwise(policy_versions)
    ):
        problems.append(
            issue(
                "High",
                "policy-version continuity",
                f"versions {policy_versions[:5]}...{policy_versions[-5:]}",
                "an optimizer update or adapter sync may be missing",
            )
        )
    vram = [float(row.get("student/peak_vram_mib", 0.0)) for row in rows]
    vram_streak = increasing_streak(vram)
    if vram_streak > 50:
        problems.append(
            issue(
                "High",
                "VRAM leak kill switch",
                f"peak VRAM rose for {vram_streak} consecutive steps",
                "the run may eventually OOM",
            )
        )
    evaluation_dir = next(
        (
            path
            for path in (run / "evaluation_v2", run / "evaluation")
            if (path / "evaluation_summary.json").is_file()
        ),
        None,
    )
    if evaluation_dir is not None:
        evaluation = evaluation_dir / "evaluation_summary.json"
        summary = json.loads(evaluation.read_text())
        records = metrics(evaluation_dir / "evaluation_records.jsonl")
        ids = [row["id"] for row in records]
        if len(ids) != len(set(ids)) or len(ids) != int(summary["tasks"]):
            problems.append(
                issue(
                    "High",
                    "evaluation coverage",
                    f"records={len(ids)}, unique={len(set(ids))}, summary={summary['tasks']}",
                    "pass@1 denominator is unreliable",
                )
            )
    run_summary = {
        "run": str(run),
        "expected_steps": expected,
        "observed_steps": len(rows),
        "unique_steps": len(set(steps)),
        "max_student_residual_mass": max_residual,
        "max_policy_staleness": staleness,
        "longest_increasing_vram_streak": vram_streak,
        "ready": not any(item["severity"] in ("Critical", "High") for item in problems),
    }
    return run_summary, problems


def validate_dataset(root: Path) -> tuple[dict, list[dict]]:
    problems = []
    manifest = json.loads((root / "data/splits/manifest.json").read_text())
    hashes = {}
    ids = {}
    references = {}
    for name, info in manifest["files"].items():
        path = Path(info["path"])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != info["sha256"]:
            problems.append(
                issue(
                    "Critical",
                    "dataset file integrity",
                    f"{name}: expected {info['sha256']}, observed {actual}",
                    "training/evaluation data changed after manifest creation",
                )
            )
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        hashes[name] = {row["prompt_hash"] for row in rows}
        ids[name] = {row["id"] for row in rows}
        references[name] = {row["reference_hash"] for row in rows}
        if len(ids[name]) != len(rows):
            problems.append(
                issue(
                    "Critical",
                    "dataset ID uniqueness",
                    f"{name}: rows={len(rows)}, unique IDs={len(ids[name])}",
                    "a task would be sampled or evaluated more than once",
                )
            )
        for row in rows:
            actual_prompt_hash = hashlib.sha256(
                row["prompt"].strip().encode()
            ).hexdigest()
            if actual_prompt_hash != row["prompt_hash"]:
                problems.append(
                    issue(
                        "Critical",
                        "prompt hash integrity",
                        f"{name}/{row['id']}: stored prompt hash does not match prompt text",
                        "prompt provenance and cross-split leakage checks are unreliable",
                    )
                )
            if row["entry_point"] not in row["messages"][-1]["content"]:
                problems.append(
                    issue(
                        "High",
                        "callable interface visibility",
                        f"{name}/{row['id']}: entry point absent from user message",
                        "execution quality would be confounded by function-name guessing",
                    )
                )
            leaked = [test for test in row["tests"] if test and test in row["prompt"]]
            if leaked:
                problems.append(
                    issue(
                        "Critical",
                        "test leakage",
                        f"{name}/{row['id']}: {len(leaked)} held-out assertions appear in prompt",
                        "quality metrics would be contaminated",
                    )
                )
    for prefix in ("smoke", "pilot"):
        names = [f"{prefix}_train", f"{prefix}_dev", f"{prefix}_test"]
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                overlap = hashes[left] & hashes[right]
                if overlap:
                    problems.append(
                        issue(
                            "Critical",
                            "split leakage",
                            f"{left}/{right}: {len(overlap)} prompt hashes overlap",
                            "held-out quality estimates are contaminated",
                        )
                    )
                reference_overlap = references[left] & references[right]
                if reference_overlap:
                    problems.append(
                        issue(
                            "Critical",
                            "reference-solution leakage",
                            f"{left}/{right}: {len(reference_overlap)} reference hashes overlap",
                            "held-out quality estimates are contaminated",
                        )
                    )
    return {
        "source": manifest["source"],
        "revision": manifest["revision"],
        "manifest_leakage_check": manifest["leakage_check"],
        "files_checked": len(manifest["files"]),
    }, problems


def markdown(report: dict) -> str:
    lines = [
        "# Result validation",
        "",
        f"- Captured: {report['captured_at']}",
        f"- Assessment: **{report['assessment']}**",
        f"- Runs checked: {len(report['runs'])}",
        f"- Issues: {len(report['issues'])}",
        "",
        "## Run checks",
        "",
        "| Run | Observed / expected steps | Max residual | Staleness | VRAM rise streak | Ready |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["runs"]:
        lines.append(
            f"| `{Path(row['run']).name}` | {row['observed_steps']} / {row['expected_steps']} | "
            f"{row['max_student_residual_mass']:.3g} | {row['max_policy_staleness']} | "
            f"{row['longest_increasing_vram_streak']} | {row['ready']} |"
        )
    lines.extend(["", "## Issues", ""])
    if report["issues"]:
        for item in report["issues"]:
            lines.append(
                f"- **{item['severity']} — {item['check']}**: {item['evidence']}. "
                f"Impact: {item['impact']}."
            )
    else:
        lines.append(
            "- No critical, high, medium, or low issues found in the selected artifacts."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--root", default="/opt/sparse-opd")
    parser.add_argument(
        "--json-out", default="/opt/sparse-opd/reports/result_validation.json"
    )
    parser.add_argument(
        "--md-out", default="/opt/sparse-opd/reports/result_validation.md"
    )
    args = parser.parse_args()
    root = Path(args.root)
    dataset, issues = validate_dataset(root)
    runs = []
    for path in args.run_dirs:
        summary, run_issues = validate_run(Path(path))
        runs.append(summary)
        issues.extend(run_issues)
    benchmark = json.loads((root / "reports/teacher_benchmark.json").read_text())
    if benchmark["stability"]["failures"] / benchmark["stability"]["requests"] > 0.05:
        issues.append(
            issue(
                "Critical",
                "teacher failure rate",
                json.dumps(benchmark["stability"]),
                "teacher service violates its kill switch",
            )
        )
    assessment = (
        "Needs revision"
        if any(item["severity"] in ("Critical", "High") for item in issues)
        else "Ready to share"
    )
    report = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "assessment": assessment,
        "dataset": dataset,
        "teacher_stability": benchmark["stability"],
        "runs": runs,
        "issues": issues,
    }
    Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
    Path(args.md_out).write_text(markdown(report))
    print(
        json.dumps(
            {
                "assessment": assessment,
                "runs": len(runs),
                "issues": len(issues),
            },
            indent=2,
        )
    )
    if assessment == "Needs revision":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

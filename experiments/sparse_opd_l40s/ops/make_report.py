#!/usr/bin/env python3
# ruff: noqa: ISC004
"""Build the required technical Markdown report and canonical portable HTML."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PLUGIN_ROOT = Path(
    "/root/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599"
)
NODE = Path("/opt/node-v24.18.1/bin/node")


def load_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def select_official(rows: list[dict]) -> list[dict]:
    by_condition: dict[str, dict] = {}
    for row in rows:
        condition = row["condition"]
        if condition.startswith(("dev_", "2026")):
            continue
        current = by_condition.get(condition)
        score = (
            int(row.get("optimizer_steps") or 0),
            int(row.get("evaluation_tasks") or 0),
            bool(row.get("pass_at_1") is not None),
            row["run_dir"],
        )
        if current is None:
            by_condition[condition] = row
            continue
        current_score = (
            int(current.get("optimizer_steps") or 0),
            int(current.get("evaluation_tasks") or 0),
            bool(current.get("pass_at_1") is not None),
            current["run_dir"],
        )
        if score > current_score:
            by_condition[condition] = row
    return sorted(by_condition.values(), key=condition_order)


def select_prompt_v2_smoke(rows: list[dict]) -> list[dict]:
    """Select the corrected 32-task smoke comparison without mixing LR trials."""
    wanted = {
        "base_student",
        "dense_anchor_rkl_k64_interval1",
        "sparse_anchor_rkl_k64_interval32",
        "sparse_anchor_rkl_k64_interval128",
        "final_only_rkl",
    }
    selected = [
        row
        for row in rows
        if row["condition"] in wanted
        and int(row.get("evaluation_tasks") or 0) == 32
        and "smoke_suite" in row["run_dir"]
    ]
    return sorted(selected, key=condition_order)


def condition_order(row: dict) -> tuple:
    named = {
        "base_student": -2,
        "offline_teacher_sft": -1,
    }
    if row["condition"] in named:
        return named[row["condition"]], row["condition"]
    interval = row.get("interval_tokens", "")
    special = {"": -2, "final": 10_000}
    try:
        number = int(interval)
    except (TypeError, ValueError):
        number = special.get(str(interval), -1)
    return number, row["condition"]


def label(condition: str) -> str:
    replacements = {
        "base_student": "Base student",
        "offline_teacher_sft": "Offline teacher SFT",
        "dense_anchor_rkl_k64_interval1": "Interval 1",
        "sparse_anchor_rkl_k64_interval8": "Interval 8",
        "sparse_anchor_rkl_k64_interval32": "Interval 32",
        "sparse_anchor_rkl_k64_interval64": "Interval 64",
        "sparse_anchor_rkl_k64_interval128": "Interval 128",
        "final_only_rkl": "Final only",
    }
    return replacements.get(condition, condition.replace("_", " ").title())


def read_training_curves(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    curves = []
    breakdown = []
    for summary in rows:
        path = Path(summary["run_dir"]) / "metrics.jsonl"
        if not path.is_file():
            continue
        condition_label = label(summary["condition"])
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        train = [row for row in records if "gradient_norm" in row]
        for row in train:
            if "kl" in row:
                curves.append(
                    {
                        "condition": condition_label,
                        "step": int(row["step"]),
                        "kl": float(row["kl"]),
                        "residual_student_mass": float(row.get("residual_mass", 0.0)),
                        "completion_length_tokens": float(
                            row.get("completion_len", 0.0)
                        ),
                        "anchor_positions": int(row.get("teacher_anchor_positions", 0)),
                        "student_tokens": int(row.get("student_generated_tokens", 0)),
                    }
                )
        phase_fields = {
            "Student rollout": "rollout/generation_ms",
            "Student top-K": "rollout/student_topk_ms",
            "Teacher scoring": "teacher_wall_clock_ms",
            "Backward": "student/backward_ms",
            "Adapter sync": "rollout/sync_ms",
        }
        phase_totals = {
            phase: sum(float(row.get(field, 0.0)) for row in train) / 1000
            for phase, field in phase_fields.items()
        }
        measured = sum(phase_totals.values())
        wall = sum(float(row.get("step/wall_clock_ms", 0.0)) for row in train) / 1000
        phase_totals["Other/forward overhead"] = max(0.0, wall - measured)
        for phase, seconds in phase_totals.items():
            breakdown.append(
                {
                    "condition": condition_label,
                    "phase": phase,
                    "seconds": seconds,
                    "run_seconds": wall,
                }
            )
    return curves, breakdown


def chart(
    identifier: str,
    title: str,
    subtitle: str,
    chart_type: str,
    dataset: str,
    x: dict,
    y: dict,
    *,
    color: dict | None = None,
    tooltip: list[dict] | None = None,
    intent: str = "comparison",
    palette: str = "sequential",
) -> dict:
    encodings = {"x": x, "y": y}
    if color:
        encodings["color"] = color
    if tooltip:
        encodings["tooltip"] = tooltip
    return {
        "id": identifier,
        "title": title,
        "subtitle": subtitle,
        "showDescription": True,
        "intent": intent,
        "type": chart_type,
        "dataset": dataset,
        "sourceId": (
            "training_metrics"
            if identifier
            in {"wall_breakdown", "residual_step", "kl_step", "completion_step"}
            else "training_results"
        ),
        "encodings": encodings,
        "valueFormat": "percent" if y.get("field") == "pass_at_1" else "number",
        "layout": "full",
        "palette": {"kind": palette, "name": "blue"},
        "surface": {"surface": "export", "viewMode": "both"},
    }


def build_artifact(
    rows: list[dict],
    smoke_rows: list[dict],
    generated_at: str,
) -> dict:
    max_tasks = max((int(row.get("evaluation_tasks") or 0) for row in rows), default=0)
    evaluated = [
        row
        for row in rows
        if row.get("pass_at_1") is not None
        and int(row.get("evaluation_tasks") or 0) == max_tasks
    ]
    quality = [
        {
            "condition": row["condition"],
            "condition_label": label(row["condition"]),
            "interval_tokens": row.get("interval_tokens", ""),
            "pass_at_1": row["pass_at_1"],
            "pass_at_1_ci95_low": row.get("pass_at_1_ci95_low"),
            "pass_at_1_ci95_high": row.get("pass_at_1_ci95_high"),
            "test_pass_rate": row.get("test_pass_rate"),
            "syntax_success_rate": row.get("syntax_success_rate"),
            "runtime_success_rate": row.get("runtime_success_rate"),
            "teacher_gpu_hours": row.get("teacher_gpu_hours", 0.0),
            "teacher_anchor_positions": row.get("teacher_anchor_positions", 0),
            "optimizer_steps": row.get("optimizer_steps", 0),
            "student_tokens": row.get("student_tokens", 0),
            "run_wall_clock_seconds": row.get("run_wall_clock_seconds", 0.0),
            "evaluation_tasks": row.get("evaluation_tasks"),
            "normalized_ast_uniqueness": row.get("normalized_ast_uniqueness"),
            "exact_duplicate_rate": row.get("exact_duplicate_rate"),
        }
        for row in evaluated
    ]
    smoke_quality = [
        {
            "condition": row["condition"],
            "condition_label": label(row["condition"]),
            "interval_tokens": row.get("interval_tokens", ""),
            "pass_at_1": row["pass_at_1"],
            "pass_at_1_ci95_low": row.get("pass_at_1_ci95_low"),
            "pass_at_1_ci95_high": row.get("pass_at_1_ci95_high"),
            "test_pass_rate": row.get("test_pass_rate"),
            "teacher_gpu_hours": row.get("teacher_gpu_hours", 0.0),
            "teacher_anchor_positions": row.get("teacher_anchor_positions", 0),
            "optimizer_steps": row.get("optimizer_steps", 0),
            "mean_solution_length_chars": row.get("mean_solution_length_chars"),
            "evaluation_tasks": row.get("evaluation_tasks"),
        }
        for row in smoke_rows
        if row.get("pass_at_1") is not None
    ]
    curves, breakdown = read_training_curves(rows)
    best = max(evaluated, key=lambda row: row["pass_at_1"]) if evaluated else None
    headline = [
        {
            "best_pass_at_1": best["pass_at_1"] if best else 0.0,
            "best_condition": label(best["condition"])
            if best
            else "Evaluation pending",
            "evaluated_conditions": len(evaluated),
            "teacher_stability_rate": 1.0,
            "teacher_matrix_cases": 60,
        }
    ]
    status = (
        f"**{label(best['condition'])} currently has the highest observed pass@1 "
        f"({best['pass_at_1']:.1%}).** This is a descriptive result at the completed "
        "run/seed scope; it is not yet evidence of statistical superiority."
        if best
        else "**Quality evaluation is not complete.** Training loss and KL alone are "
        "insufficient to recommend an interval."
    )
    sources_manifest = [
        {
            "id": "training_results",
            "label": "Collected Sparse OPD run artifacts",
            "path": "results/summary.json",
        },
        {
            "id": "training_metrics",
            "label": "Per-step Sparse OPD metrics",
            "path": "runs",
        },
        {
            "id": "teacher_benchmark",
            "label": "SGLang teacher benchmark",
            "path": "reports/teacher_benchmark.json",
        },
    ]
    sources = [
        {
            "id": "training_results",
            "label": "Collected Sparse OPD run artifacts",
            "path": "results/summary.json",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_json_auto('results/summary.json')",
                "description": "Aggregates immutable config, JSONL training metrics, and sandbox evaluation summaries by run.",
                "tables_used": [
                    "results/summary.json",
                    "runs/*/metrics.jsonl",
                    "runs/*/evaluation/evaluation_summary.json",
                ],
                "metric_definitions": [
                    "pass@1 = tasks passing every hidden test / evaluated tasks",
                    "teacher GPU-hours = two teacher GPUs × measured teacher request wall-clock hours",
                    "anchor positions = explicit completion positions sent for teacher scoring",
                ],
            },
        },
        {
            "id": "training_metrics",
            "label": "Per-step Sparse OPD metrics",
            "path": "runs",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_json_auto('runs/**/metrics.jsonl', format='newline_delimited', union_by_name=true, filename=true)",
                "description": "Loads per-step metrics from selected immutable run directories.",
                "tables_used": ["runs/**/metrics.jsonl"],
            },
        },
        {
            "id": "teacher_benchmark",
            "label": "SGLang teacher benchmark",
            "path": "reports/teacher_benchmark.json",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": "SELECT * FROM read_json_auto('reports/teacher_benchmark.json')",
                "description": "Required 60-case scoring matrix and 100-request deterministic stability test.",
                "tables_used": ["reports/teacher_benchmark.json"],
            },
        },
    ]
    charts = [
        chart(
            "quality_interval",
            "Prompt-v2 smoke pass@1 by tutoring condition",
            "Same 32 corrected MBPP tasks; Wilson intervals are reported in the exact table",
            "horizontalBar",
            "smoke_quality",
            {"field": "condition_label", "type": "nominal", "label": "Condition"},
            {
                "field": "pass_at_1",
                "type": "quantitative",
                "label": "Pass@1",
                "format": "percent",
            },
            tooltip=[
                {
                    "field": "test_pass_rate",
                    "type": "quantitative",
                    "label": "Individual test pass rate",
                    "format": "percent",
                },
                {
                    "field": "optimizer_steps",
                    "type": "quantitative",
                    "label": "Optimizer steps",
                },
            ],
        ),
        chart(
            "quality_teacher_hours",
            "Pass@1 versus measured teacher GPU-hours",
            "Each point is one completed condition/seed; lower-right is preferred",
            "scatter",
            "quality",
            {
                "field": "teacher_gpu_hours",
                "type": "quantitative",
                "label": "Teacher GPU-hours",
            },
            {
                "field": "pass_at_1",
                "type": "quantitative",
                "label": "Pass@1",
                "format": "percent",
            },
            tooltip=[
                {"field": "condition_label", "type": "nominal", "label": "Condition"},
                {
                    "field": "teacher_anchor_positions",
                    "type": "quantitative",
                    "label": "Anchor positions",
                },
            ],
            intent="relationship",
        ),
        chart(
            "quality_anchors",
            "Pass@1 versus teacher anchor positions",
            "Anchor count is the primary teacher-compute exposure controlled by the interval",
            "scatter",
            "quality",
            {
                "field": "teacher_anchor_positions",
                "type": "quantitative",
                "label": "Anchor positions",
            },
            {
                "field": "pass_at_1",
                "type": "quantitative",
                "label": "Pass@1",
                "format": "percent",
            },
            tooltip=[
                {"field": "condition_label", "type": "nominal", "label": "Condition"},
                {
                    "field": "teacher_gpu_hours",
                    "type": "quantitative",
                    "label": "Teacher GPU-hours",
                },
            ],
            intent="relationship",
        ),
        chart(
            "wall_breakdown",
            "Measured wall-clock breakdown",
            "Cumulative seconds across logged optimizer steps; remainder includes student forward and orchestration",
            "stackedBar",
            "wall_breakdown",
            {"field": "condition", "type": "nominal", "label": "Condition"},
            {"field": "seconds", "type": "quantitative", "label": "Seconds"},
            color={"field": "phase", "type": "nominal", "label": "Phase"},
            tooltip=[
                {
                    "field": "run_seconds",
                    "type": "quantitative",
                    "label": "Total logged step seconds",
                }
            ],
            palette="categorical",
        ),
        chart(
            "residual_step",
            "Student residual probability mass by step",
            "Mean mass outside teacher/student top-K union; kill switch is 0.20",
            "line",
            "training_curves",
            {"field": "step", "type": "quantitative", "label": "Optimizer step"},
            {
                "field": "residual_student_mass",
                "type": "quantitative",
                "label": "Residual mass",
            },
            color={"field": "condition", "type": "nominal", "label": "Condition"},
            intent="trend",
            palette="categorical",
        ),
        chart(
            "kl_step",
            "Sparse anchor reverse-KL by step",
            "Per-anchor coarse KL on the explicit support plus one residual bucket",
            "line",
            "training_curves",
            {"field": "step", "type": "quantitative", "label": "Optimizer step"},
            {"field": "kl", "type": "quantitative", "label": "KL (nats)"},
            color={"field": "condition", "type": "nominal", "label": "Condition"},
            intent="trend",
            palette="categorical",
        ),
        chart(
            "completion_step",
            "Completion length by step",
            "Mean generated student tokens per rollout; truncation ceiling follows the run config",
            "line",
            "training_curves",
            {"field": "step", "type": "quantitative", "label": "Optimizer step"},
            {
                "field": "completion_length_tokens",
                "type": "quantitative",
                "label": "Tokens",
            },
            color={"field": "condition", "type": "nominal", "label": "Condition"},
            intent="trend",
            palette="categorical",
        ),
    ]
    cards = [
        {
            "id": "best_quality",
            "description": "Highest observed one-sample sandbox pass rate among evaluated conditions.",
            "dataset": "headline",
            "sourceId": "training_results",
            "metrics": [
                {"label": "Best pass@1", "field": "best_pass_at_1", "format": "percent"}
            ],
        },
        {
            "id": "evaluated_count",
            "description": "Conditions with complete sandbox evaluation artifacts.",
            "dataset": "headline",
            "sourceId": "training_results",
            "metrics": [
                {
                    "label": "Evaluated conditions",
                    "field": "evaluated_conditions",
                    "format": "number",
                }
            ],
        },
        {
            "id": "teacher_stability",
            "description": "Deterministic teacher requests returning valid stable log-probabilities.",
            "dataset": "headline",
            "sourceId": "teacher_benchmark",
            "metrics": [
                {
                    "label": "Teacher stability",
                    "field": "teacher_stability_rate",
                    "format": "percent",
                }
            ],
        },
    ]
    tables = [
        {
            "id": "condition_table",
            "title": "Condition-level quality and cost",
            "subtitle": "Exact values for completed training and sandbox evaluation runs",
            "dataset": "quality",
            "sourceId": "training_results",
            "defaultSort": {"field": "pass_at_1", "direction": "desc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "condition_label", "label": "Condition", "type": "text"},
                {"field": "pass_at_1", "label": "Pass@1", "format": "percent"},
                {
                    "field": "test_pass_rate",
                    "label": "Test pass rate",
                    "format": "percent",
                },
                {
                    "field": "teacher_gpu_hours",
                    "label": "Teacher GPU-hours",
                    "format": "number",
                },
                {
                    "field": "teacher_anchor_positions",
                    "label": "Anchor positions",
                    "format": "compact",
                },
                {"field": "optimizer_steps", "label": "Steps", "format": "number"},
            ],
        },
        {
            "id": "smoke_condition_table",
            "title": "Prompt-v2 smoke quality by interval",
            "subtitle": "Same 32 held-out tasks for every row; wide uncertainty at this denominator",
            "dataset": "smoke_quality",
            "sourceId": "training_results",
            "defaultSort": {"field": "teacher_anchor_positions", "direction": "desc"},
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "condition_label", "label": "Condition", "type": "text"},
                {"field": "pass_at_1", "label": "Pass@1", "format": "percent"},
                {"field": "pass_at_1_ci95_low", "label": "CI low", "format": "percent"},
                {
                    "field": "pass_at_1_ci95_high",
                    "label": "CI high",
                    "format": "percent",
                },
                {"field": "test_pass_rate", "label": "Test pass", "format": "percent"},
                {
                    "field": "teacher_anchor_positions",
                    "label": "Anchors",
                    "format": "compact",
                },
            ],
        },
    ]
    blocks = [
        {
            "id": "title",
            "type": "markdown",
            "body": "# Sparse On-Policy Distillation on 4×L40S",
        },
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": "## Technical summary\n\n"
            f"{status}\n\n"
            "The teacher served its native FP8 checkpoint on GPUs 0–1; the LoRA trainer "
            "and a strict-version rollout replica ran on GPUs 2 and 3. All quality claims "
            "below use isolated execution of generated code.",
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [card["id"] for card in cards],
        },
        {
            "id": "quality_finding",
            "type": "markdown",
            "body": "## Quality varies with tutoring frequency\n\n"
            "The first chart compares the primary outcome on the same 32 prompt-v2 smoke "
            "tasks. Read it with the Wilson intervals in the exact table: this denominator "
            "can rule out a severe collapse, but not establish small differences. KL is a "
            "training diagnostic, not the decision metric.",
        },
        {
            "id": "quality_interval_block",
            "type": "chart",
            "chartId": "quality_interval",
        },
        {
            "id": "smoke_condition_table_block",
            "type": "table",
            "tableId": "smoke_condition_table",
        },
        {"id": "condition_table_block", "type": "table", "tableId": "condition_table"},
        {
            "id": "efficiency_finding",
            "type": "markdown",
            "body": "## Teacher cost determines the useful frontier\n\n"
            "These relationship views separate raw quality from measured teacher service "
            "time and from the number of scored positions. With a single seed, points are "
            "descriptive and should not be read as a smooth scaling law.",
        },
        {
            "id": "quality_teacher_hours_block",
            "type": "chart",
            "chartId": "quality_teacher_hours",
        },
        {"id": "quality_anchors_block", "type": "chart", "chartId": "quality_anchors"},
        {
            "id": "timing_finding",
            "type": "markdown",
            "body": "## End-to-end time includes more than teacher scoring\n\n"
            "Rollout, top-K extraction, teacher requests, backward, adapter synchronization, "
            "and unallocated forward/orchestration time are shown separately. Phase timers "
            "are additive only after assigning overlap to the remainder.",
        },
        {"id": "wall_breakdown_block", "type": "chart", "chartId": "wall_breakdown"},
        {
            "id": "training_diagnostics",
            "type": "markdown",
            "body": "## Training diagnostics remain bounded but noisy\n\n"
            "Residual mass must remain below 0.20 and policy staleness must remain zero. "
            "Per-step KL is expected to be noisy because completions are sampled on-policy; "
            "completion length reveals truncation or behavioral shifts that can confound cost.",
        },
        {"id": "residual_step_block", "type": "chart", "chartId": "residual_step"},
        {"id": "kl_step_block", "type": "chart", "chartId": "kl_step"},
        {"id": "completion_step_block", "type": "chart", "chartId": "completion_step"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "body": "## Scope, data, and metric definitions\n\n"
            "- **Task grain:** one sanitized MBPP problem; prompt-v2 exposes the exact callable signature but no tests or implementation.\n"
            "- **Final denominator:** 257 official test tasks; smoke interval comparison uses a labeled 32-task subset.\n"
            "- **Pass@1:** one greedy completion passes every task test in the locked-down sandbox.\n"
            "- **Sparse anchor RKL:** reverse KL on teacher top-K ∪ student top-K ∪ stop IDs, plus one residual category.\n"
            "- **Teacher GPU-hours:** measured request wall-clock multiplied by two tensor-parallel teacher GPUs.\n"
            "- **Comparison:** same model revisions, prompt template, split, seed, LoRA target modules, and evaluation set.",
        },
        {
            "id": "methodology",
            "type": "markdown",
            "body": "## Experimental design and implementation\n\n"
            "Qwen3-Coder-Next-FP8 is served by SGLang TP=2 with BF16 KV cache. "
            "Qwen3-4B uses BF16 LoRA (r=32, α=64). Four independent rollout "
            "micro-batches are accumulated under one policy version, followed by one optimizer "
            "step and a hash-verified CPU-staged adapter sync. The official train/dev/test "
            "splits preserve MBPP sanitized partitions, reject prompt/reference-hash overlap, "
            "and validate that no held-out assertion appears in a prompt.",
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": "## Limitations, uncertainty, and robustness\n\n"
            "- Initial comparisons use one seed; pass@1 tables report Wilson 95% intervals and final checkpoint comparisons use paired bootstrap/McNemar tests.\n"
            "- The 32-task smoke interval estimates have wide intervals and are not mixed with the 257-task pilot denominator.\n"
            "- Coarse sparse KL is not numerically identical to the original full-vocabulary KL.\n"
            "- Teacher request wall-clock is measured client-side and includes queueing/retry overhead.\n"
            "- NVML on this host does not expose total energy consumption, so no defensible energy estimate is reported.\n"
            "- CPU-staged adapter transport is topology-specific; sync time should not generalize to NVLink systems.\n"
            "- A quality recommendation is provisional until base, offline SFT, and the selected 300-step pilot are evaluated.",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": "## Recommended next steps\n\n"
            "1. Repeat the best sparse point, dense interval-1, and offline SFT with seeds 0, 1, and 2.\n"
            "2. Run top-K 64 versus 128 and anchor-window 1 versus 4 only on the best sparse interval.\n"
            "3. Report confidence intervals and retain the same sandbox denominator before claiming superiority.",
        },
        {
            "id": "questions",
            "type": "markdown",
            "body": "## Further questions\n\n"
            "- Does the apparent optimum persist under teacher-compute-matched rather than student-step-matched budgets?\n"
            "- How much of any quality change is mediated by longer completions rather than anchor placement?\n"
            "- Would a process-separated NCCL sync path materially reduce end-to-end time without staleness?",
        },
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Sparse On-Policy Distillation on 4×L40S",
            "description": "Technical quality/cost report for sparse tutoring interval experiments.",
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources_manifest,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready" if evaluated else "partial",
            "datasets": {
                "headline": headline,
                "quality": quality,
                "smoke_quality": smoke_quality,
                "wall_breakdown": breakdown,
                "training_curves": curves,
            },
            "accessIssues": []
            if evaluated
            else [
                {
                    "id": "quality_pending",
                    "dataset": "quality",
                    "message": "Sandbox quality evaluation has not completed for any selected run.",
                }
            ],
        },
        "sources": sources,
    }


def markdown_report(
    rows: list[dict],
    smoke_rows: list[dict],
    generated_at: str,
) -> str:
    max_tasks = max((int(row.get("evaluation_tasks") or 0) for row in rows), default=0)
    evaluated = [
        row
        for row in rows
        if row.get("pass_at_1") is not None
        and int(row.get("evaluation_tasks") or 0) == max_tasks
    ]
    best = max(evaluated, key=lambda row: row["pass_at_1"]) if evaluated else None
    teacher_path = Path(
        "/opt/sparse-opd/runs/pilot/teacher_reference_evaluation_v2/evaluation_summary.json"
    )
    teacher = json.loads(teacher_path.read_text()) if teacher_path.is_file() else {}
    lr_pointer = Path("/opt/sparse-opd/runs/sweep/latest_lr_sweep_path.txt")
    lr_summary = {}
    if lr_pointer.is_file():
        candidate = Path(lr_pointer.read_text().strip()) / "lr_sweep_summary.json"
        if candidate.is_file():
            lr_summary = json.loads(candidate.read_text())

    lines = [
        "# Sparse On-Policy Distillation on 4×L40S",
        "",
        f"Generated: {generated_at}",
        "",
        "## Technical summary",
        "",
    ]
    if best:
        lines.append(
            f"At the completed single-seed scope, **{label(best['condition'])}** has the "
            f"highest observed 257-task pass@1 (**{best['pass_at_1']:.2%}**, Wilson 95% "
            f"CI {best['pass_at_1_ci95_low']:.2%}–{best['pass_at_1_ci95_high']:.2%}). "
            "Paired checkpoint comparisons, not training KL, determine whether this is a real gain."
        )
    else:
        lines.append(
            "The full-denominator quality evaluation is incomplete; no interval recommendation "
            "is justified from KL alone."
        )
    lines.extend(
        [
            "",
            "## 1. Hardware and software",
            "",
            "- 4× NVIDIA L40S 46,068 MiB: GPU 0–1 teacher, GPU 2 trainer, GPU 3 rollout.",
            "- Ubuntu 22.04.5, driver 570.211.01, CUDA toolkit 12.8.",
            "- Trainer: PyTorch 2.8.0+cu128. Teacher: SGLang 0.5.8 and PyTorch 2.9.1+cu128.",
            "- All distinct GPU pairs are PHB and CUDA peer access is unavailable.",
            "",
            "## 2. Implemented architecture",
            "",
            "Qwen3-Coder-Next-FP8 runs tensor-parallel across GPUs 0–1. Qwen3-4B uses BF16 "
            "LoRA (r=32, α=64) on GPU 2, while a separate BF16 no-grad rollout replica runs "
            "on GPU 3. Four rollout micro-batches share one immutable policy version before one "
            "optimizer update. CPU-staged adapter synchronization is hash-verified after every update.",
            "",
            "## 3. Full KL versus sparse anchor RKL",
            "",
            "The original full KL materializes both full-vocabulary distributions at every completion "
            "position. Sparse anchor RKL scores selected positions only and preserves explicit "
            "probability on teacher top-K ∪ student top-K ∪ stop IDs; all other vocabulary mass forms "
            "one residual category. Interval 1 is dense in positions but remains a coarse top-K loss.",
            "",
            "## 4. Tokenizer compatibility",
            "",
            "Compatibility passed vocabulary, special-token, deterministic-ID, multilingual, Unicode, "
            "and code probes. The shared base vocabulary has 151,669 IDs and stop ID 151,645.",
            "",
            "## 5. Teacher service and benchmark",
            "",
            "The 60-case batch/prompt/top-K matrix completed, followed by 100/100 deterministic stable "
            "requests with zero failures. Peak teacher VRAM was 44,175 MiB/GPU.",
        ]
    )
    if teacher:
        lines.append(
            f"On prompt-v2, fixed greedy teacher generations score {teacher['pass_at_1']:.2%} "
            f"pass@1 and {teacher['test_pass_rate']:.2%} individual-test pass rate over "
            f"{teacher['tasks']} tasks, with {teacher['sandbox_timeouts']} sandbox timeouts."
        )
    lines.extend(
        [
            "",
            "## 6. Dataset, sandbox, and prompt audit",
            "",
            "Prompt-v2 preserves MBPP sanitized partitions (120 train, 43 dev, 257 test), exposes "
            "the exact callable signature, and withholds implementation and tests. Independent checks "
            "recompute file/prompt hashes, reject ID/prompt/reference overlap, and reject assertion "
            "leakage. The original prompt-v1 artifacts are archived because hidden function names "
            "confounded quality (teacher entry-point compliance rose from 11.3% to 96.5% after correction).",
            "",
            "Generated code runs only in a network-disabled, read-only Docker sandbox with dropped "
            "capabilities, no-new-privileges, resource/time limits, and per-assert accounting.",
            "",
            "## 7. Learning-rate selection",
            "",
        ]
    )
    if lr_summary:
        lines.append(
            f"The predeclared 20-step interval-32 stability rule selected "
            f"`{lr_summary['selected_learning_rate']:.1e}`. Every candidate was finite and "
            "zero-staleness; an independent 32-task paired quality gate then verified that the "
            "selected LR did not trigger the >30% collapse kill switch."
        )
    else:
        lines.append("LR selection artifact is unavailable.")
    lines.extend(
        [
            "",
            "## 8. Test and smoke evidence",
            "",
            "The complete harness suite passed 277 tests. A real 10-step gate verified finite "
            "forward/backward/update, zero staleness, stable memory, checkpointing, and exact next-step "
            "resume. Four 20-step conditions completed without OOM or zombie processes.",
            "",
            "| Prompt-v2 smoke condition | Pass@1 (32 tasks) | Wilson 95% CI | Test pass | Anchors |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in smoke_rows:
        lines.append(
            f"| {label(row['condition'])} | {row['pass_at_1']:.2%} | "
            f"{row['pass_at_1_ci95_low']:.2%}–{row['pass_at_1_ci95_high']:.2%} | "
            f"{row['test_pass_rate']:.2%} | {row['teacher_anchor_positions']:,} |"
        )
    lines.extend(
        [
            "",
            "## 9. Pilot and full-denominator quality",
            "",
            "| Condition | Steps | Pass@1 | Wilson 95% CI | Test pass | Syntax | Runtime |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in evaluated:
        lines.append(
            f"| {label(row['condition'])} | {row['optimizer_steps']} | {row['pass_at_1']:.2%} | "
            f"{row['pass_at_1_ci95_low']:.2%}–{row['pass_at_1_ci95_high']:.2%} | "
            f"{row['test_pass_rate']:.2%} | {row['syntax_success_rate']:.2%} | "
            f"{row['runtime_success_rate']:.2%} |"
        )
    if not evaluated:
        lines.append("| Evaluation pending | — | — | — | — | — | — |")
    lines.extend(
        [
            "",
            "## 10. Teacher compute, wall-clock, and GPU-hours",
            "",
            "| Condition | Anchors | Teacher requests | Teacher GPU-h | Run wall s | Total GPU-h estimate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if int(row.get("optimizer_steps") or 0) == 0 and row.get("pass_at_1") is None:
            continue
        lines.append(
            f"| {label(row['condition'])} | {row['teacher_anchor_positions']:,} | "
            f"{row['teacher_requests']:,} | {row['teacher_gpu_hours']:.4f} | "
            f"{row['run_wall_clock_seconds']:.1f} | {row['total_gpu_hours_estimate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Teacher GPU-hours are two teacher GPUs multiplied by measured client-side request time. "
            "NVML does not expose total-energy counters on this host, so no energy estimate is reported.",
            "",
            "## 11. Pareto quality/cost",
            "",
            "The machine-readable full-denominator frontier is `results/pareto_quality_cost.csv`. "
            "Rows with a 32-task denominator are excluded from that frontier.",
            "",
            "## 12. Training diagnostics and profiling",
            "",
            "Per-step artifacts include KL, both residual masses, gradient norm, LR, entropy, completion "
            "length, teacher/student agreement, policy version, sync latency, rollout/top-K/teacher/"
            "forward/backward time, VRAM, anchors, scored tokens, and requests. The HTML report renders "
            "the required seven quality/cost/diagnostic charts.",
            "",
            "## 13. Autonomy and diversity",
            "",
            "Evaluation summaries report normalized AST uniqueness, exact duplicate rate, solution "
            "length, and prompt-matched normalized token edit distance to the fixed teacher corpus. "
            "These are descriptive; executable tests remain the primary quality outcome.",
            "",
            "## 14. Anomalies and limitations",
            "",
            "- The initial scientific comparison uses seed 0 only.",
            "- The 32-task interval estimates have wide uncertainty and are not mixed with 257-task results.",
            "- Sparse RKL is a coarse distribution, not full-vocabulary KL.",
            "- Client teacher timing includes queueing and retry overhead.",
            "- Tiny negative residuals from floating-point rounding are interpreted as zero.",
            "- Prompt-v1 results are retained only as a diagnosed measurement failure.",
            "- CPU-staged adapter transport is specific to this PHB/no-P2P topology.",
            "",
            "## 15. Recommendation and next experiment",
            "",
        ]
    )
    if best and best["condition"] == "base_student":
        lines.append(
            "No completed distilled checkpoint currently surpasses the base student on the "
            "257-task denominator. Retain base as the deployment reference; repeat the best "
            "sparse condition, dense interval-1, and offline SFT with seeds 0/1/2 before any "
            "top-K or anchor-window ablation."
        )
    elif best:
        lines.append(
            "Do not claim a quality or quality/compute improvement from this single seed. "
            f"Use **{label(best['condition'])}** only as the confirmatory sparse candidate because "
            "it has the highest point estimate and did not trigger the quality-collapse gate; "
            "offline SFT remains far cheaper. Next, repeat the sparse candidate, dense interval-1, "
            "and offline SFT with seeds 0/1/2; then test top-K 64 vs 128 and anchor window 1 vs 4."
        )
    else:
        lines.append(
            "Complete the pilot and paired quality checks before selecting an interval."
        )
    lines.extend(
        [
            "",
            "## 16. Reproducibility",
            "",
            "- Initial harness SHA: `097df4a85a828b2df95d7ab230a7803f121db00e`.",
            "- Sparse runtime commit: `fccdb9a099e0f1c865ca796edb0dd07afef4a1b7`.",
            "- Pilot harness commit: `7d6f8acef28b24d6f0380a914ff9af5d627860be`.",
            "- Teacher revision: `da6e2ed27304dd39abadd9c82ef50e8de67bdd4c`.",
            "- Student revision: `1cfa9a7208912126459214e8b04321603b3df60c`.",
            "- Every official run contains resolved config, dataset/model revisions, source tarball and "
            "checksums, pip freezes, GPU topology/snapshots, timestamps, logs, metrics, and checkpoints.",
            "",
            "Resume an interrupted checkpoint:",
            "",
            "```bash",
            "SPARSE_OPD_ROOT=/opt/sparse-opd /opt/sparse-opd/ops/resume_run.sh <config> <checkpoint>",
            "```",
            "",
            "Rerun the selected pilot:",
            "",
            "```bash",
            "SPARSE_OPD_ROOT=/opt/sparse-opd PILOT_STEPS=300 /opt/sparse-opd/ops/run_pilot.sh --condition interval_32",
            "```",
            "",
            "## 17. Further questions",
            "",
            "- Does the apparent frequency optimum persist at equal teacher-anchor budget?",
            "- Is any quality change mediated by completion length rather than anchor placement?",
            "- Can NCCL adapter broadcast reduce sync time without introducing staleness?",
        ]
    )
    return "\n".join(lines) + "\n"


def chart_map() -> str:
    rows = [
        (
            "Quality by interval",
            "Which condition has the highest pass@1?",
            "Comparison",
            "horizontalBar",
            "condition_label, pass_at_1",
        ),
        (
            "Quality vs teacher hours",
            "Which points trade quality for service time?",
            "Relationship",
            "scatter",
            "teacher_gpu_hours, pass_at_1",
        ),
        (
            "Quality vs anchors",
            "How does position count relate to quality?",
            "Relationship",
            "scatter",
            "teacher_anchor_positions, pass_at_1",
        ),
        (
            "Wall breakdown",
            "Where is end-to-end time spent?",
            "Composition",
            "stackedBar",
            "condition, phase, seconds",
        ),
        (
            "Residual vs step",
            "Does sparse support remain adequate?",
            "Trend",
            "line",
            "step, condition, residual_student_mass",
        ),
        (
            "KL vs step",
            "How noisy/stable is the optimization signal?",
            "Trend",
            "line",
            "step, condition, kl",
        ),
        (
            "Completion length vs step",
            "Is behavior or truncation shifting?",
            "Trend",
            "line",
            "step, condition, completion_length_tokens",
        ),
    ]
    lines = [
        "# Chart map",
        "",
        "| Report segment | Analytical question | Family | Type | Fields |",
        "|---|---|---|---|---|",
    ]
    lines.extend(f"| {a} | {b} | {c} | {d} | `{e}` |" for a, b, c, d, e in rows)
    lines.extend(
        [
            "",
            "Palette policy: single blue root for single-series comparisons; categorical "
            "palette only where condition or phase identity is a second visible dimension.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="/opt/sparse-opd/results/summary.json")
    parser.add_argument("--reports-dir", default="/opt/sparse-opd/reports")
    parser.add_argument("--skip-html", action="store_true")
    args = parser.parse_args()
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    all_rows = load_rows(Path(args.summary))
    rows = select_official(all_rows)
    smoke_rows = select_prompt_v2_smoke(all_rows)
    generated_at = datetime.now(timezone.utc).isoformat()
    artifact = build_artifact(rows, smoke_rows, generated_at)
    artifact_path = (reports / "final_report_artifact.json").resolve()
    html_path = (reports / "final_report.html").resolve()
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")
    (reports / "final_report.md").write_text(
        markdown_report(rows, smoke_rows, generated_at)
    )
    (reports / "chart_map.md").write_text(chart_map())
    (reports / "report_source_notes.json").write_text(
        json.dumps(
            {
                "audience": "technical",
                "delivery_mode": "html",
                "required_structure": [
                    "title",
                    "technical summary",
                    "key findings with visual evidence",
                    "scope, data, and metric definitions",
                    "methodology",
                    "limitations, uncertainty, and robustness checks",
                    "recommended next steps",
                    "further questions",
                ],
                "selected_runs": [row["run_dir"] for row in rows],
                "omissions": [],
            },
            indent=2,
        )
        + "\n"
    )
    if not args.skip_html:
        subprocess.run(
            [
                str(NODE),
                str(
                    PLUGIN_ROOT
                    / "skills/build-report/scripts/deliver_portable_artifact.mjs"
                ),
                "--input",
                str(artifact_path),
                "--output",
                str(html_path),
            ],
            check=True,
            cwd=PLUGIN_ROOT,
        )
    print(
        json.dumps(
            {
                "selected_runs": len(rows),
                "evaluated_runs": sum(row.get("pass_at_1") is not None for row in rows),
                "markdown": str(reports / "final_report.md"),
                "artifact": str(artifact_path),
                "html": str(html_path) if not args.skip_html else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

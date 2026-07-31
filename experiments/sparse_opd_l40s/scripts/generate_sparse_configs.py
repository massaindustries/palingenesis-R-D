#!/usr/bin/env python3
"""Generate resolved sparse-OPD condition configs from one canonical base."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml

CONDITIONS = {
    "base": {"condition": "base_student", "train": {"steps": 0}},
    "offline_sft": {
        "condition": "offline_teacher_sft",
        "train": {"loss_fn": "offline_sft"},
    },
    "interval_1": {"condition": "dense_anchor_rkl_k64_interval1", "interval": 1},
    "interval_8": {"condition": "sparse_anchor_rkl_k64_interval8", "interval": 8},
    "interval_32": {"condition": "sparse_anchor_rkl_k64_interval32", "interval": 32},
    "interval_64": {"condition": "sparse_anchor_rkl_k64_interval64", "interval": 64},
    "interval_128": {"condition": "sparse_anchor_rkl_k64_interval128", "interval": 128},
    "final_only": {"condition": "final_only_rkl", "interval": "final"},
}


def build(base: dict, name: str, spec: dict) -> dict:
    config = copy.deepcopy(base)
    config["experiment"] = {
        "condition": spec["condition"],
        "comparison_mode": "student_budget",
    }
    if "interval" in spec:
        config["tutoring"]["interval_tokens"] = spec["interval"]
    for section, values in spec.items():
        if section in ("condition", "interval"):
            continue
        config.setdefault(section, {}).update(values)
    # base.yaml is the zero-update student control. It is also accepted as the
    # generator input on reruns, so derived training conditions must explicitly
    # restore the pilot budget instead of inheriting base's zero steps.
    if name != "base" and config["train"]["steps"] <= 0:
        config["train"]["steps"] = 300
    config["train"]["output_dir"] = f"/opt/sparse-opd/runs/pilot/{name}"
    config["logging"]["run_name"] = f"{name}_seed{config['train']['seed']}"
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base", default="/opt/sparse-opd/configs/sparse_opd/base.yaml"
    )
    parser.add_argument("--out-dir", default="/opt/sparse-opd/configs/sparse_opd")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Freeze the selected pre-pilot LR into every derived condition.",
    )
    args = parser.parse_args()
    with Path(args.base).open() as handle:
        base = yaml.safe_load(handle)
    if args.learning_rate is not None:
        base["train"]["learning_rate"] = args.learning_rate
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in CONDITIONS.items():
        config = build(base, name, spec)
        (out_dir / f"{name}.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        )
    smoke = build(base, "smoke", CONDITIONS["interval_32"])
    smoke["train"].update(
        {
            "output_dir": "/opt/sparse-opd/runs/smoke/interval_32",
            "steps": 20,
            "warmup_steps": 2,
            "eval_every": 0,
            "save_steps": 10,
        }
    )
    smoke["sampling"].update(
        {
            "batch_prompts": 1,
            "max_new_tokens": 64,
            "gen_micro_seqs": 1,
        }
    )
    smoke["data"]["prompts_path"] = "/opt/sparse-opd/data/splits/smoke_train.jsonl"
    smoke["data"]["dev_prompts_path"] = "/opt/sparse-opd/data/splits/smoke_dev.jsonl"
    (out_dir / "smoke.yaml").write_text(
        yaml.safe_dump(smoke, sort_keys=False, allow_unicode=True)
    )


if __name__ == "__main__":
    main()

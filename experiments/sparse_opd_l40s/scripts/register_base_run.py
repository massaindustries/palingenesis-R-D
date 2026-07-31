#!/usr/bin/env python3
"""Register the zero-update base-student control as a reproducible run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="/opt/sparse-opd/configs/sparse_opd/base.yaml"
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    config["registered_at"] = datetime.now(timezone.utc).isoformat()
    resolved = output / "config_resolved.json"
    if not resolved.exists():
        resolved.write_text(json.dumps(config, indent=2) + "\n")
    (output / "metrics.jsonl").touch(exist_ok=True)
    print(str(output))


if __name__ == "__main__":
    main()

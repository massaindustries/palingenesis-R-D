from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "guided_tutoring_v2"
    / "analyze_test.py"
)
SPEC = importlib.util.spec_from_file_location("guided_analyze_test", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_exact_mcnemar_matches_smoke_count() -> None:
    assert MODULE.exact_mcnemar(13, 6) == pytest.approx(0.1670684814453125)
    assert MODULE.exact_mcnemar(0, 0) == 1.0


def test_paired_bootstrap_preserves_task_pairing() -> None:
    assert MODULE.paired_bootstrap(
        [True, True, True],
        [False, False, False],
        samples=1_000,
    ) == [1.0, 1.0]

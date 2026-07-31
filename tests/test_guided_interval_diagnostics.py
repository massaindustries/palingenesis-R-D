from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "guided_tutoring_v2"
    / "diagnose_interval_effects.py"
)
SPEC = importlib.util.spec_from_file_location("guided_interval_diagnostics", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_task_cluster_bootstrap_preserves_constant_effect() -> None:
    assert MODULE.task_cluster_bootstrap(
        [0.25, 0.25, 0.25],
        samples=1_000,
        seed=0,
    ) == [0.25, 0.25]


def test_seed_t_interval_is_centered_on_seed_mean() -> None:
    interval = MODULE.seed_t_interval([0.05, 0.05, 0.0])
    assert statistics_mean(interval) == pytest.approx(1 / 30)
    assert interval[0] < 0 < interval[1]


def statistics_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def test_relative_savings_keeps_cost_direction_explicit() -> None:
    assert MODULE.relative_savings(75, 100) == pytest.approx(0.25)
    assert MODULE.relative_savings(125, 100) == pytest.approx(-0.25)


@pytest.mark.parametrize(
    ("name", "updates"),
    [("step_5", 5), ("step_15", 15), ("final", 20)],
)
def test_checkpoint_updates(name: str, updates: int) -> None:
    assert MODULE.checkpoint_updates(name) == updates

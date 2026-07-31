from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1]
    / "experiments"
    / "guided_tutoring_v2"
    / "prepare_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("guided_prepare_dataset", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(task_id: int, split: str, prompt_hash: str) -> dict:
    return {
        "id": str(task_id),
        "split": split,
        "prompt_hash": prompt_hash,
        "entry_point": "f",
        "messages": [],
        "original_prompt": "p",
        "prompt": "p",
        "reference_hash": "r",
        "source": "mbpp_sanitized",
        "tests": ["assert f() == 1"],
    }


def test_canonical_id_accepts_evalplus_prefix() -> None:
    assert MODULE.canonical_id("Mbpp/806") == "806"
    assert MODULE.canonical_id(806) == "806"


def test_learning_row_never_uses_evalplus_fields() -> None:
    source = row(1, "train", "hash-1")
    source["test"] = "SECRET_PLUS_TEST"
    clean = MODULE.sanitized_learning_row(source)
    assert clean["public_feedback_tests"] == ["assert f() == 1"]
    assert "tests" not in clean
    assert "test" not in clean
    assert "SECRET_PLUS_TEST" not in str(clean)


def test_cross_split_id_leakage_fails() -> None:
    split_maps = {
        "train": {"1": row(1, "train", "hash-1")},
        "dev": {"1": row(1, "dev", "hash-2")},
        "test": {"3": row(3, "test", "hash-3")},
    }
    with pytest.raises(ValueError, match="leakage"):
        MODULE.assert_disjoint(split_maps)


def test_cross_split_prompt_hash_leakage_fails() -> None:
    split_maps = {
        "train": {"1": row(1, "train", "shared")},
        "dev": {"2": row(2, "dev", "shared")},
        "test": {"3": row(3, "test", "hash-3")},
    }
    with pytest.raises(ValueError, match="leakage"):
        MODULE.assert_disjoint(split_maps)

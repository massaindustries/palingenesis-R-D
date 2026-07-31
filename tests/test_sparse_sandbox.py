import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def sandbox_module():
    path = Path("/opt/sparse-opd/sandbox/run_task.py")
    spec = importlib.util.spec_from_file_location("sparse_sandbox", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sandbox_executes_tests(sandbox_module):
    result = sandbox_module.run(
        {
            "code": "def add(a, b):\n    return a + b",
            "tests": ["assert add(2, 3) == 5"],
        }
    )
    assert result["syntax_success"]
    assert result["runtime_success"]
    assert result["test_pass"]
    assert not result["timeout"]


def test_sandbox_timeout(sandbox_module):
    result = sandbox_module.run(
        {
            "code": "while True:\n    pass",
            "tests": [],
            "timeout": 0.1,
        }
    )
    assert result["timeout"]
    assert not result["test_pass"]

import math

import pytest
import torch

from palingenesis.opd.sparse import select_anchor_positions, sparse_anchor_rkl
from palingenesis.opd.teacher_backend import AnchorQuery, SGLangTeacherBackend


def test_anchor_selection_interval_one_and_final():
    completion = [10, 11, 12, 99]
    assert select_anchor_positions(completion, 1, stop_ids=(99,)) == [0, 1, 2, 3]
    assert select_anchor_positions(completion, "final", stop_ids=(99,)) == [3]


def test_anchor_selection_interval_32_short_sequence_eos_and_truncation():
    short = list(range(12))
    assert select_anchor_positions(short, 32) == [11]

    completion = list(range(70))
    completion[10] = 999
    assert select_anchor_positions(completion, 32, stop_ids=(999,)) == [10, 31, 63, 69]
    assert select_anchor_positions(
        completion,
        32,
        stop_ids=(999,),
        max_anchors_per_sequence=3,
    ) == [10, 31, 69]


def test_anchor_window_empty_and_validation():
    assert select_anchor_positions([], 8) == []
    assert select_anchor_positions(list(range(9)), 8, anchor_window_tokens=4) == [4, 5, 6, 7, 8]
    with pytest.raises(ValueError):
        select_anchor_positions([1], 0)


def test_sparse_rkl_identical_distributions_is_zero_with_residual_bucket():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]], requires_grad=True)
    full_logp = torch.log_softmax(logits.detach(), dim=-1)
    ids = torch.tensor([[0, 2]])
    loss, stats = sparse_anchor_rkl(logits, ids, full_logp[:, ids[0]])
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    expected_residual = full_logp.exp()[0, [1, 3]].sum().item()
    assert stats.mean_student_residual == pytest.approx(expected_residual)
    assert stats.mean_teacher_residual == pytest.approx(expected_residual)


def test_sparse_rkl_positive_finite_gradients():
    logits = torch.tensor([[3.0, 1.0, -1.0, 0.0]], requires_grad=True)
    ids = torch.tensor([[0, 1, 3]])
    teacher_logp = torch.log(torch.tensor([[0.1, 0.6, 0.2]]))
    loss, stats = sparse_anchor_rkl(logits, ids, teacher_logp)
    assert loss.item() > 0
    assert math.isfinite(loss.item())
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert stats.mean_teacher_residual == pytest.approx(0.1, abs=1e-6)


def test_sparse_rkl_rejects_nonfinite_teacher():
    with pytest.raises(ValueError, match="finite"):
        sparse_anchor_rkl(
            torch.zeros(1, 3),
            torch.tensor([[0]]),
            torch.tensor([[float("nan")]]),
        )


class _Response:
    def __init__(self, payload, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        return self.payload


class _SequenceClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def post(self, *args, **kwargs):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)


def _valid_sglang_row():
    return {
        "meta_info": {
            "output_top_logprobs": [[[math.log(0.5), 2], [math.log(0.3), 1]]],
            "output_token_ids_logprobs": [[[math.log(0.3), 1], [math.log(0.1), 3]]],
        }
    }


def test_sglang_backend_retries_then_parses_union():
    client = _SequenceClient([TimeoutError("timeout"), _valid_sglang_row()])
    backend = SGLangTeacherBackend(
        "http://teacher",
        client=client,
        max_retries=1,
        retry_backoff_seconds=0,
    )
    query = AnchorQuery(prefix_ids=(4, 5), student_top_ids=(1, 3))
    [score] = backend.score_anchors([query], top_k=2)
    assert client.calls == 2
    assert score.token_ids == (1, 2, 3)
    assert score.teacher_top_ids == (2, 1)
    assert score.residual_mass == pytest.approx(0.1)


def test_sglang_backend_malformed_response_and_circuit_breaker():
    client = _SequenceClient([{}, {}, {}])
    backend = SGLangTeacherBackend(
        "http://teacher",
        client=client,
        max_retries=5,
        circuit_breaker_failures=3,
        retry_backoff_seconds=0,
    )
    query = AnchorQuery(prefix_ids=(1,), student_top_ids=(1,))
    with pytest.raises(RuntimeError, match="bounded retries"):
        backend.score_anchors([query], top_k=1)
    assert client.calls == 3
    with pytest.raises(RuntimeError, match="circuit breaker"):
        backend.score_anchors([query], top_k=1)

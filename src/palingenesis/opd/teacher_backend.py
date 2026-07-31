"""Teacher scoring backends for sparse anchor distillation."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class AnchorQuery:
    prefix_ids: tuple[int, ...]
    student_top_ids: tuple[int, ...]
    required_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class AnchorScores:
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    teacher_top_ids: tuple[int, ...]
    residual_mass: float

    def as_dict(self) -> dict[int, float]:
        return dict(zip(self.token_ids, self.logprobs))


@dataclass(frozen=True, slots=True)
class TeacherGeneration:
    token_ids: tuple[int, ...]
    finish_reason: str


@runtime_checkable
class TeacherBackend(Protocol):
    def score_anchors(self, queries: list[AnchorQuery], top_k: int) -> list[AnchorScores]:
        """Score next-token distributions for each prefix."""

    def generate_tokens(
        self, prefix_ids: list[tuple[int, ...]], max_new_tokens: int
    ) -> list[TeacherGeneration]:
        """Generate deterministic continuation tokens for each prefix."""


def _validate_scores(scores: AnchorScores) -> None:
    if not scores.token_ids:
        raise ValueError("teacher returned empty explicit support")
    if len(scores.token_ids) != len(scores.logprobs):
        raise ValueError("teacher token/logprob length mismatch")
    if len(set(scores.token_ids)) != len(scores.token_ids):
        raise ValueError("teacher returned duplicate token IDs")
    if not all(math.isfinite(value) and value <= 1e-5 for value in scores.logprobs):
        raise ValueError("teacher returned malformed or non-finite logprobs")
    if not math.isfinite(scores.residual_mass) or not -1e-5 <= scores.residual_mass <= 1.0 + 1e-5:
        raise ValueError("teacher residual mass is invalid")


class LocalTransformersTeacherBackend:
    """Reference backend that computes exact probabilities locally."""

    def __init__(self, model, *, device: str, pad_token_id: int):
        self.model = model
        self.device = device
        self.pad_token_id = pad_token_id
        self.model.eval().requires_grad_(False)

    @torch.inference_mode()
    def score_anchors(self, queries: list[AnchorQuery], top_k: int) -> list[AnchorScores]:
        if not queries:
            return []
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if any(not query.prefix_ids for query in queries):
            raise ValueError("anchor prefixes must not be empty")
        width = max(len(query.prefix_ids) for query in queries)
        ids = torch.full((len(queries), width), self.pad_token_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for row, query in enumerate(queries):
            prefix = torch.tensor(query.prefix_ids, dtype=torch.long, device=self.device)
            ids[row, : len(prefix)] = prefix
            mask[row, : len(prefix)] = 1
        output = self.model(input_ids=ids, attention_mask=mask)
        logits = output.logits[torch.arange(len(queries), device=self.device), mask.sum(-1) - 1]
        logp = F.log_softmax(logits.float(), dim=-1)

        results = []
        k = min(top_k, logp.shape[-1])
        for row, query in enumerate(queries):
            teacher_top = torch.topk(logp[row], k=k).indices.tolist()
            support = sorted(set(teacher_top) | set(query.student_top_ids) | set(query.required_ids))
            values = logp[row, support].tolist()
            residual = max(0.0, 1.0 - sum(math.exp(value) for value in values))
            score = AnchorScores(tuple(support), tuple(values), tuple(teacher_top), residual)
            _validate_scores(score)
            results.append(score)
        return results

    @torch.inference_mode()
    def generate_tokens(
        self, prefix_ids: list[tuple[int, ...]], max_new_tokens: int
    ) -> list[TeacherGeneration]:
        if not prefix_ids:
            return []
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        width = max(len(prefix) for prefix in prefix_ids)
        ids = torch.full(
            (len(prefix_ids), width),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.zeros_like(ids)
        for row, prefix in enumerate(prefix_ids):
            values = torch.tensor(prefix, dtype=torch.long, device=self.device)
            ids[row, width - len(prefix) :] = values
            mask[row, width - len(prefix) :] = 1
        generated = self.model.generate(
            ids,
            attention_mask=mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.pad_token_id,
        )
        return [
            TeacherGeneration(
                token_ids=tuple(generated[row, width:].tolist()),
                finish_reason="length",
            )
            for row in range(len(prefix_ids))
        ]


class SGLangTeacherBackend:
    """SGLang Native API backend with bounded retries and circuit breaking."""

    def __init__(
        self,
        endpoint: str,
        *,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        circuit_breaker_failures: int = 3,
        retry_backoff_seconds: float = 0.25,
        client=None,
    ):
        import httpx

        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.circuit_breaker_failures = circuit_breaker_failures
        self.retry_backoff_seconds = retry_backoff_seconds
        self.client = client or httpx.Client(timeout=timeout_seconds)
        self.consecutive_failures = 0
        self.request_count = 0
        self.failure_count = 0
        self.successful_request_count = 0
        self.total_latency_seconds = 0.0

    def score_anchors(self, queries: list[AnchorQuery], top_k: int) -> list[AnchorScores]:
        if not queries:
            return []
        if self.consecutive_failures >= self.circuit_breaker_failures:
            raise RuntimeError("teacher circuit breaker is open")
        requested = [sorted(set(query.student_top_ids) | set(query.required_ids)) for query in queries]
        payload = {
            "input_ids": [list(query.prefix_ids) for query in queries],
            "sampling_params": {
                "max_new_tokens": 1,
                "temperature": 0,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "top_logprobs_num": top_k,
            "token_ids_logprob": requested,
            "return_text_in_logprobs": False,
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.request_count += 1
            request_started = time.perf_counter()
            try:
                response = self.client.post(
                    f"{self.endpoint}/generate",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                raw = response.json()
                rows = raw if isinstance(raw, list) else [raw]
                if len(rows) != len(queries):
                    raise ValueError(f"teacher batch size mismatch: expected {len(queries)}, got {len(rows)}")
                scores = [self._parse_row(row, query, top_k) for row, query in zip(rows, queries)]
                self.consecutive_failures = 0
                self.successful_request_count += 1
                return scores
            except Exception as error:  # noqa: BLE001 - normalize HTTP/schema failures
                last_error = error
                self.failure_count += 1
                self.consecutive_failures += 1
                if attempt >= self.max_retries or self.consecutive_failures >= self.circuit_breaker_failures:
                    break
                time.sleep(self.retry_backoff_seconds * (2**attempt))
            finally:
                self.total_latency_seconds += time.perf_counter() - request_started
        raise RuntimeError(f"teacher request failed after bounded retries: {last_error}") from last_error

    def generate_tokens(
        self, prefix_ids: list[tuple[int, ...]], max_new_tokens: int
    ) -> list[TeacherGeneration]:
        """Generate actual teacher continuations, not next-token score tables."""
        if not prefix_ids:
            return []
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if any(not prefix for prefix in prefix_ids):
            raise ValueError("teacher prefixes must not be empty")
        if self.consecutive_failures >= self.circuit_breaker_failures:
            raise RuntimeError("teacher circuit breaker is open")
        payload = {
            "input_ids": [list(prefix) for prefix in prefix_ids],
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
            },
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.request_count += 1
            request_started = time.perf_counter()
            try:
                response = self.client.post(
                    f"{self.endpoint}/generate",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                raw = response.json()
                rows = raw if isinstance(raw, list) else [raw]
                if len(rows) != len(prefix_ids):
                    raise ValueError(
                        f"teacher batch size mismatch: expected {len(prefix_ids)}, got {len(rows)}"
                    )
                output = [
                    self._parse_generation_row(row, max_new_tokens) for row in rows
                ]
                self.consecutive_failures = 0
                self.successful_request_count += 1
                return output
            except Exception as error:  # noqa: BLE001
                last_error = error
                self.failure_count += 1
                self.consecutive_failures += 1
                if attempt >= self.max_retries or self.consecutive_failures >= self.circuit_breaker_failures:
                    break
                time.sleep(self.retry_backoff_seconds * (2**attempt))
            finally:
                self.total_latency_seconds += time.perf_counter() - request_started
        raise RuntimeError(
            f"teacher generation failed after bounded retries: {last_error}"
        ) from last_error

    @staticmethod
    def _parse_generation_row(
        row: dict, max_new_tokens: int
    ) -> TeacherGeneration:
        try:
            meta = row["meta_info"]
            # SGLang 0.5.8 returns auditable token IDs at the top level.
            # Never reconstruct them by tokenizing `text`.
            raw_ids = row["output_ids"]
        except (KeyError, TypeError) as error:
            raise ValueError("malformed SGLang generation response") from error
        if not isinstance(raw_ids, list):
            raise ValueError("SGLang output_token_ids must be a list")
        token_ids = tuple(int(token_id) for token_id in raw_ids)
        if len(token_ids) > max_new_tokens or any(token_id < 0 for token_id in token_ids):
            raise ValueError("SGLang returned invalid generated token IDs")
        finish = meta.get("finish_reason", {})
        if isinstance(finish, dict):
            finish_reason = str(finish.get("type", "unknown"))
        else:
            finish_reason = str(finish or "unknown")
        return TeacherGeneration(token_ids=token_ids, finish_reason=finish_reason)

    @staticmethod
    def _parse_row(row: dict, query: AnchorQuery, top_k: int) -> AnchorScores:
        try:
            meta = row["meta_info"]
            teacher_pairs = meta["output_top_logprobs"][0]
            requested_pairs = meta["output_token_ids_logprobs"][0]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("malformed SGLang logprob response") from error

        values: dict[int, float] = {}
        teacher_top = []
        for pair in teacher_pairs[:top_k]:
            logprob, token_id = float(pair[0]), int(pair[1])
            values[token_id] = logprob
            teacher_top.append(token_id)
        for pair in requested_pairs:
            logprob, token_id = float(pair[0]), int(pair[1])
            values[token_id] = logprob

        required = set(query.student_top_ids) | set(query.required_ids) | set(teacher_top)
        missing = required - values.keys()
        if missing:
            raise ValueError(f"SGLang response omitted requested token IDs: {sorted(missing)}")
        token_ids = tuple(sorted(required))
        logprobs = tuple(values[token_id] for token_id in token_ids)
        residual = max(0.0, 1.0 - sum(math.exp(value) for value in logprobs))
        score = AnchorScores(token_ids, logprobs, tuple(teacher_top), residual)
        _validate_scores(score)
        return score

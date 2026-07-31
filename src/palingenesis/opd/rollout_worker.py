"""Dedicated Transformers rollout replica with strict policy versioning."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import torch

from palingenesis.opd.sparse import select_anchor_positions


def transformers_top_k(value: int) -> int:
    """Translate OPD's negative "unbounded" sentinel to Transformers' zero."""
    return max(0, value)


@dataclass(slots=True)
class RolloutResult:
    completions: list[list[int]]
    student_top_ids: list[dict[int, tuple[int, ...]]]
    policy_version: int
    generation_ms: float
    student_topk_ms: float


def adapter_state_hash(model) -> str:
    """Stable SHA-256 over named trainable adapter tensors."""
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad and "lora_" not in name:
            continue
        tensor = parameter.detach().contiguous().cpu().view(torch.uint8)
        digest.update(name.encode("utf-8"))
        digest.update(bytes(tensor.numpy()))
    return digest.hexdigest()


class TransformersRolloutWorker:
    """BF16 no-grad model replica pinned to the dedicated rollout GPU."""

    def __init__(
        self,
        model,
        *,
        device: str,
        pad_token_id: int,
        stop_ids: tuple[int, ...],
        tutoring,
        sampling,
    ):
        self.model = model.to(device)
        self.model.eval().requires_grad_(False)
        self.device = device
        self.pad_token_id = pad_token_id
        self.stop_ids = stop_ids
        self.tutoring = tutoring
        self.sampling = sampling
        self.policy_version = -1

    @staticmethod
    def _base(model):
        return model.get_base_model() if hasattr(model, "get_base_model") else model

    def sync_from(self, trainer_model, policy_version: int) -> dict[str, float | int | str]:
        """Synchronize only trainable LoRA tensors through measured host staging."""
        started = time.perf_counter()
        source = {name: parameter for name, parameter in trainer_model.named_parameters() if parameter.requires_grad}
        target = {name: parameter for name, parameter in self.model.named_parameters() if "lora_" in name}
        if source.keys() != target.keys():
            raise RuntimeError(f"adapter parameter mismatch: trainer={len(source)} rollout={len(target)}")
        with torch.no_grad():
            for name in sorted(source):
                # CUDA P2P is unavailable on this host. Explicit CPU staging
                # avoids an unmeasured implicit copy path.
                staged = source[name].detach().to("cpu", non_blocking=False)
                target[name].copy_(staged.to(self.device, non_blocking=False))
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        self.policy_version = policy_version
        trainer_hash = adapter_state_hash(trainer_model)
        rollout_hash = adapter_state_hash(self.model)
        if trainer_hash != rollout_hash:
            raise RuntimeError("adapter synchronization hash mismatch")
        return {
            "policy_version": policy_version,
            "sync_ms": (time.perf_counter() - started) * 1000,
            "adapter_hash": trainer_hash,
        }

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[list[int]],
        max_new_tokens: int,
        *,
        expected_policy_version: int,
        greedy: bool = False,
    ) -> RolloutResult:
        if self.policy_version != expected_policy_version:
            raise RuntimeError(
                f"policy version mismatch: expected {expected_policy_version}, rollout has {self.policy_version}"
            )
        width = max(len(prompt) for prompt in prompt_ids)
        ids = torch.full(
            (len(prompt_ids), width),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.zeros_like(ids)
        for row, prompt in enumerate(prompt_ids):
            values = torch.tensor(prompt, dtype=torch.long, device=self.device)
            ids[row, width - len(prompt) :] = values
            mask[row, width - len(prompt) :] = 1
        decode = (
            {"do_sample": False}
            if greedy
            else {
                "do_sample": True,
                "temperature": self.sampling.temperature,
                "top_p": self.sampling.top_p,
                # OPD configs use -1 for an unbounded sampler. Transformers
                # represents the same policy with top_k=0.
                "top_k": transformers_top_k(self.sampling.top_k),
            }
        )
        generation_started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generated = self.model.generate(
                ids,
                attention_mask=mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=list(self.stop_ids),
                pad_token_id=self.pad_token_id,
                use_cache=True,
                **decode,
            )
        torch.cuda.synchronize(self.device)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        completions = [self._clean(generated[row, width:].tolist()) for row in range(len(prompt_ids))]

        topk_started = time.perf_counter()
        top_ids = self._topk_at_anchors(prompt_ids, completions)
        torch.cuda.synchronize(self.device)
        return RolloutResult(
            completions=completions,
            student_top_ids=top_ids,
            policy_version=self.policy_version,
            generation_ms=generation_ms,
            student_topk_ms=(time.perf_counter() - topk_started) * 1000,
        )

    def _clean(self, ids: list[int]) -> list[int]:
        cleaned = []
        for token_id in ids:
            cleaned.append(token_id)
            if token_id in self.stop_ids:
                break
        return cleaned

    def _topk_at_anchors(
        self,
        prompts: list[list[int]],
        completions: list[list[int]],
    ) -> list[dict[int, tuple[int, ...]]]:
        output: list[dict[int, tuple[int, ...]]] = []
        base = self._base(self.model)
        for prompt, completion in zip(prompts, completions):
            anchors = select_anchor_positions(
                completion,
                self.tutoring.interval_tokens,
                stop_ids=self.stop_ids,
                always_include_final_anchor=self.tutoring.always_include_final_anchor,
                include_eos_anchor=self.tutoring.include_eos_anchor,
                anchor_window_tokens=self.tutoring.anchor_window_tokens,
                max_anchors_per_sequence=self.tutoring.max_anchors_per_sequence,
            )
            if not anchors:
                output.append({})
                continue
            sequence = prompt + completion[:-1]
            ids = torch.tensor([sequence], dtype=torch.long, device=self.device)
            mask = torch.ones_like(ids)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = base.model(input_ids=ids, attention_mask=mask).last_hidden_state
                selected = torch.stack([hidden[0, len(prompt) + anchor - 1] for anchor in anchors])
                logits = base.lm_head(selected)
            k = min(self.tutoring.student_top_k, logits.shape[-1])
            indices = torch.topk(logits.float(), k=k, dim=-1).indices.cpu().tolist()
            output.append({anchor: tuple(token_ids) for anchor, token_ids in zip(anchors, indices)})
        return output

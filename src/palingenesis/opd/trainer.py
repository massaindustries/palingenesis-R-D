"""On-policy distillation trainer.

Per step (mirrors tinker-cookbook's on_policy_distillation, self-contained):
  1. draw pool prompts, render with the STUDENT's chat template
  2. student samples completions (temperature 1.0) with its current weights
     -> exactly on-policy, single gradient step per batch, no importance sampling
  3. teacher scores the same completions conditioned on the SAME conversation
     rendered with the TEACHER's chat template (see token_bridge for why)
  4. loss = full-distribution reverse KL over completion tokens
     sum_v p_student(v) * (log p_student(v) - log p_teacher(v))
     ("sampled_rkl" reproduces tinker's sampled-token REINFORCE variant)

In ``guided_ce`` mode, steps 3-4 are replaced by interleaving real greedy
teacher continuation groups into the sampled student trajectory and applying
cross entropy only at those provenance-tracked teacher-token positions.

The trainer is single-process. It can either generate on the training device
or synchronize a dedicated rollout replica on ``model.rollout_device``; the
teacher can be local or served by SGLang.

Launch:
    pgs distill --config configs/distill_opd.yaml
    python -m palingenesis.opd.trainer --config configs/distill_opd.yaml
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import logging
import math
import os
import random
import shutil
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from palingenesis.opd.config import OPDConfig
from palingenesis.opd.guided import (
    GuidedTrajectory,
    assign_shuffled_group_targets,
    build_guided_trajectories,
)
from palingenesis.opd.rollout_worker import TransformersRolloutWorker, transformers_top_k
from palingenesis.opd.sources import PromptSource, build_source
from palingenesis.opd.sparse import select_anchor_positions, sparse_anchor_rkl
from palingenesis.opd.teacher_backend import (
    AnchorQuery,
    LocalTransformersTeacherBackend,
    SGLangTeacherBackend,
    TeacherGeneration,
)
from palingenesis.opd.token_bridge import TokenBridge, check_compatible

logger = logging.getLogger(__name__)


def intermediate_checkpoint_name(
    zero_based_step: int, save_steps: int, total_steps: int
) -> str | None:
    """Return a human-counted checkpoint name after a completed update."""
    if save_steps <= 0:
        return None
    completed_updates = zero_based_step + 1
    if completed_updates >= total_steps or completed_updates % save_steps:
        return None
    return f"step_{completed_updates}"


def load_causal_lm(name: str, dtype: torch.dtype):
    """from_pretrained across the transformers 4.x (torch_dtype) / 5.x (dtype) rename."""
    try:
        return AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OPDTrainer:
    """The task-agnostic OPD engine; task-specific data lives in the source.

    The source decides what to roll out and how to evaluate (see
    opd.sources.PromptSource); the engine samples on-policy, teacher-scores,
    and steps. Pass a custom source for tasks the built-ins don't cover.
    """

    def __init__(self, config: OPDConfig, source: PromptSource | None = None):
        self.config = config
        self.device = config.model.train_device or pick_device()
        self.teacher_device = config.model.teacher_device or self.device
        self.rng = random.Random(config.train.seed)
        torch.manual_seed(config.train.seed)
        os.makedirs(config.train.output_dir, exist_ok=True)
        self.metrics_path = os.path.join(config.train.output_dir, "metrics.jsonl")
        self.provenance_path = os.path.join(
            config.train.output_dir, "teacher_provenance.jsonl"
        )
        resolved_path = os.path.join(config.train.output_dir, "config_resolved.json")
        if not os.path.exists(resolved_path):
            with open(resolved_path, "x") as handle:
                json.dump(dataclasses.asdict(config), handle, indent=2)

        logger.info("Loading tokenizers (%s / %s)", config.model.student, config.model.teacher)
        self.s_tok = AutoTokenizer.from_pretrained(config.model.student)
        self.t_tok = AutoTokenizer.from_pretrained(config.model.teacher)
        self.bridge = TokenBridge.from_tokenizers(
            self.s_tok,
            self.t_tok,
            eos_map=config.bridge.eos_map,
            extra_stop_tokens=tuple(config.bridge.extra_stop_tokens),
        )
        check_compatible(self.s_tok, self.t_tok, self.bridge, probe_texts=tuple(config.bridge.probe_texts))
        self.s_pad = self.s_tok.pad_token_id or self.bridge.stop_ids[-1]
        self.t_pad = self.t_tok.pad_token_id or self.t_tok.eos_token_id

        student_dtype = torch.bfloat16 if config.train.bf16 and self.device != "cpu" else torch.float32
        logger.info("Loading student (%s) on %s", student_dtype, self.device)
        self.student = load_causal_lm(config.model.student, student_dtype).to(self.device)
        if config.model.gradient_checkpointing:
            self.student.gradient_checkpointing_enable()
        self.student = self._wrap_lora(self.student)

        self.teacher = None
        if config.model.teacher_backend == "local_transformers":
            logger.info("Loading teacher (bf16, frozen) on %s", self.teacher_device)
            self.teacher = load_causal_lm(config.model.teacher, torch.bfloat16).to(self.teacher_device)
            self.teacher.eval().requires_grad_(False)
            self.teacher_backend = LocalTransformersTeacherBackend(
                self.teacher,
                device=self.teacher_device,
                pad_token_id=self.t_pad,
            )
        else:
            logger.info("Using remote SGLang teacher at %s", config.model.teacher_endpoint)
            self.teacher_backend = SGLangTeacherBackend(
                config.model.teacher_endpoint,
                timeout_seconds=config.tutoring.teacher_timeout_seconds,
                max_retries=config.tutoring.teacher_max_retries,
            )

        self.policy_version = 0
        self.rollout_worker = None
        self.last_sync_metrics = {
            "policy_version": 0,
            "sync_ms": 0.0,
            "adapter_hash": "",
        }
        if config.model.rollout_device and config.model.rollout_device != self.device:
            logger.info(
                "Loading dedicated BF16 Transformers rollout replica on %s",
                config.model.rollout_device,
            )
            rollout_model = load_causal_lm(config.model.student, student_dtype)
            rollout_model = self._wrap_lora(rollout_model)
            self.rollout_worker = TransformersRolloutWorker(
                rollout_model,
                device=config.model.rollout_device,
                pad_token_id=self.s_pad,
                stop_ids=self.bridge.stop_ids,
                tutoring=config.tutoring,
                sampling=config.sampling,
            )
            self.last_sync_metrics = self.rollout_worker.sync_from(
                self.student,
                self.policy_version,
            )

        self.source = source or build_source(config, rng=self.rng)

        trainable = [parameter for parameter in self.student.parameters() if parameter.requires_grad]
        self.opt = torch.optim.AdamW(
            trainable,
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.start_step = 0
        if config.train.resume_from:
            self._resume(config.train.resume_from)
            if self.rollout_worker is not None:
                self.last_sync_metrics = self.rollout_worker.sync_from(
                    self.student,
                    self.policy_version,
                )
        # A metrics backend must never kill a training run: init/log are guarded.
        self.wandb = None
        if config.logging.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=config.logging.project,
                    name=config.logging.run_name or None,
                    config=dataclasses.asdict(config),
                    dir=config.train.output_dir,
                )
                self.wandb = wandb
            except Exception as e:  # noqa: BLE001 — degrade to console logging
                logger.warning("wandb init failed (%s); continuing without it", e)

    def _wrap_lora(self, model):
        """Attach BF16 PEFT LoRA adapters and persist the inspected targets."""
        adapter = self.config.adapter
        if adapter.type != "lora":
            raise ValueError(f"unsupported adapter type: {adapter.type}")
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as error:
            raise RuntimeError("PEFT is required for OPD LoRA training") from error

        if adapter.target_modules:
            targets = list(adapter.target_modules)
        else:
            candidates = {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            }
            targets = sorted(
                {
                    name.rsplit(".", 1)[-1]
                    for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in candidates
                }
            )
            if not targets:
                raise RuntimeError("could not find Qwen attention/MLP projection modules for LoRA")
            adapter.target_modules = targets
        logger.info("LoRA target modules inspected from model: %s", targets)
        peft_config = LoraConfig(
            r=adapter.rank,
            lora_alpha=adapter.alpha,
            lora_dropout=adapter.dropout,
            bias=adapter.bias,
            target_modules=targets,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype != torch.bfloat16 and self.config.train.bf16:
                parameter.data = parameter.data.to(torch.bfloat16)
        return model

    # ------------------------------------------------------------------ utils

    def _lr_at(self, step: int) -> float:
        train = self.config.train
        if step < train.warmup_steps:
            return train.learning_rate * (step + 1) / train.warmup_steps
        if train.lr_scheduler == "constant":
            return train.learning_rate
        t = (step - train.warmup_steps) / max(1, train.steps - train.warmup_steps)
        return train.learning_rate * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    @staticmethod
    def _encode_prompt(tok, messages) -> list[int]:
        try:
            text = tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        ids = tok.encode(text, add_special_tokens=False)
        bos = tok.bos_token_id
        if bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
        return ids

    @staticmethod
    def _right_pad(seqs: list[list[int]], pad: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return ids.to(device), mask.to(device)

    def _left_pad(self, prompts: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Left padding so all prompts end at the same position (for generate)."""
        T = max(len(p) for p in prompts)
        ids = torch.full((len(prompts), T), self.s_pad, dtype=torch.long)
        mask = torch.zeros((len(prompts), T), dtype=torch.long)
        for j, p in enumerate(prompts):
            ids[j, T - len(p) :] = torch.tensor(p, dtype=torch.long)
            mask[j, T - len(p) :] = 1
        return ids, mask, T

    # -------------------------------------------------------------- generation

    @torch.no_grad()
    def _generate(self, prompt_ids: list[list[int]], max_new_tokens: int, greedy: bool = False) -> list[list[int]]:
        """One completion per prompt (already replicated for group_size), cleaned."""
        if self.rollout_worker is not None:
            return self.rollout_worker.generate(
                prompt_ids,
                max_new_tokens,
                expected_policy_version=self.policy_version,
                greedy=greedy,
            ).completions
        sampling = self.config.sampling
        self.student.eval()
        completions: list[list[int]] = []
        for i in range(0, len(prompt_ids), sampling.gen_micro_seqs):
            chunk = prompt_ids[i : i + sampling.gen_micro_seqs]
            ids, mask, T = self._left_pad(chunk)
            decode_kwargs = (
                dict(do_sample=False)
                if greedy
                else dict(
                    do_sample=True,
                    temperature=sampling.temperature,
                    top_p=sampling.top_p,
                    top_k=transformers_top_k(sampling.top_k),
                )
            )
            with torch.autocast(self.device.split(":")[0], dtype=torch.bfloat16, enabled=self.device != "cpu"):
                out = self.student.generate(
                    ids.to(self.device),
                    attention_mask=mask.to(self.device),
                    max_new_tokens=max_new_tokens,
                    eos_token_id=list(self.bridge.stop_ids),
                    pad_token_id=self.s_pad,
                    **decode_kwargs,
                )
            for j in range(len(chunk)):
                raw = out[j, T:].tolist()
                completions.append(self.bridge.clean_completion(raw))
        self.student.train()
        return completions

    # ------------------------------------------------- engine services (sources)

    def greedy_generate(self, messages_list, max_new_tokens: int) -> list[str]:
        """Greedy-decode one completion per conversation, decoded (stop token trimmed)."""
        prompts = [self._encode_prompt(self.s_tok, m) for m in messages_list]
        comps = self._generate(prompts, max_new_tokens, greedy=True)
        return [self.s_tok.decode(c[:-1] if c and c[-1] in self.bridge.stop_ids else c) for c in comps]

    @torch.no_grad()
    def dev_kl(self, messages_list, max_new_tokens: int) -> dict[str, float]:
        """On-policy sample + teacher-score held-out prompts; mean reverse KL/token."""
        rollouts = []
        s_prompts = [self._encode_prompt(self.s_tok, m) for m in messages_list]
        t_prompts = [self._encode_prompt(self.t_tok, m) for m in messages_list]
        comps = self._generate(s_prompts, max_new_tokens)
        for s_p, t_p, comp in zip(s_prompts, t_prompts, comps):
            if comp:
                rollouts.append(({"s_prompt": s_p, "t_prompt": t_p}, comp))
        if not rollouts:
            return {"dev_kl": float("nan"), "dev_len": 0.0}
        total_kl = total_tok = 0.0
        for i in range(0, len(rollouts), self.config.train.score_micro_seqs):
            chunk = rollouts[i : i + self.config.train.score_micro_seqs]
            if self.config.train.loss_fn == "sparse_anchor_rkl":
                _, n_tok, stats = self._sparse_loss_on_chunk(chunk)
            else:
                _, n_tok, stats = self._loss_on_chunk(*self._chunk_args(chunk))
            total_kl += stats["kl"] * n_tok
            total_tok += n_tok
        return {"dev_kl": total_kl / total_tok, "dev_len": total_tok / len(rollouts)}

    # ----------------------------------------------------------------- scoring

    @staticmethod
    def _backbone_and_head(model):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        return base.model, base.lm_head

    @staticmethod
    def _gather_logits(model, seqs, plens, lens, pad, device, autocast_dev=None):
        """Forward `seqs`, return lm_head logits only at completion positions.

        Position p predicts token p+1, so for a completion of length L starting
        at index P (= prompt length) we need hidden states at P-1 .. P+L-2.
        Full logits over a 128k vocab for every position would not fit; gathering
        hidden states first keeps memory at N_completion_tokens x vocab.

        Assumes the HF causal-LM layout (model.model backbone + model.lm_head),
        which holds for Llama/Qwen/Gemma-family architectures.
        """
        ids, mask = OPDTrainer._right_pad(seqs, pad, device)
        # grad-vs-no-grad is decided by the caller's context, not here
        ctx = torch.autocast(autocast_dev, dtype=torch.bfloat16) if autocast_dev else contextlib.nullcontext()
        with ctx:
            backbone, lm_head = OPDTrainer._backbone_and_head(model)
            h = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            B, T, H = h.shape
            flat = []
            for i, (P, L) in enumerate(zip(plens, lens)):
                flat.extend(range(i * T + P - 1, i * T + P - 1 + L))
            h_sel = h.reshape(B * T, H)[torch.tensor(flat, device=device)]
            logits = lm_head(h_sel)  # (N, vocab)
        return logits

    @staticmethod
    def _gather_anchor_logits(model, seqs, prediction_positions, pad, device, autocast_dev=None):
        """Gather next-token logits at arbitrary absolute prediction positions."""
        ids, mask = OPDTrainer._right_pad(seqs, pad, device)
        ctx = torch.autocast(autocast_dev, dtype=torch.bfloat16) if autocast_dev else contextlib.nullcontext()
        with ctx:
            backbone, lm_head = OPDTrainer._backbone_and_head(model)
            hidden = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
            selected = torch.stack(
                [hidden[row, position] for row, positions in enumerate(prediction_positions) for position in positions]
            )
            return lm_head(selected)

    def _chunk_args(self, chunk):
        """(rollout, completion) pairs -> _loss_on_chunk arguments."""
        s_seqs, t_seqs, plens_s, plens_t, lens, targets = [], [], [], [], [], []
        for b, comp in chunk:
            comp_t = self.bridge.to_teacher(comp)
            s_seqs.append(b["s_prompt"] + comp[:-1])
            t_seqs.append(b["t_prompt"] + comp_t[:-1])
            plens_s.append(len(b["s_prompt"]))
            plens_t.append(len(b["t_prompt"]))
            lens.append(len(comp))
            targets.extend(comp_t)
        return s_seqs, t_seqs, plens_s, plens_t, lens, targets

    def _loss_on_chunk(self, s_seqs, t_seqs, plens_s, plens_t, lens, targets_t):
        """Loss for a micro-batch of rollouts. Returns (loss, n_tokens, stats)."""
        train = self.config.train
        V = self.bridge.shared_vocab_size

        with torch.no_grad():
            t_logits = self._gather_logits(self.teacher, t_seqs, plens_t, lens, self.t_pad, self.teacher_device)
            logp_t = F.log_softmax(t_logits.float(), dim=-1).to(self.device)  # (N, V)

        s_logits = self._gather_logits(
            self.student,
            s_seqs,
            plens_s,
            lens,
            self.s_pad,
            self.device,
            autocast_dev=self.device.split(":")[0] if self.device != "cpu" else None,
        )
        logp_s_full = F.log_softmax(s_logits.float(), dim=-1)  # (N, student vocab)

        tgt = torch.tensor(targets_t, dtype=torch.long, device=self.device)  # (N,)

        # merge the student's end-of-turn mass into the teacher's terminator slot(s)
        p_full = logp_s_full.exp()
        p_shared = p_full[:, :V].clone()
        for s_id, t_id in self.bridge.swap.items():
            p_shared[:, t_id] += p_full[:, s_id]
        residual = 1.0 - p_shared.sum(-1)  # mass on unmapped student-only tokens, should be ~0
        logp_s = (p_shared + 1e-12).log()

        kl = (p_shared * (logp_s - logp_t)).sum(-1)  # (N,) full reverse KL
        sampled_kl = (logp_s.gather(-1, tgt[:, None]) - logp_t.gather(-1, tgt[:, None])).squeeze(-1)

        if train.loss_fn == "full_kl":
            loss = kl.sum()
        elif train.loss_fn == "sampled_rkl":
            # tinker-style: REINFORCE with per-token advantage = -sampled KL
            logp_s_tgt = logp_s.gather(-1, tgt[:, None]).squeeze(-1)
            loss = (logp_s_tgt * sampled_kl.detach()).sum()
        else:
            raise ValueError(train.loss_fn)

        stats = {
            "kl": kl.detach().mean().item(),
            "sampled_kl": sampled_kl.detach().mean().item(),
            "residual_mass": residual.detach().mean().item(),
        }
        return loss, len(targets_t), stats

    def _sparse_loss_on_chunk(self, chunk):
        """Sparse anchor coarse reverse-KL for a rollout microbatch."""
        tutoring = self.config.tutoring
        seqs = []
        prediction_positions = []
        anchor_meta = []
        for batch, completion in chunk:
            anchors = select_anchor_positions(
                completion,
                tutoring.interval_tokens,
                stop_ids=self.bridge.stop_ids,
                always_include_final_anchor=tutoring.always_include_final_anchor,
                include_eos_anchor=tutoring.include_eos_anchor,
                anchor_window_tokens=tutoring.anchor_window_tokens,
                max_anchors_per_sequence=tutoring.max_anchors_per_sequence,
            )
            if not anchors:
                continue
            seqs.append(batch["s_prompt"] + completion[:-1])
            prediction_positions.append([len(batch["s_prompt"]) + anchor - 1 for anchor in anchors])
            completion_t = self.bridge.to_teacher(completion)
            anchor_meta.extend((batch["t_prompt"] + completion_t[:anchor], anchor, batch) for anchor in anchors)
        if not anchor_meta:
            zero = next(self.student.parameters()).sum() * 0.0
            return (
                zero,
                0,
                {
                    "kl": 0.0,
                    "sampled_kl": 0.0,
                    "residual_mass": 0.0,
                    "teacher_residual_mass": 0.0,
                    "clipped_probabilities": 0.0,
                    "output_entropy": 0.0,
                    "teacher_student_agreement": 0.0,
                    "teacher_scored_tokens": 0,
                },
            )

        logits = self._gather_anchor_logits(
            self.student,
            seqs,
            prediction_positions,
            self.s_pad,
            self.device,
            autocast_dev=self.device.split(":")[0] if self.device != "cpu" else None,
        )
        student_top = torch.topk(
            logits.detach().float(),
            k=min(tutoring.student_top_k, logits.shape[-1]),
            dim=-1,
        ).indices
        required_teacher_ids = tuple(sorted({self.t_tok.eos_token_id} | set(self.bridge.swap.values()) - {None}))
        queries = []
        for row, (prefix, anchor, batch) in enumerate(anchor_meta):
            provided = batch.get("student_top_ids", {}).get(anchor)
            top_ids = provided if provided is not None else student_top[row].tolist()
            student_ids = []
            for token_id in top_ids:
                if token_id in self.bridge.swap:
                    student_ids.append(self.bridge.swap[token_id])
                elif token_id < self.bridge.shared_vocab_size:
                    student_ids.append(token_id)
            queries.append(
                AnchorQuery(
                    prefix_ids=tuple(prefix),
                    student_top_ids=tuple(sorted(set(student_ids))),
                    required_ids=required_teacher_ids,
                )
            )
        scores = self.teacher_backend.score_anchors(queries, tutoring.teacher_top_k)

        losses = []
        stat_weighted = {
            "kl": 0.0,
            "residual_mass": 0.0,
            "teacher_residual_mass": 0.0,
            "clipped_probabilities": 0.0,
            "output_entropy": 0.0,
            "teacher_student_agreement": 0.0,
        }
        teacher_scored_tokens = 0
        inverse_swap = {teacher: student for student, teacher in self.bridge.swap.items()}
        for row, score in enumerate(scores):
            student_ids = [inverse_swap.get(token_id, token_id) for token_id in score.token_ids]
            ids = torch.tensor([student_ids], dtype=torch.long, device=self.device)
            teacher_logp = torch.tensor(
                [score.logprobs],
                dtype=torch.float32,
                device=self.device,
            )
            anchor_loss, stats = sparse_anchor_rkl(
                logits[row : row + 1],
                ids,
                teacher_logp,
                epsilon=tutoring.epsilon,
            )
            losses.append(anchor_loss)
            stat_weighted["kl"] += stats.mean_kl
            stat_weighted["residual_mass"] += stats.mean_student_residual
            stat_weighted["teacher_residual_mass"] += stats.mean_teacher_residual
            stat_weighted["clipped_probabilities"] += stats.clipped_probabilities
            logp = F.log_softmax(logits[row].detach().float(), dim=-1)
            stat_weighted["output_entropy"] += float(-(logp.exp() * logp).sum().item())
            student_argmax_teacher = self.bridge.swap.get(
                int(student_top[row, 0]),
                int(student_top[row, 0]),
            )
            stat_weighted["teacher_student_agreement"] += float(
                bool(score.teacher_top_ids) and student_argmax_teacher == score.teacher_top_ids[0]
            )
            teacher_scored_tokens += len(score.token_ids)
        count = len(losses)
        stats = {key: value / count for key, value in stat_weighted.items()}
        stats["sampled_kl"] = stats["kl"]
        stats["teacher_scored_tokens"] = teacher_scored_tokens
        return torch.stack(losses).mean(), count, stats

    def _guided_loss_on_chunk(self, chunk):
        """Cross entropy only at teacher-inserted token positions.

        Matched NLL is always measured against the true teacher tokens. In the
        shuffled control, only the optimization targets are replaced.
        """
        seqs = []
        prediction_positions = []
        matched_targets = []
        training_targets = []
        for batch, trajectory in chunk:
            positions = trajectory.teacher_positions
            if not positions:
                continue
            seqs.append(batch["s_prompt"] + trajectory.completion[:-1])
            prediction_positions.append(
                [len(batch["s_prompt"]) + position - 1 for position in positions]
            )
            matched_targets.extend(trajectory.completion[position] for position in positions)
            if self.config.tutoring.shuffle_teacher_groups:
                training_targets.extend(
                    trajectory.shuffled_targets[position] for position in positions
                )
            else:
                training_targets.extend(
                    trajectory.completion[position] for position in positions
                )
        if not matched_targets:
            zero = next(self.student.parameters()).sum() * 0.0
            return zero, 0, {
                "teacher_token_nll": 0.0,
                "training_target_nll": 0.0,
                "teacher_token_top1_agreement": 0.0,
            }

        logits = self._gather_anchor_logits(
            self.student,
            seqs,
            prediction_positions,
            self.s_pad,
            self.device,
            autocast_dev=self.device.split(":")[0] if self.device != "cpu" else None,
        ).float()
        matched = torch.tensor(matched_targets, dtype=torch.long, device=self.device)
        training = torch.tensor(training_targets, dtype=torch.long, device=self.device)
        matched_nll = F.cross_entropy(logits, matched, reduction="none")
        training_nll = F.cross_entropy(logits, training, reduction="none")
        return training_nll.sum(), len(matched_targets), {
            "teacher_token_nll": matched_nll.detach().mean().item(),
            "training_target_nll": training_nll.detach().mean().item(),
            "teacher_token_top1_agreement": (
                logits.detach().argmax(dim=-1) == matched
            )
            .float()
            .mean()
            .item(),
        }

    @torch.no_grad()
    def _measure_guided_rollouts(self, rollouts) -> dict[str, float]:
        weighted = {
            "teacher_token_nll": 0.0,
            "training_target_nll": 0.0,
            "teacher_token_top1_agreement": 0.0,
        }
        total = 0
        for offset in range(0, len(rollouts), self.config.train.score_micro_seqs):
            _, count, stats = self._guided_loss_on_chunk(
                rollouts[offset : offset + self.config.train.score_micro_seqs]
            )
            total += count
            for key in weighted:
                weighted[key] += stats[key] * count
        if total <= 0:
            raise RuntimeError("guided post-update measurement has no teacher tokens")
        return {key: value / total for key, value in weighted.items()}

    def _track_guided_provenance(
        self,
        rollouts: list[tuple[dict, GuidedTrajectory]],
        paired_completions: list[list[int]],
    ) -> None:
        records = []
        for rollout_index, ((batch, trajectory), paired) in enumerate(
            zip(rollouts, paired_completions)
        ):
            groups = []
            for group_index, group in enumerate(trajectory.teacher_groups):
                token_ids = trajectory.completion[group.start : group.end]
                groups.append(
                    {
                        "group_index": group_index,
                        "start": group.start,
                        "end": group.end,
                        "token_ids": token_ids,
                        "token_sha256": hashlib.sha256(
                            json.dumps(token_ids).encode("utf-8")
                        ).hexdigest(),
                        "finish_reason": group.finish_reason,
                        "nonempty": bool(token_ids),
                    }
                )
            records.append(
                {
                    "step": getattr(self, "current_step", -1),
                    "microstep": getattr(self, "current_microstep", -1),
                    "rollout_index": rollout_index,
                    "policy_version": self.policy_version,
                    "task_id": batch["meta"].get("id"),
                    "prompt_hash": batch["meta"].get("prompt_hash"),
                    "interval_tokens": self.config.tutoring.interval_tokens,
                    "guidance_group_tokens": self.config.tutoring.guidance_group_tokens,
                    "completion_token_ids": trajectory.completion,
                    "teacher_mask": trajectory.teacher_mask,
                    "student_tokens": trajectory.student_tokens,
                    "teacher_tokens": trajectory.teacher_tokens,
                    "teacher_groups": groups,
                    "paired_student_only_token_ids": paired,
                    "shuffled_control": self.config.tutoring.shuffle_teacher_groups,
                }
            )
        with open(self.provenance_path, "a") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _train_guided_microstep(self, batch, gradient_scale: float, micro_started: float):
        config = self.config
        request_before = getattr(self.teacher_backend, "request_count", 0)
        success_before = getattr(self.teacher_backend, "successful_request_count", 0)
        failure_before = getattr(self.teacher_backend, "failure_count", 0)
        latency_before = getattr(self.teacher_backend, "total_latency_seconds", 0.0)
        rollout_generation_ms = 0.0

        def student_generate(indices, completions, limit):
            nonlocal rollout_generation_ms
            prompts = [
                batch[index]["s_prompt"] + completion
                for index, completion in zip(indices, completions)
            ]
            if self.rollout_worker is not None:
                result = self.rollout_worker.generate(
                    prompts,
                    limit,
                    expected_policy_version=self.policy_version,
                    compute_topk=False,
                )
                if result.policy_version != self.policy_version:
                    raise RuntimeError(
                        f"stale rollout {result.policy_version}, expected {self.policy_version}"
                    )
                rollout_generation_ms += result.generation_ms
                return result.completions
            started = time.perf_counter()
            output = self._generate(prompts, limit)
            rollout_generation_ms += (time.perf_counter() - started) * 1000
            return output

        def teacher_generate(indices, completions, limit):
            prefixes = [
                tuple(batch[index]["t_prompt"] + self.bridge.to_teacher(completion))
                for index, completion in zip(indices, completions)
            ]
            generated = self.teacher_backend.generate_tokens(prefixes, limit)
            return [
                TeacherGeneration(
                    token_ids=tuple(self.bridge.to_student(list(item.token_ids))),
                    finish_reason=item.finish_reason,
                )
                for item in generated
            ]

        trajectories = build_guided_trajectories(
            count=len(batch),
            max_completion_tokens=[item["mnt"] for item in batch],
            interval_tokens=int(config.tutoring.interval_tokens),
            guidance_group_tokens=config.tutoring.guidance_group_tokens,
            stop_ids=self.bridge.stop_ids,
            student_generate=student_generate,
            teacher_generate=teacher_generate,
        )
        rollouts = [
            (item, trajectory)
            for item, trajectory in zip(batch, trajectories)
            if trajectory.completion and trajectory.teacher_tokens
        ]
        if not rollouts:
            self.last_empty_guided_stats = {
                "rejected_no_teacher_rollouts": len(trajectories),
                "rejected_no_teacher_student_tokens": sum(
                    trajectory.student_tokens for trajectory in trajectories
                ),
                "rejected_no_teacher_trajectory_tokens": sum(
                    len(trajectory.completion) for trajectory in trajectories
                ),
            }
            return None
        if config.tutoring.shuffle_teacher_groups:
            assign_shuffled_group_targets([trajectory for _, trajectory in rollouts])

        paired_completions = [[] for _ in rollouts]
        if config.tutoring.paired_student_only:
            prompts = [item["s_prompt"] for item, _ in rollouts]
            limits = {item["mnt"] for item, _ in rollouts}
            if len(limits) != 1:
                raise ValueError("paired student-only rollout requires one token budget")
            limit = limits.pop()
            if self.rollout_worker is not None:
                paired_result = self.rollout_worker.generate(
                    prompts,
                    limit,
                    expected_policy_version=self.policy_version,
                    compute_topk=False,
                )
                paired_completions = paired_result.completions
                rollout_generation_ms += paired_result.generation_ms
            else:
                paired_completions = self._generate(prompts, limit)
        self._track_guided_provenance(rollouts, paired_completions)

        teacher_tokens = sum(trajectory.teacher_tokens for _, trajectory in rollouts)
        student_tokens = sum(trajectory.student_tokens for trajectory in trajectories)
        trajectory_tokens = sum(len(trajectory.completion) for trajectory in trajectories)
        paired_student_tokens = sum(len(completion) for completion in paired_completions)
        no_teacher_rollouts = len(trajectories) - len(rollouts)
        no_teacher_student_tokens = sum(
            trajectory.student_tokens
            for trajectory in trajectories
            if not trajectory.teacher_tokens
        )
        groups = sum(len(trajectory.teacher_groups) for _, trajectory in rollouts)
        nonempty_groups = sum(
            group.end > group.start
            for _, trajectory in rollouts
            for group in trajectory.teacher_groups
        )
        weighted = {
            "teacher_token_nll_pre": 0.0,
            "training_target_nll_pre": 0.0,
            "teacher_token_top1_pre": 0.0,
        }
        backward_ms = 0.0
        scoring_started = time.perf_counter()
        for offset in range(0, len(rollouts), config.train.score_micro_seqs):
            chunk = rollouts[offset : offset + config.train.score_micro_seqs]
            loss, count, stats = self._guided_loss_on_chunk(chunk)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite guided CE loss")
            backward_started = time.perf_counter()
            (loss / teacher_tokens * gradient_scale).backward()
            backward_ms += (time.perf_counter() - backward_started) * 1000
            weighted["teacher_token_nll_pre"] += stats["teacher_token_nll"] * count / teacher_tokens
            weighted["training_target_nll_pre"] += stats["training_target_nll"] * count / teacher_tokens
            weighted["teacher_token_top1_pre"] += (
                stats["teacher_token_top1_agreement"] * count / teacher_tokens
            )
        source_stats = self.source.batch_stats(
            [
                (item["meta"], self.s_tok.decode(trajectory.completion))
                for item, trajectory in rollouts
            ]
        )
        return {
            **weighted,
            **source_stats,
            "kl": weighted["teacher_token_nll_pre"],
            "sampled_kl": weighted["teacher_token_nll_pre"],
            "residual_mass": 0.0,
            "teacher_residual_mass": 0.0,
            "output_entropy": 0.0,
            "teacher_student_agreement": weighted["teacher_token_top1_pre"],
            "student_generated_tokens": student_tokens,
            "trajectory_tokens": trajectory_tokens,
            "teacher_anchor_positions": teacher_tokens,
            "teacher_inserted_tokens": teacher_tokens,
            "teacher_scored_tokens": teacher_tokens,
            "teacher_guidance_groups": groups,
            "teacher_nonempty_groups": nonempty_groups,
            "rollout_count": len(trajectories),
            "guided_rollout_count": len(rollouts),
            "rejected_no_teacher_rollouts": no_teacher_rollouts,
            "rejected_no_teacher_student_tokens": no_teacher_student_tokens,
            "paired_student_only_tokens": paired_student_tokens,
            "teacher_requests": (
                getattr(self.teacher_backend, "request_count", 0) - request_before
            ),
            "teacher_successful_requests": (
                getattr(self.teacher_backend, "successful_request_count", 0)
                - success_before
            ),
            "teacher_failures": (
                getattr(self.teacher_backend, "failure_count", 0) - failure_before
            ),
            "teacher_wall_clock_ms": (
                getattr(self.teacher_backend, "total_latency_seconds", 0.0)
                - latency_before
            )
            * 1000,
            "rollout_generation_ms": rollout_generation_ms,
            "rollout_student_topk_ms": 0.0,
            "scoring_and_forward_ms": (time.perf_counter() - scoring_started) * 1000,
            "backward_ms": backward_ms,
            "microstep_wall_clock_ms": (time.perf_counter() - micro_started) * 1000,
            "rollout_policy_version": self.policy_version,
            "_guided_rollouts": rollouts,
        }

    # ------------------------------------------------------------------- train

    def _train_microstep(self, gradient_scale: float) -> dict | None:
        """Sample one on-policy batch and accumulate its scaled gradients."""
        config = self.config
        micro_started = time.perf_counter()
        batch = []
        for _ in range(config.sampling.batch_prompts):
            messages, mnt, meta = self.source.sample()
            s_prompt = self._encode_prompt(self.s_tok, messages)
            t_prompt = self._encode_prompt(self.t_tok, messages)
            for _ in range(config.sampling.group_size):
                batch.append(
                    {
                        "s_prompt": s_prompt,
                        "t_prompt": t_prompt,
                        "mnt": mnt,
                        "meta": meta,
                    }
                )

        if config.train.loss_fn == "guided_ce":
            return self._train_guided_microstep(batch, gradient_scale, micro_started)

        completions: dict[int, list[int]] = {}
        rollout_top_ids: dict[int, dict[int, tuple[int, ...]]] = {}
        rollout_generation_ms = 0.0
        rollout_student_topk_ms = 0.0
        for mnt in sorted({item["mnt"] for item in batch}):
            indices = [i for i, item in enumerate(batch) if item["mnt"] == mnt]
            prompts = [batch[i]["s_prompt"] for i in indices]
            if self.rollout_worker is not None:
                result = self.rollout_worker.generate(
                    prompts,
                    mnt,
                    expected_policy_version=self.policy_version,
                )
                if result.policy_version != self.policy_version:
                    raise RuntimeError(f"stale rollout {result.policy_version}, expected {self.policy_version}")
                comps = result.completions
                rollout_top_ids.update(dict(zip(indices, result.student_top_ids)))
                rollout_generation_ms += result.generation_ms
                rollout_student_topk_ms += result.student_topk_ms
            else:
                comps = self._generate(prompts, mnt)
            completions.update(dict(zip(indices, comps)))
        for index, top_ids in rollout_top_ids.items():
            batch[index]["student_top_ids"] = top_ids

        rollouts = [(batch[i], completions[i]) for i in range(len(batch)) if completions[i]]
        if not rollouts:
            return None

        student_tokens = sum(len(completion) for _, completion in rollouts)
        anchors = (
            sum(
                len(
                    select_anchor_positions(
                        completion,
                        config.tutoring.interval_tokens,
                        stop_ids=self.bridge.stop_ids,
                        always_include_final_anchor=config.tutoring.always_include_final_anchor,
                        include_eos_anchor=config.tutoring.include_eos_anchor,
                        anchor_window_tokens=config.tutoring.anchor_window_tokens,
                        max_anchors_per_sequence=config.tutoring.max_anchors_per_sequence,
                    )
                )
                for _, completion in rollouts
            )
            if config.train.loss_fn == "sparse_anchor_rkl"
            else student_tokens
        )
        if anchors <= 0:
            return None

        teacher_requests_before = getattr(self.teacher_backend, "request_count", 0)
        teacher_latency_before = getattr(self.teacher_backend, "total_latency_seconds", 0.0)
        weighted = {
            "kl": 0.0,
            "sampled_kl": 0.0,
            "residual_mass": 0.0,
            "teacher_residual_mass": 0.0,
            "output_entropy": 0.0,
            "teacher_student_agreement": 0.0,
        }
        teacher_scored_tokens = 0
        backward_ms = 0.0
        scoring_started = time.perf_counter()
        for offset in range(0, len(rollouts), config.train.score_micro_seqs):
            chunk = rollouts[offset : offset + config.train.score_micro_seqs]
            if config.train.loss_fn == "sparse_anchor_rkl":
                loss, count, stats = self._sparse_loss_on_chunk(chunk)
                normalized_loss = loss * count / anchors
            else:
                loss, count, stats = self._loss_on_chunk(*self._chunk_args(chunk))
                normalized_loss = loss / student_tokens
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite OPD loss")
            backward_started = time.perf_counter()
            (normalized_loss * gradient_scale).backward()
            backward_ms += (time.perf_counter() - backward_started) * 1000
            for key in weighted:
                weighted[key] += stats.get(key, 0.0) * count / anchors
            teacher_scored_tokens += int(stats.get("teacher_scored_tokens", 0))

        source_stats = self.source.batch_stats(
            [(item["meta"], self.s_tok.decode(completion)) for item, completion in rollouts]
        )
        return {
            **weighted,
            **source_stats,
            "student_generated_tokens": student_tokens,
            "teacher_anchor_positions": anchors,
            "teacher_scored_tokens": teacher_scored_tokens,
            "rollout_count": len(rollouts),
            "teacher_requests": (getattr(self.teacher_backend, "request_count", 0) - teacher_requests_before),
            "teacher_wall_clock_ms": (
                getattr(self.teacher_backend, "total_latency_seconds", 0.0) - teacher_latency_before
            )
            * 1000,
            "rollout_generation_ms": rollout_generation_ms,
            "rollout_student_topk_ms": rollout_student_topk_ms,
            "scoring_and_forward_ms": (time.perf_counter() - scoring_started) * 1000,
            "backward_ms": backward_ms,
            "microstep_wall_clock_ms": (time.perf_counter() - micro_started) * 1000,
        }

    def train(self):
        config = self.config
        run_started = time.time()
        grad_accum = config.train.gradient_accumulation_steps
        for step in range(self.start_step, config.train.steps):
            self.current_step = step
            step_started = time.perf_counter()
            for group in self.opt.param_groups:
                group["lr"] = self._lr_at(step)
            self.opt.zero_grad(set_to_none=True)
            if str(self.device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(self.device)
            if config.model.rollout_device and str(config.model.rollout_device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(config.model.rollout_device)

            microsteps = []
            for microstep_index in range(grad_accum):
                self.current_microstep = microstep_index
                result = None
                rejected_rollouts = 0
                rejected_tokens = 0
                retries = (
                    config.tutoring.empty_microstep_retries
                    if config.train.loss_fn == "guided_ce"
                    else 1
                )
                for retry in range(retries + 1):
                    result = self._train_microstep(gradient_scale=1.0 / grad_accum)
                    if result is not None:
                        break
                    empty_stats = getattr(self, "last_empty_guided_stats", {})
                    rejected_rollouts += empty_stats.get(
                        "rejected_no_teacher_rollouts", 0
                    )
                    rejected_tokens += empty_stats.get(
                        "rejected_no_teacher_student_tokens", 0
                    )
                    if retry < retries:
                        logger.warning(
                            "step %d microstep %d: no teacher intervention; "
                            "retrying (%d/%d)",
                            step,
                            microstep_index,
                            retry + 1,
                            retries,
                        )
                if result is None:
                    raise RuntimeError(
                        "guided microstep exhausted bounded no-intervention retries"
                    )
                result["rejected_no_teacher_rollouts"] = (
                    result.get("rejected_no_teacher_rollouts", 0)
                    + rejected_rollouts
                )
                result["rejected_no_teacher_student_tokens"] = (
                    result.get("rejected_no_teacher_student_tokens", 0)
                    + rejected_tokens
                )
                result["rollout_count"] += rejected_rollouts
                result["student_generated_tokens"] += rejected_tokens
                result["trajectory_tokens"] = (
                    result.get("trajectory_tokens", 0) + rejected_tokens
                )
                result["empty_microstep_retries"] = retry
                microsteps.append(result)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.student.parameters(),
                config.train.max_grad_norm,
            )
            grad_norm_value = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
            if not math.isfinite(grad_norm_value) or grad_norm_value <= 0:
                raise FloatingPointError(f"invalid LoRA gradient norm: {grad_norm_value}")
            self.opt.step()

            guided_post = None
            if config.train.loss_fn == "guided_ce":
                guided_rollouts = [
                    rollout
                    for item in microsteps
                    for rollout in item["_guided_rollouts"]
                ]
                guided_post = self._measure_guided_rollouts(guided_rollouts)
            if self.rollout_worker is not None:
                self.policy_version += 1
                self.last_sync_metrics = self.rollout_worker.sync_from(
                    self.student,
                    self.policy_version,
                )

            total_anchors = sum(item["teacher_anchor_positions"] for item in microsteps)
            total_tokens = sum(item["student_generated_tokens"] for item in microsteps)
            total_rollouts = sum(item["rollout_count"] for item in microsteps)
            mean_keys = {
                "kl",
                "sampled_kl",
                "residual_mass",
                "teacher_residual_mass",
                "output_entropy",
                "teacher_student_agreement",
            }
            metrics = {
                key: sum(item.get(key, 0.0) * item["teacher_anchor_positions"] for item in microsteps) / total_anchors
                for key in mean_keys
            }
            if metrics["residual_mass"] > 0.20:
                raise RuntimeError(f"student residual-mass kill switch tripped: {metrics['residual_mass']:.4f}")
            for key in set().union(*(item.keys() for item in microsteps)) - mean_keys:
                if key.startswith("format_"):
                    metrics[key] = sum(item[key] for item in microsteps) / len(microsteps)
            metrics.update(
                {
                    "completion_len": total_tokens / total_rollouts,
                    "gradient_norm": grad_norm_value,
                    "gradient_accumulation_steps": grad_accum,
                    "student_generated_tokens": total_tokens,
                    "teacher_anchor_positions": total_anchors,
                    "teacher_scored_tokens": sum(item["teacher_scored_tokens"] for item in microsteps),
                    "teacher_inserted_tokens": sum(
                        item.get("teacher_inserted_tokens", 0) for item in microsteps
                    ),
                    "teacher_guidance_groups": sum(
                        item.get("teacher_guidance_groups", 0) for item in microsteps
                    ),
                    "teacher_nonempty_groups": sum(
                        item.get("teacher_nonempty_groups", 0) for item in microsteps
                    ),
                    "guided_rollout_count": sum(
                        item.get("guided_rollout_count", item["rollout_count"])
                        for item in microsteps
                    ),
                    "rejected_no_teacher_rollouts": sum(
                        item.get("rejected_no_teacher_rollouts", 0)
                        for item in microsteps
                    ),
                    "rejected_no_teacher_student_tokens": sum(
                        item.get("rejected_no_teacher_student_tokens", 0)
                        for item in microsteps
                    ),
                    "paired_student_only_tokens": sum(
                        item.get("paired_student_only_tokens", 0)
                        for item in microsteps
                    ),
                    "empty_microstep_retries": sum(
                        item.get("empty_microstep_retries", 0)
                        for item in microsteps
                    ),
                    "teacher_requests": sum(item["teacher_requests"] for item in microsteps),
                    "teacher_successful_requests": sum(
                        item.get("teacher_successful_requests", 0) for item in microsteps
                    ),
                    "teacher_failures": sum(
                        item.get("teacher_failures", 0) for item in microsteps
                    ),
                    "teacher_wall_clock_ms": sum(item["teacher_wall_clock_ms"] for item in microsteps),
                    "rollout/generation_ms": sum(item["rollout_generation_ms"] for item in microsteps),
                    "rollout/student_topk_ms": sum(item["rollout_student_topk_ms"] for item in microsteps),
                    "student/scoring_and_forward_ms": sum(item["scoring_and_forward_ms"] for item in microsteps),
                    "student/backward_ms": sum(item["backward_ms"] for item in microsteps),
                    "rollout/policy_version": self.policy_version,
                    "rollout/staleness": 0,
                    "rollout/sync_ms": self.last_sync_metrics["sync_ms"],
                    "step/wall_clock_ms": (time.perf_counter() - step_started) * 1000,
                    "run/wall_clock_seconds": time.time() - run_started,
                    "lr": self.opt.param_groups[0]["lr"],
                }
            )
            if config.train.loss_fn == "guided_ce" and guided_post is not None:
                pre_teacher_nll = sum(
                    item["teacher_token_nll_pre"] * item["teacher_inserted_tokens"]
                    for item in microsteps
                ) / total_anchors
                pre_training_nll = sum(
                    item["training_target_nll_pre"] * item["teacher_inserted_tokens"]
                    for item in microsteps
                ) / total_anchors
                pre_top1 = sum(
                    item["teacher_token_top1_pre"] * item["teacher_inserted_tokens"]
                    for item in microsteps
                ) / total_anchors
                metrics.update(
                    {
                        "teacher_token_nll_pre": pre_teacher_nll,
                        "teacher_token_nll_post": guided_post["teacher_token_nll"],
                        "teacher_token_nll_delta": (
                            guided_post["teacher_token_nll"] - pre_teacher_nll
                        ),
                        "teacher_token_nll_decreased": float(
                            guided_post["teacher_token_nll"] < pre_teacher_nll
                        ),
                        "training_target_nll_pre": pre_training_nll,
                        "training_target_nll_post": guided_post["training_target_nll"],
                        "training_target_nll_delta": (
                            guided_post["training_target_nll"] - pre_training_nll
                        ),
                        "teacher_token_top1_pre": pre_top1,
                        "teacher_token_top1_post": guided_post[
                            "teacher_token_top1_agreement"
                        ],
                        "trajectory_tokens": sum(
                            item["trajectory_tokens"] for item in microsteps
                        ),
                        "teacher_group_nonempty_rate": (
                            sum(item["teacher_nonempty_groups"] for item in microsteps)
                            / sum(item["teacher_guidance_groups"] for item in microsteps)
                        ),
                        "teacher_intervention_coverage": (
                            sum(item["guided_rollout_count"] for item in microsteps)
                            / total_rollouts
                        ),
                        "teacher_request_success_rate": (
                            sum(item["teacher_successful_requests"] for item in microsteps)
                            / max(1, sum(item["teacher_requests"] for item in microsteps))
                        ),
                        "teacher_circuit_breaker_open": float(
                            getattr(self.teacher_backend, "consecutive_failures", 0)
                            >= getattr(self.teacher_backend, "circuit_breaker_failures", 3)
                        ),
                    }
                )
            if str(self.device).startswith("cuda"):
                metrics["student/peak_vram_mib"] = torch.cuda.max_memory_allocated(self.device) / 1024**2
            if config.model.rollout_device and str(config.model.rollout_device).startswith("cuda"):
                metrics["rollout/peak_vram_mib"] = (
                    torch.cuda.max_memory_allocated(config.model.rollout_device) / 1024**2
                )

            if step % config.logging.log_every == 0:
                logger.info(
                    "step %d | kl/anchor=%.4f anchors=%d tokens=%d grad_norm=%.4f len=%.1f lr=%.2e (%.0fs)",
                    step,
                    metrics["kl"],
                    total_anchors,
                    total_tokens,
                    grad_norm_value,
                    metrics["completion_len"],
                    self.opt.param_groups[0]["lr"],
                    time.time() - run_started,
                )
                self._track(metrics, step)

            if config.train.eval_every and step and step % config.train.eval_every == 0:
                eval_metrics = self._evaluate()
                logger.info(
                    "step %d | %s",
                    step,
                    " ".join(f"{key}={value:.4f}" for key, value in eval_metrics.items()),
                )
                self._track(eval_metrics, step)

            checkpoint_name = intermediate_checkpoint_name(
                step, config.train.save_steps, config.train.steps
            )
            if checkpoint_name is not None:
                self._save(checkpoint_name, step=step)

        eval_metrics = self._evaluate()
        final_step = max(self.start_step - 1, config.train.steps - 1)
        logger.info(
            "final | %s",
            " ".join(f"{key}={value:.4f}" for key, value in eval_metrics.items()),
        )
        self._track({"event": "final_eval", **eval_metrics}, final_step)
        self._save("final", step=final_step)

    # -------------------------------------------------------------------- eval

    def _evaluate(self) -> dict[str, float]:
        """Held-out evaluation, delegated to the source (metric is task-specific)."""
        self.student.eval()
        try:
            return self.source.evaluate(self)
        finally:
            self.student.train()

    # ------------------------------------------------------------- bookkeeping

    def _track(self, metrics: dict, step: int):
        record = {
            "step": step,
            "timestamp": time.time(),
            **metrics,
        }
        with open(self.metrics_path, "a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self.wandb:
            try:
                self.wandb.log(metrics, step=step)
            except Exception as e:  # noqa: BLE001 — a metrics backend must never kill training
                logger.warning("wandb.log failed (%s); disabling wandb", e)
                self.wandb = None

    def _save(self, name: str, step: int | None = None):
        path = os.path.join(self.config.train.output_dir, name)
        logger.info("Saving checkpoint -> %s", path)
        self.student.save_pretrained(path)
        self.s_tok.save_pretrained(path)
        with open(os.path.join(path, "opd_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=2)
        torch.save(
            {
                "optimizer": self.opt.state_dict(),
                "step": step,
                "policy_version": self.policy_version,
                "python_rng_state": self.rng.getstate(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(self.device)
                if str(self.device).startswith("cuda")
                else None,
            },
            os.path.join(path, "trainer_state.pt"),
        )
        self._prune_checkpoints()

    def _resume(self, checkpoint: str) -> None:
        """Restore adapter, optimizer, RNG, step, and policy version exactly."""
        from peft import load_peft_weights, set_peft_model_state_dict

        state_path = os.path.join(checkpoint, "trainer_state.pt")
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"missing checkpoint trainer state: {state_path}")
        adapter_state = load_peft_weights(checkpoint, device=self.device)
        result = set_peft_model_state_dict(self.student, adapter_state)
        if getattr(result, "unexpected_keys", None):
            raise RuntimeError(f"unexpected adapter keys during resume: {result.unexpected_keys}")
        state = torch.load(state_path, map_location=self.device, weights_only=False)
        self.opt.load_state_dict(state["optimizer"])
        saved_step = state.get("step")
        self.start_step = 0 if saved_step is None else int(saved_step) + 1
        self.policy_version = int(state["policy_version"])
        self.rng.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if state.get("cuda_rng_state") is not None and str(self.device).startswith("cuda"):
            torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), self.device)
        logger.info(
            "Resumed %s at next_step=%d policy_version=%d",
            checkpoint,
            self.start_step,
            self.policy_version,
        )

    def _prune_checkpoints(self):
        """Keep only the newest keep_checkpoints step_* dirs ("final" is exempt)."""
        keep = self.config.train.keep_checkpoints
        if keep <= 0:
            return
        steps = sorted(
            (d for d in os.listdir(self.config.train.output_dir) if d.startswith("step_")),
            key=lambda d: int(d.split("_")[1]),
        )
        for d in steps[:-keep]:
            victim = os.path.join(self.config.train.output_dir, d)
            logger.info("Pruning old checkpoint %s", victim)
            shutil.rmtree(victim, ignore_errors=True)


def main():
    from palingenesis.logging import setup_logging

    setup_logging(rank=0)
    config = OPDConfig.from_cli()
    for warning in config.validate():
        logger.warning(warning)
    OPDTrainer(config).train()


if __name__ == "__main__":
    main()

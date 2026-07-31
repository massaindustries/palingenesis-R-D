#!/usr/bin/env python3
"""LoRA SFT baseline over the immutable teacher-generated MBPP dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import torch
import yaml
from peft import (
    LoraConfig,
    get_peft_model,
    load_peft_weights,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from palingenesis.loss import IGNORE_INDEX, chunked_cross_entropy_loss


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    try:
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    return tokenizer.encode(text, add_special_tokens=False)


def encode_rows(path: str, tokenizer) -> list[tuple[list[int], int]]:
    encoded = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row["messages"]
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError(
                "offline SFT rows must end with a teacher assistant message"
            )
        prompt = render_prompt(tokenizer, messages[:-1])
        completion = tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
        if not completion or completion[-1] != tokenizer.eos_token_id:
            completion.append(tokenizer.eos_token_id)
        encoded.append((prompt + completion, len(prompt)))
    if not encoded:
        raise ValueError("offline SFT dataset is empty")
    return encoded


def collate(samples, pad_token_id: int, device: str):
    width = max(len(ids) for ids, _ in samples)
    input_ids = torch.full((len(samples), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    for row, (ids, prompt_len) in enumerate(samples):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids)
        attention_mask[row, :length] = 1
        labels[row, prompt_len:length] = torch.tensor(ids[prompt_len:])
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def learning_rate(step: int, *, peak: float, warmup: int, total: int) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def adapter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(
                bytes(parameter.detach().contiguous().cpu().view(torch.uint8).numpy())
            )
    return digest.hexdigest()


def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    rng,
    output: Path,
    step: int,
    keep: int,
    device: str,
) -> None:
    path = output / f"step_{step}"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "python_rng_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device),
            "adapter_hash": adapter_hash(model),
        },
        path / "trainer_state.pt",
    )
    if keep > 0:
        checkpoints = sorted(
            (item for item in output.glob("step_*") if item.is_dir()),
            key=lambda item: int(item.name.split("_")[1]),
        )
        for victim in checkpoints[:-keep]:
            shutil.rmtree(victim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="/opt/sparse-opd/configs/sparse_opd/offline_sft.yaml"
    )
    parser.add_argument(
        "--dataset", default="/opt/sparse-opd/data/offline/teacher_sft_train_v2.jsonl"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--resume-from", default="")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved = {
        **config,
        "offline_runtime": {
            "dataset": args.dataset,
            "steps": args.steps,
            "micro_batch_size": args.micro_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "global_prompts_per_optimizer_step": (
                args.micro_batch_size * args.gradient_accumulation_steps
            ),
            "device": args.device,
        },
    }
    resolved_path = output / "config_resolved.json"
    if not resolved_path.exists():
        resolved_path.write_text(json.dumps(resolved, indent=2) + "\n")

    seed = int(config["train"]["seed"])
    rng = random.Random(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["student"], local_files_only=True
    )
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    dataset = encode_rows(args.dataset, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["student"],
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    if config["model"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    adapter = config["adapter"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=adapter["rank"],
            lora_alpha=adapter["alpha"],
            lora_dropout=adapter["dropout"],
            bias=adapter["bias"],
            target_modules=adapter["target_modules"],
            task_type="CAUSAL_LM",
        ),
    ).to(args.device)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    start_step = 0
    if args.resume_from:
        state = torch.load(
            Path(args.resume_from) / "trainer_state.pt",
            map_location=args.device,
            weights_only=False,
        )
        adapter_state = load_peft_weights(args.resume_from, device=args.device)
        result = set_peft_model_state_dict(model, adapter_state)
        if getattr(result, "unexpected_keys", None):
            raise RuntimeError(f"unexpected adapter keys: {result.unexpected_keys}")
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng_state"].cpu(), args.device)
        start_step = int(state["step"]) + 1

    metrics_path = output / "metrics.jsonl"
    run_started = time.perf_counter()
    for step in range(start_step, args.steps):
        step_started = time.perf_counter()
        lr = learning_rate(
            step,
            peak=float(config["train"]["learning_rate"]),
            warmup=int(config["train"]["warmup_steps"]),
            total=args.steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(args.device)
        accumulated_loss = 0.0
        trained_tokens = 0
        for _ in range(args.gradient_accumulation_steps):
            samples = [
                dataset[rng.randrange(len(dataset))]
                for _ in range(args.micro_batch_size)
            ]
            ids, mask, labels = collate(samples, pad_token_id, args.device)
            valid = int((labels[:, 1:] != IGNORE_INDEX).sum().item())
            base = model.get_base_model()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = base.model(
                    input_ids=ids[:, :-1],
                    attention_mask=mask[:, :-1],
                ).last_hidden_state
                loss = chunked_cross_entropy_loss(
                    hidden,
                    labels[:, 1:],
                    base.lm_head,
                    num_chunks=min(8, hidden.shape[1]),
                    global_valid_tokens=max(1, valid),
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("offline SFT loss is non-finite")
            (loss / args.gradient_accumulation_steps).backward()
            accumulated_loss += float(loss.detach().item())
            trained_tokens += valid
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            float(config["train"]["max_grad_norm"]),
        )
        grad_norm_value = float(grad_norm.item())
        if not math.isfinite(grad_norm_value) or grad_norm_value <= 0:
            raise FloatingPointError(
                f"invalid offline gradient norm: {grad_norm_value}"
            )
        optimizer.step()
        record = {
            "step": step,
            "loss": accumulated_loss / args.gradient_accumulation_steps,
            "gradient_norm": grad_norm_value,
            "learning_rate": lr,
            "student_training_tokens": trained_tokens,
            "teacher_requests": 0,
            "teacher_anchor_positions": 0,
            "teacher_scored_tokens": 0,
            "student/peak_vram_mib": torch.cuda.max_memory_allocated(args.device)
            / 1024**2,
            "step/wall_clock_ms": (time.perf_counter() - step_started) * 1000,
            "run/wall_clock_seconds": time.perf_counter() - run_started,
        }
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(
            f"step {step} loss={record['loss']:.4f} grad_norm={grad_norm_value:.4f} "
            f"tokens={trained_tokens} lr={lr:.2e}",
            flush=True,
        )
        save_steps = int(config["train"]["save_steps"])
        if save_steps and step and step % save_steps == 0:
            save_checkpoint(
                model,
                tokenizer,
                optimizer,
                rng,
                output,
                step,
                int(config["train"]["keep_checkpoints"]),
                args.device,
            )
    final = output / "final"
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    torch.save(
        {
            "step": args.steps - 1,
            "optimizer": optimizer.state_dict(),
            "python_rng_state": rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(args.device),
            "adapter_hash": adapter_hash(model),
        },
        final / "trainer_state.pt",
    )


if __name__ == "__main__":
    main()

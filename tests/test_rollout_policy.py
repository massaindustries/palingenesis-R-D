import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import GPT2Config, GPT2LMHeadModel

from palingenesis.opd.config import OPDConfig
from palingenesis.opd.rollout_worker import (
    TransformersRolloutWorker,
    adapter_state_hash,
    transformers_top_k,
)
from palingenesis.opd.trainer import OPDTrainer


class _AdapterOnly(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.randn(3, 2))
        self.lora_B = torch.nn.Parameter(torch.randn(2, 3))
        self.frozen = torch.nn.Parameter(torch.randn(2), requires_grad=False)


def test_adapter_synchronization_hash_and_policy_version():
    trainer = _AdapterOnly()
    replica = _AdapterOnly()
    worker = TransformersRolloutWorker.__new__(TransformersRolloutWorker)
    worker.model = replica
    worker.device = "cpu"
    worker.policy_version = -1
    result = worker.sync_from(trainer, policy_version=7)
    assert result["policy_version"] == 7
    assert worker.policy_version == 7
    assert adapter_state_hash(trainer) == adapter_state_hash(replica)


def test_rollout_rejects_policy_version_mismatch():
    worker = TransformersRolloutWorker.__new__(TransformersRolloutWorker)
    worker.policy_version = 2
    worker.device = "cpu"
    with pytest.raises(RuntimeError, match="policy version mismatch"):
        worker.generate([[1]], 1, expected_policy_version=3)


@pytest.mark.parametrize(("configured", "transformers_value"), [(-1, 0), (0, 0), (64, 64)])
def test_transformers_top_k_normalizes_unbounded_sentinel(configured, transformers_value):
    assert transformers_top_k(configured) == transformers_value


class _TokenizerStub:
    def save_pretrained(self, path):
        return (path,)


def _tiny_peft():
    base = GPT2LMHeadModel(GPT2Config(
        vocab_size=32,
        n_positions=16,
        n_embd=16,
        n_layer=1,
        n_head=2,
    ))
    return get_peft_model(
        base,
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["c_attn"],
            task_type="CAUSAL_LM",
        ),
    )


def _bare_checkpoint_trainer(path):
    trainer = OPDTrainer.__new__(OPDTrainer)
    trainer.config = OPDConfig()
    trainer.config.train.output_dir = str(path)
    trainer.config.train.keep_checkpoints = 0
    trainer.student = _tiny_peft()
    trainer.s_tok = _TokenizerStub()
    trainer.device = "cpu"
    trainer.policy_version = 4
    trainer.rng = __import__("random").Random(123)
    trainer.opt = torch.optim.AdamW(
        [p for p in trainer.student.parameters() if p.requires_grad],
        lr=1e-3,
    )
    return trainer


def test_opd_checkpoint_resume_roundtrip(tmp_path):
    original = _bare_checkpoint_trainer(tmp_path)
    loss = sum(parameter.sum() for parameter in original.student.parameters() if parameter.requires_grad)
    loss.backward()
    original.opt.step()
    original._save("step_6", step=6)

    restored = _bare_checkpoint_trainer(tmp_path)
    restored._resume(str(tmp_path / "step_6"))
    assert restored.start_step == 7
    assert restored.policy_version == 4
    for left, right in zip(
        (p for p in original.student.parameters() if p.requires_grad),
        (p for p in restored.student.parameters() if p.requires_grad),
    ):
        torch.testing.assert_close(left, right)
    assert restored.opt.state_dict()["state"]

"""Tests for adaptive curriculum training control."""

from __future__ import annotations

from types import SimpleNamespace

from scripts.train_chamelia import run_training
from training.curriculum.control import AdaptiveTrainingController, EvalPoint
from training.curriculum.domains.stage1_reasoning import ReasoningCurriculumDomain


def test_controller_extends_when_stage_is_improving() -> None:
    """Improving stage scores should trigger an extension at budget exhaustion."""
    controller = AdaptiveTrainingController(extension_factor=2.0)
    domain = ReasoningCurriculumDomain(domain_variant="basic_arithmetic", batch_size=8, seq_len=8, vocab_size=64)
    history = [
        EvalPoint(step=20, mean_loss=3.0, stage_score=0.20, metrics={domain.domain_name(): {"accuracy": 0.20}}),
        EvalPoint(step=40, mean_loss=2.2, stage_score=0.45, metrics={domain.domain_name(): {"accuracy": 0.45}}),
        EvalPoint(step=60, mean_loss=1.8, stage_score=0.70, metrics={domain.domain_name(): {"accuracy": 0.70}}),
    ]
    decision = controller.decide(
        history=history,
        stage_passed=False,
        total_steps=60,
        current_budget_steps=60,
        max_total_stage_steps=200,
        extensions_used=0,
        max_extensions=2,
        retunes_used=0,
        max_retunes=2,
    )
    assert decision.action == "extend"
    assert decision.new_budget_steps == 120


def test_train_chamelia_arithmetic_stage_passes_locally(tmp_path) -> None:
    """Selected arithmetic stage should pass under the adaptive controller."""
    args = SimpleNamespace(
        curriculum_config="/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/configs/curriculum.yaml",
        chamelia_config="/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/configs/chamelia.yaml",
        stage=["1"],
        domain=["basic_arithmetic"],
        device="cpu",
        seed=11,
        backbone_mode="stub",
        lr=3e-3,
        initial_stage_steps=160,
        eval_every=40,
        max_total_stage_steps=320,
        max_extensions=1,
        max_retunes=1,
        extension_factor=1.5,
        retune_lr_factors="0.5,1.5",
        clip_grad=1.0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )
    assert run_training(args) == 0

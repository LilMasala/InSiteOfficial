"""Real-HJEPA curriculum training integration test."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from scripts.train_chamelia import build_model, build_optimizer, build_stage_domains, run_training
from training.curriculum.graduation import GraduationManager
from training.curriculum.stage_runner import CurriculumStageRunner


CURRICULUM_CONFIG = Path("/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/configs/curriculum.yaml")
TINY_HJEPA_CONFIG = Path("/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/configs/chamelia_hjepa_tiny.yaml")


def test_real_hjepa_backbone_runs_one_curriculum_step() -> None:
    """Verify the curriculum runner can execute one step with the real HJEPA backbone."""
    curriculum_config = yaml.safe_load(CURRICULUM_CONFIG.read_text())
    stages = build_stage_domains(
        curriculum_config,
        selected_stages=[1],
        selected_domains=["basic_arithmetic"],
    )

    chamelia_config = yaml.safe_load(TINY_HJEPA_CONFIG.read_text())

    device = torch.device("cpu")
    model = build_model(
        chamelia_config,
        stages,
        device=device,
        backbone_mode="hjepa",
    )
    optimizer = build_optimizer(model, chamelia_config["training"], 1.0e-3)

    graduation_manager = GraduationManager(stages, curriculum_config)
    runner = CurriculumStageRunner(
        model=model,
        stages=stages,
        graduation_manager=graduation_manager,
        config=curriculum_config,
        device=device,
        optimizer=optimizer,
    )
    runner._runtime_domains[stages[0][0].domain_name()] = model.domain

    domain = stages[0][0]
    batch = next(iter(domain.get_data_loader(domain.cost.current_level, split="train"))).to_device(device)
    batch.domain_state["level"] = domain.cost.current_level
    if batch.tokens is not None:
        batch.domain_state["tokens"] = batch.tokens
    step_batch = runner._to_step_batch(batch, domain)

    loss = runner._default_train_step(step_batch, domain)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    assert loss.dim() == 0


def test_build_stage_domains_honors_configured_stage_window() -> None:
    """Configured stage windows should apply when no CLI stage filter is passed."""
    curriculum_config = yaml.safe_load(CURRICULUM_CONFIG.read_text())
    curriculum_config["curriculum"]["start_stage"] = 1
    curriculum_config["curriculum"]["end_stage"] = 4

    stages = build_stage_domains(
        curriculum_config,
        selected_stages=[],
        selected_domains=[],
    )
    assert [stage[0].stage() for stage in stages] == [1, 2, 3, 4]


def test_real_hjepa_basic_arithmetic_emits_bridge_artifacts_locally(tmp_path) -> None:
    """Verify tiny real-HJEPA curriculum training emits bridge-loadable checkpoints."""
    args = SimpleNamespace(
        curriculum_config=str(CURRICULUM_CONFIG),
        chamelia_config=str(TINY_HJEPA_CONFIG),
        stage=["1"],
        domain=["basic_arithmetic"],
        device="cpu",
        data_root=None,
        seed=11,
        backbone_mode="hjepa",
        lr=1.0e-3,
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
    exit_code = run_training(args)
    assert exit_code in {0, 1}

    bridge_artifacts = list((Path(args.checkpoint_dir) / "bridge_artifacts").glob("*.pth"))
    assert bridge_artifacts
    payload = torch.load(bridge_artifacts[0], map_location="cpu")
    assert "model_state_dict" in payload
    assert "config" in payload
    assert payload["bridge_backbone_mode"] == "hjepa"

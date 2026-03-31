#!/usr/bin/env python3
"""Adaptive curriculum trainer for Chamelia V2."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
import random
import sys
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chamelia.actor import Actor
from src.chamelia.chamelia import Chamelia
from src.chamelia.configurator import Configurator
from src.chamelia.cost import CostModule, IntrinsicCost, TrainableCritic
from src.chamelia.memory import LatentMemory
from src.losses.combined import CombinedLoss
from src.losses.hjepa_loss import HJEPALoss
from src.models.hjepa import HJEPA
from training.curriculum.control import AdaptiveTrainingController, EvalPoint
from training.curriculum.domains.base import CurriculumDomain
from training.curriculum.domains.stage0_language import LanguageCurriculumDomain
from training.curriculum.domains.stage1_reasoning import ReasoningCurriculumDomain
from training.curriculum.domains.stage2_patterns import PatternCurriculumDomain
from training.curriculum.domains.stage3_games import GamesCurriculumDomain
from training.curriculum.domains.stage4_collaborative import CollaborativeCurriculumDomain
from training.curriculum.domains.stage5_health import HealthCurriculumDomain
from training.curriculum.graduation import GraduationManager
from training.curriculum.stage_runner import CurriculumStageRunner


class StubSequenceHJEPA(torch.nn.Module):
    """Small pre-embedded-token backbone for curriculum training."""

    def __init__(self, embed_dim: int) -> None:
        """Initialize the stub backbone.

        Args:
            embed_dim: Shared latent dimension D.

        Returns:
            None.
        """
        super().__init__()
        self.embed_dim = embed_dim

    def _apply_fpn(self, features: torch.Tensor, is_prediction: bool = False) -> list[torch.Tensor]:
        """Construct a simple three-level token pyramid.

        Args:
            features: Token features [B, N, D].
            is_prediction: Unused compatibility flag.

        Returns:
            Pyramid features [[B, N, D], [B, ceil(N/2), D], [B, 1, D]].
        """
        _ = is_prediction
        return [features, features[:, ::2, :], features.mean(dim=1, keepdim=True)]

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass over pre-embedded tokens.

        Args:
            tokens: Embedded tokens [B, N, D].
            mask: Binary mask [B, N].

        Returns:
            HJEPA-compatible output dict.
        """
        masked_tokens = tokens * (1.0 - mask.unsqueeze(-1))
        cls = masked_tokens.mean(dim=1, keepdim=True)
        target_features = torch.cat([cls, tokens], dim=1)
        predictions = self._apply_fpn(masked_tokens, is_prediction=True)
        targets = self._apply_fpn(tokens, is_prediction=False)
        masks_valid = [
            torch.ones(tokens.shape[0], level.shape[1], dtype=torch.bool, device=tokens.device)
            for level in predictions
        ]
        return {
            "predictions": predictions,
            "targets": targets,
            "masks_valid": masks_valid,
            "context_features": target_features,
            "target_features": target_features,
        }


@dataclass
class StageSnapshot:
    """Restorable stage state for bounded retunes."""

    model_state: dict[str, Any]
    domain_cost_states: list[dict[str, Any]]
    score: float


def parse_csv_arg(values: list[str] | None) -> list[str]:
    """Split repeated or comma-separated CLI arguments.

    Args:
        values: Raw argument values.

    Returns:
        Flat list of strings.
    """
    if not values:
        return []
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def set_seed(seed: int) -> None:
    """Seed Python and PyTorch RNGs.

    Args:
        seed: Global seed.

    Returns:
        None.
    """
    random.seed(seed)
    torch.manual_seed(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML config file.

    Args:
        path: Config path.

    Returns:
        Parsed config dictionary.
    """
    return yaml.safe_load(Path(path).read_text())


def _resolve_training_config(chamelia_config: dict[str, Any], backbone_mode: str) -> dict[str, Any]:
    """Resolve the effective training config for the requested backbone mode.

    Args:
        chamelia_config: Parsed Chamelia config.
        backbone_mode: Requested backbone mode.

    Returns:
        Effective training configuration.
    """
    if backbone_mode == "stub":
        return chamelia_config.get(
            "stub_training",
            {
                "optimizer": "adam",
                "weight_decay": 0.0,
                "learning_rate": 3e-3,
                "store_to_memory": True,
                "critic_train_interval": 20,
                "critic_loss_weight": 1.0,
                "mode1_distill_interval": 0,
                "mode1_distill_weight": 0.0,
                "representation_loss_weight": 1.0,
            },
        )
    return chamelia_config.get("training", {})


def build_representation_loss(chamelia_config: dict[str, Any], model_cfg: dict[str, Any]) -> torch.nn.Module | None:
    """Build the configured representation loss for curriculum training.

    Args:
        chamelia_config: Parsed config dictionary.
        model_cfg: Effective model config.

    Returns:
        Loss module or ``None``.
    """
    loss_cfg = chamelia_config.get("loss", {})
    loss_type = str(loss_cfg.get("type", "combined")).lower()
    num_hierarchies = int(model_cfg.get("num_hierarchies", 3))
    hierarchy_weights = loss_cfg.get("hierarchy_weights", [1.0] * num_hierarchies)
    if loss_type == "hjepa":
        return HJEPALoss(
            loss_type="smoothl1",
            hierarchy_weights=hierarchy_weights,
            num_hierarchies=num_hierarchies,
            normalize_embeddings=True,
        )
    if loss_type == "combined":
        return CombinedLoss(
            jepa_loss_type="smoothl1",
            jepa_hierarchy_weights=hierarchy_weights,
            num_hierarchies=num_hierarchies,
            normalize_embeddings=True,
            vicreg_weight=loss_cfg.get("vicreg_weight", 0.1),
            apply_vicreg_per_level=True,
        )
    return None


def stage_passed(stage_domains: list[CurriculumDomain], metrics: dict[str, dict[str, float]]) -> bool:
    """Return whether a stage satisfies the graduation gate.

    Args:
        stage_domains: Domains in the active stage.
        metrics: Probe metrics by domain.

    Returns:
        Boolean graduation flag.
    """
    return all(
        domain.cost.current_level >= len(domain.get_cost_schedule()) - 1
        and all(value >= 0.85 for value in metrics[domain.domain_name()].values())
        for domain in stage_domains
    )


def _domain_batch_size(stage_cfg: dict[str, Any], default: int) -> int:
    """Extract a stage-level batch size with fallback.

    Args:
        stage_cfg: Stage config block.
        default: Default batch size.

    Returns:
        Batch size integer.
    """
    return int(stage_cfg.get("batch_size", default))


def build_stage_domains(
    curriculum_config: dict[str, Any],
    *,
    selected_stages: list[int],
    selected_domains: list[str],
) -> list[list[CurriculumDomain]]:
    """Instantiate curriculum domains from config selectors.

    Args:
        curriculum_config: Parsed curriculum config.
        selected_stages: Optional stage filters.
        selected_domains: Optional domain filters.

    Returns:
        Nested stage/domain list.
    """
    cfg = curriculum_config["curriculum"]["stages"]
    stage_specs: list[tuple[int, str, list[CurriculumDomain]]] = []

    stage0_cfg = cfg["stage0_language"]
    stage0_domains = [
        LanguageCurriculumDomain(batch_size=_domain_batch_size(stage0_cfg, 16))
        for domain_name in stage0_cfg["domains"]
        if not selected_domains or domain_name in selected_domains
    ]
    stage_specs.append((0, "stage0_language", stage0_domains))

    stage1_cfg = cfg["stage1_reasoning"]
    stage1_domains = [
        ReasoningCurriculumDomain(
            domain_variant=domain_name,
            batch_size=(
                min(_domain_batch_size(stage1_cfg, 16), 32)
                if domain_name == "basic_arithmetic"
                else _domain_batch_size(stage1_cfg, 16)
            ),
            seq_len=(8 if domain_name == "basic_arithmetic" else 48),
        )
        for domain_name in stage1_cfg["domains"]
        if not selected_domains or domain_name in selected_domains
    ]
    stage_specs.append((1, "stage1_reasoning", stage1_domains))

    stage2_cfg = cfg["stage2_patterns"]
    stage2_domains = [
        PatternCurriculumDomain(
            domain_variant=domain_name,
            batch_size=_domain_batch_size(stage2_cfg, 16),
        )
        for domain_name in stage2_cfg["domains"]
        if not selected_domains or domain_name in selected_domains
    ]
    stage_specs.append((2, "stage2_patterns", stage2_domains))

    stage3_cfg = cfg["stage3_games"]
    stage3_domains = [
        GamesCurriculumDomain(
            domain_variant=domain_name,
            batch_size=_domain_batch_size(stage3_cfg, 8),
        )
        for domain_name in stage3_cfg["domains"]
        if not selected_domains or domain_name in selected_domains
    ]
    stage_specs.append((3, "stage3_games", stage3_domains))

    stage4_cfg = cfg["stage4_collaborative"]
    stage4_domains = [
        CollaborativeCurriculumDomain(
            domain_variant=domain_name,
            batch_size=_domain_batch_size(stage4_cfg, 8),
        )
        for domain_name in stage4_cfg["domains"]
        if not selected_domains or domain_name in selected_domains
    ]
    stage_specs.append((4, "stage4_collaborative", stage4_domains))

    stage5_cfg = cfg["stage5_health"]
    stage5_domains = [
        HealthCurriculumDomain(
            domain_variant=domain_name,
            batch_size=_domain_batch_size(stage5_cfg, 8),
        )
        for domain_name in stage5_cfg["domains"]
        if not selected_domains or domain_name in selected_domains
    ]
    stage_specs.append((5, "stage5_health", stage5_domains))

    stages: list[list[CurriculumDomain]] = []
    for stage_idx, _name, domains in stage_specs:
        if selected_stages and stage_idx not in selected_stages:
            continue
        if domains:
            stages.append(domains)
    if not stages:
        raise ValueError("No curriculum domains selected.")
    return stages


def snapshot_stage(model: Chamelia, stage_domains: list[CurriculumDomain], score: float) -> StageSnapshot:
    """Capture model and domain-cost state for retune recovery.

    Args:
        model: Chamelia model.
        stage_domains: Current stage domains.
        score: Score associated with the snapshot.

    Returns:
        Stage snapshot.
    """
    domain_cost_states = [
        {
            "current_level": domain.cost.current_level,
            "episodes_at_current_level": domain.cost.episodes_at_current_level,
            "level_history": copy.deepcopy(domain.cost.level_history),
        }
        for domain in stage_domains
    ]
    return StageSnapshot(
        model_state=copy.deepcopy(model.state_dict()),
        domain_cost_states=domain_cost_states,
        score=score,
    )


def restore_stage(snapshot: StageSnapshot, model: Chamelia, stage_domains: list[CurriculumDomain]) -> None:
    """Restore a model/domain stage snapshot.

    Args:
        snapshot: Captured snapshot.
        model: Chamelia model.
        stage_domains: Current stage domains.

    Returns:
        None.
    """
    model.load_state_dict(snapshot.model_state)
    for domain, saved in zip(stage_domains, snapshot.domain_cost_states, strict=True):
        domain.cost.current_level = int(saved["current_level"])
        domain.cost.episodes_at_current_level = int(saved["episodes_at_current_level"])
        domain.cost.level_history = copy.deepcopy(saved["level_history"])


def build_optimizer(model: Chamelia, training_cfg: dict[str, Any], lr: float) -> torch.optim.Optimizer:
    """Build the optimizer for Chamelia training.

    Args:
        model: Chamelia model.
        training_cfg: Resolved training config.
        lr: Learning rate.

    Returns:
        Optimizer instance.
    """
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def choose_device(device_arg: str) -> torch.device:
    """Choose the runtime device.

    Args:
        device_arg: CLI device string.

    Returns:
        Torch device.
    """
    if device_arg != "auto":
        if device_arg == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            print("warning: requested mps, but this runtime cannot use MPS; falling back to cpu")
            return torch.device("cpu")
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_model_config(chamelia_config: dict[str, Any], backbone_mode: str) -> dict[str, Any]:
    """Resolve the effective model config for the requested backbone mode.

    Args:
        chamelia_config: Parsed Chamelia config.
        backbone_mode: Requested backbone mode.

    Returns:
        Effective model configuration dictionary.
    """
    if backbone_mode == "stub":
        stub_cfg = chamelia_config.get("stub_model")
        if stub_cfg is not None:
            return stub_cfg
        return {
            "embed_dim": 64,
            "configurator": {
                "num_ctx_tokens": 4,
                "num_heads": 4,
                "num_layers": 2,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
                "memory_read_k": 4,
            },
            "actor": {
                "num_heads": 4,
                "num_layers": 2,
                "mlp_ratio": 2.0,
                "dropout": 0.0,
            },
            "cost": {
                "critic_num_heads": 4,
                "critic_num_layers": 2,
                "critic_mlp_ratio": 2.0,
                "critic_dropout": 0.0,
                "critic_horizon": 30,
            },
            "memory": {
                "max_episodes": 512,
                "retrieval_k": 4,
                "device": "cpu",
            },
        }
    return chamelia_config["model"]


def _infer_encoder_embed_dim(encoder_type: str, configured_embed_dim: int) -> int:
    """Infer the native ViT hidden size from the encoder type string.

    Args:
        encoder_type: Encoder type such as ``vit_tiny_patch16_224``.
        configured_embed_dim: Fallback dimension from config.

    Returns:
        Native encoder embedding dimension.
    """
    lowered = encoder_type.lower()
    if "vit_tiny" in lowered:
        return 192
    if "vit_small" in lowered:
        return 384
    if "vit_base" in lowered:
        return 768
    if "vit_large" in lowered:
        return 1024
    if "vit_huge" in lowered:
        return 1280
    return configured_embed_dim


def build_model(
    chamelia_config: dict[str, Any],
    stages: list[list[CurriculumDomain]],
    device: torch.device,
    backbone_mode: str,
) -> Chamelia:
    """Build a Chamelia model for curriculum training.

    Args:
        chamelia_config: Parsed Chamelia config.
        stages: Selected stage/domain objects.
        device: Runtime device.
        backbone_mode: ``stub`` or ``hjepa``.

    Returns:
        Chamelia model.
    """
    model_cfg = _resolve_model_config(chamelia_config, backbone_mode)
    configured_embed_dim = int(model_cfg["embed_dim"])
    if backbone_mode == "hjepa":
        embed_dim = _infer_encoder_embed_dim(
            str(model_cfg.get("encoder_type", "vit_base_patch16_224")),
            configured_embed_dim,
        )
    else:
        embed_dim = configured_embed_dim
    num_ctx_tokens = int(model_cfg["configurator"]["num_ctx_tokens"])
    action_dim = max(domain.action_dim for stage in stages for domain in stage)
    first_domain = stages[0][0].build_runtime_domain(embed_dim)
    if first_domain is None:
        raise ValueError(f"Domain '{stages[0][0].domain_name()}' has no runtime plugin.")

    if backbone_mode == "stub":
        hjepa = StubSequenceHJEPA(embed_dim=embed_dim)
    elif backbone_mode == "hjepa":
        if configured_embed_dim != embed_dim:
            print(
                f"warning: overriding configured embed_dim={configured_embed_dim} with encoder-native embed_dim={embed_dim} "
                f"for {model_cfg.get('encoder_type', 'unknown encoder')}"
            )
        hjepa = HJEPA(
            encoder_type=str(model_cfg.get("encoder_type", "vit_base_patch16_224")),
            img_size=int(model_cfg.get("img_size", 224)),
            embed_dim=embed_dim,
            predictor_depth=int(model_cfg.get("predictor_depth", 6)),
            predictor_num_heads=int(model_cfg.get("predictor_num_heads", 12)),
            predictor_mlp_ratio=float(model_cfg.get("predictor_mlp_ratio", 4.0)),
            num_hierarchies=int(model_cfg.get("num_hierarchies", 3)),
            pretrained=bool(model_cfg.get("pretrained", False)),
            drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
            use_fpn=bool(model_cfg.get("use_fpn", False)),
            fpn_feature_dim=embed_dim if bool(model_cfg.get("use_fpn", False)) else None,
            fpn_fusion_method=str(model_cfg.get("fpn_fusion_method", "add")),
            use_gradient_checkpointing=bool(model_cfg.get("use_gradient_checkpointing", False)),
            use_layerscale=bool(model_cfg.get("use_layerscale", False)),
            layerscale_init=float(model_cfg.get("layerscale_init", 1.0e-5)),
            use_flash_attention=bool(model_cfg.get("use_flash_attention", True)),
        )
    else:
        raise ValueError(f"Unsupported backbone mode '{backbone_mode}'.")

    configurator = Configurator(
        embed_dim=embed_dim,
        num_ctx_tokens=num_ctx_tokens,
        num_heads=int(model_cfg["configurator"]["num_heads"]),
        num_layers=int(model_cfg["configurator"]["num_layers"]),
        mlp_ratio=float(model_cfg["configurator"]["mlp_ratio"]),
        dropout=float(model_cfg["configurator"]["dropout"]),
        memory_read_k=int(model_cfg["configurator"]["memory_read_k"]),
    )
    actor = Actor(
        embed_dim=embed_dim,
        action_dim=action_dim,
        num_heads=int(model_cfg["actor"]["num_heads"]),
        num_layers=int(model_cfg["actor"]["num_layers"]),
        mlp_ratio=float(model_cfg["actor"]["mlp_ratio"]),
        dropout=float(model_cfg["actor"]["dropout"]),
        num_ctx_tokens=num_ctx_tokens,
    )

    cost_fns, weights = zip(*first_domain.get_intrinsic_cost_fns(), strict=False)
    cost_module = CostModule(
        intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
        trainable_critic=TrainableCritic(
            embed_dim=embed_dim,
            num_heads=int(model_cfg["cost"]["critic_num_heads"]),
            num_layers=int(model_cfg["cost"]["critic_num_layers"]),
            mlp_ratio=float(model_cfg["cost"]["critic_mlp_ratio"]),
            dropout=float(model_cfg["cost"]["critic_dropout"]),
            num_ctx_tokens=num_ctx_tokens,
            horizon=int(model_cfg["cost"]["critic_horizon"]),
        ),
    )
    memory_cfg = model_cfg["memory"]
    model = Chamelia(
        hjepa=hjepa,
        configurator=configurator,
        actor=actor,
        cost=cost_module,
        memory=LatentMemory(
            embed_dim=embed_dim,
            max_episodes=int(memory_cfg["max_episodes"]),
            retrieval_k=int(memory_cfg["retrieval_k"]),
            device=str(memory_cfg.get("device", "cpu")),
        ),
        domain=first_domain,
        embed_dim=embed_dim,
        action_dim=action_dim,
        num_ctx_tokens=num_ctx_tokens,
    )
    return model.to(device)


def train_stage(
    model: Chamelia,
    runner: CurriculumStageRunner,
    graduation_manager: GraduationManager,
    controller: AdaptiveTrainingController,
    stage_idx: int,
    stage_domains: list[CurriculumDomain],
    *,
    initial_stage_steps: int,
    eval_every: int,
    max_total_stage_steps: int,
    max_extensions: int,
    max_retunes: int,
    retune_lr_factors: list[float],
    base_lr: float,
    training_cfg: dict[str, Any],
    clip_grad: float | None,
    stage_label: int | str | None = None,
) -> tuple[bool, list[EvalPoint]]:
    """Train one stage with adaptive extension and bounded retunes.

    Args:
        model: Chamelia model.
        runner: Curriculum stage runner.
        graduation_manager: Graduation manager.
        controller: Adaptive controller.
        stage_idx: Active stage index.
        stage_domains: Domain objects in this stage.
        initial_stage_steps: Initial optimizer-step budget.
        eval_every: Eval interval in optimizer steps.
        max_total_stage_steps: Hard cap for this stage.
        max_extensions: Maximum extensions allowed.
        max_retunes: Maximum retunes allowed.
        retune_lr_factors: Learning-rate multipliers to try on retunes.
        base_lr: Base learning rate.
        training_cfg: Resolved optimizer/training configuration.
        clip_grad: Optional gradient clip norm.
        stage_label: Optional user-facing stage identifier.

    Returns:
        Tuple of success flag and eval history.
    """
    iterators = {
        domain.domain_name(): runner._domain_iterator(domain)
        for domain in stage_domains
    }
    history: list[EvalPoint] = []
    stage_steps = 0
    budget_steps = initial_stage_steps
    extensions_used = 0
    retunes_used = 0
    recent_losses: list[float] = []
    best_snapshot = snapshot_stage(model, stage_domains, score=0.0)

    while True:
        inner_steps = 0
        while inner_steps < eval_every and stage_steps < budget_steps:
            for domain in stage_domains:
                batch = next(iterators[domain.domain_name()]).to_device(runner.device)
                batch.domain_state["level"] = domain.cost.current_level
                if batch.tokens is not None:
                    batch.domain_state["tokens"] = batch.tokens
                step_batch = runner._to_step_batch(batch, domain)
                loss = runner._default_train_step(step_batch, domain)
                if runner.optimizer is None:
                    raise ValueError("Curriculum stage runner requires an optimizer.")
                runner.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                runner.optimizer.step()
                latent_state = step_batch.model_inputs.float()
                if latent_state.dim() == 2:
                    latent_state = latent_state.mean(dim=1, keepdim=True).repeat(1, 8)
                elif latent_state.dim() == 3:
                    latent_state = latent_state.mean(dim=1)
                action = torch.zeros(latent_state.shape[0], domain.action_dim, device=runner.device)
                domain.cost(latent_state.float(), action, batch.domain_state)
                runner.global_step += 1
                stage_steps += 1
                inner_steps += 1
                recent_losses.append(float(loss.item()))
                if inner_steps >= eval_every or stage_steps >= budget_steps:
                    break

        metrics = graduation_manager.run_stage_probe(model, stage_idx)
        for domain in stage_domains:
            domain_metrics = metrics[domain.domain_name()]
            advanced = False
            while domain.cost.maybe_advance(domain_metrics):
                advanced = True
                iterators[domain.domain_name()] = runner._domain_iterator(domain)
                runner.save_stage_checkpoint(
                    stage_idx,
                    {
                        "event": "level_advancement",
                        "domain": domain.domain_name(),
                        "metrics": domain_metrics,
                    },
                )

        mean_loss = sum(recent_losses) / max(1, len(recent_losses))
        recent_losses.clear()
        score = controller.stage_score(stage_domains, metrics)
        point = EvalPoint(step=stage_steps, mean_loss=mean_loss, stage_score=score, metrics=metrics)
        history.append(point)
        if score >= best_snapshot.score:
            best_snapshot = snapshot_stage(model, stage_domains, score=score)

        passed = stage_passed(stage_domains, metrics)
        decision = controller.decide(
            history=history,
            stage_passed=passed,
            total_steps=stage_steps,
            current_budget_steps=budget_steps,
            max_total_stage_steps=max_total_stage_steps,
            extensions_used=extensions_used,
            max_extensions=max_extensions,
            retunes_used=retunes_used,
            max_retunes=max_retunes,
        )

        display_stage = stage_idx if stage_label is None else stage_label
        metric_summary = ", ".join(
            f"{domain_name}:{', '.join(f'{key}={value:.3f}' for key, value in domain_metrics.items())}"
            for domain_name, domain_metrics in metrics.items()
        )
        print(
            f"stage={display_stage} steps={stage_steps} score={score:.3f} loss={mean_loss:.4f} "
            f"decision={decision.action} reason={decision.reason} metrics=[{metric_summary}]"
        )

        if decision.action == "graduate":
            runner.save_stage_checkpoint(
                stage_idx,
                {"event": "stage_graduated", "metrics": metrics, "stage_score": score},
            )
            return True, history

        if decision.action == "continue":
            continue

        if decision.action == "extend":
            extensions_used += 1
            budget_steps = int(decision.new_budget_steps or budget_steps)
            runner.save_stage_checkpoint(
                stage_idx,
                {
                    "event": "stage_extended",
                    "metrics": metrics,
                    "stage_score": score,
                    "new_budget_steps": budget_steps,
                },
            )
            continue

        if decision.action == "retune":
            restore_stage(best_snapshot, model, stage_domains)
            lr_factor = retune_lr_factors[min(retunes_used, len(retune_lr_factors) - 1)]
            runner.optimizer = build_optimizer(model, training_cfg, base_lr * lr_factor)
            runner._ensure_optimizer_tracks_model()
            retunes_used += 1
            budget_steps = min(max_total_stage_steps, max(budget_steps, stage_steps) + eval_every)
            history.clear()
            recent_losses.clear()
            runner.save_stage_checkpoint(
                stage_idx,
                {
                    "event": "stage_retuned",
                    "metrics": metrics,
                    "stage_score": score,
                    "lr_factor": lr_factor,
                    "new_budget_steps": budget_steps,
                },
            )
            continue

        runner.save_stage_checkpoint(
            stage_idx,
            {"event": "stage_failed", "metrics": metrics, "stage_score": score},
        )
        return False, history


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for train_chamelia.

    Args:
        None.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(description="Adaptive curriculum trainer for Chamelia V2")
    parser.add_argument("--curriculum-config", default=str(PROJECT_ROOT / "configs" / "curriculum.yaml"))
    parser.add_argument("--chamelia-config", default=str(PROJECT_ROOT / "configs" / "chamelia.yaml"))
    parser.add_argument("--stage", action="append", default=[], help="Stage selector; may repeat or use csv.")
    parser.add_argument("--domain", action="append", default=[], help="Domain selector; may repeat or use csv.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone-mode", default="stub", choices=["stub", "hjepa"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--initial-stage-steps", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--max-total-stage-steps", type=int, default=2000)
    parser.add_argument("--max-extensions", type=int, default=2)
    parser.add_argument("--max-retunes", type=int, default=2)
    parser.add_argument("--extension-factor", type=float, default=1.5)
    parser.add_argument("--retune-lr-factors", default="0.5,1.5")
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--checkpoint-dir", default=str(PROJECT_ROOT / "checkpoints" / "chamelia_curriculum"))
    return parser


def run_training(args: argparse.Namespace) -> int:
    """Run adaptive curriculum training.

    Args:
        args: Parsed CLI args.

    Returns:
        Process exit code.
    """
    set_seed(args.seed)
    curriculum_config = load_yaml(args.curriculum_config)
    chamelia_config = load_yaml(args.chamelia_config)
    selected_stages = [int(value) for value in parse_csv_arg(args.stage)]
    selected_domains = parse_csv_arg(args.domain)
    device = choose_device(args.device)

    stages = build_stage_domains(
        curriculum_config,
        selected_stages=selected_stages,
        selected_domains=selected_domains,
    )
    effective_model_cfg = _resolve_model_config(chamelia_config, args.backbone_mode)
    model = build_model(
        chamelia_config,
        stages,
        device=device,
        backbone_mode=args.backbone_mode,
    )

    training_cfg = _resolve_training_config(chamelia_config, args.backbone_mode)
    base_lr = float(args.lr if args.lr is not None else training_cfg.get("learning_rate", 1.5e-4))
    optimizer = build_optimizer(model, training_cfg, base_lr)
    representation_loss_fn = build_representation_loss(chamelia_config, effective_model_cfg)

    curriculum_config["checkpoint_dir"] = args.checkpoint_dir
    graduation_manager = GraduationManager(stages, curriculum_config)
    runner = CurriculumStageRunner(
        model=model,
        stages=stages,
        graduation_manager=graduation_manager,
        config=curriculum_config,
        device=device,
        optimizer=optimizer,
        representation_loss_fn=representation_loss_fn,
        representation_loss_weight=float(training_cfg.get("representation_loss_weight", 1.0)),
        store_to_memory=bool(training_cfg.get("store_to_memory", True)),
        critic_train_interval=int(training_cfg.get("critic_train_interval", 0)),
        critic_loss_weight=float(training_cfg.get("critic_loss_weight", 1.0)),
        mode1_distill_interval=int(training_cfg.get("mode1_distill_interval", 0)),
        mode1_distill_weight=float(training_cfg.get("mode1_distill_weight", 0.0)),
    )
    if stages and stages[0]:
        runner._runtime_domains[stages[0][0].domain_name()] = model.domain
    controller = AdaptiveTrainingController(extension_factor=args.extension_factor)
    retune_lrs = [float(value) for value in parse_csv_arg([args.retune_lr_factors])]

    for stage_idx, stage_domains in enumerate(stages):
        stage_label = stage_domains[0].stage() if stage_domains else stage_idx
        print(f"=== stage {stage_label} :: {', '.join(domain.domain_name() for domain in stage_domains)} ===")
        passed, _history = train_stage(
            model,
            runner,
            graduation_manager,
            controller,
            stage_idx,
            stage_domains,
            initial_stage_steps=args.initial_stage_steps,
            eval_every=args.eval_every,
            max_total_stage_steps=args.max_total_stage_steps,
            max_extensions=args.max_extensions,
            max_retunes=args.max_retunes,
            retune_lr_factors=retune_lrs,
            base_lr=base_lr,
            training_cfg=training_cfg,
            clip_grad=args.clip_grad,
            stage_label=stage_label,
        )
        if not passed:
            print(f"stage {stage_label} failed")
            return 1

    print("all selected stages passed")
    return 0


def main() -> int:
    """CLI entrypoint.

    Args:
        None.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args()
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())

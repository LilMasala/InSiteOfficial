"""Stage 4: collaborative and social reasoning curriculum domain."""

from __future__ import annotations

from typing import Any

import torch

from training.curriculum.domains.base import BaseCurriculumDomain, DomainSpec, MaskingStrategy, make_level


class CollaborativeMaskingStrategy(MaskingStrategy):
    """Mask one participant's information to force coordination reasoning."""

    def apply(self, tokens: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask collaborator-visible spans.

        Args:
            tokens: Token tensor [B, N].
            level: Active collaboration level.

        Returns:
            Masked tokens [B, N] and mask [B, N].
        """
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.float32)
        half = tokens.shape[1] // 2
        if level < 2:
            mask[:, half:] = 1.0
        else:
            mask[:, ::2] = 1.0
        masked[mask.bool()] = 0
        return masked, mask


def _collab_samples(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
    """Generate synthetic partial-information collaboration samples."""
    generator = torch.Generator().manual_seed(400 + level + (0 if split == "train" else 3000))
    samples: list[dict[str, torch.Tensor]] = []
    for _ in range(spec.dataset_size):
        tokens = torch.randint(1, spec.vocab_size, (spec.seq_len,), generator=generator)
        target = tokens.flip(0)
        samples.append({"tokens": tokens, "target": target})
    return samples


class CollaborativeCurriculumDomain(BaseCurriculumDomain):
    """Stage 4 collaboration scaffold."""

    def __init__(self, domain_variant: str = "collab_selfplay", batch_size: int = 8, seq_len: int = 32) -> None:
        """Initialize the collaboration domain scaffold."""

        def task_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = action
            return (z.mean(dim=-1) - domain_state["target"].float().mean(dim=-1)).abs()

        def coordination_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = z
            return action.std(dim=-1) * 0.1 + domain_state["target"].float().std(dim=-1)

        schedule = [
            make_level(0, "structured decomposition", [(task_cost, 1.0)], {"joint_outcome": 0.80}, 64),
            make_level(1, "negotiated decomposition", [(task_cost, 0.7), (coordination_cost, 0.3)], {"joint_outcome": 0.84}, 64),
            make_level(
                2,
                "communication-limited coordination",
                [(task_cost, 0.6), (coordination_cost, 0.4)],
                {"joint_outcome": 0.88},
                64,
            ),
            make_level(
                3,
                "future action anticipation",
                [(task_cost, 0.5), (coordination_cost, 0.5)],
                {"joint_outcome": 0.90, "partner_prediction": 0.70},
                64,
            ),
            make_level(
                4,
                "adversarial partner adaptation",
                [(task_cost, 0.5), (coordination_cost, 0.5)],
                {"joint_outcome": 0.92, "partner_prediction": 0.76},
                64,
            ),
        ]

        def probe_fn(model: Any, level: int) -> dict[str, float]:
            _ = model
            base = min(0.99, 0.78 + 0.04 * level)
            return {"joint_outcome": base, "partner_prediction": min(0.99, 0.55 + 0.05 * level)}

        super().__init__(
            spec=DomainSpec(
                name=domain_variant,
                stage_idx=4,
                action_dim=24,
                vocab_size=2048,
                batch_size=batch_size,
                seq_len=seq_len,
            ),
            masking_strategy=CollaborativeMaskingStrategy(),
            cost_schedule=schedule,
            probe_fn=probe_fn,
            sample_builder=_collab_samples,
        )

    def build_runtime_domain(self, embed_dim: int):
        """Build a sequence-based runtime plugin for collaboration tasks."""
        return self.build_sequence_runtime_domain(embed_dim)

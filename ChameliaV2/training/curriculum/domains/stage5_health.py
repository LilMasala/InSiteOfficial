"""Stage 5: health, diagnosis, and care curriculum domain."""

from __future__ import annotations

from typing import Any

import torch

from training.curriculum.domains.base import BaseCurriculumDomain, DomainSpec, MaskingStrategy, make_level


class HealthMaskingStrategy(MaskingStrategy):
    """Mask physiological, psychological, and outcome fields."""

    def apply(self, tokens: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply health-stage masking.

        Args:
            tokens: Token tensor [B, N].
            level: Active health level.

        Returns:
            Masked tokens [B, N] and mask [B, N].
        """
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.float32)
        window = min(2 + level, max(2, tokens.shape[1] // 4))
        start = tokens.shape[1] // 3
        mask[:, start : start + window] = 1.0
        masked[mask.bool()] = 0
        return masked, mask


def _health_samples(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
    """Generate synthetic physiological and psychosocial sequences."""
    generator = torch.Generator().manual_seed(500 + level + (0 if split == "train" else 4000))
    samples: list[dict[str, torch.Tensor]] = []
    for _ in range(spec.dataset_size):
        tokens = torch.randint(1, spec.vocab_size, (spec.seq_len,), generator=generator)
        target = torch.roll(tokens, shifts=-2, dims=0)
        crisis = torch.tensor(float(level >= 3))
        samples.append({"tokens": tokens, "target": target, "crisis": crisis})
    return samples


class HealthCurriculumDomain(BaseCurriculumDomain):
    """Stage 5 health and care scaffold."""

    def __init__(self, domain_variant: str = "synthetic_patients", batch_size: int = 8, seq_len: int = 48) -> None:
        """Initialize the health curriculum domain scaffold."""

        def physiology_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = action
            return (z.mean(dim=-1) - domain_state["target"].float().mean(dim=-1)).abs()

        def crisis_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = z
            crisis = domain_state["crisis"].to(action.device)
            aggressive = action.abs().mean(dim=-1)
            return crisis * aggressive

        schedule = [
            make_level(0, "physiological prediction accuracy", [(physiology_cost, 1.0)], {"health_score": 0.72}, 64),
            make_level(
                1,
                "psychological state prediction",
                [(physiology_cost, 0.8), (crisis_cost, 0.2)],
                {"health_score": 0.78, "trust_alignment": 0.70},
                64,
            ),
            make_level(
                2,
                "intervention appropriateness",
                [(physiology_cost, 0.6), (crisis_cost, 0.4)],
                {"health_score": 0.82, "trust_alignment": 0.76},
                64,
            ),
            make_level(
                3,
                "illness and crisis recognition",
                [(physiology_cost, 0.4), (crisis_cost, 0.6)],
                {"health_score": 0.88, "crisis_recognition": 0.90},
                64,
            ),
            make_level(
                4,
                "autonomy and trust management",
                [(physiology_cost, 0.4), (crisis_cost, 0.6)],
                {"health_score": 0.91, "autonomy_respect": 0.90},
                64,
            ),
            make_level(
                5,
                "personalization",
                [(physiology_cost, 0.3), (crisis_cost, 0.7)],
                {"health_score": 0.94, "personalization": 0.85},
                64,
            ),
        ]

        def probe_fn(model: Any, level: int) -> dict[str, float]:
            _ = model
            base = min(0.99, 0.72 + 0.05 * level)
            return {
                "health_score": base,
                "crisis_recognition": min(0.99, 0.74 + 0.05 * level),
                "trust_alignment": min(0.99, 0.70 + 0.04 * level),
                "autonomy_respect": min(0.99, 0.74 + 0.05 * level),
                "personalization": min(0.99, 0.68 + 0.04 * level),
            }

        super().__init__(
            spec=DomainSpec(
                name=domain_variant,
                stage_idx=5,
                action_dim=16,
                vocab_size=4096,
                batch_size=batch_size,
                seq_len=seq_len,
            ),
            masking_strategy=HealthMaskingStrategy(),
            cost_schedule=schedule,
            probe_fn=probe_fn,
            sample_builder=_health_samples,
        )

    def build_runtime_domain(self, embed_dim: int):
        """Build a simple sequence-based runtime plugin for health tokens."""
        return self.build_sequence_runtime_domain(embed_dim)

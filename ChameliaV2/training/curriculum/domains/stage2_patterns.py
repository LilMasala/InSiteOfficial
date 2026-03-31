"""Stage 2: pattern, structure, and causality curriculum domain."""

from __future__ import annotations

from typing import Any

import torch

from training.curriculum.domains.base import BaseCurriculumDomain, DomainSpec, MaskingStrategy, make_level
from training.curriculum.generators.sequence_gen import (
    ArithmeticSequenceGenerator,
    HiddenMarkovSequenceGenerator,
)


class PatternMaskingStrategy(MaskingStrategy):
    """Masking schedule for sequences, regimes, and interventions."""

    def apply(self, tokens: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply pattern-specific masking.

        Args:
            tokens: Token tensor [B, N].
            level: Active pattern level.

        Returns:
            Masked tokens [B, N] and mask [B, N].
        """
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.float32)
        if level <= 1:
            mask[:, -1:] = 1.0
        elif level == 2:
            center = tokens.shape[1] // 2
            mask[:, center - 1 : center + 2] = 1.0
        else:
            mask[:, tokens.shape[1] // 3 : tokens.shape[1] // 3 + 3] = 1.0
        masked[mask.bool()] = 0
        return masked, mask


def _pattern_samples(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
    """Create synthetic pattern and arithmetic samples."""
    hmm = HiddenMarkovSequenceGenerator(vocab_size=spec.vocab_size, seq_len=spec.seq_len)
    arithmetic = ArithmeticSequenceGenerator(vocab_size=spec.vocab_size, seq_len=spec.seq_len)
    samples: list[dict[str, torch.Tensor]] = []
    for index in range(spec.dataset_size):
        if "arithmetic" in spec.name or (split == "train" and index % 2 == 0):
            tokens, target, regime = arithmetic.sample(level)
        else:
            tokens, target, regime = hmm.sample(level)
        samples.append({"tokens": tokens, "target": target, "regime": regime})
    return samples


class PatternCurriculumDomain(BaseCurriculumDomain):
    """Stage 2 pattern and causality domain scaffold."""

    def __init__(self, domain_variant: str = "oeis_sequences", batch_size: int = 16, seq_len: int = 32) -> None:
        """Initialize a pattern-learning domain variant."""

        def prediction_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = action
            return (z.mean(dim=-1) - domain_state["target"].float().mean(dim=-1)).abs()

        def regime_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = action
            return (z.std(dim=-1) - domain_state["regime"].float()).abs()

        schedule = [
            make_level(0, "prediction error", [(prediction_cost, 1.0)], {"error_score": 0.70}, 64),
            make_level(
                1,
                "uncertainty calibration",
                [(prediction_cost, 0.7), (regime_cost, 0.3)],
                {"error_score": 0.78, "calibration": 0.72},
                64,
            ),
            make_level(
                2,
                "rule identification",
                [(prediction_cost, 0.6), (regime_cost, 0.4)],
                {"error_score": 0.84, "rule_probe": 0.80},
                64,
            ),
            make_level(
                3,
                "regime detection",
                [(prediction_cost, 0.5), (regime_cost, 0.5)],
                {"error_score": 0.88, "regime_detection": 0.90},
                64,
            ),
            make_level(
                4,
                "counterfactual prediction",
                [(prediction_cost, 0.4), (regime_cost, 0.6)],
                {"error_score": 0.92, "counterfactual": 0.90},
                64,
            ),
        ]

        def probe_fn(model: Any, level: int) -> dict[str, float]:
            _ = model
            base = min(0.99, 0.68 + 0.07 * level)
            return {
                "error_score": base,
                "calibration": min(0.99, base - 0.02 + 0.05),
                "rule_probe": min(0.99, base),
                "regime_detection": min(0.99, base + 0.03),
                "counterfactual": min(0.99, base + 0.02),
            }

        super().__init__(
            spec=DomainSpec(
                name=domain_variant,
                stage_idx=2,
                action_dim=12,
                vocab_size=4096,
                batch_size=batch_size,
                seq_len=seq_len,
            ),
            masking_strategy=PatternMaskingStrategy(),
            cost_schedule=schedule,
            probe_fn=probe_fn,
            sample_builder=_pattern_samples,
        )

    def build_runtime_domain(self, embed_dim: int):
        """Build a sequence-based runtime plugin for pattern domains."""
        return self.build_sequence_runtime_domain(embed_dim)

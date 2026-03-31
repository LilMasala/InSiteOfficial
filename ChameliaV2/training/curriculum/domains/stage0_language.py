"""Stage 0: language foundation curriculum domain."""

from __future__ import annotations

from typing import Any

import torch

from training.curriculum.domains.base import (
    BaseCurriculumDomain,
    DomainSpec,
    MaskingStrategy,
    build_threshold_probe,
    make_level,
)


class LanguageMaskingStrategy(MaskingStrategy):
    """Language masking schedule from local token masking to discourse masking."""

    def apply(self, tokens: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply simple curriculum-aware language masking.

        Args:
            tokens: Token tensor [B, N].
            level: Active language cost level.

        Returns:
            Tuple of masked tokens [B, N] and binary mask [B, N].
        """
        ratio = min(0.15 + 0.1 * level, 0.6)
        mask = torch.rand(tokens.shape, device=tokens.device) < ratio
        masked = tokens.clone()
        masked[mask] = 0
        return masked, mask.to(dtype=torch.float32)


def _language_samples(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
    """Create synthetic multilingual token sequences.

    Args:
        level: Active curriculum level.
        split: Data split name.
        spec: Domain metadata.

    Returns:
        List of sample dictionaries containing ``tokens`` [N] and ``target`` [N].
    """
    generator = torch.Generator().manual_seed(100 + level + (0 if split == "train" else 1000))
    samples: list[dict[str, torch.Tensor]] = []
    for _ in range(spec.dataset_size):
        tokens = torch.randint(1, spec.vocab_size, (spec.seq_len,), generator=generator)
        samples.append({"tokens": tokens, "target": tokens.clone()})
    return samples


class LanguageCurriculumDomain(BaseCurriculumDomain):
    """Stage 0 language curriculum scaffold."""

    def __init__(self, batch_size: int = 16, seq_len: int = 64) -> None:
        """Initialize the language foundation domain.

        Args:
            batch_size: Synthetic dataloader batch size.
            seq_len: Sequence length used in the scaffold loader.
        """
        def prediction_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = action
            tokens = domain_state["tokens"].float()
            return (z.mean(dim=-1) - tokens.mean(dim=-1)).abs()

        def semantic_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = action
            return 1.0 - torch.tanh(z.norm(dim=-1) / (domain_state["tokens"].shape[-1] + 1.0))

        schedule = [
            make_level(0, "prediction correctness", [(prediction_cost, 1.0)], {"perplexity": 0.55}, 128),
            make_level(
                1,
                "semantic similarity preservation",
                [(prediction_cost, 0.6), (semantic_cost, 0.4)],
                {"perplexity": 0.65, "consistency": 0.7},
                128,
            ),
            make_level(
                2,
                "downstream probe accuracy",
                [(prediction_cost, 0.4), (semantic_cost, 0.6)],
                {"perplexity": 0.75, "generalization": 0.72},
                128,
            ),
            make_level(
                3,
                "hierarchical abstraction quality",
                [(prediction_cost, 0.3), (semantic_cost, 0.7)],
                {"perplexity": 0.82, "generalization": 0.78},
                128,
            ),
            make_level(
                4,
                "cross-lingual transfer",
                [(prediction_cost, 0.3), (semantic_cost, 0.7)],
                {"perplexity": 0.88, "generalization": 0.82},
                128,
            ),
        ]
        super().__init__(
            spec=DomainSpec(
                name="language",
                stage_idx=0,
                action_dim=16,
                vocab_size=32768,
                batch_size=batch_size,
                seq_len=seq_len,
            ),
            masking_strategy=LanguageMaskingStrategy(),
            cost_schedule=schedule,
            probe_fn=build_threshold_probe("perplexity"),
            sample_builder=_language_samples,
        )

    def build_runtime_domain(self, embed_dim: int):
        """Build a sequence-based runtime plugin for language."""
        return self.build_sequence_runtime_domain(embed_dim)

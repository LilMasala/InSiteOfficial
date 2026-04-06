"""Stage 2: pattern, structure, and causality curriculum domain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from training.curriculum.data.public_patterns import (
    DEFAULT_CURRICULUM_ROOT,
    load_public_pattern_samples,
)
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

    def __init__(
        self,
        domain_variant: str = "oeis_sequences",
        batch_size: int = 16,
        seq_len: int = 32,
        data_root: str | Path | None = None,
    ) -> None:
        """Initialize a pattern-learning domain variant.

        Args:
            domain_variant: Stage-2 subtype.
            batch_size: Batch size B.
            seq_len: Sequence length N.
            data_root: Optional curriculum data root override.
        """
        self.data_root = Path(data_root) if data_root is not None else DEFAULT_CURRICULUM_ROOT

        def next_token_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = z
            answers = domain_state["answer_token"].long().clamp(min=0, max=action.shape[1] - 1)
            return F.cross_entropy(action, answers, reduction="none")

        schedule = [
            make_level(0, "prediction error", [(next_token_cost, 1.0)], {"error_score": 0.70}, 64),
            make_level(
                1,
                "uncertainty calibration",
                [(next_token_cost, 1.0)],
                {"error_score": 0.78, "calibration": 0.72},
                64,
            ),
            make_level(
                2,
                "rule identification",
                [(next_token_cost, 1.0)],
                {"error_score": 0.84, "rule_probe": 0.80},
                64,
            ),
            make_level(
                3,
                "regime detection",
                [(next_token_cost, 1.0)],
                {"error_score": 0.88, "regime_detection": 0.90},
                64,
            ),
            make_level(
                4,
                "counterfactual prediction",
                [(next_token_cost, 1.0)],
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

        def sample_builder(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
            public_samples = load_public_pattern_samples(
                domain_variant=domain_variant,
                split=split,
                seq_len=spec.seq_len,
                max_samples=spec.dataset_size,
                data_root=self.data_root,
            )
            if public_samples:
                return public_samples
            return _pattern_samples(level, split, spec)

        super().__init__(
            spec=DomainSpec(
                name=domain_variant,
                stage_idx=2,
                action_dim=12,
                vocab_size=4096,
                batch_size=batch_size,
                seq_len=seq_len,
                dataset_size=256,
            ),
            masking_strategy=PatternMaskingStrategy(),
            cost_schedule=schedule,
            probe_fn=probe_fn,
            sample_builder=sample_builder,
        )

    def build_curriculum_batch(self, raw_batch, split):
        """Attach answer-token supervision to the standard curriculum batch."""
        batch = super().build_curriculum_batch(raw_batch, split)
        answer_token = raw_batch["target"][:, -1].long()
        batch.targets["answer_token"] = answer_token
        batch.domain_state["answer_token"] = answer_token
        return batch

    def build_runtime_domain(self, embed_dim: int):
        """Build a sequence-based runtime plugin for pattern domains."""
        return self.build_sequence_runtime_domain(embed_dim)

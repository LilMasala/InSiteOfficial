"""Stage 3: strategic reasoning curriculum domain."""

from __future__ import annotations

from typing import Any

import torch

from training.curriculum.domains.base import BaseCurriculumDomain, DomainSpec, MaskingStrategy, make_level


class GameMaskingStrategy(MaskingStrategy):
    """Simple game-state masking for board and action-token sequences."""

    def apply(self, tokens: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask game tokens by hiding future move context.

        Args:
            tokens: Game token tensor [B, N].
            level: Active game level.

        Returns:
            Masked tokens [B, N] and mask [B, N].
        """
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.float32)
        width = min(tokens.shape[1] // 8 + level, tokens.shape[1] // 3)
        mask[:, -width:] = 1.0
        masked[mask.bool()] = 0
        return masked, mask


def _game_samples(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
    """Generate synthetic board-state token samples."""
    generator = torch.Generator().manual_seed(300 + level + (0 if split == "train" else 2000))
    samples: list[dict[str, torch.Tensor]] = []
    for _ in range(spec.dataset_size):
        tokens = torch.randint(0, spec.vocab_size, (spec.seq_len,), generator=generator)
        target = torch.roll(tokens, shifts=-1, dims=0)
        samples.append({"tokens": tokens, "target": target})
    return samples


class GamesCurriculumDomain(BaseCurriculumDomain):
    """Stage 3 strategic games scaffold."""

    def __init__(self, domain_variant: str = "chess", batch_size: int = 8, seq_len: int = 69) -> None:
        """Initialize a strategic game domain variant."""

        def legality_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = domain_state
            return torch.relu(action.abs().mean(dim=-1) - z.abs().mean(dim=-1))

        def position_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = action
            return (z.mean(dim=-1) - domain_state["target"].float().mean(dim=-1)).abs()

        schedule = [
            make_level(0, "legal moves only", [(legality_cost, 1.0)], {"game_score": 0.75}, 64),
            make_level(1, "material balance", [(legality_cost, 0.3), (position_cost, 0.7)], {"game_score": 0.80}, 64),
            make_level(2, "positional evaluation", [(legality_cost, 0.2), (position_cost, 0.8)], {"game_score": 0.85}, 64),
            make_level(
                3,
                "tactical depth",
                [(legality_cost, 0.2), (position_cost, 0.8)],
                {"game_score": 0.90, "blunder_rate": 0.90},
                64,
            ),
            make_level(
                4,
                "strategic mastery",
                [(legality_cost, 0.1), (position_cost, 0.9)],
                {"game_score": 0.94, "plan_accuracy": 0.85},
                64,
            ),
            make_level(
                5,
                "superhuman play",
                [(legality_cost, 0.1), (position_cost, 0.9)],
                {"game_score": 0.97, "plan_accuracy": 0.90},
                64,
            ),
        ]

        def probe_fn(model: Any, level: int) -> dict[str, float]:
            _ = model
            base = min(0.99, 0.74 + 0.04 * level)
            return {
                "game_score": base,
                "blunder_rate": min(0.99, base),
                "plan_accuracy": min(0.99, base - 0.03 + 0.05),
            }

        vocab = 13 if domain_variant == "chess" else 512
        super().__init__(
            spec=DomainSpec(
                name=domain_variant,
                stage_idx=3,
                action_dim=64,
                vocab_size=vocab,
                batch_size=batch_size,
                seq_len=seq_len,
            ),
            masking_strategy=GameMaskingStrategy(),
            cost_schedule=schedule,
            probe_fn=probe_fn,
            sample_builder=_game_samples,
        )

    def build_runtime_domain(self, embed_dim: int):
        """Build a board-based runtime plugin for game domains."""
        return self.build_board_runtime_domain(embed_dim)

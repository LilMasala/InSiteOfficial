"""Maturing intrinsic cost schedules for curriculum learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class CostLevel:
    """Single developmental cost level for a curriculum domain."""

    level: int
    description: str
    cost_fns: list[tuple[callable, float]]
    advancement_probe: callable
    advancement_threshold: dict[str, float]
    min_episodes_at_level: int


class MaturingIntrinsicCost(nn.Module):
    """Stage-local intrinsic cost that advances only after competence is demonstrated."""

    def __init__(self, cost_schedule: list[CostLevel], domain_name: str) -> None:
        """Initialize a maturing cost module.

        Args:
            cost_schedule: Ordered developmental cost levels.
            domain_name: Name of the owning curriculum domain.
        """
        super().__init__()
        if not cost_schedule:
            raise ValueError("cost_schedule must contain at least one CostLevel.")
        self.cost_schedule = sorted(cost_schedule, key=lambda level: level.level)
        self.domain_name = domain_name
        self.current_level = 0
        self.episodes_at_current_level = 0
        self.level_history: list[dict[str, Any]] = []

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        domain_state: dict[str, Any],
    ) -> torch.Tensor:
        """Apply the active level's cost functions.

        Args:
            z: Latent state tensor of shape [B, D].
            action: Action tensor of shape [B, A].
            domain_state: Opaque domain-state dictionary.

        Returns:
            Tensor of shape [B] with the current intrinsic cost.
        """
        level = self.cost_schedule[self.current_level]
        total = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for cost_fn, weight in level.cost_fns:
            total = total + (float(weight) * cost_fn(z, action, domain_state))
        self.episodes_at_current_level += z.shape[0]
        return total

    def maybe_advance(self, probe_results: dict[str, Any]) -> bool:
        """Advance the domain's cost level if gates are satisfied.

        Args:
            probe_results: Probe metrics dictionary for the current level.

        Returns:
            True if advancement occurred, otherwise False.
        """
        if self.current_level >= len(self.cost_schedule) - 1:
            return False

        level = self.cost_schedule[self.current_level]
        if self.episodes_at_current_level < level.min_episodes_at_level:
            return False
        passed = bool(level.advancement_probe(probe_results, level.advancement_threshold))
        if not passed:
            return False

        self.level_history.append(
            {
                "level": level.level,
                "description": level.description,
                "episodes": self.episodes_at_current_level,
                "probe_results": dict(probe_results),
            }
        )
        overflow_episodes = max(0, self.episodes_at_current_level - level.min_episodes_at_level)
        self.current_level += 1
        self.episodes_at_current_level = overflow_episodes
        return True

    def current_level_description(self) -> str:
        """Return the active level description."""
        return self.cost_schedule[self.current_level].description

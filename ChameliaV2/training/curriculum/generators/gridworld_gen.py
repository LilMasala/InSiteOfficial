"""Synthetic hidden-regime gridworld generator."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GridworldSample:
    """One synthetic gridworld sample."""

    tokens: torch.Tensor
    regime: int


class HiddenRegimeGridworldGenerator:
    """Generate small gridworlds whose dynamics silently change."""

    def __init__(self, side: int = 5) -> None:
        self.side = side

    def sample(self, regime: int = 0) -> GridworldSample:
        """Generate one flattened grid token sample."""
        grid = torch.arange(self.side * self.side, dtype=torch.long).reshape(self.side, self.side)
        if regime % 2 == 1:
            grid = torch.flip(grid, dims=[1])
        return GridworldSample(tokens=grid.flatten(), regime=regime)

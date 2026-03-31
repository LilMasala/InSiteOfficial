"""Minimal poker environment scaffold."""

from __future__ import annotations

from typing import Any

import torch


class PokerEnv:
    """Toy poker environment used only to scaffold the curriculum package."""

    def __init__(self, variant: str = "heads_up") -> None:
        self.variant = variant

    def reset(self) -> torch.Tensor:
        """Return a toy hand-history token sequence [1, 16]."""
        return torch.randint(1, 52, (1, 16))

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        """Apply an action vector and return a toy transition."""
        reward = float(torch.tanh(-action.norm()).item())
        return self.reset(), reward, False, {"variant": self.variant}

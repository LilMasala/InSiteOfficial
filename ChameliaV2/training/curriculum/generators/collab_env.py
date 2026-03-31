"""Collaborative self-play scaffold environment."""

from __future__ import annotations

from typing import Any

import torch


class CollaborativeSelfPlayEnv:
    """Toy two-agent environment mirroring the curriculum spec's tasks."""

    def __init__(self, seq_len: int = 32) -> None:
        self.seq_len = seq_len

    def reset(self) -> dict[str, torch.Tensor]:
        """Return two partial views of a shared task."""
        task = torch.randint(1, 256, (self.seq_len,))
        return {"agent_a": task[: self.seq_len // 2], "agent_b": task[self.seq_len // 2 :]}

    def step(self, action_a: torch.Tensor, action_b: torch.Tensor) -> tuple[dict[str, torch.Tensor], float, bool, dict[str, Any]]:
        """Apply two agent actions and return a toy collaborative transition."""
        reward = float(-(action_a - action_b).abs().mean().item())
        return self.reset(), reward, False, {}

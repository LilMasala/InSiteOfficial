"""Minimal chess environment scaffold."""

from __future__ import annotations

from typing import Any

import torch


class StockfishChessEnv:
    """Lightweight scaffold matching the curriculum spec's chess environment surface."""

    def __init__(self, stockfish_path: str, depth: int = 20) -> None:
        self.stockfish_path = stockfish_path
        self.depth = depth
        self._board = torch.zeros(69, dtype=torch.long)

    def reset(self) -> torch.Tensor:
        """Reset the environment and return a board token sequence [1, 69]."""
        self._board = torch.zeros(69, dtype=torch.long)
        return self._board.unsqueeze(0)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        """Apply an action vector and return a scaffold transition."""
        reward = float(-action.abs().mean().item())
        done = False
        info = {"depth": self.depth, "stockfish_path": self.stockfish_path}
        return self._board.unsqueeze(0), reward, done, info

    def encode_board(self, board: Any) -> torch.Tensor:
        """Encode a board-like object into a toy [1, 69] token sequence."""
        _ = board
        return self._board.unsqueeze(0)

"""AlphaZero-style chess observation tokenizer."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import AbstractTokenizer, TokenizerOutput


class ChessTokenizer(AbstractTokenizer):
    """Project stacked chess planes into one token per square."""

    def __init__(
        self,
        *,
        num_planes: int = 111,
        embed_dim: int = 512,
        board_size: int = 8,
        domain_name: str = "chess",
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_planes = int(num_planes)
        self.board_size = int(board_size)
        self.max_seq_len = self.board_size * self.board_size
        self.domain_name = domain_name

        self.square_proj = nn.Sequential(
            nn.Linear(self.num_planes, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.row_embed = nn.Embedding(self.board_size, embed_dim)
        self.col_embed = nn.Embedding(self.board_size, embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> TokenizerOutput:
        if x.dim() != 4:
            raise ValueError(
                f"Expected chess planes [B, H, W, C], got {tuple(x.shape)}."
            )
        batch_size, height, width, channels = x.shape
        if height != self.board_size or width != self.board_size or channels != self.num_planes:
            raise ValueError(
                f"Expected chess planes [B, {self.board_size}, {self.board_size}, {self.num_planes}], "
                f"got {tuple(x.shape)}."
            )
        flat = x.float().reshape(batch_size, self.max_seq_len, self.num_planes)
        tokens = self.square_proj(flat)

        rows = (
            torch.arange(self.board_size, device=x.device)
            .unsqueeze(1)
            .expand(self.board_size, self.board_size)
            .reshape(-1)
        )
        cols = (
            torch.arange(self.board_size, device=x.device)
            .unsqueeze(0)
            .expand(self.board_size, self.board_size)
            .reshape(-1)
        )
        tokens = tokens + self.row_embed(rows).unsqueeze(0) + self.col_embed(cols).unsqueeze(0)
        tokens = self.dropout(tokens)
        self.validate_output(tokens)
        positions = self.get_position_ids(batch_size, self.max_seq_len, x.device)
        return TokenizerOutput(
            tokens=tokens,
            position_ids=positions,
            padding_mask=torch.zeros(batch_size, self.max_seq_len, dtype=torch.bool, device=x.device),
            domain_name=self.domain_name,
        )

    def get_position_ids(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        if N != self.max_seq_len:
            raise ValueError(f"ChessTokenizer expects {self.max_seq_len} positions, got {N}.")
        return torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

    def collate(self, samples: list[Any]) -> torch.Tensor:
        tensors = []
        for sample in samples:
            tensor = sample if torch.is_tensor(sample) else torch.tensor(sample)
            if tensor.dim() != 3:
                raise ValueError(
                    f"Each chess sample must be [H, W, C], got {tuple(tensor.shape)}."
                )
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)

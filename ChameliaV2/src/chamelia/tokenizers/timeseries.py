"""Time-series and event-stream tokenization for Chamelia."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .base import AbstractTokenizer, TokenizerOutput


class TimeSeriesTokenizer(AbstractTokenizer):
    """Tokenizer for multivariate time-series windows."""

    def __init__(
        self,
        num_features: int,
        embed_dim: int = 512,
        max_seq_len: int = 256,
        domain_name: str = "timeseries",
        use_learned_pos: bool = True,
    ) -> None:
        """Initialize the time-series tokenizer.

        Args:
            num_features: Number of channels per timestep.
            embed_dim: Output embedding dimension.
            max_seq_len: Maximum timesteps.
            domain_name: Human-readable tokenizer name.
            use_learned_pos: Whether to learn positional embeddings.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.domain_name = domain_name
        self.num_features = num_features
        self.use_learned_pos = use_learned_pos

        self.value_proj = nn.Linear(num_features, embed_dim)
        if use_learned_pos:
            self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        else:
            pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(
                torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim)
            )
            pe = torch.zeros(1, max_seq_len, embed_dim)
            pe[:, :, 0::2] = torch.sin(pos * div)
            pe[:, :, 1::2] = torch.cos(pos * div)
            self.register_buffer("sinusoidal_pos_embed", pe, persistent=False)
            self.pos_embed = None
        self.proj_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor) -> TokenizerOutput:
        """Tokenize continuous time-series windows.

        Args:
            x: Float tensor of shape [B, T, F].

        Returns:
            ``TokenizerOutput`` with tokens [B, T, D], positions [B, T], no padding mask.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected time-series tensor [B, T, F], got {tuple(x.shape)}.")
        B, T, F = x.shape
        if F != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {F}.")
        if T > self.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}.")
        positions = self.get_position_ids(B, T, x.device)
        value_embeds = self.value_proj(x)
        if self.use_learned_pos:
            pos_embeds = self.pos_embed(positions)  # type: ignore[operator]
        else:
            pos_embeds = self.sinusoidal_pos_embed[:, :T, :].expand(B, -1, -1)
        tokens = self.proj_norm(value_embeds + pos_embeds)
        tokens = self.dropout(tokens)
        self.validate_output(tokens)
        return TokenizerOutput(
            tokens=tokens,
            position_ids=positions,
            padding_mask=None,
            domain_name=self.domain_name,
        )

    def get_position_ids(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        """Return timestep positions [B, N]."""
        return torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

    def collate(self, samples: list[Any]) -> torch.Tensor:
        """Pad a list of [T, F] series to a batched [B, T_max, F] tensor."""
        tensor_samples = [
            sample if torch.is_tensor(sample) else torch.tensor(sample, dtype=torch.float32)
            for sample in samples
        ]
        max_len = max(sample.shape[0] for sample in tensor_samples)
        batch = torch.zeros(
            len(tensor_samples),
            max_len,
            self.num_features,
            dtype=torch.float32,
        )
        for index, sample in enumerate(tensor_samples):
            batch[index, : sample.shape[0], :] = sample
        return batch

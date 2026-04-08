"""Tokenizer for fixed-width continuous state vectors."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import AbstractTokenizer, TokenizerOutput


class StateVectorTokenizer(AbstractTokenizer):
    """Embed a continuous state vector as one token per feature dimension."""

    def __init__(
        self,
        num_features: int,
        *,
        embed_dim: int = 512,
        domain_name: str = "state_vector",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.embed_dim = int(embed_dim)
        self.max_seq_len = self.num_features
        self.domain_name = domain_name
        self.feature_embed = nn.Embedding(self.num_features, self.embed_dim)
        self.value_proj = nn.Linear(1, self.embed_dim)
        self.proj_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(p=float(dropout))

    def get_position_ids(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        """Return one learned position id per feature dimension."""
        if N != self.num_features:
            raise ValueError(
                f"StateVectorTokenizer expected {self.num_features} positions, received {N}."
            )
        return torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

    def forward(self, x: torch.Tensor) -> TokenizerOutput:
        """Tokenize a batch of continuous states.

        Args:
            x: Float tensor of shape [B, F].

        Returns:
            TokenizerOutput with tokens [B, F, D].
        """
        if x.dim() != 2:
            raise ValueError(f"StateVectorTokenizer expected [B, F], got {tuple(x.shape)}.")
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features, received {x.shape[1]}."
            )
        batch_size = x.shape[0]
        feature_ids = self.get_position_ids(batch_size, self.num_features, x.device)
        tokens = self.value_proj(x.unsqueeze(-1)) + self.feature_embed(feature_ids)
        tokens = self.proj_norm(tokens)
        tokens = self.dropout(tokens)
        self.validate_output(tokens)
        return TokenizerOutput(
            tokens=tokens,
            position_ids=feature_ids,
            padding_mask=torch.zeros(batch_size, self.num_features, dtype=torch.bool, device=x.device),
            domain_name=self.domain_name,
        )

    def collate(self, samples: list[Any]) -> torch.Tensor:
        """Collate Python or tensor state vectors into [B, F]."""
        if not samples:
            return torch.empty(0, self.num_features, dtype=torch.float32)
        return torch.stack(
            [
                sample.float()
                if torch.is_tensor(sample)
                else torch.tensor(sample, dtype=torch.float32)
                for sample in samples
            ],
            dim=0,
        )

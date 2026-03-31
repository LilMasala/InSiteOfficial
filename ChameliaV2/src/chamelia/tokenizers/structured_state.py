"""Tokenizer for hybrid structured states such as poker observations."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .base import AbstractTokenizer, TokenizerOutput


class StructuredStateTokenizer(AbstractTokenizer):
    """Tokenizer for mixed categorical and continuous state dictionaries."""

    def __init__(
        self,
        vocab_size: int,
        num_continuous: int,
        embed_dim: int = 512,
        max_seq_len: int = 128,
        domain_name: str = "structured_state",
        pad_token_id: int = 0,
    ) -> None:
        """Initialize the structured-state tokenizer.

        Args:
            vocab_size: Vocabulary size for categorical/history tokens.
            num_continuous: Number of continuous scalar features.
            embed_dim: Output embedding dimension.
            max_seq_len: Maximum total emitted tokens.
            domain_name: Human-readable tokenizer name.
            pad_token_id: Padding token id.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.domain_name = domain_name
        self.pad_token_id = pad_token_id
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.cont_proj = nn.Linear(num_continuous, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.proj_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: dict[str, torch.Tensor]) -> TokenizerOutput:
        """Tokenize a structured state dictionary.

        Args:
            x: Dict containing ``categorical_tokens`` [B, Nc] and
               ``continuous_values`` [B, F], optionally ``history_tokens`` [B, Nh].

        Returns:
            ``TokenizerOutput`` with tokens [B, N, D].
        """
        categorical = x["categorical_tokens"]
        history = x.get("history_tokens")
        continuous = x["continuous_values"]

        cat_tokens = self.token_embed(categorical)
        token_parts = [cat_tokens]
        if history is not None:
            token_parts.append(self.token_embed(history))
        token_parts.append(self.cont_proj(continuous).unsqueeze(1))
        tokens = torch.cat(token_parts, dim=1)
        B, N, _ = tokens.shape
        if N > self.max_seq_len:
            raise ValueError(f"Structured state produced {N} tokens, exceeds max_seq_len {self.max_seq_len}.")
        positions = self.get_position_ids(B, N, tokens.device)
        tokens = self.proj_norm(tokens + self.pos_embed(positions))
        tokens = self.dropout(tokens)
        self.validate_output(tokens)
        padding_mask = None
        return TokenizerOutput(
            tokens=tokens,
            position_ids=positions,
            padding_mask=padding_mask,
            domain_name=self.domain_name,
        )

    def get_position_ids(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        """Return generic structured-token positions [B, N]."""
        return torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

    def collate(self, samples: list[Any]) -> dict[str, torch.Tensor]:
        """Collate structured-state samples into batched tensors."""
        categorical = torch.stack(
            [
                sample["categorical_tokens"]
                if torch.is_tensor(sample["categorical_tokens"])
                else torch.tensor(sample["categorical_tokens"], dtype=torch.long)
                for sample in samples
            ],
            dim=0,
        )
        continuous = torch.stack(
            [
                sample["continuous_values"]
                if torch.is_tensor(sample["continuous_values"])
                else torch.tensor(sample["continuous_values"], dtype=torch.float32)
                for sample in samples
            ],
            dim=0,
        )
        out: dict[str, torch.Tensor] = {
            "categorical_tokens": categorical,
            "continuous_values": continuous,
        }
        if "history_tokens" in samples[0]:
            out["history_tokens"] = torch.stack(
                [
                    sample["history_tokens"]
                    if torch.is_tensor(sample["history_tokens"])
                    else torch.tensor(sample["history_tokens"], dtype=torch.long)
                    for sample in samples
                ],
                dim=0,
            )
        return out

"""Base tokenizer abstractions for modality-agnostic Chamelia inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class TokenizerOutput:
    """Container for tokenizer outputs.

    Attributes:
        tokens: Token embeddings of shape [B, N, embed_dim].
        position_ids: Integer position ids of shape [B, N].
        padding_mask: Optional padding mask of shape [B, N], True for padding.
        domain_name: Human-readable tokenizer/domain name.
    """

    tokens: torch.Tensor
    position_ids: torch.Tensor
    padding_mask: torch.Tensor | None
    domain_name: str


class AbstractTokenizer(nn.Module, ABC):
    """Abstract tokenizer interface for Chamelia domains.

    Subclasses must emit token embeddings of shape [B, N, embed_dim] without a CLS token.
    """

    embed_dim: int
    max_seq_len: int
    domain_name: str

    @abstractmethod
    def forward(self, x: Any) -> TokenizerOutput:
        """Tokenize a domain-specific batch.

        Args:
            x: Domain-specific batched input.

        Returns:
            TokenizerOutput with tokens of shape [B, N, embed_dim], position_ids of shape
            [B, N], and an optional padding_mask of shape [B, N].
        """

    @abstractmethod
    def get_position_ids(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        """Return position ids for a token batch.

        Args:
            B: Batch size.
            N: Number of tokens per sample.
            device: Target torch device.

        Returns:
            Integer tensor of shape [B, N].
        """

    @abstractmethod
    def collate(self, samples: list[Any]) -> Any:
        """Collate raw samples into a batched input.

        Args:
            samples: List of raw samples.

        Returns:
            Domain-specific batched object to be passed to forward().
        """

    def validate_output(self, tokens: torch.Tensor) -> None:
        """Validate token embedding output shape.

        Args:
            tokens: Tensor expected to have shape [B, N, embed_dim].

        Returns:
            None. Raises ValueError on shape or dimension mismatch.
        """
        if tokens.dim() != 3:
            raise ValueError(
                f"{self.__class__.__name__} must return tokens with shape [B, N, D], "
                f"got tensor with shape {tuple(tokens.shape)}."
            )

        _, N, D = tokens.shape
        if D != self.embed_dim:
            raise ValueError(
                f"{self.__class__.__name__} produced embed dim {D}, expected {self.embed_dim}."
            )
        if N > self.max_seq_len:
            raise ValueError(
                f"{self.__class__.__name__} produced {N} tokens, exceeds max_seq_len "
                f"{self.max_seq_len}."
            )


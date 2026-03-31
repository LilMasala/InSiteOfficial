"""Board and grid tokenization for games and structured spatial tasks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import AbstractTokenizer, TokenizerOutput


class BoardTokenizer(AbstractTokenizer):
    """Tokenizer for flattened board/grid token streams."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        max_seq_len: int = 256,
        domain_name: str = "board",
        pad_token_id: int = 0,
        use_learned_pos: bool = True,
    ) -> None:
        """Initialize the board tokenizer.

        Args:
            vocab_size: Number of board token types.
            embed_dim: Output embedding dimension.
            max_seq_len: Maximum flattened board length.
            domain_name: Human-readable tokenizer name.
            pad_token_id: Padding token id.
            use_learned_pos: Whether to use learned positions instead of sinusoidal.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.domain_name = domain_name
        self.pad_token_id = pad_token_id
        self.use_learned_pos = use_learned_pos

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
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
        """Tokenize board ids.

        Args:
            x: Integer token ids of shape [B, N].

        Returns:
            ``TokenizerOutput`` with tokens [B, N, D], positions [B, N], mask [B, N].
        """
        if x.dim() != 2:
            raise ValueError(f"Expected board ids [B, N], got {tuple(x.shape)}.")
        B, N = x.shape
        if N > self.max_seq_len:
            raise ValueError(f"Board length {N} exceeds max_seq_len {self.max_seq_len}.")
        positions = self.get_position_ids(B, N, x.device)
        token_embeds = self.token_embed(x)
        if self.use_learned_pos:
            pos_embeds = self.pos_embed(positions)  # type: ignore[operator]
        else:
            pos_embeds = self.sinusoidal_pos_embed[:, :N, :].expand(B, -1, -1)
        tokens = self.proj_norm(token_embeds + pos_embeds)
        tokens = self.dropout(tokens)
        self.validate_output(tokens)
        return TokenizerOutput(
            tokens=tokens,
            position_ids=positions,
            padding_mask=(x == self.pad_token_id),
            domain_name=self.domain_name,
        )

    def get_position_ids(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        """Return flattened board positions [B, N]."""
        return torch.arange(N, device=device).unsqueeze(0).expand(B, -1)

    def collate(self, samples: list[list[int] | torch.Tensor]) -> torch.Tensor:
        """Pad a list of flattened boards to a batch tensor.

        Args:
            samples: List of board token sequences.

        Returns:
            Batched ids of shape [B, N_max].
        """
        tensor_samples = [
            sample if torch.is_tensor(sample) else torch.tensor(sample, dtype=torch.long)
            for sample in samples
        ]
        max_len = max(sample.shape[0] for sample in tensor_samples)
        padded = torch.full((len(tensor_samples), max_len), self.pad_token_id, dtype=torch.long)
        for index, sample in enumerate(tensor_samples):
            padded[index, : sample.shape[0]] = sample
        return padded

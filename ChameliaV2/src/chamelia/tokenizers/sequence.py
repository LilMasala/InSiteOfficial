"""Sequence tokenizer for 1D token domains."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import AbstractTokenizer, TokenizerOutput


class SequenceTokenizer(AbstractTokenizer):
    """Tokenizer for 1D token sequences.

    Produces contextualized token embeddings of shape [B, N, embed_dim].
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        max_seq_len: int = 256,
        domain_name: str = "sequence",
        pad_token_id: int = 0,
        use_learned_pos: bool = False,
    ) -> None:
        """Initialize the sequence tokenizer.

        Args:
            vocab_size: Number of token ids.
            embed_dim: Output token embedding dimension.
            max_seq_len: Maximum padded sequence length.
            domain_name: Human-readable tokenizer/domain name.
            pad_token_id: Padding token id.
            use_learned_pos: Whether to use learned positional embeddings.

        Returns:
            None.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.domain_name = domain_name
        self.pad_token_id = pad_token_id
        self.use_learned_pos = use_learned_pos

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        if use_learned_pos:
            self.pos_embed: nn.Module | None = nn.Embedding(max_seq_len, embed_dim)
        else:
            self.pos_embed = None
            self.register_buffer(
                "sinusoidal_pos",
                self._build_sinusoidal_encoding(max_seq_len, embed_dim),
                persistent=False,
            )

        self.proj_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=0.1)

    def _build_sinusoidal_encoding(self, max_seq_len: int, embed_dim: int) -> torch.Tensor:
        """Construct sinusoidal position embeddings.

        Args:
            max_seq_len: Maximum sequence length.
            embed_dim: Embedding dimension.

        Returns:
            Tensor of shape [1, max_seq_len, embed_dim].
        """
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        encoding = torch.zeros(max_seq_len, embed_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> TokenizerOutput:
        """Embed a padded token id batch.

        Args:
            x: Token ids of shape [B, N].

        Returns:
            TokenizerOutput with tokens of shape [B, N, embed_dim], position_ids of shape
            [B, N], padding_mask of shape [B, N], and the tokenizer domain name.
        """
        if x.dim() != 2:
            raise ValueError(f"SequenceTokenizer expected [B, N] ids, got {tuple(x.shape)}.")
        B, N = x.shape
        if N > self.max_seq_len:
            raise ValueError(
                f"Sequence length {N} exceeds max_seq_len {self.max_seq_len}."
            )

        token_embeds = self.token_embed(x)
        positions = self.get_position_ids(B, N, x.device)
        if self.use_learned_pos:
            assert self.pos_embed is not None
            pos_embeds = self.pos_embed(positions)
        else:
            sinusoidal_pos = self.sinusoidal_pos[:, :N, :].to(x.device)
            pos_embeds = sinusoidal_pos.expand(B, -1, -1)

        tokens = self.proj_norm(token_embeds + pos_embeds)
        tokens = self.dropout(tokens)
        self.validate_output(tokens)
        padding_mask = x == self.pad_token_id

        return TokenizerOutput(
            tokens=tokens,
            position_ids=positions,
            padding_mask=padding_mask,
            domain_name=self.domain_name,
        )

    def get_position_ids(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        """Return sequence position ids.

        Args:
            B: Batch size.
            N: Sequence length.
            device: Output device.

        Returns:
            Integer tensor of shape [B, N].
        """
        return torch.arange(N, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

    def collate(self, samples: list[list[int]]) -> torch.Tensor:
        """Pad raw sequences on the right.

        Args:
            samples: Python token id sequences, each length <= max_seq_len.

        Returns:
            Padded int64 tensor of shape [B, N_max].
        """
        if not samples:
            return torch.empty(0, 0, dtype=torch.long)

        max_len = max(len(sample) for sample in samples)
        if max_len > self.max_seq_len:
            raise ValueError(
                f"Collated sequence length {max_len} exceeds max_seq_len {self.max_seq_len}."
            )

        batch = torch.full(
            (len(samples), max_len),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        for idx, sample in enumerate(samples):
            if sample:
                batch[idx, : len(sample)] = torch.tensor(sample, dtype=torch.long)
        return batch


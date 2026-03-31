"""Sequence generators for pattern and regime learning."""

from __future__ import annotations

import torch


class ArithmeticSequenceGenerator:
    """Generate simple arithmetic and linear-rule sequences."""

    def __init__(self, vocab_size: int = 4096, seq_len: int = 32) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def sample(self, level: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one arithmetic sequence.

        Returns:
            Tuple of tokens [N], target [N], and regime label scalar [].
        """
        start = torch.randint(1, 20, (1,)).item()
        step = max(1, level + 1)
        seq = torch.arange(start, start + step * self.seq_len, step)[: self.seq_len] % self.vocab_size
        target = torch.roll(seq, shifts=-1, dims=0)
        regime = torch.tensor(float(step))
        return seq.long(), target.long(), regime


class HiddenMarkovSequenceGenerator:
    """Generate toy sequences with hidden regime changes."""

    def __init__(self, vocab_size: int = 4096, seq_len: int = 32) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def sample(self, level: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one HMM-like token sequence."""
        change_point = min(self.seq_len - 2, max(2, 4 + level))
        first = torch.randint(1, self.vocab_size // 4, (change_point,))
        second = torch.randint(self.vocab_size // 4, self.vocab_size // 2, (self.seq_len - change_point,))
        seq = torch.cat([first, second], dim=0)
        target = torch.roll(seq, shifts=-1, dims=0)
        regime = torch.tensor(float(change_point))
        return seq.long(), target.long(), regime

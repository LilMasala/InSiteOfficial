"""Logic and arithmetic generators for formal reasoning curricula."""

from __future__ import annotations

import torch


class BasicArithmeticGenerator:
    """Generate very simple arithmetic expressions and targets."""

    def __init__(self, vocab_size: int = 8192, seq_len: int = 48) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def sample(self, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate one arithmetic sample."""
        terms = 2 + min(level, 3)
        values = torch.randint(1, 10 + level, (terms,))
        total = int(values.sum().item()) % self.vocab_size
        tokens = torch.zeros(self.seq_len, dtype=torch.long)
        tokens[:terms] = values
        tokens[terms] = 11  # "+" marker in toy vocab
        tokens[terms + 1] = total
        target = tokens.clone()
        return tokens, target


class LogicProblemGenerator:
    """Generate toy logic-like token traces."""

    def __init__(self, vocab_size: int = 8192, seq_len: int = 48) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def sample(self, level: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate one synthetic premise/conclusion example.

        Returns:
            Tuple of (tokens, target, answer) where answer is the conclusion token.
        """
        premise_len = min(self.seq_len - 4, 8 + level)
        premise = torch.randint(1, self.vocab_size // 8, (premise_len,))
        conclusion_val = int(premise.sum().item() % max(1, self.vocab_size - 1)) + 1
        conclusion = torch.tensor([conclusion_val], dtype=torch.long)
        tokens = torch.zeros(self.seq_len, dtype=torch.long)
        tokens[:premise_len] = premise
        tokens[premise_len] = 99  # toy implication token
        tokens[premise_len + 1] = conclusion
        return tokens, tokens.clone(), conclusion.squeeze(0)

"""Reusable masking helpers for scaffold curriculum domains."""

from __future__ import annotations

import torch


def random_mask(tokens: torch.Tensor, ratio: float, mask_token_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply Bernoulli random masking to token ids."""
    mask = (torch.rand(tokens.shape, device=tokens.device) < ratio).to(dtype=torch.float32)
    masked = tokens.clone()
    masked[mask.bool()] = mask_token_id
    return masked, mask


def contiguous_span_mask(
    tokens: torch.Tensor,
    span: int,
    start: int | None = None,
    mask_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask a contiguous span of token ids."""
    if start is None:
        start = max(0, tokens.shape[1] // 2 - span // 2)
    mask = torch.zeros_like(tokens, dtype=torch.float32)
    mask[:, start : start + span] = 1.0
    masked = tokens.clone()
    masked[mask.bool()] = mask_token_id
    return masked, mask

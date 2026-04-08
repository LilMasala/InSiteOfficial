"""Latent macro-action encoding for procedural skill compilation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class _TransformerBlock(nn.Module):
    """Small transformer block used by the latent action encoder."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            spectral_norm(nn.Linear(embed_dim, hidden_dim)),
            nn.GELU(),
            spectral_norm(nn.Linear(hidden_dim, embed_dim)),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(inputs)
        inputs = inputs + self.drop(self.attn(hidden, hidden, hidden)[0])
        inputs = inputs + self.drop(self.mlp(self.norm2(inputs)))
        return inputs


@dataclass(frozen=True)
class LatentSkillCandidate:
    """Sleep-phase skill candidate prior to indexing."""

    action_path: torch.Tensor
    symbolic_codes: torch.Tensor | None
    target_delta: torch.Tensor | None
    source_weight: float
    source_episodes: tuple[int, ...]


class LatentActionEncoder(nn.Module):
    """Encode action sequences into fixed-size latent macro-actions."""

    def __init__(
        self,
        action_dim: int,
        skill_dim: int,
        *,
        symbolic_vocab_size: int = 1024,
        max_path_length: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.skill_dim = skill_dim
        self.max_path_length = max_path_length
        self.action_proj = spectral_norm(nn.Linear(action_dim, skill_dim))
        self.symbolic_embed = nn.Embedding(symbolic_vocab_size, skill_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, skill_dim))
        self.position_embed = nn.Parameter(torch.zeros(1, max_path_length + 1, skill_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embed, std=0.02)
        self.layers = nn.ModuleList(
            [
                _TransformerBlock(
                    embed_dim=skill_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(skill_dim)
        self.skill_head = spectral_norm(nn.Linear(skill_dim, skill_dim))
        self.transition_head = spectral_norm(nn.Linear(skill_dim, skill_dim))

    def forward(
        self,
        action_path: torch.Tensor,
        symbolic_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action_path.dim() == 2:
            action_path = action_path.unsqueeze(0)
        if action_path.dim() != 3:
            raise ValueError("action_path must be [B, P, A] or [P, A].")
        batch_size, path_length, _ = action_path.shape
        if path_length > self.max_path_length:
            raise ValueError(
                f"path_length={path_length} exceeds max_path_length={self.max_path_length}."
            )
        tokens = self.action_proj(action_path)
        if symbolic_codes is not None:
            if symbolic_codes.dim() == 1:
                symbolic_codes = symbolic_codes.unsqueeze(0)
            if symbolic_codes.shape[0] != batch_size or symbolic_codes.shape[1] != path_length:
                raise ValueError(
                    "symbolic_codes must align with action_path as [B, P] or [P]."
                )
            symbolic_codes = torch.remainder(
                symbolic_codes.long(),
                self.symbolic_embed.num_embeddings,
            )
            tokens = tokens + self.symbolic_embed(symbolic_codes)
        cls = self.cls_token.expand(batch_size, -1, -1)
        hidden = torch.cat([cls, tokens], dim=1)
        hidden = hidden + self.position_embed[:, : path_length + 1, :]
        for layer in self.layers:
            hidden = layer(hidden)
        pooled = self.norm(hidden[:, 0, :])
        return self.skill_head(pooled)

    def predict_transition_delta(
        self,
        action_path: torch.Tensor,
        symbolic_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.transition_head(self.forward(action_path, symbolic_codes=symbolic_codes))

    def compute_loss(
        self,
        action_path: torch.Tensor,
        target_delta: torch.Tensor,
        symbolic_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        predicted = self.predict_transition_delta(
            action_path,
            symbolic_codes=symbolic_codes,
        )
        return torch.nn.functional.mse_loss(predicted, target_delta.detach())

    def encode_candidate(self, candidate: LatentSkillCandidate) -> torch.Tensor:
        return self.forward(candidate.action_path, symbolic_codes=candidate.symbolic_codes)


def estimate_target_delta(
    start_latent: torch.Tensor,
    goal_latent: torch.Tensor,
) -> torch.Tensor:
    """Compute the latent transition target used for procedural retrieval."""
    if start_latent.shape != goal_latent.shape:
        raise ValueError("start_latent and goal_latent must share a shape.")
    delta = goal_latent - start_latent
    scale = delta.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    return delta / scale

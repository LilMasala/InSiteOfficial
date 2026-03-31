"""Configurator module for Chamelia."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Transformer block used inside Chamelia modules.

    Inputs and outputs have shape [B, N, D].
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the transformer block.

        Args:
            embed_dim: Embedding dimension D.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion ratio for the MLP hidden dimension.
            dropout: Dropout probability.

        Returns:
            None.
        """
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and MLP.

        Args:
            x: Tensor of shape [B, N, D].

        Returns:
            Tensor of shape [B, N, D].
        """
        x_norm = self.norm1(x)
        x = x + self.drop(self.attn(x_norm, x_norm, x_norm)[0])
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class Configurator(nn.Module):
    """Metacognitive context generator for Chamelia.

    Reads hierarchical latent state plus latent memory and emits context tokens [B, C, D].
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_ctx_tokens: int = 16,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        memory_read_k: int = 8,
    ) -> None:
        """Initialize the configurator.

        Args:
            embed_dim: Shared embedding dimension D.
            num_ctx_tokens: Number of context tokens C emitted by the configurator.
            num_heads: Number of attention heads.
            num_layers: Number of transformer blocks over context tokens.
            mlp_ratio: Expansion ratio for transformer MLP blocks.
            dropout: Dropout probability.
            memory_read_k: Maximum number of memory episodes consumed per sample.

        Returns:
            None.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_ctx_tokens = num_ctx_tokens
        self.memory_read_k = memory_read_k

        self.ctx_tokens = nn.Parameter(torch.empty(1, num_ctx_tokens, embed_dim))
        nn.init.trunc_normal_(self.ctx_tokens, std=0.02)

        self.level_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                )
                for _ in range(3)
            ]
        )
        self.level_embed = nn.Embedding(3, embed_dim)
        self.memory_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.self_attn_layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.cross_attn_to_latent = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_to_memory = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_latent = nn.LayerNorm(embed_dim)
        self.norm_memory = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        hjepa_outputs: dict,
        memory_keys: torch.Tensor | None,
    ) -> torch.Tensor:
        """Generate context tokens from hierarchical latent state and memory.

        Args:
            hjepa_outputs: Dict containing "target_features_per_level" as a list of 3 tensors,
                each of shape [B, N_i, D].
            memory_keys: Optional memory tensor of shape [B, K, D], where K <= memory_read_k.

        Returns:
            Context tensor of shape [B, num_ctx_tokens, D].
        """
        if "target_features_per_level" not in hjepa_outputs:
            raise KeyError("hjepa_outputs must contain 'target_features_per_level'.")

        target_features_per_level = hjepa_outputs["target_features_per_level"]
        if len(target_features_per_level) != 3:
            raise ValueError("Configurator expects exactly 3 hierarchy levels.")

        level_features: list[torch.Tensor] = []
        B = target_features_per_level[0].shape[0]
        device = target_features_per_level[0].device

        for level_idx, feats in enumerate(target_features_per_level):
            projected = self.level_proj[level_idx](feats)
            _, N_i, _ = projected.shape
            level_ids = torch.full((B, N_i), level_idx, device=device, dtype=torch.long)
            projected = projected + self.level_embed(level_ids)
            level_features.append(projected)

        all_latents = torch.cat(level_features, dim=1)
        all_latents = self.norm_latent(all_latents)

        ctx = self.ctx_tokens.expand(B, -1, -1)
        for layer in self.self_attn_layers:
            ctx = layer(ctx)

        ctx = ctx + self.cross_attn_to_latent(
            query=ctx,
            key=all_latents,
            value=all_latents,
        )[0]

        if memory_keys is not None:
            mem = self.memory_proj(memory_keys)
            mem = self.norm_memory(mem)
            ctx = ctx + self.cross_attn_to_memory(
                query=ctx,
                key=mem,
                value=mem,
            )[0]

        ctx = self.norm_out(ctx)
        ctx = self.output_proj(ctx)
        return ctx


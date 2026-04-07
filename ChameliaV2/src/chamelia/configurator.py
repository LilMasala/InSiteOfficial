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
        num_hierarchies: int = 3,
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
            num_hierarchies: Number of structural hierarchy levels expected from the backbone.

        Returns:
            None.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_ctx_tokens = num_ctx_tokens
        self.memory_read_k = memory_read_k
        self.num_hierarchies = num_hierarchies

        self.ctx_tokens = nn.Parameter(torch.empty(1, num_ctx_tokens, embed_dim))
        nn.init.trunc_normal_(self.ctx_tokens, std=0.02)

        self.level_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                )
                for _ in range(num_hierarchies)
            ]
        )
        self.level_embed = nn.Embedding(num_hierarchies, embed_dim)
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
        memory_tokens: torch.Tensor | None,
        memory_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate context tokens from hierarchical latent state and memory.

        Args:
            hjepa_outputs: Dict containing "target_features_per_level" as a list of tensors,
                each of shape [B, N_i, D].
            memory_tokens: Optional memory-summary tensor of shape [B, K, D], where
                K <= memory_read_k.
            memory_scores: Optional memory quality scores [B, K] used to emphasize
                better retrieved summaries.

        Returns:
            Context tensor of shape [B, num_ctx_tokens, D].
        """
        if "target_features_per_level" not in hjepa_outputs:
            raise KeyError("hjepa_outputs must contain 'target_features_per_level'.")

        target_features_per_level = hjepa_outputs["target_features_per_level"]

        level_features: list[torch.Tensor] = []
        B = target_features_per_level[0].shape[0]
        device = target_features_per_level[0].device

        # Dynamically loop through however many levels HJEPA emitted
        for level_idx, feats in enumerate(target_features_per_level):
            proj_idx = min(level_idx, len(self.level_proj) - 1)
            embed_idx = min(level_idx, self.level_embed.num_embeddings - 1)
            
            projected = self.level_proj[proj_idx](feats)
            _, N_i, _ = projected.shape
            level_ids = torch.full((B, N_i), embed_idx, device=device, dtype=torch.long)
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

        if memory_tokens is not None:
            mem = self.memory_proj(memory_tokens)
            if memory_scores is not None:
                if memory_scores.dim() != 2:
                    raise ValueError("memory_scores must have shape [B, K].")
                valid_mask = torch.isfinite(memory_scores)
                safe_scores = memory_scores.masked_fill(~valid_mask, -1.0e4)
                weights = torch.softmax(safe_scores, dim=1) * valid_mask.float()
                weights = torch.where(
                    weights.sum(dim=1, keepdim=True) > 0,
                    weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6),
                    weights,
                )
                mem = mem * weights.unsqueeze(-1)
            mem = self.norm_memory(mem)
            ctx = ctx + self.cross_attn_to_memory(
                query=ctx,
                key=mem,
                value=mem,
            )[0]

        ctx = self.norm_out(ctx)
        ctx = self.output_proj(ctx)
        return ctx
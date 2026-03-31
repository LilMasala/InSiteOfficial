"""Actor module for Chamelia."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Transformer block used by the actor.

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
        """Apply transformer block.

        Args:
            x: Tensor of shape [B, N, D].

        Returns:
            Tensor of shape [B, N, D].
        """
        x_norm = self.norm1(x)
        x = x + self.drop(self.attn(x_norm, x_norm, x_norm)[0])
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class Actor(nn.Module):
    """Action proposal module conditioned on configurator context tokens."""

    def __init__(
        self,
        embed_dim: int = 512,
        action_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_ctx_tokens: int = 16,
    ) -> None:
        """Initialize the actor.

        Args:
            embed_dim: Input latent dimension D.
            action_dim: Continuous action vector dimension A.
            num_heads: Number of attention heads.
            num_layers: Number of transformer blocks in deliberate mode.
            mlp_ratio: Expansion ratio for transformer MLPs.
            dropout: Dropout probability.
            num_ctx_tokens: Expected number of configurator context tokens.

        Returns:
            None.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.num_ctx_tokens = num_ctx_tokens

        self.state_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.transformer_layers = nn.ModuleList(
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
        self.cross_attn_to_ctx = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ctx = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, action_dim),
        )
        self.mode_1_policy = nn.Linear(embed_dim, action_dim)

    def forward(
        self,
        z: torch.Tensor,
        ctx_tokens: torch.Tensor,
        mode: str = "mode2",
    ) -> torch.Tensor:
        """Produce a continuous action vector.

        Args:
            z: Scene summary tensor of shape [B, D].
            ctx_tokens: Configurator context tokens of shape [B, C, D].
            mode: "mode1" for reactive policy or "mode2" for deliberate policy.

        Returns:
            Action tensor of shape [B, A].
        """
        if mode == "mode1":
            return self.mode_1_policy(z)
        if mode != "mode2":
            raise ValueError(f"Unsupported actor mode '{mode}'.")

        x = self.state_proj(z).unsqueeze(1)
        for layer in self.transformer_layers:
            x = layer(x)

        ctx = self.norm_ctx(ctx_tokens)
        x = x + self.cross_attn_to_ctx(query=x, key=ctx, value=ctx)[0]
        x = self.norm_out(x).squeeze(1)
        return self.action_head(x)

    def distill_from_mode2(
        self,
        states: torch.Tensor,
        ctx_tokens: torch.Tensor,
        mode2_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss from deliberate to reactive policy.

        Args:
            states: State tensor of shape [B, D].
            ctx_tokens: Context tokens of shape [B, C, D].
            mode2_actions: Deliberate-mode target actions of shape [B, A].

        Returns:
            Scalar tensor [] containing the MSE distillation loss.
        """
        _ = ctx_tokens
        mode1_actions = self.mode_1_policy(states)
        return F.mse_loss(mode1_actions, mode2_actions.detach())


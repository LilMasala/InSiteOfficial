"""Cost modules for Chamelia."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Transformer block used by the trainable critic.

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


class IntrinsicCost(nn.Module):
    """Immutable intrinsic cost aggregator.

    Aggregates domain-provided cost terms into a scalar [B] cost vector.
    """

    def __init__(
        self,
        cost_fns: list[Callable[[torch.Tensor, torch.Tensor, dict], torch.Tensor]],
        weights: list[float],
    ) -> None:
        """Initialize intrinsic cost aggregation.

        Args:
            cost_fns: Domain cost callables, each returning [B].
            weights: Fixed scalar weights, one per callable.

        Returns:
            None.
        """
        super().__init__()
        if len(cost_fns) != len(weights):
            raise ValueError("cost_fns and weights must have the same length.")
        self.cost_fns = cost_fns
        self.weights = weights

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        domain_state: dict,
    ) -> torch.Tensor:
        """Compute immutable intrinsic cost.

        Args:
            z: Current latent state tensor of shape [B, D].
            action: Proposed action tensor of shape [B, A].
            domain_state: Opaque domain state payload.

        Returns:
            Intrinsic cost tensor of shape [B].
        """
        costs = []
        for fn, weight in zip(self.cost_fns, self.weights):
            c = fn(z, action, domain_state)
            costs.append(weight * c)
        return torch.stack(costs, dim=0).sum(dim=0)


class TrainableCritic(nn.Module):
    """Context-conditioned learned future cost estimator."""

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_ctx_tokens: int = 16,
        horizon: int = 30,
    ) -> None:
        """Initialize the critic.

        Args:
            embed_dim: Latent embedding dimension D.
            num_heads: Number of attention heads.
            num_layers: Number of transformer blocks.
            mlp_ratio: Expansion ratio for transformer MLPs.
            dropout: Dropout probability.
            num_ctx_tokens: Expected number of context tokens.
            horizon: Future horizon metadata in steps.

        Returns:
            None.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_ctx_tokens = num_ctx_tokens
        self.horizon = horizon

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
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor, ctx_tokens: torch.Tensor) -> torch.Tensor:
        """Predict future intrinsic cost.

        Args:
            z: Current latent state tensor of shape [B, D].
            ctx_tokens: Context tokens of shape [B, C, D].

        Returns:
            Predicted future cost tensor of shape [B].
        """
        x = self.state_proj(z).unsqueeze(1)
        for layer in self.transformer_layers:
            x = layer(x)
        ctx = self.norm_ctx(ctx_tokens)
        x = x + self.cross_attn_to_ctx(query=x, key=ctx, value=ctx)[0]
        x = self.norm_out(x).squeeze(1)
        return self.value_head(x).squeeze(-1)

    def compute_critic_loss(
        self,
        predicted_value: torch.Tensor,
        realized_ic: torch.Tensor,
    ) -> torch.Tensor:
        """Compute critic regression loss.

        Args:
            predicted_value: Predicted critic value tensor of shape [B].
            realized_ic: Realized intrinsic cost target tensor of shape [B].

        Returns:
            Scalar tensor [] containing smooth L1 loss.
        """
        return F.smooth_l1_loss(predicted_value, realized_ic.detach())


class CostModule(nn.Module):
    """Combine immutable intrinsic cost and trainable future cost."""

    def __init__(
        self,
        intrinsic_cost: IntrinsicCost,
        trainable_critic: TrainableCritic,
    ) -> None:
        """Initialize the combined cost module.

        Args:
            intrinsic_cost: Immutable intrinsic cost module.
            trainable_critic: Context-conditioned learned critic.

        Returns:
            None.
        """
        super().__init__()
        self.intrinsic_cost = intrinsic_cost
        self.trainable_critic = trainable_critic

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        ctx_tokens: torch.Tensor,
        domain_state: dict,
    ) -> dict[str, torch.Tensor]:
        """Compute intrinsic, trainable, and total cost terms.

        Args:
            z: Current latent state tensor of shape [B, D].
            action: Action tensor of shape [B, A].
            ctx_tokens: Context tokens of shape [B, C, D].
            domain_state: Opaque domain state payload.

        Returns:
            Dict with tensors:
                - "ic": [B]
                - "tc": [B]
                - "total": [B]
        """
        ic = self.intrinsic_cost(z, action, domain_state)
        tc = self.trainable_critic(z, ctx_tokens)
        return {"ic": ic, "tc": tc, "total": ic + tc}


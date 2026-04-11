"""Cost modules for Chamelia."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _repeat_domain_state(domain_state: dict[str, Any], repeats: int) -> dict[str, Any]:
    """Repeat batch-shaped domain-state tensors to match flattened candidates."""
    repeated: dict[str, Any] = {}
    for key, value in domain_state.items():
        if torch.is_tensor(value) and value.dim() > 0:
            repeated[key] = value.repeat_interleave(repeats, dim=0)
        else:
            repeated[key] = copy(value)
    return repeated


def _maybe_build_imagined_domain_state(
    builder: Callable[[dict[str, Any], torch.Tensor, int], dict[str, Any]] | None,
    domain_state: dict[str, Any],
    future_z: torch.Tensor,
    step_idx: int,
) -> dict[str, Any]:
    if builder is None:
        return domain_state
    return builder(domain_state, future_z, step_idx)


def _flatten_path_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Flatten a [B, K, P, ...] tensor into [B*K*P, ...] with shape metadata."""
    batch_size, num_candidates, path_length = tensor.shape[:3]
    return tensor.reshape(batch_size * num_candidates * path_length, *tensor.shape[3:]), (
        batch_size,
        num_candidates,
        path_length,
    )


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
    """Context-conditioned learned future cost-to-go estimator."""

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
        self.output_activation = nn.Identity()
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, z: torch.Tensor, ctx_tokens: torch.Tensor) -> torch.Tensor:
        """Predict future signed cost-to-go.

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
        raw_value = self.value_head(x).squeeze(-1)
        # TC is a signed value-to-go, so let the head represent both good and bad futures.
        return self.output_activation(raw_value)

    def compute_critic_loss(
        self,
        predicted_value: torch.Tensor,
        realized_ic: torch.Tensor,
    ) -> torch.Tensor:
        """Compute critic regression loss.

        Args:
            predicted_value: Predicted critic value tensor of shape [B].
            realized_ic: Realized signed cost-to-go target tensor of shape [B].

        Returns:
            Scalar tensor [] containing smooth L1 loss.
        """
        target = realized_ic.detach()
        base_loss = F.smooth_l1_loss(predicted_value, target)
        if predicted_value.numel() < 2:
            return base_loss

        pair_mask = torch.triu(
            torch.ones(
                predicted_value.numel(),
                predicted_value.numel(),
                dtype=torch.bool,
                device=predicted_value.device,
            ),
            diagonal=1,
        )
        pred_diff = predicted_value.unsqueeze(1) - predicted_value.unsqueeze(0)
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)
        pairwise_loss = F.smooth_l1_loss(pred_diff[pair_mask], target_diff[pair_mask])

        pred_std = predicted_value.std(unbiased=False).unsqueeze(0)
        target_std = target.std(unbiased=False).unsqueeze(0)
        spread_loss = F.smooth_l1_loss(pred_std, target_std)

        return base_loss + (0.25 * pairwise_loss) + (0.10 * spread_loss)


class CostModule(nn.Module):
    """Combine immutable intrinsic cost and trainable future cost."""

    def __init__(
        self,
        intrinsic_cost: IntrinsicCost,
        trainable_critic: TrainableCritic,
        gamma: float = 0.99,
    ) -> None:
        """Initialize the combined cost module.

        Args:
            intrinsic_cost: Immutable intrinsic cost module.
            trainable_critic: Context-conditioned learned critic.
            gamma: Discount factor applied to path cost and tail critic value.

        Returns:
            None.
        """
        super().__init__()
        self.intrinsic_cost = intrinsic_cost
        self.trainable_critic = trainable_critic
        self.gamma = float(gamma)

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        ctx_tokens: torch.Tensor,
        domain_state: dict,
        future_z: torch.Tensor | None = None,
        horizon: int = 1,
        imagined_domain_state_builder: Callable[[dict[str, Any], torch.Tensor, int], dict[str, Any]] | None = None,
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
        critic_latent = z if future_z is None else future_z
        ic_domain_state = _maybe_build_imagined_domain_state(
            imagined_domain_state_builder,
            domain_state,
            critic_latent,
            0,
        )
        ic_latent = critic_latent if imagined_domain_state_builder is not None and future_z is not None else z
        ic = self.intrinsic_cost(ic_latent, action, ic_domain_state)
        tc = self.trainable_critic(critic_latent, ctx_tokens)
        total = ic + ((self.gamma**max(1, int(horizon))) * tc)
        return {"ic": ic, "tc": tc, "total": total}

    def score_candidates(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        ctx_tokens: torch.Tensor,
        domain_state: dict[str, Any],
        future_z: torch.Tensor | None = None,
        future_trajectory: torch.Tensor | None = None,
        imagined_domain_state_builder: Callable[[dict[str, Any], torch.Tensor, int], dict[str, Any]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score a batch of candidate actions.

        Args:
            z: Current latent state [B, D].
            actions: Candidate actions [B, K, A] or [B, A].
            ctx_tokens: Context tokens [B, C, D].
            domain_state: Opaque domain state.
            future_z: Optional candidate terminal latents [B, K, D].
            future_trajectory: Optional imagined path latents [B, K, P, D].

        Returns:
            Dict with ``ic``, ``tc``, and ``total`` as [B, K].
        """
        if actions.dim() == 2:
            return {
                key: value.unsqueeze(1)
                for key, value in self.forward(
                    z=z,
                    action=actions,
                    ctx_tokens=ctx_tokens,
                    domain_state=domain_state,
                    future_z=future_z,
                    horizon=1,
                    imagined_domain_state_builder=imagined_domain_state_builder,
                ).items()
            }

        if actions.dim() == 4:
            _, (batch_size, num_candidates, path_length) = _flatten_path_tensor(actions)
            if future_trajectory is None:
                raise ValueError("future_trajectory is required when scoring action paths.")
            repeated_domain_state = _repeat_domain_state(domain_state, num_candidates)
            path_costs = []
            for step_idx in range(path_length):
                flat_actions = actions[:, :, step_idx, :].reshape(batch_size * num_candidates, -1)
                flat_future = future_trajectory[:, :, step_idx, :].reshape(batch_size * num_candidates, -1)
                step_domain_state = _maybe_build_imagined_domain_state(
                    imagined_domain_state_builder,
                    repeated_domain_state,
                    flat_future,
                    step_idx,
                )
                path_costs.append(self.intrinsic_cost(flat_future, flat_actions, step_domain_state))
            path_ic = torch.stack(path_costs, dim=-1)
            discounts = torch.pow(
                torch.full(
                    (path_length,),
                    self.gamma,
                    device=path_ic.device,
                    dtype=path_ic.dtype,
                ),
                torch.arange(path_length, device=path_ic.device, dtype=path_ic.dtype),
            )
            ic = (
                path_ic.view(batch_size, num_candidates, path_length)
                * discounts.view(1, 1, path_length)
            ).sum(dim=-1)

            if future_z is None:
                future_z = future_trajectory[:, :, -1, :]
            flat_ctx = (
                ctx_tokens.unsqueeze(1)
                .expand(-1, num_candidates, -1, -1)
                .reshape(
                    batch_size * num_candidates,
                    ctx_tokens.shape[1],
                    ctx_tokens.shape[2],
                )
            )
            tc = self.trainable_critic(
                future_z.reshape(batch_size * num_candidates, -1),
                flat_ctx,
            ).view(batch_size, num_candidates)
            tail_discount = self.gamma**path_length
            return {"ic": ic, "tc": tc, "total": ic + (tail_discount * tc)}

        batch_size, num_candidates, _ = actions.shape
        flat_actions = actions.reshape(batch_size * num_candidates, -1)
        flat_z = (
            z.unsqueeze(1)
            .expand(-1, num_candidates, -1)
            .reshape(batch_size * num_candidates, -1)
        )
        flat_ctx = (
            ctx_tokens.unsqueeze(1)
            .expand(-1, num_candidates, -1, -1)
            .reshape(
                batch_size * num_candidates,
                ctx_tokens.shape[1],
                ctx_tokens.shape[2],
            )
        )
        if future_z is None:
            raise ValueError("future_z is required when scoring candidate actions [B, K, A].")
        flat_future = future_z.reshape(batch_size * num_candidates, -1)
        repeated_domain_state = _repeat_domain_state(domain_state, num_candidates)
        ic_domain_state = _maybe_build_imagined_domain_state(
            imagined_domain_state_builder,
            repeated_domain_state,
            flat_future,
            0,
        )
        ic_latent = flat_future if imagined_domain_state_builder is not None else flat_z
        ic = self.intrinsic_cost(ic_latent, flat_actions, ic_domain_state).view(batch_size, num_candidates)
        tc = self.trainable_critic(flat_future, flat_ctx).view(batch_size, num_candidates)
        return {"ic": ic, "tc": tc, "total": ic + (self.gamma * tc)}

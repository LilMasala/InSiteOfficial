"""Action-conditioned latent rollout model for Chamelia."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.chamelia.action_spec import ActionKind, ActionPath, ActionSpec, coerce_action_path

if TYPE_CHECKING:
    from src.chamelia.session_geometry import SessionGeometry


class TransformerBlock(nn.Module):
    """Transformer block used by the latent world model."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
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
        x_norm = self.norm1(x)
        x = x + self.drop(self.attn(x_norm, x_norm, x_norm)[0])
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class ActionConditionedWorldModel(nn.Module):
    """Predict future latent states conditioned on action and context."""

    def __init__(
        self,
        embed_dim: int = 512,
        action_dim: int = 64,
        posture_dim: int = 16,
        num_heads: int = 8,
        num_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_horizon: int = 8,
        ensemble_size: int = 4,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.action_spec = ActionSpec.continuous(action_dim)
        self.posture_dim = posture_dim
        self.max_horizon = max_horizon
        self.ensemble_size = max(1, int(ensemble_size))

        self.state_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        # LazyLinear defers in_features (= A) to the first forward pass so
        # the world model does not need to know the action dimension at
        # construction time.  posture_proj uses a regular Linear because P
        # is known at construction.
        self.action_proj = nn.Sequential(
            nn.LazyLinear(embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.posture_proj = nn.Sequential(
            nn.Linear(posture_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.time_embed = nn.Embedding(max_horizon, embed_dim)
        self.cross_attn_to_ctx = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ctx_norm = nn.LayerNorm(embed_dim)
        self.rollout_layers = nn.ModuleList(
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
        self.summary_norm = nn.LayerNorm(embed_dim)
        self.transition_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim),
                )
                for _ in range(self.ensemble_size)
            ]
        )
        self.posture_transition = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.state_norm = nn.LayerNorm(embed_dim)

    def _build_action_proj(
        self,
        action_spec: ActionSpec,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        if action_spec.kind == ActionKind.DISCRETE:
            return nn.Embedding(action_spec.primary_width, self.embed_dim).to(device=device)
        return nn.Sequential(
            nn.Linear(action_spec.primary_width, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        ).to(device=device, dtype=dtype)

    def _project_actions(self, flat_actions: torch.Tensor) -> torch.Tensor:
        if self.action_spec.kind == ActionKind.DISCRETE:
            action_ids = flat_actions.long()
            if action_ids.dim() > 1 and action_ids.shape[-1] == 1:
                action_ids = action_ids.squeeze(-1)
            return self.action_proj(action_ids)
        return self.action_proj(flat_actions)

    def bind_geometry(self, geometry: "SessionGeometry") -> None:
        """Bind or rebind the action projection to a concrete action dimension.

        The world model is constructed with ``action_proj`` as a
        ``LazyLinear`` so that A does not need to be known at construction
        time.  This method replaces it with a materialised ``Linear(A, D)``
        once A is known.

        Idempotent: a no-op when A is unchanged and ``action_proj`` has
        already been materialised (i.e. is no longer a ``LazyLinear``).
        On a domain switch with a different A, the projection is rebuilt
        with fresh weights — the world model must re-learn the new action
        embedding from that point onward.

        Args:
            geometry: SessionGeometry describing {D, A, P, K, H, T}.

        Returns:
            None.
        """
        action_spec = geometry.action_spec
        A = action_spec.primary_width
        is_lazy = isinstance(self.action_proj, nn.Sequential) and isinstance(
            self.action_proj[0], nn.LazyLinear
        )
        if not is_lazy and self.action_dim == A and self.action_spec == action_spec:
            return

        try:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.float32

        self.action_proj = self._build_action_proj(action_spec, device=device, dtype=dtype)
        self.action_dim = A
        self.action_spec = action_spec

    def forward(
        self,
        z: torch.Tensor,
        actions: torch.Tensor | ActionPath,
        ctx_tokens: torch.Tensor,
        candidate_postures: torch.Tensor | None = None,
        reasoning_states: torch.Tensor | None = None,
        horizon: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Roll out imagined latent futures.

        Args:
            z: Current latent state [B, D].
            actions: Candidate actions [B, A] or [B, K, A].
            ctx_tokens: Context tokens [B, C, D].
            candidate_postures: Optional candidate postures [B, K, posture_dim].
            reasoning_states: Optional candidate reasoning states [B, K, D].
            horizon: Number of rollout steps.

        Returns:
            Dict with:
                - trajectory: [B, K, H, D]
                - terminal_latents: [B, K, D]
                - summary_tokens: [B, K, D]
        """
        if horizon < 1:
            raise ValueError("horizon must be at least 1.")
        if horizon > self.max_horizon:
            raise ValueError(
                f"horizon={horizon} exceeds max_horizon={self.max_horizon}."
            )

        action_path = coerce_action_path(actions, self.action_spec)
        actions_tensor = action_path.as_tensor()
        if actions_tensor.dim() == 2:
            actions_tensor = actions_tensor.unsqueeze(1).unsqueeze(2)
        elif actions_tensor.dim() == 3:
            actions_tensor = actions_tensor.unsqueeze(2)
        elif actions_tensor.dim() != 4:
            raise ValueError(
                "actions must have shape [B, A], [B, K, A], or [B, K, P, A]."
            )

        batch_size, num_candidates, path_length, _ = actions_tensor.shape
        if path_length > self.max_horizon:
            raise ValueError(
                f"path_length={path_length} exceeds max_horizon={self.max_horizon}."
            )
        effective_horizon = path_length
        flat_state = z.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, z.shape[-1])
        flat_ctx = (
            ctx_tokens.unsqueeze(1)
            .expand(-1, num_candidates, -1, -1)
            .reshape(-1, ctx_tokens.shape[1], ctx_tokens.shape[2])
        )
        flat_reasoning = None
        if reasoning_states is not None:
            if reasoning_states.dim() == 2:
                reasoning_states = reasoning_states.unsqueeze(1)
            flat_reasoning = reasoning_states.reshape(-1, reasoning_states.shape[-1])
        flat_postures = None
        if candidate_postures is not None:
            if candidate_postures.dim() == 2:
                candidate_postures = candidate_postures.unsqueeze(1)
            flat_postures = candidate_postures.reshape(-1, candidate_postures.shape[-1])

        current = flat_state
        trajectory_steps: list[torch.Tensor] = []
        summary_steps: list[torch.Tensor] = []
        ctx = self.ctx_norm(flat_ctx)

        for step_idx in range(effective_horizon):
            flat_actions = actions_tensor[:, :, step_idx, :].reshape(-1, actions_tensor.shape[-1])
            step_ids = torch.full(
                (current.shape[0],),
                step_idx,
                device=current.device,
                dtype=torch.long,
            )
            x = self.state_proj(current) + self._project_actions(flat_actions) + self.time_embed(step_ids)
            if flat_reasoning is not None:
                x = x + flat_reasoning
            posture_token = None
            if flat_postures is not None:
                posture_token = self.posture_proj(flat_postures) + self.time_embed(step_ids)
                x = x + posture_token
            x = x.unsqueeze(1)
            x = x + self.cross_attn_to_ctx(query=x, key=ctx, value=ctx)[0]
            for layer in self.rollout_layers:
                x = layer(x)
            summary = self.summary_norm(x.squeeze(1))
            head_deltas = torch.stack([head(summary) for head in self.transition_heads], dim=0)
            delta = head_deltas.mean(dim=0)
            if posture_token is not None:
                delta = delta + self.posture_transition(torch.cat([summary, posture_token], dim=-1))
            current = self.state_norm(current + delta)
            trajectory_steps.append(current.view(batch_size, num_candidates, self.embed_dim))
            summary_steps.append(summary.view(batch_size, num_candidates, self.embed_dim))
            uncertainty_steps = head_deltas.std(dim=0, unbiased=False).mean(dim=-1).view(
                batch_size,
                num_candidates,
            )
            if step_idx == 0:
                uncertainty = uncertainty_steps.unsqueeze(-1)
            else:
                uncertainty = torch.cat([uncertainty, uncertainty_steps.unsqueeze(-1)], dim=-1)

        trajectory = torch.stack(trajectory_steps, dim=2)
        summary_tokens = torch.stack(summary_steps, dim=2).mean(dim=2)
        terminal_latents = trajectory[:, :, -1, :]

        return {
            "trajectory": trajectory,
            "terminal_latents": terminal_latents,
            "summary_tokens": summary_tokens,
            "uncertainty": uncertainty,
            "ensemble_deltas": head_deltas.view(
                self.ensemble_size,
                batch_size,
                num_candidates,
                self.embed_dim,
            ),
        }

    def compute_transition_loss(
        self,
        z_t: torch.Tensor,
        actions: torch.Tensor,
        z_tH: torch.Tensor,
        ctx_tokens: torch.Tensor,
        candidate_postures: torch.Tensor | None = None,
        horizon: int = 1,
    ) -> torch.Tensor:
        """Train the rollout model on stored transitions."""
        outputs = self(
            z_t,
            actions,
            ctx_tokens,
            candidate_postures=candidate_postures,
            horizon=horizon,
        )
        predicted = outputs["terminal_latents"]
        if predicted.dim() == 3 and predicted.shape[1] == 1:
            predicted = predicted[:, 0, :]
        base_loss = F.smooth_l1_loss(predicted, z_tH.detach())
        diversity_loss = self.compute_diversity_loss(outputs["ensemble_deltas"])
        return base_loss + (0.05 * diversity_loss)

    def compute_diversity_loss(self, ensemble_deltas: torch.Tensor) -> torch.Tensor:
        """Penalize collapsed ensemble transition heads."""
        if ensemble_deltas.shape[0] < 2:
            return ensemble_deltas.new_zeros(())
        flattened = ensemble_deltas.reshape(ensemble_deltas.shape[0], -1)
        normalized = F.normalize(flattened, dim=-1)
        similarity = normalized @ normalized.transpose(0, 1)
        pair_mask = torch.triu(
            torch.ones(
                similarity.shape[0],
                similarity.shape[1],
                dtype=torch.bool,
                device=similarity.device,
            ),
            diagonal=1,
        )
        return F.relu(similarity[pair_mask] - 0.25).mean()

    def compute_trajectory_loss(
        self,
        z_t: torch.Tensor,
        actions: torch.Tensor,
        target_trajectory: torch.Tensor,
        ctx_tokens: torch.Tensor,
        *,
        candidate_postures: torch.Tensor | None = None,
        reasoning_states: torch.Tensor | None = None,
        step_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Train the rollout model against a full target latent trajectory."""
        if target_trajectory.dim() == 3:
            target_trajectory = target_trajectory.unsqueeze(1)
        horizon = int(target_trajectory.shape[2])
        outputs = self(
            z_t,
            actions,
            ctx_tokens,
            candidate_postures=candidate_postures,
            reasoning_states=reasoning_states,
            horizon=horizon,
        )
        predicted = outputs["trajectory"]
        target = target_trajectory.detach().to(predicted)
        per_step = F.smooth_l1_loss(predicted, target, reduction="none").mean(dim=-1)
        if step_weights is None:
            step_weights = torch.full(
                (horizon,),
                fill_value=1.0 / max(1, horizon),
                dtype=per_step.dtype,
                device=per_step.device,
            )
        else:
            step_weights = step_weights.to(per_step.device, per_step.dtype)
        loss = (per_step * step_weights.view(1, 1, horizon)).sum(dim=-1).mean()
        per_transition_l2 = torch.sqrt(
            torch.mean((predicted - target).pow(2), dim=-1).clamp_min(1.0e-12)
        )
        return loss, predicted, per_transition_l2

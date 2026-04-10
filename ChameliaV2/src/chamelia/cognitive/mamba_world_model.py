"""State-space world-model aligned with the phase-6 Mamba path."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from mamba_ssm import Mamba2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency.
    Mamba2 = None


@dataclass(frozen=True)
class WorldModelBenchmark:
    """Measured comparison between two world-model implementations."""

    reference_loss: float
    candidate_loss: float
    reference_latency_ms: float
    candidate_latency_ms: float
    backend: str


class _SelectiveStateSpaceBlock(nn.Module):
    """Portable fallback state-space block for non-Mamba environments."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.state_gate = nn.Linear(embed_dim, embed_dim)
        self.input_gate = nn.Linear(embed_dim, embed_dim)
        self.output_gate = nn.Linear(embed_dim, embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = inputs.shape
        state = torch.zeros(batch_size, embed_dim, device=inputs.device, dtype=inputs.dtype)
        outputs: list[torch.Tensor] = []
        for step in range(seq_len):
            x_t = inputs[:, step, :]
            a_t = torch.sigmoid(self.state_gate(x_t))
            b_t = torch.tanh(self.input_gate(x_t))
            c_t = torch.sigmoid(self.output_gate(x_t))
            state = (a_t * state) + ((1.0 - a_t) * b_t)
            outputs.append(c_t * state)
        return torch.stack(outputs, dim=1)


class _SequenceMixerBlock(nn.Module):
    """Residual sequence-mixer block using native Mamba2 when available."""

    def __init__(
        self,
        *,
        embed_dim: int,
        mlp_ratio: float,
        dropout: float,
        use_native_mamba: bool,
        d_state: int,
        d_conv: int,
        expand: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        if use_native_mamba and Mamba2 is not None:
            self.mixer: nn.Module = Mamba2(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.backend = "mamba2"
        else:
            self.mixer = _SelectiveStateSpaceBlock(embed_dim=embed_dim)
            self.backend = "fallback_ssm"
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.dropout(self.mixer(self.norm1(inputs)))
        inputs = inputs + self.dropout(self.mlp(self.norm2(inputs)))
        return inputs


class MambaActionConditionedWorldModel(nn.Module):
    """Action-conditioned rollout model with a selectable Mamba2 backend."""

    def __init__(
        self,
        *,
        embed_dim: int = 512,
        action_dim: int = 64,
        posture_dim: int = 16,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_horizon: int = 8,
        use_native_mamba: bool = True,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.posture_dim = posture_dim
        self.max_horizon = max_horizon
        self.state_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.action_proj = nn.Sequential(nn.Linear(action_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.posture_proj = nn.Sequential(nn.Linear(posture_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.reasoning_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.ctx_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.time_embed = nn.Embedding(max_horizon, embed_dim)
        self.sequence_model = nn.ModuleList(
            [
                _SequenceMixerBlock(
                    embed_dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_native_mamba=use_native_mamba,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_layers)
            ]
        )
        self.backend = self.sequence_model[0].backend if self.sequence_model else "fallback_ssm"
        self.summary_norm = nn.LayerNorm(embed_dim)
        self.transition_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.posture_transition = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.state_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        ctx_tokens: torch.Tensor,
        candidate_postures: torch.Tensor | None = None,
        reasoning_states: torch.Tensor | None = None,
        horizon: int = 1,
    ) -> dict[str, torch.Tensor]:
        if horizon < 1:
            raise ValueError("horizon must be at least 1.")
        if horizon > self.max_horizon:
            raise ValueError(
                f"horizon={horizon} exceeds max_horizon={self.max_horizon}."
            )
        if actions.dim() == 2:
            actions = actions.unsqueeze(1).unsqueeze(2)
        elif actions.dim() == 3:
            actions = actions.unsqueeze(2)
        elif actions.dim() != 4:
            raise ValueError(
                f"actions must have shape [B, A], [B, K, A], or [B, K, P, A], got {tuple(actions.shape)}."
            )
        batch_size, num_candidates, path_length, _ = actions.shape
        if path_length > self.max_horizon:
            raise ValueError(
                f"path_length={path_length} exceeds max_horizon={self.max_horizon}."
            )
        current = z.unsqueeze(1).expand(-1, num_candidates, -1).reshape(-1, z.shape[-1])
        flat_ctx = (
            ctx_tokens.unsqueeze(1)
            .expand(-1, num_candidates, -1, -1)
            .reshape(-1, ctx_tokens.shape[1], ctx_tokens.shape[2])
        )
        ctx_summary = self.ctx_proj(flat_ctx.mean(dim=1))
        if reasoning_states is not None:
            if reasoning_states.dim() == 2:
                reasoning_states = reasoning_states.unsqueeze(1)
            reasoning_flat = self.reasoning_proj(
                reasoning_states.reshape(-1, reasoning_states.shape[-1])
            )
        else:
            reasoning_flat = None
        if candidate_postures is not None:
            if candidate_postures.dim() == 2:
                candidate_postures = candidate_postures.unsqueeze(1)
            posture_flat = self.posture_proj(
                candidate_postures.reshape(-1, candidate_postures.shape[-1])
            )
        else:
            posture_flat = None
        token_sequence: list[torch.Tensor] = []
        posture_tokens: list[torch.Tensor | None] = []
        for step_idx in range(path_length):
            step_action = actions[:, :, step_idx, :].reshape(-1, self.action_dim)
            token = self.state_proj(current) + self.action_proj(step_action) + ctx_summary
            token = token + self.time_embed(
                torch.full((token.shape[0],), step_idx, device=token.device, dtype=torch.long)
            )
            posture_token = None
            if posture_flat is not None:
                posture_token = posture_flat + self.time_embed(
                    torch.full((token.shape[0],), step_idx, device=token.device, dtype=torch.long)
                )
                token = token + posture_token
            if reasoning_flat is not None:
                token = token + reasoning_flat
            token_sequence.append(token)
            posture_tokens.append(posture_token)
        sequence = torch.stack(token_sequence, dim=1)
        modeled = sequence
        for layer in self.sequence_model:
            modeled = layer(modeled)
        deltas = self.transition_head(self.summary_norm(modeled))
        trajectory = []
        state = current
        for step_idx in range(path_length):
            delta = deltas[:, step_idx, :]
            posture_token = posture_tokens[step_idx]
            if posture_token is not None:
                delta = delta + self.posture_transition(
                    torch.cat([modeled[:, step_idx, :], posture_token], dim=-1)
                )
            state = self.state_norm(state + delta)
            trajectory.append(state.view(batch_size, num_candidates, self.embed_dim))
        trajectory_tensor = torch.stack(trajectory, dim=2)
        return {
            "trajectory": trajectory_tensor,
            "terminal_latents": trajectory_tensor[:, :, -1, :],
            "summary_tokens": self.summary_norm(modeled.mean(dim=1)).view(
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
        return F.smooth_l1_loss(predicted, z_tH.detach())

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        return loss, predicted


def benchmark_world_models(
    *,
    reference_model: Any,
    candidate_model: Any,
    z_t: torch.Tensor,
    actions: torch.Tensor,
    z_tH: torch.Tensor,
    ctx_tokens: torch.Tensor,
    candidate_postures: torch.Tensor | None = None,
    horizon: int = 1,
) -> WorldModelBenchmark:
    """Compare a reference world model against the Mamba prototype on identical data."""
    started = time.perf_counter()
    reference_loss = reference_model.compute_transition_loss(
        z_t=z_t,
        actions=actions,
        z_tH=z_tH,
        ctx_tokens=ctx_tokens,
        candidate_postures=candidate_postures,
        horizon=horizon,
    )
    reference_latency_ms = (time.perf_counter() - started) * 1000.0
    started = time.perf_counter()
    candidate_loss = candidate_model.compute_transition_loss(
        z_t=z_t,
        actions=actions,
        z_tH=z_tH,
        ctx_tokens=ctx_tokens,
        candidate_postures=candidate_postures,
        horizon=horizon,
    )
    candidate_latency_ms = (time.perf_counter() - started) * 1000.0
    return WorldModelBenchmark(
        reference_loss=float(reference_loss.item()),
        candidate_loss=float(candidate_loss.item()),
        reference_latency_ms=reference_latency_ms,
        candidate_latency_ms=candidate_latency_ms,
        backend=getattr(candidate_model, "backend", "unknown"),
    )

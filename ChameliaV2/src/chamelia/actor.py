"""Actor module for Chamelia."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from src.chamelia.session_geometry import SessionGeometry


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
        num_candidates: int = 6,
        path_length: int = 3,
        posture_dim: int = 16,
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
            posture_dim: Behavioural-intent bottleneck dimension P (default 16).
                Kept intentionally small so postures act as compressed strategy
                codes rather than full latent vectors.  ``MemoryRelevanceScorer``
                handles the P → D projection internally before comparing postures
                during retrieval, so no external projection is required.

        Returns:
            None.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.num_ctx_tokens = num_ctx_tokens
        self.num_candidates = num_candidates
        self.path_length = path_length
        self.posture_dim = posture_dim

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
        self.posture_seeds = nn.Parameter(torch.empty(1, num_candidates, posture_dim))
        nn.init.trunc_normal_(self.posture_seeds, std=0.02)
        self.reasoning_tokens = nn.Parameter(torch.empty(1, num_candidates, embed_dim))
        nn.init.trunc_normal_(self.reasoning_tokens, std=0.02)
        self.candidate_layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(max(1, num_layers - 1))
            ]
        )
        self.cross_attn_to_state = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_to_ctx = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_to_rollout = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_to_reasoning = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_to_posture_memory = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        ) 
        self.norm_ctx = nn.LayerNorm(embed_dim)
        self.norm_state = nn.LayerNorm(embed_dim)
        self.norm_posture = nn.LayerNorm(posture_dim)
        self.norm_reasoning = nn.LayerNorm(embed_dim)
        self.norm_rollout = nn.LayerNorm(embed_dim)
        self.posture_to_reasoning = nn.Linear(posture_dim, embed_dim)
        self.posture_memory_proj = nn.Sequential(
            nn.Linear(posture_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.posture_from_memory = nn.Linear(embed_dim, posture_dim)
        self.posture_from_state = nn.Linear(embed_dim, posture_dim)
        self.posture_from_ctx = nn.Linear(embed_dim, posture_dim)
        self.posture_refinement = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, posture_dim),
        )
        self.posture_path_head = nn.Sequential(
            nn.Linear(posture_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, action_dim),
        )
        self.step_posture_embeddings = nn.Parameter(torch.empty(1, 1, path_length, posture_dim))
        nn.init.trunc_normal_(self.step_posture_embeddings, std=0.02)
        self.score_proj = nn.Linear(1, embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, action_dim * path_length),
        )
        self.refinement_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, action_dim * path_length),
        )
        self.candidate_selection_head = nn.Sequential(
            nn.Linear(embed_dim + posture_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.mode_1_policy = nn.Linear(embed_dim, action_dim)

    def bind_geometry(self, geometry: "SessionGeometry") -> None:
        """Rebind action-dimension-dependent heads from a SessionGeometry.

        Idempotent: if ``geometry.A == self.action_dim`` and
        ``geometry.H == self.path_length`` the call is a no-op so that the
        existing initialised weights are preserved (e.g. after ``.to(device)``
        has already placed them on the target device).

        Only the four action-output heads (``action_head``, ``refinement_head``,
        ``posture_path_head``, ``mode_1_policy``) and, when H changes,
        ``step_posture_embeddings`` are rebuilt.

        ``candidate_selection_head`` is NOT rebuilt here because its input
        shape is ``D + P``, which depends only on ``embed_dim`` and
        ``posture_dim`` — both are fixed for the lifetime of the model and
        independent of the domain's action space.

        Args:
            geometry: SessionGeometry describing {D, A, P, K, H, T}.

        Raises:
            ValueError: If ``geometry.P != self.posture_dim``.  The posture
                bottleneck is a model-level constant; it cannot be changed by
                switching domains.

        Returns:
            None.
        """
        if geometry.P != self.posture_dim:
            raise ValueError(
                f"geometry.P={geometry.P} does not match "
                f"actor.posture_dim={self.posture_dim}.  "
                "The posture bottleneck is fixed at construction time."
            )

        A = geometry.A
        H = geometry.H

        # Idempotent guard: skip rebuild if nothing action-relevant changed.
        if self.action_dim == A and self.path_length == H:
            return

        D = self.embed_dim
        P = self.posture_dim
        try:
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
        except StopIteration:
            device = torch.device("cpu")
            dtype = torch.float32

        self.action_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, A * H),
        ).to(device=device, dtype=dtype)

        self.refinement_head = nn.Sequential(
            nn.Linear(D, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, A * H),
        ).to(device=device, dtype=dtype)

        self.posture_path_head = nn.Sequential(
            nn.Linear(P, D // 2),
            nn.GELU(),
            nn.Linear(D // 2, A),
        ).to(device=device, dtype=dtype)

        self.mode_1_policy = nn.Linear(D, A).to(device=device, dtype=dtype)

        if H != self.path_length:
            self.path_length = H
            self.step_posture_embeddings = nn.Parameter(
                torch.empty(1, 1, H, P, device=device, dtype=dtype)
            )
            nn.init.trunc_normal_(self.step_posture_embeddings, std=0.02)

        self.action_dim = A

    def _encode_state(self, z: torch.Tensor) -> torch.Tensor:
        """Encode the current latent state into a planning token."""
        state = self.state_proj(z).unsqueeze(1)
        for layer in self.transformer_layers:
            state = layer(state)
        return self.norm_state(state)

    def _enforce_simple_baseline(
        self,
        candidate_paths: torch.Tensor,
        candidate_postures: torch.Tensor,
        reasoning_states: torch.Tensor,
        state_token: torch.Tensor,
        simple_baseline_path: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep candidate 0 as the simple/null baseline across reasoning rounds.

        The baseline uses the raw state encoding (no candidate-specific elaboration)
        but can optionally use a domain-provided baseline path.
        """
        baseline_paths = candidate_paths.clone()
        baseline_postures = candidate_postures.clone()
        baseline_reasoning = reasoning_states.clone()

        if simple_baseline_path is not None:
            if simple_baseline_path.dim() != 3:
                raise ValueError("simple_baseline_path must be [B, H, A].")
            baseline_paths[:, 0, :, :] = simple_baseline_path.to(candidate_paths)
        else:
            baseline_paths[:, 0, :, :] = 0.0
        baseline_postures[:, 0, :] = 0.0
        baseline_reasoning[:, 0, :] = state_token.squeeze(1)

        return baseline_paths, baseline_postures, baseline_reasoning

    def _selection_logits(
        self,
        candidate_postures: torch.Tensor,
        reasoning_states: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([reasoning_states, candidate_postures], dim=-1)
        return self.candidate_selection_head(combined).squeeze(-1)

    def compute_posture_diversity_loss(
        self,
        candidate_postures: torch.Tensor,
        candidate_paths: torch.Tensor,
        max_posture_similarity: float = 0.90,
        max_path_similarity: float = 0.95,
    ) -> torch.Tensor:
        """Penalize collapse across non-baseline candidate postures."""
        if candidate_postures.dim() != 3 or candidate_paths.dim() != 4:
            raise ValueError("candidate_postures must be [B, K, posture_dim] and candidate_paths must be [B, K, H, A].")
        if candidate_postures.shape[1] < 3:
            return candidate_postures.new_zeros(())

        nonbaseline_postures = candidate_postures[:, 1:, :]
        nonbaseline_paths = candidate_paths[:, 1:, :, :].reshape(
            candidate_paths.shape[0],
            candidate_paths.shape[1] - 1,
            -1,
        )

        posture_norm = F.normalize(nonbaseline_postures, dim=-1)
        path_norm = F.normalize(nonbaseline_paths, dim=-1)
        posture_similarity = posture_norm @ posture_norm.transpose(1, 2)
        path_similarity = path_norm @ path_norm.transpose(1, 2)

        pair_mask = torch.triu(
            torch.ones(
                posture_similarity.shape[1],
                posture_similarity.shape[2],
                device=candidate_postures.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        posture_penalty = F.relu(posture_similarity - max_posture_similarity)[:, pair_mask]
        path_penalty = F.relu(path_similarity - max_path_similarity)[:, pair_mask]
        return posture_penalty.mean() + path_penalty.mean()

    def _posture_path_bias(self, candidate_postures: torch.Tensor) -> torch.Tensor:
        """Project candidate postures directly into path-space biases."""
        step_postures = candidate_postures.unsqueeze(2) + self.step_posture_embeddings[:, :, : self.path_length, :]
        return self.posture_path_head(step_postures)

    def _memory_posture_bias(
        self,
        postures: torch.Tensor,
        retrieved_postures: torch.Tensor | None,
        retrieved_posture_scores: torch.Tensor | None,
    ) -> torch.Tensor:
        """Project retrieved successful postures into an initial candidate-posture bias."""
        if retrieved_postures is None:
            return torch.zeros_like(postures)
        if retrieved_postures.dim() != 3:
            raise ValueError("retrieved_postures must have shape [B, M, posture_dim].")

        memory_tokens = self.posture_memory_proj(retrieved_postures)
        if retrieved_posture_scores is not None:
            if retrieved_posture_scores.dim() != 2:
                raise ValueError("retrieved_posture_scores must have shape [B, M].")
            valid_mask = torch.isfinite(retrieved_posture_scores)
            safe_scores = retrieved_posture_scores.masked_fill(~valid_mask, -1.0e4)
            weights = torch.softmax(safe_scores, dim=1) * valid_mask.float()
            weights = torch.where(
                weights.sum(dim=1, keepdim=True) > 0,
                weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6),
                weights,
            )
            memory_tokens = memory_tokens * weights.unsqueeze(-1)

        memory_delta = self.cross_attn_to_posture_memory(
            query=self.posture_to_reasoning(postures),
            key=memory_tokens,
            value=memory_tokens,
        )[0]
        return self.posture_from_memory(memory_delta)

    def _memory_refinement_bias(
        self,
        rollout_summary: torch.Tensor,
        retrieved_postures: torch.Tensor | None,
        retrieved_episode_summaries: torch.Tensor | None,
        retrieved_memory_scores: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pull retrieved framings back into later reasoning rounds based on rollout shape."""
        if retrieved_postures is None and retrieved_episode_summaries is None:
            return torch.zeros(
                rollout_summary.shape[0],
                rollout_summary.shape[1],
                self.posture_dim,
                device=rollout_summary.device,
                dtype=rollout_summary.dtype,
            )

        memory_tokens = None
        if retrieved_postures is not None:
            if retrieved_postures.dim() != 3:
                raise ValueError("retrieved_postures must have shape [B, M, posture_dim].")
            memory_tokens = self.posture_memory_proj(retrieved_postures)
        if retrieved_episode_summaries is not None:
            if retrieved_episode_summaries.dim() != 3:
                raise ValueError(
                    "retrieved_episode_summaries must have shape [B, M, embed_dim]."
                )
            memory_tokens = (
                retrieved_episode_summaries
                if memory_tokens is None
                else memory_tokens + retrieved_episode_summaries
            )
        if memory_tokens is None:
            return torch.zeros(
                rollout_summary.shape[0],
                rollout_summary.shape[1],
                self.posture_dim,
                device=rollout_summary.device,
                dtype=rollout_summary.dtype,
            )
        if retrieved_memory_scores is not None:
            if retrieved_memory_scores.dim() != 2:
                raise ValueError("retrieved_memory_scores must have shape [B, M].")
            valid_mask = torch.isfinite(retrieved_memory_scores)
            safe_scores = retrieved_memory_scores.masked_fill(~valid_mask, -1.0e4)
            weights = torch.softmax(safe_scores, dim=1) * valid_mask.float()
            weights = torch.where(
                weights.sum(dim=1, keepdim=True) > 0,
                weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6),
                weights,
            )
            memory_tokens = memory_tokens * weights.unsqueeze(-1)

        memory_delta = self.cross_attn_to_posture_memory(
            query=rollout_summary,
            key=memory_tokens,
            value=memory_tokens,
        )[0]
        return self.posture_from_memory(memory_delta)

    def propose(
        self,
        z: torch.Tensor,
        ctx_tokens: torch.Tensor,
        retrieved_postures: torch.Tensor | None = None,
        retrieved_posture_scores: torch.Tensor | None = None,
        simple_baseline_path: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate candidate actions and candidate planning states."""
        batch_size = z.shape[0]
        state = self._encode_state(z)
        ctx = self.norm_ctx(ctx_tokens)
        postures = self.posture_seeds.expand(batch_size, -1, -1)
        posture_state = self.cross_attn_to_state(
            query=self.posture_to_reasoning(postures),
            key=state,
            value=state,
        )[0]
        posture_ctx = self.cross_attn_to_ctx(
            query=self.posture_to_reasoning(postures),
            key=ctx,
            value=ctx,
        )[0]
        postures = (
            postures
            + self.posture_from_state(posture_state)
            + self.posture_from_ctx(posture_ctx)
            + self._memory_posture_bias(
                postures,
                retrieved_postures=retrieved_postures,
                retrieved_posture_scores=retrieved_posture_scores,
            )
        )
        candidate_postures = self.norm_posture(postures)
        reasoning = self.reasoning_tokens.expand(batch_size, -1, -1) + self.posture_to_reasoning(candidate_postures)
        reasoning = reasoning + self.cross_attn_to_state(
            query=reasoning,
            key=state,
            value=state,
        )[0]
        reasoning = reasoning + self.cross_attn_to_ctx(
            query=reasoning,
            key=ctx,
            value=ctx,
        )[0]
        for layer in self.candidate_layers:
            reasoning = layer(reasoning)
        reasoning_states = self.norm_reasoning(reasoning)
        reasoning_paths = self.action_head(reasoning_states).view(
            batch_size,
            self.num_candidates,
            self.path_length,
            self.action_dim,
        )
        posture_paths = self._posture_path_bias(candidate_postures)
        candidate_paths = reasoning_paths + posture_paths
        candidate_paths, candidate_postures, reasoning_states = self._enforce_simple_baseline(
            candidate_paths,
            candidate_postures,
            reasoning_states,
            state,
            simple_baseline_path=simple_baseline_path,
        )
        candidate_actions = candidate_paths[:, :, 0, :]
        return {
            "candidate_postures": candidate_postures,
            "reasoning_states": reasoning_states,
            "candidate_states": reasoning_states,
            "candidate_paths": candidate_paths,
            "candidate_actions": candidate_actions,
            "candidate_selection_logits": self._selection_logits(
                candidate_postures,
                reasoning_states,
            ),
        }

    def refine(
        self,
        z: torch.Tensor,
        ctx_tokens: torch.Tensor,
        candidate_paths: torch.Tensor,
        candidate_postures: torch.Tensor,
        reasoning_states: torch.Tensor,
        rollout_summary: torch.Tensor,
        candidate_scores: torch.Tensor,
        retrieved_postures: torch.Tensor | None = None,
        retrieved_posture_scores: torch.Tensor | None = None,
        retrieved_episode_summaries: torch.Tensor | None = None,
        retrieved_episode_scores: torch.Tensor | None = None,
        simple_baseline_path: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine candidate actions after scoring imagined futures."""
        state = self._encode_state(z)
        ctx = self.norm_ctx(ctx_tokens)
        rollout = self.norm_rollout(rollout_summary)
        score_tokens = self.score_proj(candidate_scores.unsqueeze(-1))
        previous_posture_paths = self._posture_path_bias(candidate_postures)
        memory_scores = (
            retrieved_episode_scores
            if retrieved_episode_scores is not None
            else retrieved_posture_scores
        )
        posture_delta = self.posture_refinement(rollout + score_tokens) + self._memory_refinement_bias(
            rollout_summary=rollout,
            retrieved_postures=retrieved_postures,
            retrieved_episode_summaries=retrieved_episode_summaries,
            retrieved_memory_scores=memory_scores,
        )
        candidate_postures = self.norm_posture(candidate_postures + posture_delta)
        refined = reasoning_states + self.posture_to_reasoning(candidate_postures) + score_tokens
        refined = refined + self.cross_attn_to_rollout(
            query=refined,
            key=rollout,
            value=rollout,
        )[0]
        refined = refined + self.cross_attn_to_reasoning(
            query=refined,
            key=state,
            value=state,
        )[0]
        refined = refined + self.cross_attn_to_ctx(
            query=refined,
            key=ctx,
            value=ctx,
        )[0]
        for layer in self.candidate_layers:
            refined = layer(refined)
        refined = self.norm_out(refined)
        path_delta = self.refinement_head(refined).view(
            refined.shape[0],
            refined.shape[1],
            self.path_length,
            self.action_dim,
        )
        updated_posture_paths = self._posture_path_bias(candidate_postures)
        candidate_paths = candidate_paths + path_delta + (updated_posture_paths - previous_posture_paths)
        candidate_paths, candidate_postures, refined = self._enforce_simple_baseline(
            candidate_paths,
            candidate_postures,
            refined,
            state,
            simple_baseline_path=simple_baseline_path,
        )
        return candidate_paths, candidate_postures, refined

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
        proposal = self.propose(z, ctx_tokens)
        candidate_idx = 1 if self.num_candidates > 1 else 0
        return proposal["candidate_actions"][:, candidate_idx, :]

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

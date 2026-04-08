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
    prompt_vector: torch.Tensor | None = None


class LatentActionEncoder(nn.Module):
    """Encode action sequences into fixed-size latent macro-actions."""

    def __init__(
        self,
        action_dim: int,
        skill_dim: int,
        *,
        latent_prompt_dim: int | None = None,
        symbolic_vocab_size: int = 1024,
        max_path_length: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.skill_dim = skill_dim
        self.latent_prompt_dim = int(latent_prompt_dim or skill_dim)
        self.max_path_length = max_path_length
        self.action_proj = spectral_norm(nn.Linear(action_dim, skill_dim))
        self.symbolic_embed = nn.Embedding(symbolic_vocab_size, skill_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, skill_dim))
        self.position_embed = nn.Parameter(torch.zeros(1, max_path_length + 1, skill_dim))
        self.prompt_norm = nn.LayerNorm(self.latent_prompt_dim)
        self.prompt_proj = spectral_norm(nn.Linear(self.latent_prompt_dim, skill_dim))
        self.path_decoder = spectral_norm(
            nn.Linear(self.latent_prompt_dim, max_path_length * action_dim)
        )
        self.skill_to_prompt = spectral_norm(nn.Linear(skill_dim, self.latent_prompt_dim))
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

    def _prepare_prompt(self, prompt_vector: torch.Tensor) -> torch.Tensor:
        if prompt_vector.dim() == 1:
            prompt_vector = prompt_vector.unsqueeze(0)
        if prompt_vector.dim() != 2 or prompt_vector.shape[-1] != self.latent_prompt_dim:
            raise ValueError(
                "prompt_vector must be [B, latent_prompt_dim] or [latent_prompt_dim]."
            )
        return self.prompt_norm(prompt_vector)

    def decode_prompt(
        self,
        prompt_vector: torch.Tensor,
        *,
        path_length: int | None = None,
    ) -> torch.Tensor:
        """Decode a latent prompt into an executable action path."""
        prompt_batch = self._prepare_prompt(prompt_vector)
        resolved_length = int(path_length or self.max_path_length)
        if resolved_length > self.max_path_length:
            raise ValueError(
                f"path_length={resolved_length} exceeds max_path_length={self.max_path_length}."
            )
        decoded = torch.tanh(self.path_decoder(prompt_batch))
        decoded = decoded.view(prompt_batch.shape[0], self.max_path_length, self.action_dim)
        return decoded[:, :resolved_length, :]

    def forward(
        self,
        action_path: torch.Tensor | None,
        symbolic_codes: torch.Tensor | None = None,
        *,
        prompt_vector: torch.Tensor | None = None,
        path_length: int | None = None,
    ) -> torch.Tensor:
        prompt_batch = None
        if prompt_vector is not None:
            prompt_batch = self._prepare_prompt(prompt_vector)
        if action_path is None:
            if prompt_batch is None:
                raise ValueError("Either action_path or prompt_vector must be provided.")
            action_path = self.decode_prompt(prompt_vector, path_length=path_length)
        if action_path.dim() == 2:
            action_path = action_path.unsqueeze(0)
        if action_path.dim() != 3:
            raise ValueError("action_path must be [B, P, A] or [P, A].")
        batch_size, path_length, _ = action_path.shape
        if path_length > self.max_path_length:
            raise ValueError(
                f"path_length={path_length} exceeds max_path_length={self.max_path_length}."
            )
        if prompt_batch is not None and prompt_batch.shape[0] != batch_size:
            raise ValueError("prompt_vector batch dimension must match action_path.")
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
        if prompt_batch is not None:
            cls = cls + self.prompt_proj(prompt_batch).unsqueeze(1)
        hidden = torch.cat([cls, tokens], dim=1)
        hidden = hidden + self.position_embed[:, : path_length + 1, :]
        for layer in self.layers:
            hidden = layer(hidden)
        pooled = self.norm(hidden[:, 0, :])
        return self.skill_head(pooled)

    def predict_transition_delta(
        self,
        action_path: torch.Tensor | None,
        symbolic_codes: torch.Tensor | None = None,
        *,
        prompt_vector: torch.Tensor | None = None,
        path_length: int | None = None,
    ) -> torch.Tensor:
        return self.transition_head(
            self.forward(
                action_path,
                symbolic_codes=symbolic_codes,
                prompt_vector=prompt_vector,
                path_length=path_length,
            )
        )

    def infer_prompt(
        self,
        action_path: torch.Tensor,
        symbolic_codes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map an action path back into the latent prompt space used by BODE-GEN."""
        skill = self.forward(action_path, symbolic_codes=symbolic_codes)
        return self.prompt_norm(self.skill_to_prompt(skill))

    def lipschitz_penalty(
        self,
        prompt_vectors: torch.Tensor,
        skill_embeddings: torch.Tensor,
        *,
        max_ratio: float = 1.0,
    ) -> torch.Tensor:
        """Penalize prompt-to-skill mappings whose local stretch exceeds the target ratio."""
        if prompt_vectors.dim() == 1:
            prompt_vectors = prompt_vectors.unsqueeze(0)
        if skill_embeddings.dim() == 1:
            skill_embeddings = skill_embeddings.unsqueeze(0)
        if prompt_vectors.shape[0] <= 1:
            return torch.zeros((), dtype=skill_embeddings.dtype, device=skill_embeddings.device)
        prompt_dist = torch.cdist(prompt_vectors.float(), prompt_vectors.float())
        skill_dist = torch.cdist(skill_embeddings.float(), skill_embeddings.float())
        pair_mask = torch.triu(
            torch.ones_like(prompt_dist, dtype=torch.bool),
            diagonal=1,
        )
        if not pair_mask.any():
            return torch.zeros((), dtype=skill_embeddings.dtype, device=skill_embeddings.device)
        stretch = skill_dist[pair_mask] / prompt_dist[pair_mask].clamp_min(1.0e-6)
        return torch.relu(stretch - max_ratio).mean()

    def compute_loss(
        self,
        action_path: torch.Tensor | None,
        target_delta: torch.Tensor,
        symbolic_codes: torch.Tensor | None = None,
        *,
        prompt_vector: torch.Tensor | None = None,
        path_length: int | None = None,
        lipschitz_weight: float = 0.0,
    ) -> torch.Tensor:
        predicted = self.predict_transition_delta(
            action_path,
            symbolic_codes=symbolic_codes,
            prompt_vector=prompt_vector,
            path_length=path_length,
        )
        loss = torch.nn.functional.mse_loss(predicted, target_delta.detach())
        if prompt_vector is not None and lipschitz_weight > 0.0:
            embedding = self.forward(
                action_path,
                symbolic_codes=symbolic_codes,
                prompt_vector=prompt_vector,
                path_length=path_length,
            )
            loss = loss + (
                float(lipschitz_weight)
                * self.lipschitz_penalty(prompt_vector, embedding)
            )
        return loss

    def encode_candidate(self, candidate: LatentSkillCandidate) -> torch.Tensor:
        return self.forward(
            candidate.action_path,
            symbolic_codes=candidate.symbolic_codes,
            prompt_vector=candidate.prompt_vector,
            path_length=int(candidate.action_path.shape[0]),
        )


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

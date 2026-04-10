"""Learned retrieval relevance scoring for Chamelia memory."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_cosine_similarity(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine similarity with broadcasted batch dimensions."""
    left_norm = left / left.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    right_norm = right / right.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    return (left_norm * right_norm).sum(dim=-1)


class MemoryRelevanceScorer(nn.Module):
    """Learn a relevance score over an explicitly retrieved memory shortlist."""

    def __init__(
        self,
        embed_dim: int = 512,
        posture_dim: int = 16,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.posture_dim = posture_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.summary_proj = nn.Linear(embed_dim, embed_dim)
        self.posture_query_proj = nn.Linear(posture_dim, embed_dim)
        self.posture_memory_proj = nn.Linear(posture_dim, embed_dim)
        self.feature_head = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query_key: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_summaries: torch.Tensor,
        memory_quality: torch.Tensor | None = None,
        query_posture: torch.Tensor | None = None,
        memory_postures: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score each shortlisted memory item.

        Args:
            query_key: Current latent state [B, D].
            memory_keys: Retrieved latent keys [B, M, D].
            memory_summaries: Retrieved summary tokens [B, M, D].
            memory_quality: Optional outcome-quality prior [B, M].
            query_posture: Optional current posture query [B, P].
            memory_postures: Optional retrieved selected postures [B, M, P].

        Returns:
            Dict containing:
                - ``scores``: learned relevance logits [B, M]
                - ``weights``: softmax over scores [B, M]
                - ``features``: explicit scalar feature stack [B, M, 5]
        """
        if query_key.dim() != 2 or memory_keys.dim() != 3 or memory_summaries.dim() != 3:
            raise ValueError(
                "query_key must be [B, D], memory_keys [B, M, D], and memory_summaries [B, M, D]."
            )

        query_key_expanded = query_key.unsqueeze(1).expand_as(memory_keys)
        key_similarity = _safe_cosine_similarity(query_key_expanded, memory_keys)
        summary_similarity = _safe_cosine_similarity(query_key.unsqueeze(1).expand_as(memory_summaries), memory_summaries)

        projected_query = self.query_proj(query_key).unsqueeze(1)
        projected_summary = self.summary_proj(memory_summaries)
        interaction = (
            projected_query * projected_summary
        ).sum(dim=-1) / math.sqrt(float(self.embed_dim))

        quality_feature = (
            memory_quality
            if memory_quality is not None
            else torch.zeros_like(key_similarity)
        )

        posture_similarity = torch.zeros_like(key_similarity)
        if query_posture is not None and memory_postures is not None:
            if query_posture.dim() != 2 or memory_postures.dim() != 3:
                raise ValueError(
                    "query_posture must be [B, P] and memory_postures must be [B, M, P]."
                )
            projected_posture_query = self.posture_query_proj(query_posture).unsqueeze(1)
            projected_posture_memory = self.posture_memory_proj(memory_postures)
            posture_similarity = _safe_cosine_similarity(
                projected_posture_query.expand_as(projected_posture_memory),
                projected_posture_memory,
            )

        explicit_features = torch.stack(
            [
                key_similarity,
                summary_similarity,
                quality_feature,
                posture_similarity,
                interaction,
            ],
            dim=-1,
        )
        scores = self.feature_head(explicit_features).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return {
            "scores": scores,
            "weights": weights,
            "features": explicit_features,
        }


class ProceduralRelevanceScorer(nn.Module):
    """Learn to rerank retrieved procedural skills from a shortlist."""

    def __init__(
        self,
        embed_dim: int = 512,
        retrieval_dim: int = 512,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.retrieval_dim = retrieval_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.query_retrieval_proj = nn.Linear(embed_dim, retrieval_dim)
        self.skill_proj = nn.Linear(embed_dim, embed_dim)
        self.retrieval_proj = nn.Linear(retrieval_dim, retrieval_dim)
        self.feature_head = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        query_latent: torch.Tensor,
        skill_embeddings: torch.Tensor,
        retrieval_vectors: torch.Tensor,
        retrieval_similarity: torch.Tensor,
        confidence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score a procedural skill shortlist.

        Args:
            query_latent: Query latent [B, D].
            skill_embeddings: Dense skill embeddings [B, K, D].
            retrieval_vectors: Retrieval vectors [B, K, R].
            retrieval_similarity: Base retrieval similarity [B, K].
            confidence: Stored skill confidence [B, K].

        Returns:
            Dict containing:
                - ``scores``: learned relevance logits [B, K]
                - ``weights``: softmax over scores [B, K]
                - ``features``: explicit scalar features [B, K, 5]
        """
        if query_latent.dim() != 2:
            raise ValueError("query_latent must be [B, D].")
        if skill_embeddings.dim() != 3 or retrieval_vectors.dim() != 3:
            raise ValueError("skill_embeddings and retrieval_vectors must be [B, K, *].")
        if retrieval_similarity.dim() != 2 or confidence.dim() != 2:
            raise ValueError("retrieval_similarity and confidence must be [B, K].")
        if skill_embeddings.shape[:2] != retrieval_vectors.shape[:2]:
            raise ValueError("skill_embeddings and retrieval_vectors must share [B, K].")

        query_dense = self.query_proj(query_latent).unsqueeze(1)
        skill_dense = self.skill_proj(skill_embeddings)
        dense_similarity = _safe_cosine_similarity(
            query_dense.expand_as(skill_dense),
            skill_dense,
        )

        query_sparse = self.query_retrieval_proj(query_latent).unsqueeze(1)
        skill_sparse = self.retrieval_proj(retrieval_vectors)
        sparse_similarity = _safe_cosine_similarity(
            query_sparse.expand_as(skill_sparse),
            skill_sparse,
        )

        interaction = (
            query_dense.expand_as(skill_dense) * skill_dense
        ).sum(dim=-1) / math.sqrt(float(self.embed_dim))
        explicit_features = torch.stack(
            [
                dense_similarity,
                sparse_similarity,
                retrieval_similarity,
                confidence,
                interaction,
            ],
            dim=-1,
        )
        scores = self.feature_head(explicit_features).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return {
            "scores": scores,
            "weights": weights,
            "features": explicit_features,
        }


def compute_retrieval_relevance_loss(
    learned_scores: torch.Tensor,
    retrieved_postures: torch.Tensor,
    selected_posture: torch.Tensor,
    base_quality_scores: torch.Tensor,
    realized_ic: torch.Tensor,
    temperature: float = 0.25,
) -> torch.Tensor | None:
    """Directly supervise retrieval relevance from delayed outcomes.

    Args:
        learned_scores: Learned reranker logits [B, M].
        retrieved_postures: Retrieved posture bank [B, M, P].
        selected_posture: Posture ultimately selected for the episode [B, P].
        base_quality_scores: Stored shortlist quality priors [B, M].
        realized_ic: Realized intrinsic cost [B].
        temperature: Softmax temperature.

    Returns:
        Scalar loss or ``None`` when inputs are malformed.
    """
    if (
        learned_scores.dim() != 2
        or retrieved_postures.dim() != 3
        or selected_posture.dim() != 2
        or base_quality_scores.dim() != 2
    ):
        return None

    common_count = min(
        learned_scores.shape[1],
        base_quality_scores.shape[1],
        retrieved_postures.shape[1],
    )
    if common_count < 1:
        return None

    learned_scores = learned_scores[:, :common_count]
    base_quality_scores = base_quality_scores[:, :common_count]
    retrieved_postures = retrieved_postures[:, :common_count, :]
    realized_ic = realized_ic.flatten()
    if realized_ic.shape[0] != learned_scores.shape[0]:
        return None

    posture_alignment = F.cosine_similarity(
        selected_posture.unsqueeze(1).expand_as(retrieved_postures),
        retrieved_postures,
        dim=-1,
    )
    target_scores = base_quality_scores + posture_alignment
    temperature = max(float(temperature), 1.0e-4)
    target_distribution = F.softmax(target_scores / temperature, dim=1).detach()
    predicted_log_probs = F.log_softmax(learned_scores / temperature, dim=1)
    current_quality = -realized_ic
    best_reference_quality = base_quality_scores.max(dim=1).values.detach()
    regret = F.relu(best_reference_quality - current_quality)
    weight = 0.25 + regret
    return (-(target_distribution * predicted_log_probs).sum(dim=1) * weight).mean()


def compute_procedural_reranker_loss(
    learned_scores: torch.Tensor,
    target_scores: torch.Tensor,
    *,
    temperature: float = 0.25,
) -> torch.Tensor | None:
    """Train the procedural reranker from sleep-time shortlist scores."""
    if learned_scores.dim() != 2 or target_scores.dim() != 2:
        return None
    common_width = min(learned_scores.shape[1], target_scores.shape[1])
    if common_width < 2:
        return None
    learned_scores = learned_scores[:, :common_width]
    target_scores = target_scores[:, :common_width]
    temperature = max(float(temperature), 1.0e-4)
    target_distribution = F.softmax(target_scores.detach() / temperature, dim=1)
    predicted_log_probs = F.log_softmax(learned_scores / temperature, dim=1)
    return -(target_distribution * predicted_log_probs).sum(dim=1).mean()

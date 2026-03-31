"""Latent episodic memory for Chamelia."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EpisodeRecord:
    """Latent-indexed episode record.

    Attributes are stored as:
        key: [D]
        action: [A]
        ctx_tokens: [C, D]
        outcome_key: Optional [D]
    """

    key: torch.Tensor
    action: torch.Tensor
    ctx_tokens: torch.Tensor
    ic_at_decision: float
    ic_realized: float | None
    tc_predicted: float
    outcome_key: torch.Tensor | None
    step: int
    domain_name: str


class LatentMemory:
    """CPU-backed latent key-value episodic memory."""

    def __init__(
        self,
        embed_dim: int = 512,
        max_episodes: int = 10000,
        retrieval_k: int = 8,
        device: str = "cpu",
    ) -> None:
        """Initialize latent memory.

        Args:
            embed_dim: Key embedding dimension D.
            max_episodes: Maximum circular-buffer capacity.
            retrieval_k: Default retrieval count K.
            device: Storage device for keys, typically "cpu".

        Returns:
            None.
        """
        self.embed_dim = embed_dim
        self.max_episodes = max_episodes
        self.retrieval_k = retrieval_k
        self.device = device
        self.keys = torch.zeros(max_episodes, embed_dim, device=device, dtype=torch.float32)
        self.records: list[EpisodeRecord] = []
        self.size = 0
        self.head = 0

    def store(self, record: EpisodeRecord) -> int:
        """Store an episode record in the circular buffer.

        Args:
            record: EpisodeRecord with key [D], action [A], and ctx_tokens [C, D].

        Returns:
            Integer storage index.
        """
        idx = self.head
        self.keys[idx] = record.key.detach().cpu()
        stored_record = EpisodeRecord(
            key=record.key.detach().cpu(),
            action=record.action.detach().cpu(),
            ctx_tokens=record.ctx_tokens.detach().cpu(),
            ic_at_decision=record.ic_at_decision,
            ic_realized=record.ic_realized,
            tc_predicted=record.tc_predicted,
            outcome_key=None if record.outcome_key is None else record.outcome_key.detach().cpu(),
            step=record.step,
            domain_name=record.domain_name,
        )
        if len(self.records) < self.max_episodes:
            self.records.append(stored_record)
        else:
            self.records[idx] = stored_record
        self.size = min(self.size + 1, self.max_episodes)
        self.head = (self.head + 1) % self.max_episodes
        return idx

    def fill_outcome(self, idx: int, ic_realized: float, outcome_key: torch.Tensor) -> None:
        """Fill in delayed outcome information for a stored episode.

        Args:
            idx: Integer record index.
            ic_realized: Realized intrinsic cost scalar.
            outcome_key: Outcome latent key tensor of shape [D].

        Returns:
            None.
        """
        self.records[idx].ic_realized = ic_realized
        self.records[idx].outcome_key = outcome_key.detach().cpu()

    def retrieve(
        self,
        query_key: torch.Tensor,
        k: int | None = None,
    ) -> tuple[torch.Tensor | None, list[list[EpisodeRecord]]]:
        """Retrieve the top-k nearest episodes by cosine similarity.

        Args:
            query_key: Query tensor of shape [D] or [B, D].
            k: Optional retrieval override.

        Returns:
            Tuple of:
                - retrieved_keys: [B, k, D] or None if memory empty
                - episode_lists: Python lists of EpisodeRecord, one list per batch element
        """
        k = k or self.retrieval_k
        if self.size == 0:
            return None, []

        if query_key.dim() == 1:
            query_key = query_key.unsqueeze(0)
        stored = self.keys[: self.size]

        q_norm = F.normalize(query_key.detach().cpu(), dim=-1)
        k_norm = F.normalize(stored, dim=-1)
        sims = torch.mm(q_norm, k_norm.T)
        top_k_indices = sims.topk(min(k, self.size), dim=-1).indices
        retrieved_keys = stored[top_k_indices]
        episode_lists = [
            [self.records[idx.item()] for idx in top_k_indices[b]]
            for b in range(query_key.shape[0])
        ]
        return retrieved_keys, episode_lists

    def get_critic_training_pairs(
        self,
        min_outcome_delay: int = 1,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return state keys and realized intrinsic costs for critic training.

        Args:
            min_outcome_delay: Unused placeholder for API compatibility.

        Returns:
            Tuple of:
                - state_keys: [N, D] or None
                - realized_ics: [N] or None
        """
        _ = min_outcome_delay
        valid = [r for r in self.records[: self.size] if r.ic_realized is not None]
        if not valid:
            return None, None
        keys = torch.stack([r.key for r in valid])
        ics = torch.tensor([float(r.ic_realized) for r in valid], dtype=torch.float32)
        return keys, ics

    def get_jepa_transition_pairs(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Return JEPA transition triples.

        Args:
            None.

        Returns:
            Tuple of:
                - z_t: [N, D] or None
                - actions: [N, A] or None
                - z_tH: [N, D] or None
        """
        valid = [r for r in self.records[: self.size] if r.outcome_key is not None]
        if not valid:
            return None, None, None
        z_t = torch.stack([r.key for r in valid])
        actions = torch.stack([r.action for r in valid])
        z_tH = torch.stack([r.outcome_key for r in valid if r.outcome_key is not None])
        return z_t, actions, z_tH


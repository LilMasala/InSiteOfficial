"""Latent episodic memory for Chamelia."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
    record_id: int = 0
    model_version: str | None = None
    candidate_postures: torch.Tensor | None = None
    selected_posture: torch.Tensor | None = None
    candidate_reasoning_states: torch.Tensor | None = None
    candidate_paths: torch.Tensor | None = None
    selected_path: torch.Tensor | None = None
    candidate_actions: torch.Tensor | None = None
    candidate_ic: torch.Tensor | None = None
    candidate_tc: torch.Tensor | None = None
    candidate_total: torch.Tensor | None = None
    candidate_terminal_latents: torch.Tensor | None = None
    selected_candidate_idx: int | None = None
    retrieval_trace: tuple[RetrievalTraceStep, ...] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class RetrievalTraceStep:
    """Stored retrieval decision for one planner round."""

    query_key: torch.Tensor
    memory_keys: torch.Tensor
    memory_summaries: torch.Tensor
    base_quality_scores: torch.Tensor
    query_posture: torch.Tensor | None = None
    memory_postures: torch.Tensor | None = None
    base_scores: torch.Tensor | None = None
    relevance_scores: torch.Tensor | None = None
    relevance_weights: torch.Tensor | None = None


@dataclass
class RetrievalReplayExample:
    """Offline replay sample for retrieval-reranker training."""

    query_key: torch.Tensor
    memory_keys: torch.Tensor
    memory_summaries: torch.Tensor
    base_quality_scores: torch.Tensor
    selected_posture: torch.Tensor
    realized_ic: float
    query_posture: torch.Tensor | None = None
    memory_postures: torch.Tensor | None = None


def _cpu_optional_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
    """Detach and move optional tensors to CPU."""
    if value is None:
        return None
    return value.detach().cpu()


def _cpu_optional_retrieval_trace(
    trace: tuple[RetrievalTraceStep, ...] | None,
) -> tuple[RetrievalTraceStep, ...] | None:
    """Detach and move an optional retrieval trace to CPU."""
    if trace is None:
        return None
    stored_trace: list[RetrievalTraceStep] = []
    for step in trace:
        stored_trace.append(
            RetrievalTraceStep(
                query_key=step.query_key.detach().cpu(),
                memory_keys=step.memory_keys.detach().cpu(),
                memory_summaries=step.memory_summaries.detach().cpu(),
                base_quality_scores=step.base_quality_scores.detach().cpu(),
                query_posture=_cpu_optional_tensor(step.query_posture),
                memory_postures=_cpu_optional_tensor(step.memory_postures),
                base_scores=_cpu_optional_tensor(step.base_scores),
                relevance_scores=_cpu_optional_tensor(step.relevance_scores),
                relevance_weights=_cpu_optional_tensor(step.relevance_weights),
            )
        )
    return tuple(stored_trace)


def _lift_to_embed(value: torch.Tensor | None, embed_dim: int) -> torch.Tensor | None:
    """Flatten and pad/truncate a tensor to the shared embed width."""
    if value is None:
        return None
    flat = value.detach().float().reshape(-1)
    if flat.numel() >= embed_dim:
        return flat[:embed_dim]
    lifted = torch.zeros(embed_dim, dtype=torch.float32)
    lifted[: flat.numel()] = flat
    return lifted


def _normalize_vector(value: torch.Tensor) -> torch.Tensor:
    """Normalize a 1D vector without producing NaNs for zero vectors."""
    norm = value.norm(p=2)
    if float(norm.item()) == 0.0:
        return value
    return value / norm


def _record_quality_score(record: EpisodeRecord) -> float:
    """Return an outcome-oriented score where larger means better."""
    realized_cost = (
        float(record.ic_realized)
        if record.ic_realized is not None
        else float(record.ic_at_decision + record.tc_predicted)
    )
    return -realized_cost


def _normalize_score_vector(values: torch.Tensor) -> torch.Tensor:
    """Squash a score vector into a comparable range without changing its ordering."""
    if values.numel() == 0:
        return values
    centered = values - values.mean()
    scale = centered.std(unbiased=False)
    if float(scale.item()) < 1.0e-6:
        return torch.zeros_like(values)
    return torch.tanh(centered / scale)


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
        self._next_record_id = 0
        self._id_to_slot: dict[int, int] = {}

    def store(self, record: EpisodeRecord) -> int:
        """Store an episode record in the circular buffer.

        Args:
            record: EpisodeRecord with key [D], action [A], and ctx_tokens [C, D].

        Returns:
            Stable integer record ID. This ID remains valid until the slot is
            overwritten by a future store when the buffer is full. Callers must
            use this ID (not a raw slot index) when calling fill_outcome.
        """
        idx = self.head
        record_id = self._next_record_id
        self._next_record_id += 1

        # If overwriting an existing slot, evict its stable ID from the lookup.
        if len(self.records) >= self.max_episodes:
            evicted_id = self.records[idx].record_id
            self._id_to_slot.pop(evicted_id, None)

        self.keys[idx] = record.key.detach().cpu()
        stored_record = EpisodeRecord(
            key=record.key.detach().cpu(),
            action=record.action.detach().cpu(),
            ctx_tokens=record.ctx_tokens.detach().cpu(),
            ic_at_decision=record.ic_at_decision,
            ic_realized=record.ic_realized,
            tc_predicted=record.tc_predicted,
            outcome_key=_cpu_optional_tensor(record.outcome_key),
            step=record.step,
            domain_name=record.domain_name,
            record_id=record_id,
            model_version=record.model_version,
            candidate_postures=_cpu_optional_tensor(record.candidate_postures),
            selected_posture=_cpu_optional_tensor(record.selected_posture),
            candidate_reasoning_states=_cpu_optional_tensor(record.candidate_reasoning_states),
            candidate_paths=_cpu_optional_tensor(record.candidate_paths),
            selected_path=_cpu_optional_tensor(record.selected_path),
            candidate_actions=_cpu_optional_tensor(record.candidate_actions),
            candidate_ic=_cpu_optional_tensor(record.candidate_ic),
            candidate_tc=_cpu_optional_tensor(record.candidate_tc),
            candidate_total=_cpu_optional_tensor(record.candidate_total),
            candidate_terminal_latents=_cpu_optional_tensor(record.candidate_terminal_latents),
            selected_candidate_idx=record.selected_candidate_idx,
            retrieval_trace=_cpu_optional_retrieval_trace(record.retrieval_trace),
            metadata=dict(record.metadata) if record.metadata is not None else None,
        )
        if len(self.records) < self.max_episodes:
            self.records.append(stored_record)
        else:
            self.records[idx] = stored_record
        self._id_to_slot[record_id] = idx
        self.size = min(self.size + 1, self.max_episodes)
        self.head = (self.head + 1) % self.max_episodes
        return record_id

    def fill_outcome(self, record_id: int, ic_realized: float, outcome_key: torch.Tensor) -> bool:
        """Fill in delayed outcome information for a stored episode.

        If the record has been evicted (buffer wrapped past it), this is a
        silent no-op and returns False. Callers should not use raw slot indices
        here — only the stable record_id returned by store().

        Args:
            record_id: Stable record ID returned by store().
            ic_realized: Realized intrinsic cost scalar.
            outcome_key: Outcome latent key tensor of shape [D].

        Returns:
            True if the record was found and updated, False if it was evicted.
        """
        slot = self._id_to_slot.get(record_id)
        if slot is None:
            return False
        self.records[slot].ic_realized = ic_realized
        self.records[slot].outcome_key = outcome_key.detach().cpu()
        return True

    def get_record_by_id(self, record_id: int) -> EpisodeRecord | None:
        """Return the record for a given stable record_id, or None if evicted.

        Args:
            record_id: Stable record ID returned by store().

        Returns:
            EpisodeRecord or None.
        """
        slot = self._id_to_slot.get(record_id)
        if slot is None:
            return None
        return self.records[slot]

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
        retrieved_keys, episode_lists, _ = self.retrieve_scored(query_key, k=k)
        return retrieved_keys, episode_lists

    def retrieve_scored(
        self,
        query_key: torch.Tensor,
        k: int | None = None,
        query_posture: torch.Tensor | None = None,
        quality_weight: float = 0.15,
        posture_weight: float = 0.25,
    ) -> tuple[torch.Tensor | None, list[list[EpisodeRecord]], torch.Tensor | None]:
        """Retrieve top-k episodes using state similarity plus optional quality/posture reranking."""
        k = k or self.retrieval_k
        if self.size == 0:
            return None, [], None

        if query_key.dim() == 1:
            query_key = query_key.unsqueeze(0)
        stored = self.keys[: self.size]

        q_norm = F.normalize(query_key.detach().cpu(), dim=-1)
        k_norm = F.normalize(stored, dim=-1)
        combined_scores = torch.mm(q_norm, k_norm.T)

        if quality_weight != 0.0:
            quality_scores = torch.tensor(
                [_record_quality_score(record) for record in self.records[: self.size]],
                dtype=torch.float32,
            )
            combined_scores = combined_scores + quality_weight * _normalize_score_vector(
                quality_scores
            ).unsqueeze(0)

        if query_posture is not None and posture_weight != 0.0:
            if query_posture.dim() == 1:
                query_posture = query_posture.unsqueeze(0)
            posture_dim = query_posture.shape[-1]
            posture_bank = torch.zeros(self.size, posture_dim, dtype=torch.float32)
            valid_mask = torch.zeros(self.size, dtype=torch.bool)
            for idx, record in enumerate(self.records[: self.size]):
                if (
                    record.selected_posture is not None
                    and record.selected_posture.shape[-1] == posture_dim
                ):
                    posture_bank[idx] = record.selected_posture.float()
                    valid_mask[idx] = True
            if valid_mask.any():
                posture_query = F.normalize(query_posture.detach().cpu(), dim=-1)
                posture_bank = F.normalize(posture_bank, dim=-1)
                posture_scores = posture_query @ posture_bank.T
                posture_scores = posture_scores.masked_fill(~valid_mask.unsqueeze(0), 0.0)
                combined_scores = combined_scores + posture_weight * posture_scores

        top_k_indices = combined_scores.topk(min(k, self.size), dim=-1).indices
        retrieved_keys = stored[top_k_indices]
        episode_lists = [
            [self.records[idx.item()] for idx in top_k_indices[b]]
            for b in range(query_key.shape[0])
        ]
        retrieved_scores = combined_scores.gather(1, top_k_indices)
        return retrieved_keys, episode_lists, retrieved_scores

    def summarize_retrieved_postures(
        self,
        episode_lists: list[list[EpisodeRecord]],
        posture_dim: int,
        retrieval_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Convert retrieved episodes into a posture bank and outcome-based scores."""
        if posture_dim < 1 or not episode_lists:
            return None, None

        max_count = max(
            sum(
                1
                for record in episodes
                if record.selected_posture is not None
                and record.selected_posture.shape[-1] == posture_dim
            )
            for episodes in episode_lists
        )
        if max_count == 0:
            return None, None

        posture_bank = torch.zeros(
            len(episode_lists),
            max_count,
            posture_dim,
            dtype=torch.float32,
        )
        posture_scores = torch.full(
            (len(episode_lists), max_count),
            float("-inf"),
            dtype=torch.float32,
        )
        for batch_idx, episodes in enumerate(episode_lists):
            insert_idx = 0
            for episode_idx, record in enumerate(episodes):
                if record.selected_posture is None:
                    continue
                if record.selected_posture.shape[-1] != posture_dim:
                    continue
                posture_bank[batch_idx, insert_idx] = record.selected_posture.float()
                if retrieval_scores is not None and episode_idx < retrieval_scores.shape[1]:
                    posture_scores[batch_idx, insert_idx] = retrieval_scores[batch_idx, episode_idx]
                else:
                    posture_scores[batch_idx, insert_idx] = _record_quality_score(record)
                insert_idx += 1
                if insert_idx >= max_count:
                    break

        if not torch.isfinite(posture_scores).any():
            return None, None
        return posture_bank, posture_scores

    def summarize_retrieved_episodes(
        self,
        episode_lists: list[list[EpisodeRecord]],
        retrieval_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Convert retrieved episodes into richer summary tokens and outcome-quality scores (Vectorized)."""
        if not episode_lists:
            return None, None

        B = len(episode_lists)
        max_count = max(len(episodes) for episodes in episode_lists)
        if max_count == 0:
            return None, None

        # Pre-allocate tensors for up to 6 components: [Key, Ctx, Action, Posture, Outcome, Diff]
        # Shape: [Batch, K_Retrievals, 6_Components, Embed_Dim]
        comp_vals = torch.zeros(B, max_count, 6, self.embed_dim, dtype=torch.float32)
        comp_mask = torch.zeros(B, max_count, 6, 1, dtype=torch.float32)
        summary_scores = torch.full((B, max_count), float("-inf"), dtype=torch.float32)

        for b, episodes in enumerate(episode_lists):
            for k, record in enumerate(episodes[:max_count]):
                # 1. Key
                comp_vals[b, k, 0] = record.key.float()
                comp_mask[b, k, 0] = 1.0
                
                # 2. Ctx tokens (mean)
                comp_vals[b, k, 1] = record.ctx_tokens.float().mean(dim=0)
                comp_mask[b, k, 1] = 1.0
                
                # 3. Action
                a_flat = record.action.detach().float().reshape(-1)
                a_len = min(a_flat.numel(), self.embed_dim)
                comp_vals[b, k, 2, :a_len] = a_flat[:a_len]
                comp_mask[b, k, 2] = 1.0
                
                # 4. Posture
                if record.selected_posture is not None:
                    p_flat = record.selected_posture.detach().float().reshape(-1)
                    p_len = min(p_flat.numel(), self.embed_dim)
                    comp_vals[b, k, 3, :p_len] = p_flat[:p_len]
                    comp_mask[b, k, 3] = 1.0
                    
                # 5 & 6. Outcome and Difference
                if record.outcome_key is not None:
                    out_key = record.outcome_key.float()
                    comp_vals[b, k, 4] = out_key
                    comp_mask[b, k, 4] = 1.0
                    comp_vals[b, k, 5] = out_key - record.key.float()
                    comp_mask[b, k, 5] = 1.0

                # Determine Score
                if retrieval_scores is not None and k < retrieval_scores.shape[1]:
                    summary_scores[b, k] = retrieval_scores[b, k]
                else:
                    summary_scores[b, k] = _record_quality_score(record)

        # ------------------------------------------------------------------
        # Vectorized Math Operations (replaces inner-loop stacking/normalization)
        # ------------------------------------------------------------------
        
        # 1. Normalize individual components (safe against div-by-zero)
        norms = comp_vals.norm(p=2, dim=-1, keepdim=True)
        norms = torch.where(norms == 0.0, torch.ones_like(norms), norms)
        comp_vals_normalized = comp_vals / norms
        
        # 2. Compute mean across the valid components
        sum_components = (comp_vals_normalized * comp_mask).sum(dim=2)
        valid_counts = comp_mask.sum(dim=2).clamp_min(1.0)
        summary_mean = sum_components / valid_counts
        
        # 3. Final normalization of the resulting summary token
        final_norms = summary_mean.norm(p=2, dim=-1, keepdim=True)
        final_norms = torch.where(final_norms == 0.0, torch.ones_like(final_norms), final_norms)
        summary_tokens = summary_mean / final_norms

        if not torch.isfinite(summary_scores).any():
            return None, None
            
        return summary_tokens, summary_scores

    def get_critic_training_pairs(
        self,
        min_outcome_delay: int = 1,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Return future latents, stored context, and realized intrinsic costs for critic training.

        Args:
            min_outcome_delay: Unused placeholder for API compatibility.

        Returns:
            Tuple of:
                - future_keys: [N, D] or None
                - ctx_tokens: [N, C, D] or None
                - realized_ics: [N] or None
        """
        _ = min_outcome_delay
        valid = [
            r
            for r in self.records[: self.size]
            if r.ic_realized is not None and r.outcome_key is not None
        ]
        if not valid:
            return None, None, None
        keys = torch.stack([r.outcome_key for r in valid if r.outcome_key is not None])
        ctx = torch.stack([r.ctx_tokens for r in valid])
        ics = torch.tensor([float(r.ic_realized) for r in valid], dtype=torch.float32)
        return keys, ctx, ics

    def get_world_model_training_pairs(
        self,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Return stored transitions for world-model training."""
        valid = [r for r in self.records[: self.size] if r.outcome_key is not None]
        if not valid:
            return None, None, None, None, None
        z_t = torch.stack([r.key for r in valid])
        actions = torch.stack([r.action for r in valid])
        ctx_tokens = torch.stack([r.ctx_tokens for r in valid])
        z_tH = torch.stack([r.outcome_key for r in valid if r.outcome_key is not None])
        selected_postures = None
        if all(r.selected_posture is not None for r in valid):
            selected_postures = torch.stack([r.selected_posture for r in valid if r.selected_posture is not None])
        return z_t, actions, ctx_tokens, z_tH, selected_postures

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
        z_t, actions, _, z_tH, _ = self.get_world_model_training_pairs()
        if z_t is None or actions is None or z_tH is None:
            return None, None, None
        return z_t, actions, z_tH

    def get_retrieval_training_examples(self) -> list[RetrievalReplayExample]:
        """Return replayable retrieval decisions with realized outcomes."""
        examples: list[RetrievalReplayExample] = []
        for record in self.records[: self.size]:
            if record.ic_realized is None or record.selected_posture is None:
                continue
            if record.retrieval_trace is None:
                continue
            for step in record.retrieval_trace:
                if step.memory_keys.numel() == 0 or step.memory_summaries.numel() == 0:
                    continue
                if step.base_quality_scores.numel() == 0:
                    continue
                examples.append(
                    RetrievalReplayExample(
                        query_key=step.query_key.float(),
                        memory_keys=step.memory_keys.float(),
                        memory_summaries=step.memory_summaries.float(),
                        base_quality_scores=step.base_quality_scores.float(),
                        selected_posture=record.selected_posture.float(),
                        realized_ic=float(record.ic_realized),
                        query_posture=(
                            step.query_posture.float()
                            if step.query_posture is not None
                            else None
                        ),
                        memory_postures=(
                            step.memory_postures.float()
                            if step.memory_postures is not None
                            else None
                        ),
                    )
                )
        return examples

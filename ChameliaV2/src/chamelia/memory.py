"""Latent episodic memory for Chamelia."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

import torch
import torch.nn.functional as F

from src.chamelia.cognitive.representation import InformationOrderedBottleneck


def _detach_to_device(value: torch.Tensor | None, device: torch.device | str) -> torch.Tensor | None:
    """Detach and move optional tensors to the target memory device."""
    if value is None:
        return None
    return value.detach().to(device)


def _detach_trace_to_device(
    trace: tuple[RetrievalTraceStep, ...] | None,
    device: torch.device | str,
) -> tuple[RetrievalTraceStep, ...] | None:
    """Detach and move an optional retrieval trace to the target device."""
    if trace is None:
        return None
    stored_trace: list[RetrievalTraceStep] = []
    for step in trace:
        stored_trace.append(
            RetrievalTraceStep(
                query_key=step.query_key.detach().to(device),
                memory_keys=step.memory_keys.detach().to(device),
                memory_summaries=step.memory_summaries.detach().to(device),
                base_quality_scores=step.base_quality_scores.detach().to(device),
                query_posture=_detach_to_device(step.query_posture, device),
                memory_postures=_detach_to_device(step.memory_postures, device),
                base_scores=_detach_to_device(step.base_scores, device),
                relevance_scores=_detach_to_device(step.relevance_scores, device),
                relevance_weights=_detach_to_device(step.relevance_weights, device),
            )
        )
    return tuple(stored_trace)


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
    mcts_trace: dict[str, Any] | None = None
    skill_trace: tuple[int, ...] | None = None
    goal_key: torch.Tensor | None = None
    domain_cluster_id: int | None = None
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


def _lift_to_embed(value: torch.Tensor | None, embed_dim: int, device: torch.device | str) -> torch.Tensor | None:
    """Flatten and pad/truncate a tensor to the shared embed width."""
    if value is None:
        return None
    flat = value.detach().float().reshape(-1)
    if flat.numel() >= embed_dim:
        return flat[:embed_dim]
    lifted = torch.zeros(embed_dim, dtype=torch.float32, device=device)
    lifted[: flat.numel()] = flat
    return lifted


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


def _serialize_retrieval_trace_step(step: RetrievalTraceStep) -> dict[str, Any]:
    """Serialize one retrieval trace step for checkpointing."""
    return {
        "query_key": step.query_key.detach().cpu(),
        "memory_keys": step.memory_keys.detach().cpu(),
        "memory_summaries": step.memory_summaries.detach().cpu(),
        "base_quality_scores": step.base_quality_scores.detach().cpu(),
        "query_posture": _detach_to_device(step.query_posture, "cpu"),
        "memory_postures": _detach_to_device(step.memory_postures, "cpu"),
        "base_scores": _detach_to_device(step.base_scores, "cpu"),
        "relevance_scores": _detach_to_device(step.relevance_scores, "cpu"),
        "relevance_weights": _detach_to_device(step.relevance_weights, "cpu"),
    }


def _deserialize_retrieval_trace_step(payload: dict[str, Any], device: torch.device | str) -> RetrievalTraceStep:
    """Restore one retrieval trace step from a checkpoint payload."""
    return RetrievalTraceStep(
        query_key=payload["query_key"].detach().to(device),
        memory_keys=payload["memory_keys"].detach().to(device),
        memory_summaries=payload["memory_summaries"].detach().to(device),
        base_quality_scores=payload["base_quality_scores"].detach().to(device),
        query_posture=_detach_to_device(payload.get("query_posture"), device),
        memory_postures=_detach_to_device(payload.get("memory_postures"), device),
        base_scores=_detach_to_device(payload.get("base_scores"), device),
        relevance_scores=_detach_to_device(payload.get("relevance_scores"), device),
        relevance_weights=_detach_to_device(payload.get("relevance_weights"), device),
    )


def _serialize_episode_record(record: EpisodeRecord) -> dict[str, Any]:
    """Serialize one episode record for checkpointing."""
    return {
        "key": record.key.detach().cpu(),
        "action": record.action.detach().cpu(),
        "ctx_tokens": record.ctx_tokens.detach().cpu(),
        "ic_at_decision": float(record.ic_at_decision),
        "ic_realized": None if record.ic_realized is None else float(record.ic_realized),
        "tc_predicted": float(record.tc_predicted),
        "outcome_key": _detach_to_device(record.outcome_key, "cpu"),
        "step": int(record.step),
        "domain_name": record.domain_name,
        "record_id": int(record.record_id),
        "model_version": record.model_version,
        "candidate_postures": _detach_to_device(record.candidate_postures, "cpu"),
        "selected_posture": _detach_to_device(record.selected_posture, "cpu"),
        "candidate_reasoning_states": _detach_to_device(record.candidate_reasoning_states, "cpu"),
        "candidate_paths": _detach_to_device(record.candidate_paths, "cpu"),
        "selected_path": _detach_to_device(record.selected_path, "cpu"),
        "candidate_actions": _detach_to_device(record.candidate_actions, "cpu"),
        "candidate_ic": _detach_to_device(record.candidate_ic, "cpu"),
        "candidate_tc": _detach_to_device(record.candidate_tc, "cpu"),
        "candidate_total": _detach_to_device(record.candidate_total, "cpu"),
        "candidate_terminal_latents": _detach_to_device(record.candidate_terminal_latents, "cpu"),
        "selected_candidate_idx": record.selected_candidate_idx,
        "retrieval_trace": None
        if record.retrieval_trace is None
        else tuple(_serialize_retrieval_trace_step(step) for step in record.retrieval_trace),
        "mcts_trace": None if record.mcts_trace is None else dict(record.mcts_trace),
        "skill_trace": None if record.skill_trace is None else tuple(record.skill_trace),
        "goal_key": _detach_to_device(record.goal_key, "cpu"),
        "domain_cluster_id": record.domain_cluster_id,
        "metadata": None if record.metadata is None else dict(record.metadata),
    }


def _deserialize_episode_record(payload: dict[str, Any], device: torch.device | str) -> EpisodeRecord:
    """Restore one episode record from a checkpoint payload."""
    retrieval_trace_payload = payload.get("retrieval_trace")
    retrieval_trace = None
    if retrieval_trace_payload is not None:
        retrieval_trace = tuple(
            _deserialize_retrieval_trace_step(step, device)
            for step in retrieval_trace_payload
        )
    return EpisodeRecord(
        key=payload["key"].detach().to(device),
        action=payload["action"].detach().to(device),
        ctx_tokens=payload["ctx_tokens"].detach().to(device),
        ic_at_decision=float(payload["ic_at_decision"]),
        ic_realized=payload.get("ic_realized"),
        tc_predicted=float(payload["tc_predicted"]),
        outcome_key=_detach_to_device(payload.get("outcome_key"), device),
        step=int(payload["step"]),
        domain_name=str(payload["domain_name"]),
        record_id=int(payload["record_id"]),
        model_version=payload.get("model_version"),
        candidate_postures=_detach_to_device(payload.get("candidate_postures"), device),
        selected_posture=_detach_to_device(payload.get("selected_posture"), device),
        candidate_reasoning_states=_detach_to_device(payload.get("candidate_reasoning_states"), device),
        candidate_paths=_detach_to_device(payload.get("candidate_paths"), device),
        selected_path=_detach_to_device(payload.get("selected_path"), device),
        candidate_actions=_detach_to_device(payload.get("candidate_actions"), device),
        candidate_ic=_detach_to_device(payload.get("candidate_ic"), device),
        candidate_tc=_detach_to_device(payload.get("candidate_tc"), device),
        candidate_total=_detach_to_device(payload.get("candidate_total"), device),
        candidate_terminal_latents=_detach_to_device(payload.get("candidate_terminal_latents"), device),
        selected_candidate_idx=payload.get("selected_candidate_idx"),
        retrieval_trace=retrieval_trace,
        mcts_trace=None if payload.get("mcts_trace") is None else dict(payload["mcts_trace"]),
        skill_trace=None if payload.get("skill_trace") is None else tuple(payload["skill_trace"]),
        goal_key=_detach_to_device(payload.get("goal_key"), device),
        domain_cluster_id=payload.get("domain_cluster_id"),
        metadata=None if payload.get("metadata") is None else dict(payload["metadata"]),
    )


class LatentMemory:
    """Device-aware latent key-value episodic memory."""

    def __init__(
        self,
        embed_dim: int = 512,
        max_episodes: int = 10000,
        retrieval_k: int = 8,
        device: str = "cpu",
        iob_encoder: InformationOrderedBottleneck | None = None,
        iob_widths: tuple[int, ...] | None = None,
    ) -> None:
        """Initialize latent memory.

        Args:
            embed_dim: Key embedding dimension D.
            max_episodes: Maximum circular-buffer capacity.
            retrieval_k: Default retrieval count K.
            device: Storage device for keys and values (e.g. "cuda" or "cpu").
        """
        self.embed_dim = embed_dim
        self.max_episodes = max_episodes
        self.retrieval_k = retrieval_k
        self.device = device
        self.iob_encoder = iob_encoder
        self.iob_widths = iob_widths
        self.keys = torch.zeros(max_episodes, embed_dim, device=device, dtype=torch.float32)
        self.ordered_keys = None
        if self.iob_encoder is not None:
            self.ordered_keys = torch.zeros(
                max_episodes,
                self.iob_encoder.bottleneck_dim,
                device=device,
                dtype=torch.float32,
            )
        self.records: list[EpisodeRecord] = []
        self.size = 0
        self.head = 0
        self._next_record_id = 0
        self._id_to_slot: dict[int, int] = {}
        self._lock = threading.RLock()

    def _encode_iob(self, value: torch.Tensor) -> torch.Tensor:
        if self.iob_encoder is None:
            raise RuntimeError("IOB encoder is not configured.")
        encoder_device = next(self.iob_encoder.parameters()).device
        with torch.no_grad():
            encoded = self.iob_encoder(value.detach().to(encoder_device))
        return encoded.to(self.device)

    def _resolve_iob_widths(self) -> tuple[int, ...]:
        if self.iob_encoder is None:
            return ()
        bottleneck_dim = self.iob_encoder.bottleneck_dim
        if self.iob_widths:
            widths = tuple(
                max(1, min(int(width), bottleneck_dim))
                for width in self.iob_widths
            )
        else:
            widths = (
                max(1, bottleneck_dim // 4),
                max(1, bottleneck_dim // 2),
                bottleneck_dim,
            )
        return tuple(sorted(set(widths)))

    def _iob_shortlist_mask(self, query_embedding: torch.Tensor, k: int) -> torch.Tensor:
        if self.ordered_keys is None:
            raise RuntimeError("IOB storage is not initialized.")
        widths = self._resolve_iob_widths()
        batch_size = query_embedding.shape[0]
        shortlist_mask = torch.zeros(batch_size, self.size, dtype=torch.bool, device=self.device)
        if self.size == 0:
            return shortlist_mask
        for batch_idx in range(batch_size):
            candidate_indices = torch.arange(self.size, device=self.device)
            for stage_idx, width in enumerate(widths):
                if candidate_indices.numel() == 0:
                    break
                query_slice = F.normalize(
                    query_embedding[batch_idx : batch_idx + 1, :width],
                    dim=-1,
                )
                stored_slice = F.normalize(
                    self.ordered_keys[candidate_indices, :width],
                    dim=-1,
                )
                stage_scores = (query_slice @ stored_slice.T).squeeze(0)
                keep = min(
                    candidate_indices.numel(),
                    k if stage_idx == len(widths) - 1 else max(k * 4, k),
                )
                top_local = stage_scores.topk(keep).indices
                candidate_indices = candidate_indices[top_local]
            shortlist_mask[batch_idx, candidate_indices] = True
        return shortlist_mask

    def store(self, record: EpisodeRecord) -> int:
        """Store an episode record in the circular buffer."""
        with self._lock:
            idx = self.head
            record_id = self._next_record_id
            self._next_record_id += 1

            if len(self.records) >= self.max_episodes:
                evicted_id = self.records[idx].record_id
                self._id_to_slot.pop(evicted_id, None)

            self.keys[idx] = record.key.detach().to(self.device)
            if self.ordered_keys is not None:
                self.ordered_keys[idx] = self._encode_iob(record.key.unsqueeze(0)).squeeze(0)
            stored_record = EpisodeRecord(
                key=record.key.detach().to(self.device),
                action=record.action.detach().to(self.device),
                ctx_tokens=record.ctx_tokens.detach().to(self.device),
                ic_at_decision=record.ic_at_decision,
                ic_realized=record.ic_realized,
                tc_predicted=record.tc_predicted,
                outcome_key=_detach_to_device(record.outcome_key, self.device),
                step=record.step,
                domain_name=record.domain_name,
                record_id=record_id,
                model_version=record.model_version,
                candidate_postures=_detach_to_device(record.candidate_postures, self.device),
                selected_posture=_detach_to_device(record.selected_posture, self.device),
                candidate_reasoning_states=_detach_to_device(record.candidate_reasoning_states, self.device),
                candidate_paths=_detach_to_device(record.candidate_paths, self.device),
                selected_path=_detach_to_device(record.selected_path, self.device),
                candidate_actions=_detach_to_device(record.candidate_actions, self.device),
                candidate_ic=_detach_to_device(record.candidate_ic, self.device),
                candidate_tc=_detach_to_device(record.candidate_tc, self.device),
                candidate_total=_detach_to_device(record.candidate_total, self.device),
                candidate_terminal_latents=_detach_to_device(record.candidate_terminal_latents, self.device),
                selected_candidate_idx=record.selected_candidate_idx,
                retrieval_trace=_detach_trace_to_device(record.retrieval_trace, self.device),
                mcts_trace=dict(record.mcts_trace) if record.mcts_trace is not None else None,
                skill_trace=tuple(record.skill_trace) if record.skill_trace is not None else None,
                goal_key=_detach_to_device(record.goal_key, self.device),
                domain_cluster_id=record.domain_cluster_id,
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
        """Fill in delayed outcome information for a stored episode."""
        with self._lock:
            slot = self._id_to_slot.get(record_id)
            if slot is None or slot >= len(self.records):
                self._id_to_slot.pop(record_id, None)
                return False
            self.records[slot].ic_realized = ic_realized
            self.records[slot].outcome_key = outcome_key.detach().to(self.device)
            return True

    def get_record_by_id(self, record_id: int) -> EpisodeRecord | None:
        """Return the record for a given stable record_id, or None if evicted."""
        with self._lock:
            slot = self._id_to_slot.get(record_id)
            if slot is None or slot >= len(self.records):
                return None
            return self.records[slot]

    def retrieve(
        self,
        query_key: torch.Tensor,
        k: int | None = None,
    ) -> tuple[torch.Tensor | None, list[list[EpisodeRecord]]]:
        """Retrieve the top-k nearest episodes by cosine similarity."""
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
        with self._lock:
            k = k or self.retrieval_k
            if self.size == 0:
                return None, [], None

            if query_key.dim() == 1:
                query_key = query_key.unsqueeze(0)
            stored = self.keys[: self.size]

            shortlist_mask = None
            if self.iob_encoder is not None and self.ordered_keys is not None:
                encoded_query = self._encode_iob(query_key)
                q_norm = F.normalize(encoded_query, dim=-1)
                k_norm = F.normalize(self.ordered_keys[: self.size], dim=-1)
                combined_scores = torch.mm(q_norm, k_norm.T)
                shortlist_mask = self._iob_shortlist_mask(encoded_query, k)
            else:
                q_norm = F.normalize(query_key.detach().to(self.device), dim=-1)
                k_norm = F.normalize(stored, dim=-1)
                combined_scores = torch.mm(q_norm, k_norm.T)

            if quality_weight != 0.0:
                quality_scores = torch.tensor(
                    [_record_quality_score(record) for record in self.records[: self.size]],
                    dtype=torch.float32,
                    device=self.device,
                )
                combined_scores = combined_scores + quality_weight * _normalize_score_vector(
                    quality_scores
                ).unsqueeze(0)

            if query_posture is not None and posture_weight != 0.0:
                if query_posture.dim() == 1:
                    query_posture = query_posture.unsqueeze(0)
                posture_dim = query_posture.shape[-1]
                posture_bank = torch.zeros(self.size, posture_dim, dtype=torch.float32, device=self.device)
                valid_mask = torch.zeros(self.size, dtype=torch.bool, device=self.device)
                for idx, record in enumerate(self.records[: self.size]):
                    if (
                        record.selected_posture is not None
                        and record.selected_posture.shape[-1] == posture_dim
                    ):
                        posture_bank[idx] = record.selected_posture.float()
                        valid_mask[idx] = True
                if valid_mask.any():
                    posture_query = F.normalize(query_posture.detach().to(self.device), dim=-1)
                    posture_bank = F.normalize(posture_bank, dim=-1)
                    posture_scores = posture_query @ posture_bank.T
                    posture_scores = posture_scores.masked_fill(~valid_mask.unsqueeze(0), 0.0)
                    combined_scores = combined_scores + posture_weight * posture_scores

            if shortlist_mask is not None:
                combined_scores = combined_scores.masked_fill(~shortlist_mask, float("-inf"))

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
            device=self.device
        )
        posture_scores = torch.full(
            (len(episode_lists), max_count),
            float("-inf"),
            dtype=torch.float32,
            device=self.device
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
        comp_vals = torch.zeros(B, max_count, 6, self.embed_dim, dtype=torch.float32, device=self.device)
        comp_mask = torch.zeros(B, max_count, 6, 1, dtype=torch.float32, device=self.device)
        summary_scores = torch.full((B, max_count), float("-inf"), dtype=torch.float32, device=self.device)

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
        # Vectorized Math Operations
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
        """Return future latents, stored context, and realized intrinsic costs for critic training."""
        _ = min_outcome_delay
        with self._lock:
            valid = [
                r
                for r in self.records[: self.size]
                if r.ic_realized is not None and r.outcome_key is not None
            ]
        if not valid:
            return None, None, None
        keys = torch.stack([r.outcome_key for r in valid if r.outcome_key is not None])
        ctx = torch.stack([r.ctx_tokens for r in valid])
        ics = torch.tensor([float(r.ic_realized) for r in valid], dtype=torch.float32, device=self.device)
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
        with self._lock:
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
        """Return JEPA transition triples."""
        z_t, actions, _, z_tH, _ = self.get_world_model_training_pairs()
        if z_t is None or actions is None or z_tH is None:
            return None, None, None
        return z_t, actions, z_tH

    def get_retrieval_training_examples(self) -> list[RetrievalReplayExample]:
        """Return replayable retrieval decisions with realized outcomes."""
        examples: list[RetrievalReplayExample] = []
        with self._lock:
            records = list(self.records[: self.size])
        for record in records:
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

    def iter_records(self) -> list[EpisodeRecord]:
        """Return a stable snapshot of the currently retained records."""
        with self._lock:
            return list(self.records[: self.size])

    def prune(self, predicate: Any) -> int:
        """Remove records matching ``predicate`` while preserving order."""
        with self._lock:
            kept = [record for record in self.records[: self.size] if not predicate(record)]
            removed = self.size - len(kept)
            self.records = list(kept)
            self.keys.zero_()
            if self.ordered_keys is not None:
                self.ordered_keys.zero_()
            self._id_to_slot = {}
            for idx, record in enumerate(kept):
                self.keys[idx] = record.key.detach().to(self.device)
                if self.ordered_keys is not None:
                    self.ordered_keys[idx] = self._encode_iob(record.key.unsqueeze(0)).squeeze(0)
                self._id_to_slot[record.record_id] = idx
            self.size = len(kept)
            self.head = self.size % self.max_episodes
            return removed

    def state_dict(self) -> dict[str, Any]:
        """Serialize the episodic memory buffer into a checkpointable payload."""
        with self._lock:
            return {
                "embed_dim": self.embed_dim,
                "max_episodes": self.max_episodes,
                "retrieval_k": self.retrieval_k,
                "device": str(self.device),
                "size": self.size,
                "head": self.head,
                "next_record_id": self._next_record_id,
                "keys": self.keys[: self.size].detach().cpu(),
                "ordered_keys": (
                    None
                    if self.ordered_keys is None
                    else self.ordered_keys[: self.size].detach().cpu()
                ),
                "records": tuple(
                    _serialize_episode_record(record)
                    for record in self.records[: self.size]
                ),
            }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the episodic memory buffer from a checkpoint payload."""
        with self._lock:
            self.embed_dim = int(state["embed_dim"])
            self.max_episodes = int(state["max_episodes"])
            self.retrieval_k = int(state["retrieval_k"])
            self.device = str(state.get("device", self.device))
            self.keys = torch.zeros(
                self.max_episodes,
                self.embed_dim,
                device=self.device,
                dtype=torch.float32,
            )
            ordered_keys_state = state.get("ordered_keys")
            self.ordered_keys = None
            if ordered_keys_state is not None:
                self.ordered_keys = torch.zeros(
                    self.max_episodes,
                    ordered_keys_state.shape[-1],
                    device=self.device,
                    dtype=torch.float32,
                )
            self.records = []
            self._id_to_slot = {}
            self.size = int(state["size"])
            self.head = int(state["head"])
            self._next_record_id = int(state["next_record_id"])
            if self.size > 0:
                self.keys[: self.size] = state["keys"].to(self.device)
                if self.ordered_keys is not None and ordered_keys_state is not None:
                    self.ordered_keys[: self.size] = ordered_keys_state.to(self.device)
            restored_records = [
                _deserialize_episode_record(payload, self.device)
                for payload in state.get("records", ())
            ]
            self.records.extend(restored_records)
            for idx, record in enumerate(restored_records):
                self._id_to_slot[record.record_id] = idx

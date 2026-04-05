"""Top-level Chamelia assembly module."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.chamelia.actor import Actor
from src.chamelia.configurator import Configurator
from src.chamelia.cost import CostModule
from src.chamelia.hjepa_adapter import forward_hjepa
from src.chamelia.memory import (
    EpisodeRecord,
    LatentMemory,
    RetrievalTraceStep,
)
from src.chamelia.plugins.base import AbstractDomain
from src.chamelia.retrieval import (
    MemoryRelevanceScorer,
    compute_retrieval_relevance_loss,
)
from src.chamelia.world_model import ActionConditionedWorldModel
from src.models.hjepa import HJEPA


def _select_candidate_tensor(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather per-batch candidate values using selected indices."""
    if tensor.dim() == 2:
        return tensor.gather(1, indices.unsqueeze(1)).squeeze(1)
    gather_index = indices.view(-1, 1, *([1] * (tensor.dim() - 2))).expand(
        -1, 1, *tensor.shape[2:]
    )
    return tensor.gather(1, gather_index).squeeze(1)


class Chamelia(nn.Module):
    """Assembled Chamelia system: HJEPA + Configurator + Actor + Cost + Memory."""

    def __init__(
        self,
        hjepa: HJEPA,
        configurator: Configurator,
        actor: Actor,
        cost: CostModule,
        memory: LatentMemory,
        domain: AbstractDomain,
        world_model: ActionConditionedWorldModel | None = None,
        retrieval_scorer: MemoryRelevanceScorer | None = None,
        embed_dim: int = 512,
        action_dim: int = 64,
        num_ctx_tokens: int = 16,
        rollout_horizon: int = 2,
        reasoning_steps: int = 2,
        model_version: str | None = None,
    ) -> None:
        """Initialize the assembled Chamelia model.

        Args:
            hjepa: Backbone HJEPA-compatible model.
            configurator: Configurator module emitting [B, C, D].
            actor: Actor module emitting [B, A].
            cost: Cost module returning [B] scalar costs.
            memory: Latent episodic memory.
            domain: Registered domain plugin.
            embed_dim: Shared latent dimension D.
            action_dim: Actor action dimension A.
            num_ctx_tokens: Number of configurator context tokens C.

        Returns:
            None.
        """
        super().__init__()
        self.hjepa = hjepa
        self.configurator = configurator
        self.actor = actor
        self.cost = cost
        self.memory = memory
        self.world_model = (
            world_model
            if world_model is not None
            else ActionConditionedWorldModel(
                embed_dim=embed_dim,
                action_dim=action_dim,
            )
        )
        self.retrieval_scorer = (
            retrieval_scorer
            if retrieval_scorer is not None
            else MemoryRelevanceScorer(
                embed_dim=embed_dim,
                posture_dim=actor.posture_dim,
            )
        )
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.num_ctx_tokens = num_ctx_tokens
        self.rollout_horizon = rollout_horizon
        self.reasoning_steps = reasoning_steps
        self.model_version = model_version
        self._pending_record_indices: list[int] = []
        self._step_counter = 0
        self.set_domain(domain)

    def set_domain(self, domain: AbstractDomain) -> None:
        """Attach a runtime domain plugin and register its tokenizer if trainable.

        Args:
            domain: Active runtime domain.

        Returns:
            None.
        """
        self.domain = domain
        tokenizer = domain.get_tokenizer()
        if isinstance(tokenizer, nn.Module):
            self.domain_tokenizer = tokenizer

    def get_domain_tokenizer(self) -> nn.Module | None:
        """Return the registered domain tokenizer module if present.

        Args:
            None.

        Returns:
            Tokenizer module or ``None``.
        """
        tokenizer = getattr(self, "domain_tokenizer", None)
        return tokenizer if isinstance(tokenizer, nn.Module) else None

    def _extract_level_features(self, hjepa_outputs: dict) -> list[torch.Tensor]:
        """Extract per-level FPN features from HJEPA output.

        Args:
            hjepa_outputs: HJEPA output dict containing target_features [B, 197, D].

        Returns:
            List of level feature tensors [B, N_i, D].
        """
        target_features = hjepa_outputs["target_features"]
        patch_features = target_features[:, 1:, :]
        level_features = self.hjepa._apply_fpn(patch_features, is_prediction=False)
        return level_features

    def _get_scene_summary(self, hjepa_outputs: dict) -> torch.Tensor:
        """Extract the CLS scene summary vector.

        Args:
            hjepa_outputs: HJEPA output dict containing target_features [B, 197, D].

        Returns:
            Scene summary tensor [B, D].
        """
        return hjepa_outputs["target_features"][:, 0, :]

    def _rerank_retrieved_memory(
        self,
        query_key: torch.Tensor,
        episodes: list[list[EpisodeRecord]],
        retrieved_keys: torch.Tensor | None,
        query_posture: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Apply the learned retrieval scorer to an explicit shortlist."""
        if retrieved_keys is None or not episodes:
            return {
                "episode_summaries": None,
                "episode_scores": None,
                "postures": None,
                "posture_scores": None,
                "base_scores": None,
                "base_quality_scores": None,
                "relevance_scores": None,
                "relevance_weights": None,
                "relevance_features": None,
            }

        base_episode_summaries, base_episode_scores = self.memory.summarize_retrieved_episodes(
            episodes
        )
        memory_postures, _ = self.memory.summarize_retrieved_postures(
            episodes,
            posture_dim=self.actor.posture_dim,
        )
        if base_episode_summaries is None:
            return {
                "episode_summaries": None,
                "episode_scores": None,
                "postures": memory_postures,
                "posture_scores": None,
                "base_scores": None,
                "base_quality_scores": None,
                "relevance_scores": None,
                "relevance_weights": None,
                "relevance_features": None,
            }

        device = query_key.device
        base_scores = torch.nn.functional.cosine_similarity(
            query_key.unsqueeze(1).expand_as(retrieved_keys.to(device)),
            retrieved_keys.to(device),
            dim=-1,
        )
        scorer_out = self.retrieval_scorer(
            query_key=query_key,
            memory_keys=retrieved_keys.to(device),
            memory_summaries=base_episode_summaries.to(device),
            memory_quality=(
                base_episode_scores.to(device) if base_episode_scores is not None else None
            ),
            query_posture=query_posture,
            memory_postures=(
                memory_postures.to(device) if memory_postures is not None else None
            ),
        )
        return {
            "episode_summaries": base_episode_summaries.to(device),
            "episode_scores": scorer_out["scores"],
            "postures": (
                memory_postures.to(device) if memory_postures is not None else None
            ),
            "posture_scores": scorer_out["scores"],
            "base_scores": base_scores,
            "base_quality_scores": (
                base_episode_scores.to(device) if base_episode_scores is not None else None
            ),
            "relevance_scores": scorer_out["scores"],
            "relevance_weights": scorer_out["weights"],
            "relevance_features": scorer_out["features"],
        }

    def _build_retrieval_trace_step(
        self,
        batch_idx: int,
        query_key: torch.Tensor,
        retrieval_bundle: dict[str, torch.Tensor | None],
        retrieved_keys: torch.Tensor | None,
        query_posture: torch.Tensor | None = None,
    ) -> RetrievalTraceStep | None:
        """Slice one batch element of retrieval state into a persistent trace step."""
        episode_summaries = retrieval_bundle["episode_summaries"]
        base_quality_scores = retrieval_bundle["base_quality_scores"]
        if (
            retrieved_keys is None
            or not isinstance(episode_summaries, torch.Tensor)
            or not isinstance(base_quality_scores, torch.Tensor)
            or batch_idx >= query_key.shape[0]
            or batch_idx >= retrieved_keys.shape[0]
            or batch_idx >= episode_summaries.shape[0]
            or batch_idx >= base_quality_scores.shape[0]
        ):
            return None
        if retrieved_keys.shape[1] == 0 or episode_summaries.shape[1] == 0:
            return None
        return RetrievalTraceStep(
            query_key=query_key.detach()[batch_idx],
            memory_keys=retrieved_keys.detach()[batch_idx],
            memory_summaries=episode_summaries.detach()[batch_idx],
            base_quality_scores=base_quality_scores.detach()[batch_idx],
            query_posture=(
                query_posture.detach()[batch_idx]
                if query_posture is not None and batch_idx < query_posture.shape[0]
                else None
            ),
            memory_postures=(
                retrieval_bundle["postures"].detach()[batch_idx]
                if isinstance(retrieval_bundle["postures"], torch.Tensor)
                and batch_idx < retrieval_bundle["postures"].shape[0]
                else None
            ),
            base_scores=(
                retrieval_bundle["base_scores"].detach()[batch_idx]
                if isinstance(retrieval_bundle["base_scores"], torch.Tensor)
                and batch_idx < retrieval_bundle["base_scores"].shape[0]
                else None
            ),
            relevance_scores=(
                retrieval_bundle["relevance_scores"].detach()[batch_idx]
                if isinstance(retrieval_bundle["relevance_scores"], torch.Tensor)
                and batch_idx < retrieval_bundle["relevance_scores"].shape[0]
                else None
            ),
            relevance_weights=(
                retrieval_bundle["relevance_weights"].detach()[batch_idx]
                if isinstance(retrieval_bundle["relevance_weights"], torch.Tensor)
                and batch_idx < retrieval_bundle["relevance_weights"].shape[0]
                else None
            ),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        domain_state: dict,
        actor_mode: str = "mode2",
        store_to_memory: bool = True,
        input_kind: str = "auto",
    ) -> dict[str, Any]:
        """Run the full Chamelia pipeline.

        Args:
            tokens: Input tensor passed to HJEPA. This may be images [B, C, H, W] or
                pre-embedded tokens [B, N, D] depending on ``input_kind``.
            mask: Binary patch mask [B, N].
            domain_state: Opaque domain-state dict.
            actor_mode: Actor mode string, "mode1" or "mode2".
            store_to_memory: Whether to store the current episode in latent memory.
            input_kind: ``image``, ``embedded_tokens``, or ``auto``.

        Returns:
            Dict containing:
                - action: Domain-decoded action object(s)
                - action_vec: [B, A]
                - ctx_tokens: [B, C, D]
                - cost: dict with [B] tensors
                - z: [B, D]
                - hjepa_out: raw HJEPA output dict
        """
        hjepa_out = forward_hjepa(self.hjepa, tokens, mask, input_kind=input_kind)
        z = self._get_scene_summary(hjepa_out)
        level_feats = self._extract_level_features(hjepa_out)

        retrieved_keys, episodes = self.memory.retrieve(z)
        retrieval_bundle = self._rerank_retrieved_memory(
            query_key=z,
            episodes=episodes,
            retrieved_keys=retrieved_keys,
        )
        retrieval_trace_rounds: list[dict[str, Any]] = [
            {
                "query_key": z,
                "query_posture": None,
                "retrieved_keys": retrieved_keys,
                "bundle": retrieval_bundle,
            }
        ]
        retrieved_episode_summaries = retrieval_bundle["episode_summaries"]
        retrieved_episode_scores = retrieval_bundle["episode_scores"]
        retrieved_postures = retrieval_bundle["postures"]
        retrieved_posture_scores = retrieval_bundle["posture_scores"]
        retrieval_base_scores = retrieval_bundle["base_scores"]
        retrieval_base_quality_scores = retrieval_bundle["base_quality_scores"]
        retrieval_relevance_scores = retrieval_bundle["relevance_scores"]
        retrieval_relevance_weights = retrieval_bundle["relevance_weights"]
        retrieval_relevance_features = retrieval_bundle["relevance_features"]

        ctx_tokens = self.configurator(
            hjepa_outputs={"target_features_per_level": level_feats},
            memory_tokens=(
                retrieved_episode_summaries.to(z.device)
                if retrieved_episode_summaries is not None
                else None
            ),
            memory_scores=(
                retrieved_episode_scores.to(z.device)
                if retrieved_episode_scores is not None
                else None
            ),
        )
        reasoning_trace: list[dict[str, torch.Tensor]] = []
        if actor_mode == "mode1":
            candidate_actions = self.actor(z, ctx_tokens, mode="mode1").unsqueeze(1)
            candidate_paths = candidate_actions.unsqueeze(2)
            candidate_postures = torch.zeros(
                z.shape[0],
                1,
                self.actor.posture_dim,
                device=z.device,
                dtype=z.dtype,
            )
            reasoning_states = z.unsqueeze(1)
        else:
            proposal = self.actor.propose(
                z,
                ctx_tokens,
                retrieved_postures=(
                    retrieved_postures.to(z.device)
                    if retrieved_postures is not None
                    else None
                ),
                retrieved_posture_scores=(
                    retrieved_posture_scores.to(z.device)
                    if retrieved_posture_scores is not None
                    else None
                ),
            )
            candidate_paths = proposal["candidate_paths"]
            candidate_actions = proposal["candidate_actions"]
            candidate_postures = proposal["candidate_postures"]
            reasoning_states = proposal["reasoning_states"]

        rollout = None
        candidate_costs = None
        for round_idx in range(max(1, self.reasoning_steps if actor_mode == "mode2" else 1)):
            rollout = self.world_model(
                z=z,
                actions=candidate_paths,
                ctx_tokens=ctx_tokens,
                candidate_postures=candidate_postures,
                reasoning_states=reasoning_states,
                horizon=self.rollout_horizon,
            )
            candidate_costs = self.cost.score_candidates(
                z=z,
                actions=candidate_paths,
                ctx_tokens=ctx_tokens,
                domain_state=domain_state,
                future_z=rollout["terminal_latents"],
                future_trajectory=rollout["trajectory"],
            )
            reasoning_trace.append(
                {
                    "candidate_total": candidate_costs["total"].detach(),
                    "candidate_tc": candidate_costs["tc"].detach(),
                }
            )
            if actor_mode != "mode2" or round_idx + 1 >= self.reasoning_steps:
                break
            next_retrieved_episode_summaries = retrieved_episode_summaries
            next_retrieved_episode_scores = retrieved_episode_scores
            next_retrieved_postures = retrieved_postures
            next_retrieved_posture_scores = retrieved_posture_scores
            next_retrieval_base_scores = retrieval_base_scores
            next_retrieval_base_quality_scores = retrieval_base_quality_scores
            next_retrieval_relevance_scores = retrieval_relevance_scores
            next_retrieval_relevance_weights = retrieval_relevance_weights
            next_retrieval_relevance_features = retrieval_relevance_features
            if candidate_postures.shape[1] > 1:
                nonbaseline_scores = candidate_costs["total"][:, 1:].detach()
                posture_weights = torch.softmax(-nonbaseline_scores, dim=1)
                posture_query = (
                    candidate_postures[:, 1:, :].detach() * posture_weights.unsqueeze(-1)
                ).sum(dim=1)
                refreshed_keys, refreshed_episodes = self.memory.retrieve(
                    z,
                )
                refreshed_bundle = self._rerank_retrieved_memory(
                    query_key=z,
                    episodes=refreshed_episodes,
                    retrieved_keys=refreshed_keys,
                    query_posture=posture_query,
                )
                retrieval_trace_rounds.append(
                    {
                        "query_key": z,
                        "query_posture": posture_query,
                        "retrieved_keys": refreshed_keys,
                        "bundle": refreshed_bundle,
                    }
                )
                next_retrieved_episode_summaries = refreshed_bundle["episode_summaries"]
                next_retrieved_episode_scores = refreshed_bundle["episode_scores"]
                next_retrieved_postures = refreshed_bundle["postures"]
                next_retrieved_posture_scores = refreshed_bundle["posture_scores"]
                next_retrieval_base_scores = refreshed_bundle["base_scores"]
                next_retrieval_base_quality_scores = refreshed_bundle["base_quality_scores"]
                next_retrieval_relevance_scores = refreshed_bundle["relevance_scores"]
                next_retrieval_relevance_weights = refreshed_bundle["relevance_weights"]
                next_retrieval_relevance_features = refreshed_bundle["relevance_features"]
            candidate_paths, candidate_postures, reasoning_states = self.actor.refine(
                z=z,
                ctx_tokens=ctx_tokens,
                candidate_paths=candidate_paths,
                candidate_postures=candidate_postures,
                reasoning_states=reasoning_states,
                rollout_summary=rollout["summary_tokens"],
                candidate_scores=candidate_costs["total"],
                retrieved_postures=(
                    next_retrieved_postures.to(z.device)
                    if next_retrieved_postures is not None
                    else None
                ),
                retrieved_posture_scores=(
                    next_retrieved_posture_scores.to(z.device)
                    if next_retrieved_posture_scores is not None
                    else None
                ),
                retrieved_episode_summaries=(
                    next_retrieved_episode_summaries.to(z.device)
                    if next_retrieved_episode_summaries is not None
                    else None
                ),
                retrieved_episode_scores=(
                    next_retrieved_episode_scores.to(z.device)
                    if next_retrieved_episode_scores is not None
                    else None
                ),
            )
            retrieved_episode_summaries = next_retrieved_episode_summaries
            retrieved_episode_scores = next_retrieved_episode_scores
            retrieved_postures = next_retrieved_postures
            retrieved_posture_scores = next_retrieved_posture_scores
            retrieval_base_scores = next_retrieval_base_scores
            retrieval_base_quality_scores = next_retrieval_base_quality_scores
            retrieval_relevance_scores = next_retrieval_relevance_scores
            retrieval_relevance_weights = next_retrieval_relevance_weights
            retrieval_relevance_features = next_retrieval_relevance_features
            candidate_actions = candidate_paths[:, :, 0, :]

        if candidate_costs is None or rollout is None:
            raise RuntimeError("Chamelia forward must produce candidate costs and rollout outputs.")

        selected_candidate_idx = candidate_costs["total"].argmin(dim=1)
        selected_path = _select_candidate_tensor(candidate_paths, selected_candidate_idx)
        selected_posture = _select_candidate_tensor(candidate_postures, selected_candidate_idx)
        action_vec = _select_candidate_tensor(candidate_actions, selected_candidate_idx)
        action = self.domain.decode_action(action_vec)
        cost_out = {
            "ic": _select_candidate_tensor(candidate_costs["ic"], selected_candidate_idx),
            "tc": _select_candidate_tensor(candidate_costs["tc"], selected_candidate_idx),
            "total": _select_candidate_tensor(candidate_costs["total"], selected_candidate_idx),
        }

        if store_to_memory:
            self._pending_record_indices = []
            for batch_idx in range(z.shape[0]):
                record = EpisodeRecord(
                    key=z.detach()[batch_idx],
                    action=action_vec.detach()[batch_idx],
                    ctx_tokens=ctx_tokens.detach()[batch_idx],
                    ic_at_decision=float(cost_out["ic"][batch_idx].item()),
                    ic_realized=None,
                    tc_predicted=float(cost_out["tc"][batch_idx].item()),
                    outcome_key=None,
                    step=self._step_counter,
                    domain_name=self.domain.domain_name,
                    model_version=self.model_version,
                    candidate_postures=candidate_postures.detach()[batch_idx],
                    selected_posture=selected_posture.detach()[batch_idx],
                    candidate_reasoning_states=reasoning_states.detach()[batch_idx],
                    candidate_paths=candidate_paths.detach()[batch_idx],
                    selected_path=selected_path.detach()[batch_idx],
                    candidate_actions=candidate_actions.detach()[batch_idx],
                    candidate_ic=candidate_costs["ic"].detach()[batch_idx],
                    candidate_tc=candidate_costs["tc"].detach()[batch_idx],
                    candidate_total=candidate_costs["total"].detach()[batch_idx],
                    candidate_terminal_latents=rollout["terminal_latents"].detach()[batch_idx],
                    selected_candidate_idx=int(selected_candidate_idx[batch_idx].item()),
                    retrieval_trace=tuple(
                        trace_step
                        for trace_step in (
                            self._build_retrieval_trace_step(
                                batch_idx=batch_idx,
                                query_key=trace_round["query_key"],
                                query_posture=trace_round["query_posture"],
                                retrieved_keys=trace_round["retrieved_keys"],
                                retrieval_bundle=trace_round["bundle"],
                            )
                            for trace_round in retrieval_trace_rounds
                        )
                        if trace_step is not None
                    ),
                )
                self._pending_record_indices.append(self.memory.store(record))
        else:
            self._pending_record_indices = []

        self._step_counter += 1
        return {
            "action": action,
            "action_vec": action_vec,
            "ctx_tokens": ctx_tokens,
            "cost": cost_out,
            "z": z,
            "hjepa_out": hjepa_out,
            "candidate_actions": candidate_actions,
            "candidate_postures": candidate_postures,
            "reasoning_states": reasoning_states,
            "retrieved_episode_summaries": (
                retrieved_episode_summaries.to(z.device)
                if retrieved_episode_summaries is not None
                else None
            ),
            "retrieval_base_scores": retrieval_base_scores,
            "retrieval_base_quality_scores": retrieval_base_quality_scores,
            "retrieval_relevance_scores": retrieval_relevance_scores,
            "retrieval_relevance_weights": retrieval_relevance_weights,
            "retrieval_relevance_features": retrieval_relevance_features,
            "retrieved_episode_scores": (
                retrieved_episode_scores.to(z.device)
                if retrieved_episode_scores is not None
                else None
            ),
            "retrieved_postures": (
                retrieved_postures.to(z.device) if retrieved_postures is not None else None
            ),
            "retrieved_posture_scores": (
                retrieved_posture_scores.to(z.device)
                if retrieved_posture_scores is not None
                else None
            ),
            "candidate_paths": candidate_paths,
            "candidate_costs": candidate_costs,
            "selected_candidate_idx": selected_candidate_idx,
            "selected_posture": selected_posture,
            "selected_path": selected_path,
            "rollout": rollout,
            "reasoning_trace": reasoning_trace,
        }

    def fill_outcome(
        self,
        ic_realized: float | torch.Tensor | list[float],
        outcome_observation: Any,
    ) -> None:
        """Fill delayed outcome into memory.

        Args:
            ic_realized: Realized intrinsic cost scalar, list, or tensor [B].
            outcome_observation: Raw observation to tokenize and encode. May be batched.

        Returns:
            None.
        """
        if not self._pending_record_indices:
            return

        tokenizer = self.domain.get_tokenizer()
        with torch.no_grad():
            outcome_tokenized = tokenizer(outcome_observation)
            outcome_tokens = outcome_tokenized.tokens
            outcome_mask = torch.zeros(
                outcome_tokens.shape[0],
                outcome_tokens.shape[1],
                device=outcome_tokens.device,
                dtype=torch.float32,
            )
            outcome_hjepa = forward_hjepa(
                self.hjepa,
                outcome_tokens,
                mask=outcome_mask,
                input_kind="embedded_tokens",
            )
            outcome_z = self._get_scene_summary(outcome_hjepa)

        realized_tensor = torch.as_tensor(ic_realized, dtype=torch.float32).flatten()
        if realized_tensor.numel() == 1 and outcome_z.shape[0] > 1:
            realized_tensor = realized_tensor.repeat(outcome_z.shape[0])

        batch_count = min(len(self._pending_record_indices), outcome_z.shape[0], realized_tensor.shape[0])
        for batch_idx in range(batch_count):
            self.memory.fill_outcome(
                self._pending_record_indices[batch_idx],
                ic_realized=float(realized_tensor[batch_idx].item()),
                outcome_key=outcome_z.detach()[batch_idx],
            )
        self._pending_record_indices = []

    def train_critic_from_memory(self) -> torch.Tensor | None:
        """Build a critic loss from memory-stored realized outcomes.

        Args:
            None.

        Returns:
            Scalar tensor [] critic loss, or None if memory has no realized outcomes.
        """
        keys, ctx_tokens, ics = self.memory.get_critic_training_pairs()
        if keys is None or ctx_tokens is None or ics is None:
            return None

        device = next(self.parameters()).device
        predicted = self.cost.trainable_critic(keys.to(device), ctx_tokens.to(device))
        return self.cost.trainable_critic.compute_critic_loss(predicted, ics.to(device))

    def train_world_model_from_memory(self) -> torch.Tensor | None:
        """Build a world-model transition loss from stored memory."""
        z_t, actions, ctx_tokens, z_tH, selected_postures = (
            self.memory.get_world_model_training_pairs()
        )
        if z_t is None or actions is None or ctx_tokens is None or z_tH is None:
            return None

        device = next(self.parameters()).device
        return self.world_model.compute_transition_loss(
            z_t=z_t.to(device),
            actions=actions.to(device),
            z_tH=z_tH.to(device),
            ctx_tokens=ctx_tokens.to(device),
            candidate_postures=(
                selected_postures.to(device) if selected_postures is not None else None
            ),
            horizon=min(self.rollout_horizon, self.world_model.max_horizon),
        )

    def train_retrieval_from_memory(
        self,
        temperature: float = 0.25,
    ) -> torch.Tensor | None:
        """Replay stored retrieval decisions against realized outcomes."""
        examples = self.memory.get_retrieval_training_examples()
        if not examples:
            return None

        device = next(self.parameters()).device
        losses: list[torch.Tensor] = []
        for example in examples:
            scorer_out = self.retrieval_scorer(
                query_key=example.query_key.unsqueeze(0).to(device),
                memory_keys=example.memory_keys.unsqueeze(0).to(device),
                memory_summaries=example.memory_summaries.unsqueeze(0).to(device),
                memory_quality=example.base_quality_scores.unsqueeze(0).to(device),
                query_posture=(
                    example.query_posture.unsqueeze(0).to(device)
                    if example.query_posture is not None
                    else None
                ),
                memory_postures=(
                    example.memory_postures.unsqueeze(0).to(device)
                    if example.memory_postures is not None
                    else None
                ),
            )
            if example.memory_postures is None:
                continue
            loss = compute_retrieval_relevance_loss(
                learned_scores=scorer_out["scores"],
                retrieved_postures=example.memory_postures.unsqueeze(0).to(device),
                selected_posture=example.selected_posture.unsqueeze(0).to(device),
                base_quality_scores=example.base_quality_scores.unsqueeze(0).to(device),
                realized_ic=torch.tensor([example.realized_ic], dtype=torch.float32, device=device),
                temperature=temperature,
            )
            if loss is not None:
                losses.append(loss)
        if not losses:
            return None
        return torch.stack(losses).mean()

"""Top-level Chamelia assembly module."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.chamelia.actor import Actor
from src.chamelia.cognitive.clustering import DomainIndex
from src.chamelia.cognitive.latent_action import LatentActionEncoder
from src.chamelia.cognitive.mamba_world_model import MambaActionConditionedWorldModel
from src.chamelia.cognitive.planning import (
    FrozenReasoningChain,
    HighLevelPlanner,
    MCTSSearch,
    ReasoningStep,
    ThinkerOutput,
)
from src.chamelia.cognitive.procedural import (
    ProceduralMemory as CognitiveProceduralMemory,
    RetrievedSkill,
)
from src.chamelia.cognitive.representation import (
    ContrastiveSparseRepresentation,
    InformationOrderedBottleneck,
    IsotropicSkillCodec,
)
from src.chamelia.cognitive.sleep import SleepCoordinator
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
    ProceduralRelevanceScorer,
    compute_retrieval_relevance_loss,
)
from src.chamelia.session_geometry import SessionGeometry
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
        procedural_memory: CognitiveProceduralMemory | None = None,
        high_level_planner: HighLevelPlanner | None = None,
        mcts_search: MCTSSearch | None = None,
        domain_index: DomainIndex | None = None,
        sleep_coordinator: SleepCoordinator | None = None,
        world_model: ActionConditionedWorldModel | MambaActionConditionedWorldModel | None = None,
        world_model_backend: str = "mamba",
        retrieval_scorer: MemoryRelevanceScorer | None = None,
        procedural_reranker: ProceduralRelevanceScorer | None = None,
        latent_action_encoder: LatentActionEncoder | None = None,
        iob_encoder: InformationOrderedBottleneck | None = None,
        csr_encoder: ContrastiveSparseRepresentation | None = None,
        skill_codec: IsotropicSkillCodec | None = None,
        embed_dim: int = 512,
        action_dim: int = 64,
        num_ctx_tokens: int = 16,
        rollout_horizon: int = 2,
        reasoning_steps: int = 2,
        planner_backend: str = "flat",
        skill_confidence_threshold: float = 0.35,
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
        self.domain_auxiliary_modules = nn.ModuleDict()
        self.hjepa = hjepa
        self.configurator = configurator
        self.actor = actor
        self.cost = cost
        self.memory = memory
        self.procedural_memory = procedural_memory
        self.latent_action_encoder = latent_action_encoder
        self.iob_encoder = iob_encoder
        self.csr_encoder = csr_encoder
        self.skill_codec = skill_codec
        self.world_model = (
            world_model
            if world_model is not None
            else (
                MambaActionConditionedWorldModel(
                    embed_dim=embed_dim,
                    action_dim=action_dim,
                    posture_dim=actor.posture_dim,
                    max_horizon=max(rollout_horizon, actor.path_length),
                )
                if world_model_backend == "mamba"
                else ActionConditionedWorldModel(
                    embed_dim=embed_dim,
                    action_dim=action_dim,
                    posture_dim=actor.posture_dim,
                )
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
        self.procedural_reranker = (
            procedural_reranker
            if procedural_reranker is not None
            else (
                ProceduralRelevanceScorer(
                    embed_dim=embed_dim,
                    retrieval_dim=(
                        int(csr_encoder.output_dim)
                        if csr_encoder is not None
                        else embed_dim
                    ),
                )
                if procedural_memory is not None
                else None
            )
        )
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.num_ctx_tokens = num_ctx_tokens
        self.rollout_horizon = rollout_horizon
        self.reasoning_steps = reasoning_steps
        self.planner_backend = planner_backend
        self.world_model_backend = world_model_backend
        self.skill_confidence_threshold = skill_confidence_threshold
        self.model_version = model_version
        self.geometry: SessionGeometry | None = None
        self._pending_record_indices: list[int] = []
        self._step_counter = 0
        self.high_level_planner = high_level_planner
        if self.high_level_planner is None and planner_backend == "mcts":
            self.high_level_planner = HighLevelPlanner(
                embed_dim=embed_dim,
                skill_dim=embed_dim,
            )
        self.mcts_search = mcts_search
        if self.mcts_search is None and planner_backend == "mcts":
            self.mcts_search = MCTSSearch(
                actor=self.actor,
                world_model=self.world_model,
                cost_module=self.cost,
                high_level_planner=self.high_level_planner,
                rollout_horizon=rollout_horizon,
            )
        self.domain_index = domain_index
        self.sleep_coordinator = sleep_coordinator
        self.set_domain(domain)

    def set_domain(self, domain: AbstractDomain) -> None:
        """Attach a runtime domain plugin and bind geometry to all sub-modules.

        Creates a ``SessionGeometry`` from the domain (fixing A, P=D) and
        calls ``bind_geometry()`` on the Actor and World Model so that their
        action-dimension-dependent heads are correctly sized for this domain.

        Args:
            domain: Active runtime domain.

        Returns:
            None.
        """
        self.domain = domain
        tokenizer = domain.get_tokenizer()
        if isinstance(tokenizer, nn.Module):
            self.domain_tokenizer = tokenizer
            try:
                device = next(self.hjepa.parameters()).device
                self.domain_tokenizer = self.domain_tokenizer.to(device)
            except (StopIteration, AttributeError):
                pass
        auxiliary_modules = domain.get_trainable_modules()
        self.domain_auxiliary_modules = nn.ModuleDict(auxiliary_modules)
        if auxiliary_modules:
            try:
                device = next(self.hjepa.parameters()).device
                self.domain_auxiliary_modules = self.domain_auxiliary_modules.to(device)
            except (StopIteration, AttributeError):
                pass
        if self.mcts_search is not None:
            self.mcts_search.imagined_domain_state_builder = domain.build_imagined_domain_state
            self.mcts_search.simple_baseline_builder = domain.build_simple_baseline_path

        # Build SessionGeometry and propagate to geometry-aware sub-modules.
        # H is the Actor's per-candidate path length; T is the world-model's
        # rollout horizon for value estimation.  They are distinct concepts.
        self.geometry = SessionGeometry.from_domain(
            domain,
            D=self.embed_dim,
            P=self.actor.posture_dim,
            K=self.actor.num_candidates,
            H=self.actor.path_length,
            T=self.rollout_horizon,
        )
        self.action_dim = self.geometry.A
        self.actor.bind_geometry(self.geometry)
        self.world_model.bind_geometry(self.geometry)
        self.configurator.bind_geometry(self.geometry)

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
        if hasattr(self.hjepa, "_apply_fpn") and (
            getattr(self.hjepa, "use_fpn", False) or not hasattr(self.hjepa, "num_hierarchies")
        ):
            return self.hjepa._apply_fpn(patch_features, is_prediction=False)

        num_hierarchies = int(getattr(self.hjepa, "num_hierarchies", 1))
        hierarchy_projections = getattr(self.hjepa, "hierarchy_projections", None)
        if hierarchy_projections is None:
            return [patch_features]
        level_features: list[torch.Tensor] = []
        for level in range(num_hierarchies):
            projected = hierarchy_projections[level](patch_features)
            if level > 0:
                out_len = max(1, projected.shape[1] // (2**level))
                projected = F.adaptive_avg_pool1d(
                    projected.transpose(1, 2),
                    out_len,
                ).transpose(1, 2)
            level_features.append(projected)
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

    def store_decision_records(
        self,
        outputs: dict[str, Any],
        *,
        metadata: list[dict[str, Any] | None] | None = None,
        action_vec_override: torch.Tensor | None = None,
        selected_path_override: torch.Tensor | None = None,
        selected_posture_override: torch.Tensor | None = None,
        selected_candidate_idx_override: list[int | None] | None = None,
        step_override: int | None = None,
    ) -> None:
        """Persist pending decision records, optionally overriding executed actions."""
        z = outputs["z"]
        ctx_tokens = outputs["ctx_tokens"]
        candidate_postures = outputs["candidate_postures"]
        reasoning_states = outputs["reasoning_states"]
        candidate_paths = outputs["candidate_paths"]
        candidate_actions = outputs["candidate_actions"]
        candidate_costs = outputs["candidate_costs"]
        rollout = outputs["rollout"]
        planner_cluster_ids = outputs.get("planner_cluster_ids") or [None] * z.shape[0]
        planner_mcts_traces = outputs.get("mcts_traces") or [None] * z.shape[0]
        planner_skill_traces = outputs.get("skill_traces") or [()] * z.shape[0]
        goal_latents = outputs.get("_goal_latents") or [None] * z.shape[0]
        retrieval_trace_rounds = outputs.get("_retrieval_trace_rounds") or []

        action_vec = action_vec_override if action_vec_override is not None else outputs["action_vec"]
        selected_path = (
            selected_path_override if selected_path_override is not None else outputs["selected_path"]
        )
        selected_posture = (
            selected_posture_override
            if selected_posture_override is not None
            else outputs["selected_posture"]
        )
        if selected_candidate_idx_override is None:
            selected_idx_tensor = outputs["selected_candidate_idx"]
            selected_candidate_idx_override = [
                int(selected_idx_tensor[batch_idx].item())
                for batch_idx in range(selected_idx_tensor.shape[0])
            ]

        self._pending_record_indices = []
        step_value = self._step_counter if step_override is None else int(step_override)
        for batch_idx in range(z.shape[0]):
            metadata_item = None
            if metadata is not None and batch_idx < len(metadata):
                metadata_item = metadata[batch_idx]
            record = EpisodeRecord(
                key=z.detach()[batch_idx],
                action=action_vec.detach()[batch_idx],
                ctx_tokens=ctx_tokens.detach()[batch_idx],
                ic_at_decision=float(outputs["cost"]["ic"][batch_idx].item()),
                ic_realized=None,
                tc_predicted=float(outputs["cost"]["tc"][batch_idx].item()),
                outcome_key=None,
                step=step_value,
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
                candidate_terminal_latents=(
                    rollout["terminal_latents"].detach()[batch_idx]
                    if isinstance(rollout.get("terminal_latents"), torch.Tensor)
                    else None
                ),
                selected_candidate_idx=selected_candidate_idx_override[batch_idx],
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
                mcts_trace=planner_mcts_traces[batch_idx],
                skill_trace=planner_skill_traces[batch_idx],
                goal_key=(
                    goal_latents[batch_idx]
                    if self.planner_backend == "mcts"
                    and self.mcts_search is not None
                    and goal_latents[batch_idx] is not None
                    else None
                ),
                domain_cluster_id=planner_cluster_ids[batch_idx],
                metadata=None if metadata_item is None else dict(metadata_item),
            )
            self._pending_record_indices.append(self.memory.store(record))

    def _slice_domain_state(
        self,
        domain_state: dict[str, Any],
        batch_idx: int,
    ) -> dict[str, Any]:
        """Slice a possibly batched domain-state dict down to one sample."""
        sliced: dict[str, Any] = {}
        for key, value in domain_state.items():
            if torch.is_tensor(value) and value.dim() > 0 and batch_idx < value.shape[0]:
                sliced[key] = value[batch_idx : batch_idx + 1]
            else:
                sliced[key] = value
        return sliced

    def _resolve_goal_latents(
        self,
        z: torch.Tensor,
        domain_state: dict[str, Any],
    ) -> list[torch.Tensor | None]:
        """Resolve explicit goal latents for high-level planning."""
        goals: list[torch.Tensor | None] = []
        for batch_idx in range(z.shape[0]):
            goal_latent = self.domain.compute_goal_latent(
                self._slice_domain_state(domain_state, batch_idx),
                z[batch_idx : batch_idx + 1],
            )
            if goal_latent is None:
                goals.append(None)
                continue
            if goal_latent.dim() == 1:
                goal_latent = goal_latent.unsqueeze(0)
            goals.append(goal_latent.to(z.device, dtype=z.dtype)[0].detach())
        return goals

    def _rerank_procedural_skills(
        self,
        query_latent: torch.Tensor,
        retrieved_skills: list[RetrievedSkill],
    ) -> list[RetrievedSkill]:
        if (
            self.procedural_reranker is None
            or self.sleep_coordinator is None
            or self.sleep_coordinator.reranker_example_count() < 64
            or not retrieved_skills
        ):
            return retrieved_skills
        skill_embeddings = torch.stack(
            [item.record.embedding.to(query_latent.device) for item in retrieved_skills],
            dim=0,
        ).unsqueeze(0)
        retrieval_vectors = torch.stack(
            [item.record.retrieval_vector.to(query_latent.device) for item in retrieved_skills],
            dim=0,
        ).unsqueeze(0)
        base_similarity = torch.tensor(
            [float(item.similarity) for item in retrieved_skills],
            dtype=query_latent.dtype,
            device=query_latent.device,
        ).unsqueeze(0)
        confidence = torch.tensor(
            [float(item.record.confidence) for item in retrieved_skills],
            dtype=query_latent.dtype,
            device=query_latent.device,
        ).unsqueeze(0)
        reranked = self.procedural_reranker(
            query_latent=query_latent.unsqueeze(0),
            skill_embeddings=skill_embeddings,
            retrieval_vectors=retrieval_vectors,
            retrieval_similarity=base_similarity,
            confidence=confidence,
        )
        weights = reranked["weights"][0]
        order = torch.argsort(weights, descending=True)
        return [retrieved_skills[int(index.item())] for index in order]

    def _pad_candidate_tensor(
        self,
        tensors: list[torch.Tensor],
        pad_value: float = 0.0,
    ) -> torch.Tensor:
        """Pad variable-width candidate tensors to a common candidate count."""
        if not tensors:
            raise ValueError("tensors must be non-empty")
        max_candidates = max(tensor.shape[0] for tensor in tensors)
        tail_shape = tensors[0].shape[1:]
        padded = torch.full(
            (len(tensors), max_candidates, *tail_shape),
            fill_value=pad_value,
            dtype=tensors[0].dtype,
            device=tensors[0].device,
        )
        for batch_idx, tensor in enumerate(tensors):
            padded[batch_idx, : tensor.shape[0]] = tensor
        return padded

    def _run_system1_skill(
        self,
        *,
        z: torch.Tensor,
        ctx_tokens: torch.Tensor,
        domain_state: dict[str, Any],
        goal_z: torch.Tensor,
        retrieved_skills: list[RetrievedSkill],
    ) -> dict[str, Any] | None:
        """Execute a confident retrieved skill directly through the low-level planner."""
        if (
            self.high_level_planner is None
            or not retrieved_skills
            or retrieved_skills[0].score < self.skill_confidence_threshold
        ):
            return None
        plan = self.high_level_planner.plan(z=z, goal_z=goal_z, retrieved_skills=retrieved_skills)
        if plan is None:
            return None
        candidate_paths = plan.action_path.detach().to(z)
        if candidate_paths.dim() == 2:
            candidate_paths = candidate_paths.unsqueeze(0)
        candidate_postures = torch.zeros(
            1,
            self.actor.posture_dim,
            dtype=z.dtype,
            device=z.device,
        )
        reasoning_states = z.unsqueeze(0)
        rollout = self.world_model(
            z=z.unsqueeze(0),
            actions=candidate_paths.unsqueeze(0),
            ctx_tokens=ctx_tokens.unsqueeze(0),
            candidate_postures=candidate_postures.unsqueeze(0),
            reasoning_states=reasoning_states.unsqueeze(0),
            horizon=min(self.rollout_horizon, candidate_paths.shape[1]),
        )
        candidate_costs = self.cost.score_candidates(
            z=z.unsqueeze(0),
            actions=candidate_paths.unsqueeze(0),
            ctx_tokens=ctx_tokens.unsqueeze(0),
            domain_state=domain_state,
            future_z=rollout["terminal_latents"],
            future_trajectory=rollout["trajectory"],
            imagined_domain_state_builder=self.domain.build_imagined_domain_state,
        )
        reasoning_chain = FrozenReasoningChain(
            steps=(
                ReasoningStep(
                    state=z.detach(),
                    candidate_paths=candidate_paths.detach(),
                    candidate_costs=candidate_costs["total"][0].detach(),
                    selected_path=candidate_paths[0].detach(),
                    source="system1_skill",
                    depth=0,
                ),
            )
        )
        return {
            "selected_action": candidate_paths[0, 0].detach(),
            "selected_path": candidate_paths[0].detach(),
            "selected_posture": candidate_postures[0].detach(),
            "selected_candidate_idx": 0,
            "candidate_paths": candidate_paths.detach(),
            "candidate_actions": candidate_paths[:, 0, :].detach(),
            "candidate_postures": candidate_postures.detach(),
            "candidate_reasoning_states": reasoning_states.detach(),
            "candidate_ic": candidate_costs["ic"][0].detach(),
            "candidate_tc": candidate_costs["tc"][0].detach(),
            "candidate_total": candidate_costs["total"][0].detach(),
            "candidate_terminal_latents": rollout["terminal_latents"][0].detach(),
            "rollout": rollout,
            "reasoning_chain": reasoning_chain,
            "thinker_output": ThinkerOutput(
                reasoning_chain=reasoning_chain,
                action_vec=candidate_paths[0, 0].detach().unsqueeze(0),
                selected_path=candidate_paths[0].detach().unsqueeze(0),
                metadata={
                    "planner_source": "system1_skill",
                    "skill_id": retrieved_skills[0].record.skill_id,
                    "score": retrieved_skills[0].score,
                },
            ),
            "planner_source": "system1_skill",
            "mcts_trace": {
                "mode": "system1_skill",
                "skill_id": retrieved_skills[0].record.skill_id,
                "score": retrieved_skills[0].score,
            },
            "skill_trace": (retrieved_skills[0].record.skill_id,),
        }

    def _run_planner_sample(
        self,
        *,
        z: torch.Tensor,
        ctx_tokens: torch.Tensor,
        domain_state: dict[str, Any],
        goal_z: torch.Tensor,
        retrieved_skills: list[RetrievedSkill],
        retrieved_postures: torch.Tensor | None = None,
        retrieved_posture_scores: torch.Tensor | None = None,
        budget_multiplier: float = 1.0,
    ) -> dict[str, Any]:
        """Run either direct-skill execution or MCTS for one sample."""
        system1 = self._run_system1_skill(
            z=z,
            ctx_tokens=ctx_tokens,
            domain_state=domain_state,
            goal_z=goal_z,
            retrieved_skills=retrieved_skills,
        )
        if system1 is not None:
            return system1
        if self.mcts_search is None:
            raise RuntimeError("planner_backend='mcts' requires an MCTSSearch instance.")
        result = self.mcts_search.search(
            z=z,
            ctx_tokens=ctx_tokens,
            domain_state=domain_state,
            goal_z=goal_z,
            retrieved_postures=retrieved_postures,
            retrieved_posture_scores=retrieved_posture_scores,
            retrieved_skills=retrieved_skills,
            budget_multiplier=budget_multiplier,
        )
        planner_diagnostics = self.domain.analyze_planner_candidates(
            candidate_paths=result.candidate_paths.detach(),
            candidate_ic=result.candidate_ic.detach() if result.candidate_ic is not None else None,
            candidate_tc=result.candidate_tc.detach() if result.candidate_tc is not None else None,
            candidate_total=result.candidate_costs.detach(),
            candidate_terminal_latents=(
                result.candidate_terminal_latents.detach()
                if result.candidate_terminal_latents is not None
                else None
            ),
            selected_candidate_idx=result.selected_candidate_idx,
            domain_state=domain_state,
            gamma=float(self.cost.gamma),
            planner_trace=result.tree_trace,
        )
        tree_trace = dict(result.tree_trace)
        if planner_diagnostics is not None:
            tree_trace["counterfactual"] = planner_diagnostics
        zero_rollout = {
            "trajectory": result.candidate_trajectories.unsqueeze(0)
            if result.candidate_trajectories is not None
            else None,
            "terminal_latents": result.candidate_terminal_latents.unsqueeze(0)
            if result.candidate_terminal_latents is not None
            else None,
        }
        return {
            "selected_action": result.selected_action.detach(),
            "selected_path": result.selected_path.detach(),
            "selected_posture": (
                result.selected_posture.detach()
                if result.selected_posture is not None
                else torch.zeros(self.actor.posture_dim, device=z.device, dtype=z.dtype)
            ),
            "selected_candidate_idx": result.selected_candidate_idx,
            "candidate_paths": result.candidate_paths.detach(),
            "candidate_actions": result.candidate_paths[:, 0, :].detach(),
            "candidate_postures": (
                result.candidate_postures.detach()
                if result.candidate_postures is not None
                else torch.zeros(
                    result.candidate_paths.shape[0],
                    self.actor.posture_dim,
                    dtype=z.dtype,
                    device=z.device,
                )
            ),
            "candidate_reasoning_states": (
                result.candidate_reasoning_states.detach()
                if result.candidate_reasoning_states is not None
                else z.unsqueeze(0).expand(result.candidate_paths.shape[0], -1).detach()
            ),
            "candidate_ic": (
                result.candidate_ic.detach()
                if result.candidate_ic is not None
                else torch.zeros_like(result.candidate_costs)
            ),
            "candidate_tc": (
                result.candidate_tc.detach()
                if result.candidate_tc is not None
                else torch.zeros_like(result.candidate_costs)
            ),
            "candidate_total": result.candidate_costs.detach(),
            "candidate_terminal_latents": (
                result.candidate_terminal_latents.detach()
                if result.candidate_terminal_latents is not None
                else None
            ),
            "rollout": zero_rollout,
            "reasoning_chain": result.reasoning_chain,
            "thinker_output": ThinkerOutput(
                reasoning_chain=result.reasoning_chain,
                action_vec=result.selected_action.detach().unsqueeze(0),
                selected_path=result.selected_path.detach().unsqueeze(0),
                metadata={
                    "planner_source": "mcts",
                    "reused_tree": result.reused_tree,
                },
            ),
            "planner_source": "mcts",
            "mcts_trace": tree_trace,
            "planner_diagnostics": planner_diagnostics,
            "skill_trace": tuple(item.record.skill_id for item in retrieved_skills[:1]),
        }

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        domain_state: dict,
        actor_mode: str = "mode2",
        store_to_memory: bool = True,
        input_kind: str = "auto",
        advance_step: bool = True,
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
            advance_step: Whether to advance the model-side decision step counter.

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
        planner_cluster_ids: list[int | None] = [None] * z.shape[0]
        planner_skill_traces: list[tuple[int, ...] | None] = [None] * z.shape[0]
        planner_mcts_traces: list[dict[str, Any] | None] = [None] * z.shape[0]
        planner_diagnostics: list[dict[str, Any] | None] = [None] * z.shape[0]
        thinker_output: ThinkerOutput | None = None
        reasoning_trace: list[dict[str, torch.Tensor]] = []
        goal_latents: torch.Tensor | None = None
        if self.planner_backend == "mcts" and self.mcts_search is not None:
            goal_latents = self._resolve_goal_latents(z, domain_state)
            planner_results: list[dict[str, Any]] = []
            for batch_idx in range(z.shape[0]):
                domain_state_i = self._slice_domain_state(domain_state, batch_idx)
                trigger_weights: dict[int, float] | None = None
                if self.domain_index is not None:
                    route = self.domain_index.route(
                        z[batch_idx].detach(),
                        self.domain.domain_name,
                    )
                    planner_cluster_ids[batch_idx] = route.primary_cluster_id
                    trigger_weights = self.domain_index.get_route_trigger_weights(route)
                    if (
                        self.domain_index.adapter_bank is not None
                        and z.shape[0] == 1
                    ):
                        self.domain_index.adapter_bank.apply_mixture(
                            route.mixture_cluster_ids,
                            route.mixture_weights,
                        )
                retrieved_skills = (
                    self.procedural_memory.retrieve(
                        z[batch_idx].detach(),
                        k=self.actor.num_candidates,
                        domain_name=self.domain.domain_name,
                        trigger_weights=trigger_weights,
                    )
                    if self.procedural_memory is not None
                    else []
                )
                retrieved_skills = self._rerank_procedural_skills(
                    z[batch_idx].detach(),
                    retrieved_skills,
                )
                retrieved_postures_i = (
                    retrieved_postures[batch_idx].detach()
                    if retrieved_postures is not None and batch_idx < retrieved_postures.shape[0]
                    else None
                )
                retrieved_posture_scores_i = (
                    retrieved_posture_scores[batch_idx].detach()
                    if retrieved_posture_scores is not None
                    and batch_idx < retrieved_posture_scores.shape[0]
                    else None
                )
                result = self._run_planner_sample(
                    z=z[batch_idx].detach(),
                    ctx_tokens=ctx_tokens[batch_idx].detach(),
                    domain_state=domain_state_i,
                    goal_z=goal_latents[batch_idx],
                    retrieved_skills=retrieved_skills,
                    retrieved_postures=retrieved_postures_i,
                    retrieved_posture_scores=retrieved_posture_scores_i,
                    budget_multiplier=float(domain_state_i.get("_planner_budget_multiplier", 1.0)),
                )
                planner_results.append(result)
                planner_skill_traces[batch_idx] = result["skill_trace"]
                planner_mcts_traces[batch_idx] = result["mcts_trace"]
                planner_diagnostics[batch_idx] = result.get("planner_diagnostics")

            candidate_paths = self._pad_candidate_tensor(
                [result["candidate_paths"].to(z.device) for result in planner_results]
            )
            candidate_actions = self._pad_candidate_tensor(
                [result["candidate_actions"].to(z.device) for result in planner_results]
            )
            candidate_postures = self._pad_candidate_tensor(
                [result["candidate_postures"].to(z.device) for result in planner_results]
            )
            reasoning_states = self._pad_candidate_tensor(
                [
                    result["candidate_reasoning_states"].to(z.device)
                    for result in planner_results
                ]
            )
            candidate_ic = self._pad_candidate_tensor(
                [result["candidate_ic"].to(z.device).unsqueeze(-1) for result in planner_results]
            ).squeeze(-1)
            candidate_tc = self._pad_candidate_tensor(
                [result["candidate_tc"].to(z.device).unsqueeze(-1) for result in planner_results]
            ).squeeze(-1)
            candidate_total = self._pad_candidate_tensor(
                [result["candidate_total"].to(z.device).unsqueeze(-1) for result in planner_results],
                pad_value=1.0e6,
            ).squeeze(-1)
            candidate_terminal_latents = self._pad_candidate_tensor(
                [
                    (
                        result["candidate_terminal_latents"].to(z.device)
                        if result["candidate_terminal_latents"] is not None
                        else torch.zeros(
                            result["candidate_paths"].shape[0],
                            z.shape[-1],
                            device=z.device,
                            dtype=z.dtype,
                        )
                    )
                    for result in planner_results
                ]
            )
            candidate_costs = {
                "ic": candidate_ic,
                "tc": candidate_tc,
                "total": candidate_total,
            }
            selected_candidate_idx = torch.tensor(
                [int(result["selected_candidate_idx"]) for result in planner_results],
                dtype=torch.long,
                device=z.device,
            )
            selected_path = torch.stack(
                [result["selected_path"].to(z.device) for result in planner_results],
                dim=0,
            )
            selected_posture = torch.stack(
                [result["selected_posture"].to(z.device) for result in planner_results],
                dim=0,
            )
            action_vec = torch.stack(
                [result["selected_action"].to(z.device) for result in planner_results],
                dim=0,
            )
            action = self.domain.decode_action(action_vec)
            cost_out = {
                "ic": torch.tensor(
                    [
                        float(result["candidate_ic"][int(result["selected_candidate_idx"])].item())
                        for result in planner_results
                    ],
                    dtype=z.dtype,
                    device=z.device,
                ),
                "tc": torch.tensor(
                    [
                        float(result["candidate_tc"][int(result["selected_candidate_idx"])].item())
                        for result in planner_results
                    ],
                    dtype=z.dtype,
                    device=z.device,
                ),
                "total": torch.tensor(
                    [
                        float(result["candidate_total"][int(result["selected_candidate_idx"])].item())
                        for result in planner_results
                    ],
                    dtype=z.dtype,
                    device=z.device,
                ),
            }
            rollout = {
                "terminal_latents": candidate_terminal_latents,
                "trajectory": None,
                "summary_tokens": None,
            }
            reasoning_trace = [
                {
                    "candidate_total": candidate_total.detach(),
                    "candidate_tc": candidate_tc.detach(),
                }
            ]
            if len(planner_results) == 1:
                thinker_output = planner_results[0]["thinker_output"]
        elif actor_mode == "mode1":
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
                simple_baseline_path=self.domain.build_simple_baseline_path(
                    domain_state,
                    self.actor.path_length,
                    self.actor.action_dim,
                ),
            )
            candidate_paths = proposal["candidate_paths"]
            candidate_actions = proposal["candidate_actions"]
            candidate_postures = proposal["candidate_postures"]
            reasoning_states = proposal["reasoning_states"]

        if self.planner_backend == "mcts" and self.mcts_search is not None:
            pass
        else:
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
                    imagined_domain_state_builder=self.domain.build_imagined_domain_state,
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
                    simple_baseline_path=self.domain.build_simple_baseline_path(
                        domain_state,
                        self.actor.path_length,
                        self.actor.action_dim,
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

        if not (self.planner_backend == "mcts" and self.mcts_search is not None):
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

        decision_step = self._step_counter
        outputs = {
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
            "thinker_output": thinker_output,
            "planner_backend": self.planner_backend,
            "world_model_backend": self.world_model_backend,
            "planner_cluster_ids": planner_cluster_ids,
            "mcts_traces": planner_mcts_traces,
            "planner_diagnostics": planner_diagnostics,
            "skill_traces": planner_skill_traces,
            "_retrieval_trace_rounds": retrieval_trace_rounds,
            "_goal_latents": goal_latents,
            "decision_step": decision_step,
        }
        if store_to_memory:
            self.store_decision_records(outputs, step_override=decision_step)
        else:
            self._pending_record_indices = []

        if advance_step:
            self._step_counter += 1
            if self.sleep_coordinator is not None:
                self.sleep_coordinator.maybe_trigger(self._step_counter)
        return outputs

    def fill_outcome(
        self,
        ic_realized: float | torch.Tensor | list[float],
        outcome_observation: Any | None = None,
        outcome_z: torch.Tensor | None = None,
    ) -> None:
        """Fill delayed outcome into memory. (Optimized to skip redundant encoding)"""
        if not self._pending_record_indices:
            return

        # If the latent vector (z) is already provided by the rollout/training loop,
        # we skip the expensive tokenizer and HJEPA forward pass entirely.
        if outcome_z is None:
            if outcome_observation is None:
                raise ValueError("Must provide either outcome_observation or outcome_z.")
            
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
        """Replay stored retrieval decisions against realized outcomes (Vectorized)."""
        examples = self.memory.get_retrieval_training_examples()
        if not examples:
            return None

        device = next(self.parameters()).device
        
        # Group examples by the number of retrieved memories (K) to allow batching
        from collections import defaultdict
        grouped_examples = defaultdict(list)
        for ex in examples:
            if ex.memory_postures is None:
                continue
            k_size = ex.memory_keys.shape[0]
            grouped_examples[k_size].append(ex)

        losses: list[torch.Tensor] = []
        total_valid_examples = 0

        for k_size, group in grouped_examples.items():
            # Stack into full batches: [B, ...]
            query_keys = torch.stack([ex.query_key for ex in group]).to(device)
            memory_keys = torch.stack([ex.memory_keys for ex in group]).to(device)
            memory_summaries = torch.stack([ex.memory_summaries for ex in group]).to(device)
            base_quality = torch.stack([ex.base_quality_scores for ex in group]).to(device)
            selected_postures = torch.stack([ex.selected_posture for ex in group]).to(device)
            memory_postures = torch.stack([ex.memory_postures for ex in group]).to(device)
            realized_ics = torch.tensor([ex.realized_ic for ex in group], dtype=torch.float32, device=device)

            # Handle optional query_posture
            if any(ex.query_posture is not None for ex in group):
                query_postures = torch.stack([
                    ex.query_posture if ex.query_posture is not None else torch.zeros_like(selected_postures[0])
                    for ex in group
                ]).to(device)
            else:
                query_postures = None

            # Batched forward pass
            scorer_out = self.retrieval_scorer(
                query_key=query_keys,
                memory_keys=memory_keys,
                memory_summaries=memory_summaries,
                memory_quality=base_quality,
                query_posture=query_postures,
                memory_postures=memory_postures,
            )

            # Batched loss computation
            loss = compute_retrieval_relevance_loss(
                learned_scores=scorer_out["scores"],
                retrieved_postures=memory_postures,
                selected_posture=selected_postures,
                base_quality_scores=base_quality,
                realized_ic=realized_ics,
                temperature=temperature,
            )
            
            if loss is not None:
                losses.append(loss * len(group))
                total_valid_examples += len(group)

        if not losses or total_valid_examples == 0:
            return None
            
        return torch.stack(losses).sum() / total_valid_examples

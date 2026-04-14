"""Thinker/Talker, HWM, and MCTS planning components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.chamelia.cognitive.latent_action import estimate_target_delta
from src.chamelia.cognitive.procedural import RetrievedSkill


@dataclass(frozen=True)
class ReasoningStep:
    """One frozen latent reasoning step."""

    state: torch.Tensor
    candidate_paths: torch.Tensor | None
    candidate_costs: torch.Tensor | None
    selected_path: torch.Tensor | None
    source: str
    depth: int


@dataclass(frozen=True)
class FrozenReasoningChain:
    """One-way latent reasoning chain that a Talker can read but not modify."""

    steps: tuple[ReasoningStep, ...]

    def as_tensor(self) -> torch.Tensor:
        return torch.stack([step.state for step in self.steps], dim=0)


@dataclass(frozen=True)
class ThinkerOutput:
    """Frozen Thinker output handed downstream to the Talker."""

    reasoning_chain: FrozenReasoningChain
    action_vec: torch.Tensor
    selected_path: torch.Tensor
    metadata: dict[str, Any]


class Talker(nn.Module):
    """Decode a frozen latent reasoning chain into token logits on demand."""

    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        *,
        num_heads: int = 4,
        num_layers: int = 2,
        max_tokens: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, max_tokens, latent_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_head = nn.Linear(latent_dim, vocab_size)

    def forward(
        self,
        thinker_output: ThinkerOutput | FrozenReasoningChain | torch.Tensor,
        *,
        max_tokens: int | None = None,
    ) -> torch.Tensor:
        if isinstance(thinker_output, ThinkerOutput):
            memory = thinker_output.reasoning_chain.as_tensor().unsqueeze(0)
        elif isinstance(thinker_output, FrozenReasoningChain):
            memory = thinker_output.as_tensor().unsqueeze(0)
        else:
            memory = thinker_output
        if memory.dim() == 2:
            memory = memory.unsqueeze(0)
        query_len = min(int(max_tokens or self.max_tokens), self.max_tokens)
        queries = self.query_tokens[:, :query_len, :].expand(memory.shape[0], -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        return self.output_head(decoded)


@dataclass(frozen=True)
class MacroPlan:
    """High-level macro-action plan."""

    selected_skill: RetrievedSkill | None
    action_path: torch.Tensor
    subgoal: torch.Tensor
    score: float
    candidate_scores: torch.Tensor


class HighLevelPlanner(nn.Module):
    """Map retrieved skills into subgoals for the low-level world model."""

    def __init__(
        self,
        embed_dim: int,
        skill_dim: int,
    ) -> None:
        super().__init__()
        self.subgoal_head = nn.Sequential(
            nn.Linear(embed_dim + skill_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.subgoal_norm = nn.LayerNorm(embed_dim)

    def plan(
        self,
        z: torch.Tensor,
        goal_z: torch.Tensor,
        retrieved_skills: list[RetrievedSkill],
    ) -> MacroPlan | None:
        if not retrieved_skills:
            return None
        if goal_z is None:
            return None
        if z.dim() != 1 or goal_z.dim() != 1:
            raise ValueError("HighLevelPlanner.plan currently expects 1D latent vectors.")
        planner_device = z.device
        try:
            planner_device = next(self.parameters()).device
        except StopIteration:
            planner_device = z.device
        z = z.to(planner_device)
        goal_z = goal_z.to(planner_device)
        skill_embeddings = torch.stack(
            [item.record.embedding for item in retrieved_skills],
            dim=0,
        ).to(planner_device)
        repeated_state = z.unsqueeze(0).expand(skill_embeddings.shape[0], -1)
        candidate_subgoals = self.subgoal_norm(
            z.unsqueeze(0) + self.subgoal_head(torch.cat([repeated_state, skill_embeddings], dim=-1))
        )
        goal_delta = estimate_target_delta(z.unsqueeze(0), goal_z.unsqueeze(0)).squeeze(0)
        subgoal_delta = F.normalize(candidate_subgoals - z.unsqueeze(0), dim=-1)
        alignment = (subgoal_delta * goal_delta.unsqueeze(0)).sum(dim=-1)
        retrieval_scores = torch.tensor(
            [item.score for item in retrieved_skills],
            dtype=alignment.dtype,
            device=alignment.device,
        )
        scores = alignment + retrieval_scores
        best_idx = int(scores.argmax().item())
        best = retrieved_skills[best_idx]
        return MacroPlan(
            selected_skill=best,
            action_path=best.record.action_path,
            subgoal=candidate_subgoals[best_idx],
            score=float(scores[best_idx].item()),
            candidate_scores=scores,
        )


@dataclass
class MCTSNode:
    """One node in the latent MCTS tree."""

    latent_state: torch.Tensor
    ctx_tokens: torch.Tensor
    depth: int
    path_from_parent: torch.Tensor | None = None
    action_from_parent: torch.Tensor | None = None
    posture: torch.Tensor | None = None
    reasoning_state: torch.Tensor | None = None
    immediate_cost: float = 0.0
    visit_count: int = 0
    total_cost: float = 0.0
    expanded: bool = False
    safety_triggered: bool = False
    candidate_paths: torch.Tensor | None = None
    candidate_ic: torch.Tensor | None = None
    candidate_tc: torch.Tensor | None = None
    candidate_costs: torch.Tensor | None = None
    candidate_postures: torch.Tensor | None = None
    candidate_reasoning_states: torch.Tensor | None = None
    candidate_terminal_latents: torch.Tensor | None = None
    candidate_trajectories: torch.Tensor | None = None
    children: list["MCTSNode"] = field(default_factory=list)

    @property
    def mean_cost(self) -> float:
        if self.visit_count == 0:
            return self.immediate_cost
        return self.total_cost / self.visit_count

    def selection_score(self, parent_visits: int, exploration_weight: float) -> float:
        if self.visit_count == 0:
            return float("inf")
        bonus = exploration_weight * ((parent_visits + 1) ** 0.5) / (1.0 + self.visit_count)
        return (-self.mean_cost) + bonus


@dataclass(frozen=True)
class MCTSResult:
    """Planner outputs from a completed MCTS search."""

    selected_path: torch.Tensor
    selected_action: torch.Tensor
    selected_posture: torch.Tensor | None
    candidate_paths: torch.Tensor
    candidate_ic: torch.Tensor | None
    candidate_tc: torch.Tensor | None
    candidate_costs: torch.Tensor
    candidate_postures: torch.Tensor | None
    candidate_reasoning_states: torch.Tensor | None
    candidate_terminal_latents: torch.Tensor | None
    candidate_trajectories: torch.Tensor | None
    selected_candidate_idx: int
    root: MCTSNode
    reasoning_chain: FrozenReasoningChain
    tree_trace: dict[str, Any]
    reused_tree: bool


def export_tree(node: MCTSNode) -> dict[str, Any]:
    """Export a tree into a JSON-friendly trace."""
    return {
        "depth": node.depth,
        "visit_count": node.visit_count,
        "mean_cost": node.mean_cost,
        "immediate_cost": node.immediate_cost,
        "safety_triggered": node.safety_triggered,
        "children": [export_tree(child) for child in node.children],
    }


class MCTSSearch:
    """UCT-guided latent MCTS over actor proposals and world-model rollouts."""

    def __init__(
        self,
        *,
        actor: Any,
        world_model: Any,
        cost_module: Any,
        high_level_planner: HighLevelPlanner | None = None,
        simulations: int = 32,
        max_depth: int = 4,
        exploration_weight: float = 1.25,
        rollout_horizon: int = 3,
        tree_reuse_similarity: float = 0.98,
        safety_cost_ceiling: float | None = None,
        use_baseline_guard: bool = True,
        baseline_cost_margin: float = 0.25,
        baseline_uncertainty_scale: float = 3.0,
        imagined_domain_state_builder: Callable[[dict[str, Any], torch.Tensor, int], dict[str, Any]] | None = None,
        simple_baseline_builder: Callable[[dict[str, Any], int, int], torch.Tensor | None] | None = None,
    ) -> None:
        self.actor = actor
        self.world_model = world_model
        self.cost_module = cost_module
        self.high_level_planner = high_level_planner
        self.simulations = simulations
        self.max_depth = max_depth
        self.exploration_weight = exploration_weight
        self.rollout_horizon = rollout_horizon
        self.tree_reuse_similarity = tree_reuse_similarity
        self.safety_cost_ceiling = safety_cost_ceiling
        self.use_baseline_guard = use_baseline_guard
        self.baseline_cost_margin = baseline_cost_margin
        self.baseline_uncertainty_scale = baseline_uncertainty_scale
        self.imagined_domain_state_builder = imagined_domain_state_builder
        self.simple_baseline_builder = simple_baseline_builder
        self._previous_root: MCTSNode | None = None

    def _maybe_reuse_root(self, z: torch.Tensor, ctx_tokens: torch.Tensor) -> tuple[MCTSNode, bool]:
        if self._previous_root is None:
            return MCTSNode(latent_state=z, ctx_tokens=ctx_tokens, depth=0), False
        similarity = F.cosine_similarity(
            z.unsqueeze(0),
            self._previous_root.latent_state.unsqueeze(0),
            dim=-1,
        )
        if float(similarity.item()) >= self.tree_reuse_similarity:
            self._previous_root.latent_state = z
            self._previous_root.ctx_tokens = ctx_tokens
            return self._previous_root, True
        return MCTSNode(latent_state=z, ctx_tokens=ctx_tokens, depth=0), False

    def _re_root_subtree(self, node: MCTSNode, *, depth: int = 0) -> MCTSNode:
        node.depth = depth
        if depth == 0:
            node.immediate_cost = 0.0
            node.path_from_parent = None
            node.action_from_parent = None
            node.total_cost = 0.0
            node.visit_count = 0
        for child in node.children:
            self._re_root_subtree(child, depth=depth + 1)
        return node

    def _make_skill_seed(
        self,
        *,
        z: torch.Tensor,
        goal_z: torch.Tensor | None,
        retrieved_skills: list[RetrievedSkill] | None,
    ) -> torch.Tensor | None:
        if self.high_level_planner is None or goal_z is None or not retrieved_skills:
            return None
        plan = self.high_level_planner.plan(z=z, goal_z=goal_z, retrieved_skills=retrieved_skills)
        if plan is None:
            return None
        path = plan.action_path
        if path.dim() == 2:
            return path.unsqueeze(0)
        return path

    def _expand_node(
        self,
        node: MCTSNode,
        *,
        domain_state: dict[str, Any],
        retrieved_postures: torch.Tensor | None = None,
        retrieved_posture_scores: torch.Tensor | None = None,
        skill_seed_path: torch.Tensor | None = None,
    ) -> None:
        simple_baseline_path = None
        if self.simple_baseline_builder is not None:
            simple_baseline_path = self.simple_baseline_builder(
                domain_state,
                self.actor.path_length,
                self.actor.action_dim,
            )
        proposal = self.actor.propose(
            node.latent_state.unsqueeze(0),
            node.ctx_tokens.unsqueeze(0),
            retrieved_postures=(
                retrieved_postures.unsqueeze(0)
                if retrieved_postures is not None and retrieved_postures.dim() == 2
                else retrieved_postures
            ),
            retrieved_posture_scores=(
                retrieved_posture_scores.unsqueeze(0)
                if retrieved_posture_scores is not None and retrieved_posture_scores.dim() == 1
                else retrieved_posture_scores
            ),
            simple_baseline_path=simple_baseline_path,
        )
        candidate_paths = proposal["candidate_paths"]
        candidate_postures = proposal["candidate_postures"]
        reasoning_states = proposal["reasoning_states"]
        if skill_seed_path is not None:
            if skill_seed_path.dim() == 1:
                skill_seed_path = skill_seed_path.unsqueeze(0).unsqueeze(0)
            if skill_seed_path.dim() == 2:
                skill_seed_path = skill_seed_path.unsqueeze(0)
            seed = skill_seed_path.to(candidate_paths)
            target_path_len = int(candidate_paths.shape[2])
            target_action_dim = int(candidate_paths.shape[3])
            if seed.shape[-1] != target_action_dim:
                if seed.shape[-1] > target_action_dim:
                    seed = seed[..., :target_action_dim]
                else:
                    pad_width = target_action_dim - int(seed.shape[-1])
                    seed = F.pad(seed, (0, pad_width))
            if seed.shape[1] != target_path_len:
                if seed.shape[1] > target_path_len:
                    seed = seed[:, :target_path_len, :]
                else:
                    pad_steps = target_path_len - int(seed.shape[1])
                    pad_value = seed[:, -1:, :].expand(-1, pad_steps, -1)
                    seed = torch.cat([seed, pad_value], dim=1)
            candidate_paths = torch.cat([candidate_paths, seed.unsqueeze(1)], dim=1)
            zero_posture = torch.zeros(
                1,
                1,
                candidate_postures.shape[-1],
                device=candidate_postures.device,
                dtype=candidate_postures.dtype,
            )
            candidate_postures = torch.cat([candidate_postures, zero_posture], dim=1)
            seed_reasoning = node.latent_state.unsqueeze(0).unsqueeze(0)
            candidate_reasoning = seed_reasoning.expand(-1, 1, reasoning_states.shape[-1])
            reasoning_states = torch.cat([reasoning_states, candidate_reasoning], dim=1)
        rollout = self.world_model(
            z=node.latent_state.unsqueeze(0),
            actions=candidate_paths,
            ctx_tokens=node.ctx_tokens.unsqueeze(0),
            candidate_postures=candidate_postures,
            reasoning_states=reasoning_states,
            horizon=min(self.rollout_horizon, candidate_paths.shape[2]),
        )
        candidate_costs = self.cost_module.score_candidates(
            z=node.latent_state.unsqueeze(0),
            actions=candidate_paths,
            ctx_tokens=node.ctx_tokens.unsqueeze(0),
            domain_state=domain_state,
            future_z=rollout["terminal_latents"],
            future_trajectory=rollout["trajectory"],
            imagined_domain_state_builder=self.imagined_domain_state_builder,
        )
        total_costs = candidate_costs["total"][0]
        node.candidate_paths = candidate_paths[0].detach()
        node.candidate_ic = candidate_costs["ic"][0].detach()
        node.candidate_tc = candidate_costs["tc"][0].detach()
        node.candidate_costs = total_costs.detach()
        node.candidate_postures = candidate_postures[0].detach()
        node.candidate_reasoning_states = reasoning_states[0].detach()
        node.candidate_terminal_latents = rollout["terminal_latents"][0].detach()
        node.candidate_trajectories = rollout["trajectory"][0].detach()
        node.expanded = True
        node.children = []
        for idx in range(candidate_paths.shape[1]):
            branch_cost = float(total_costs[idx].item())
            safety_triggered = False
            if self.safety_cost_ceiling is not None and branch_cost > self.safety_cost_ceiling:
                branch_cost += self.safety_cost_ceiling
                safety_triggered = True
            child = MCTSNode(
                latent_state=rollout["terminal_latents"][0, idx].detach(),
                ctx_tokens=node.ctx_tokens.detach(),
                depth=node.depth + 1,
                path_from_parent=candidate_paths[0, idx].detach(),
                action_from_parent=candidate_paths[0, idx, 0].detach(),
                posture=candidate_postures[0, idx].detach(),
                reasoning_state=reasoning_states[0, idx].detach(),
                immediate_cost=branch_cost,
                safety_triggered=safety_triggered,
            )
            node.children.append(child)

    def _select_child(self, node: MCTSNode) -> MCTSNode:
        if not node.children:
            raise RuntimeError("Cannot select a child from an unexpanded node.")
        parent_visits = max(1, node.visit_count)
        return max(
            node.children,
            key=lambda child: child.selection_score(parent_visits, self.exploration_weight),
        )

    def _backpropagate(self, selection_path: list[MCTSNode], leaf_cost: float) -> None:
        gamma = float(getattr(self.cost_module, "gamma", 1.0))
        if len(selection_path) == 1:
            selection_path[0].visit_count += 1
            selection_path[0].total_cost += float(leaf_cost)
            return

        cumulative_return = 0.0
        for path_idx in range(len(selection_path) - 1, 0, -1):
            node = selection_path[path_idx]
            cumulative_return = float(node.immediate_cost) + (gamma * cumulative_return)
            node.visit_count += 1
            node.total_cost += cumulative_return
        selection_path[0].visit_count += 1
        selection_path[0].total_cost += cumulative_return

    def _select_root_child(self, root: MCTSNode) -> tuple[int, MCTSNode, dict[str, Any]]:
        if root.candidate_costs is None or not root.children:
            raise RuntimeError("MCTS root never expanded.")
        best_idx, best_child = min(
            enumerate(root.children),
            key=lambda item: item[1].mean_cost,
        )
        selection_reason = "lowest_mean_cost"
        predicted_costs = root.candidate_costs.detach().float()
        mean_costs = torch.tensor(
            [child.mean_cost for child in root.children],
            dtype=predicted_costs.dtype,
            device=predicted_costs.device,
        )
        predicted_cost_std = (
            float(predicted_costs.std(unbiased=False).item())
            if predicted_costs.numel() > 1
            else 0.0
        )
        mean_cost_std = (
            float(mean_costs.std(unbiased=False).item())
            if mean_costs.numel() > 1
            else 0.0
        )
        actual_predicted_improvement: float | None = None
        required_predicted_improvement: float | None = None
        baseline_predicted_total: float | None = None

        if self.use_baseline_guard and self.simple_baseline_builder is not None and root.children:
            baseline_child = root.children[0]
            if baseline_child.path_from_parent is not None:
                baseline_predicted_total = float(predicted_costs[0].item())
                selected_predicted_total = float(predicted_costs[best_idx].item())
                actual_predicted_improvement = baseline_predicted_total - selected_predicted_total
                required_predicted_improvement = max(
                    float(self.baseline_cost_margin),
                    float(self.baseline_uncertainty_scale) * predicted_cost_std,
                )
                if (
                    best_idx != 0
                    and actual_predicted_improvement <= required_predicted_improvement
                ):
                    best_idx = 0
                    best_child = baseline_child
                    selection_reason = "baseline_uncertainty_guard"
                elif baseline_child.mean_cost <= (best_child.mean_cost + self.baseline_cost_margin):
                    best_idx = 0
                    best_child = baseline_child
                    selection_reason = "baseline_margin"

        return best_idx, best_child, {
            "reason": selection_reason,
            "use_baseline_guard": bool(self.use_baseline_guard),
            "baseline_cost_margin": float(self.baseline_cost_margin),
            "baseline_uncertainty_scale": float(self.baseline_uncertainty_scale),
            "root_candidate_mean_costs": [float(child.mean_cost) for child in root.children],
            "root_candidate_visit_counts": [int(child.visit_count) for child in root.children],
            "baseline_mean_cost": float(root.children[0].mean_cost),
            "selected_mean_cost": float(best_child.mean_cost),
            "root_mean_cost_std": mean_cost_std,
            "root_predicted_cost_std": predicted_cost_std,
            "baseline_predicted_total": baseline_predicted_total,
            "selected_predicted_total": float(predicted_costs[best_idx].item()),
            "actual_predicted_improvement": actual_predicted_improvement,
            "required_predicted_improvement": required_predicted_improvement,
        }

    def search(
        self,
        *,
        z: torch.Tensor,
        ctx_tokens: torch.Tensor,
        domain_state: dict[str, Any],
        goal_z: torch.Tensor | None = None,
        retrieved_postures: torch.Tensor | None = None,
        retrieved_posture_scores: torch.Tensor | None = None,
        retrieved_skills: list[RetrievedSkill] | None = None,
        budget_multiplier: float = 1.0,
    ) -> MCTSResult:
        root, reused_tree = self._maybe_reuse_root(z.detach(), ctx_tokens.detach())
        bounded_multiplier = max(0.5, min(4.0, float(budget_multiplier)))
        simulation_budget = max(1, int(round(float(self.simulations) * bounded_multiplier)))
        skill_seed_path = self._make_skill_seed(
            z=z.detach(),
            goal_z=goal_z.detach() if goal_z is not None else None,
            retrieved_skills=retrieved_skills,
        )
        for _ in range(simulation_budget):
            node = root
            selection_path = [root]
            while node.expanded and node.children and node.depth < self.max_depth:
                node = self._select_child(node)
                selection_path.append(node)
            if node.depth < self.max_depth and not node.expanded:
                self._expand_node(
                    node,
                    domain_state=domain_state,
                    retrieved_postures=retrieved_postures,
                    retrieved_posture_scores=retrieved_posture_scores,
                    skill_seed_path=skill_seed_path if node.depth == 0 else None,
                )
                if node.children:
                    node = min(node.children, key=lambda child: child.immediate_cost)
                    selection_path.append(node)
            self._backpropagate(selection_path, node.immediate_cost)
        if root.candidate_paths is None or root.candidate_costs is None or not root.children:
            raise RuntimeError("MCTS root never expanded.")
        best_idx, best_child, selection_debug = self._select_root_child(root)
        selected_path = best_child.path_from_parent
        if selected_path is None:
            raise RuntimeError("Expanded child is missing a selected path.")
        reasoning_chain = FrozenReasoningChain(
            steps=(
                ReasoningStep(
                    state=root.latent_state.detach(),
                    candidate_paths=root.candidate_paths.detach(),
                    candidate_costs=root.candidate_costs.detach(),
                    selected_path=selected_path.detach(),
                    source="mcts_root",
                    depth=0,
                ),
                ReasoningStep(
                    state=best_child.latent_state.detach(),
                    candidate_paths=None,
                    candidate_costs=None,
                    selected_path=selected_path.detach(),
                    source="mcts_selected",
                    depth=best_child.depth,
                ),
            )
        )
        tree_trace = export_tree(root)
        tree_trace["selected_candidate_idx"] = int(best_idx)
        tree_trace["selection_debug"] = selection_debug
        tree_trace["budget_multiplier"] = bounded_multiplier
        tree_trace["simulation_budget"] = simulation_budget
        tree_trace["root_candidate_costs"] = [float(value) for value in root.candidate_costs.tolist()]
        if root.candidate_ic is not None:
            tree_trace["root_candidate_ic"] = [float(value) for value in root.candidate_ic.tolist()]
        if root.candidate_tc is not None:
            tree_trace["root_candidate_tc"] = [float(value) for value in root.candidate_tc.tolist()]
        self._previous_root = self._re_root_subtree(best_child, depth=0)
        return MCTSResult(
            selected_path=selected_path.detach(),
            selected_action=selected_path[0].detach(),
            selected_posture=best_child.posture.detach() if best_child.posture is not None else None,
            candidate_paths=root.candidate_paths.detach(),
            candidate_ic=root.candidate_ic.detach() if root.candidate_ic is not None else None,
            candidate_tc=root.candidate_tc.detach() if root.candidate_tc is not None else None,
            candidate_costs=root.candidate_costs.detach(),
            candidate_postures=(
                root.candidate_postures.detach() if root.candidate_postures is not None else None
            ),
            candidate_reasoning_states=(
                root.candidate_reasoning_states.detach()
                if root.candidate_reasoning_states is not None
                else None
            ),
            candidate_terminal_latents=(
                root.candidate_terminal_latents.detach()
                if root.candidate_terminal_latents is not None
                else None
            ),
            candidate_trajectories=(
                root.candidate_trajectories.detach()
                if root.candidate_trajectories is not None
                else None
            ),
            selected_candidate_idx=best_idx,
            root=root,
            reasoning_chain=reasoning_chain,
            tree_trace=tree_trace,
            reused_tree=reused_tree,
        )

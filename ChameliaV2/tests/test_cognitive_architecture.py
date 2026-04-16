"""Harness tests for the additive cognitive architecture."""

from __future__ import annotations

from pathlib import Path
import threading

import pytest
import torch
import torch.nn as nn

from src.chamelia.actor import Actor
from src.chamelia.chamelia import Chamelia
from src.chamelia.cognitive.clustering import DomainIndex, LoRAAdapterBank
from src.chamelia.cognitive.lancedb_assessment import assess_vector_backends
from src.chamelia.cognitive.latent_action import LatentActionEncoder, LatentSkillCandidate
from src.chamelia.cognitive.mamba_world_model import (
    MambaActionConditionedWorldModel,
    benchmark_world_models,
)
from src.chamelia.cognitive.planning import HighLevelPlanner, MCTSNode, MCTSSearch, Talker
from src.chamelia.cognitive.procedural import ProceduralMemory
from src.chamelia.cognitive.procedural import (
    SKILL_LIFECYCLE_ACTIVE,
    SKILL_LIFECYCLE_INTERNALIZABLE,
    SKILL_LIFECYCLE_PROVISIONAL,
)
from src.chamelia.cognitive.representation import (
    ContrastiveSparseRepresentation,
    InformationOrderedBottleneck,
    IsotropicSkillCodec,
    VectorQuantizer,
)
from src.chamelia.cognitive.storage import _import_lancedb, _import_pyarrow
import src.chamelia.cognitive.sleep as sleep_module
from src.chamelia.cognitive.sleep import SleepCoordinator
from src.chamelia.cognitive.sleep import (
    BODEGenOptimizer,
    GemmaAutoDocWorker,
    LILOAutoDoc,
    LOVEDecomposer,
    StitchCompressor,
)
from src.chamelia.cognitive.storage import CognitiveStorage
from src.chamelia.configurator import Configurator
from src.chamelia.cost import CostModule, IntrinsicCost, TrainableCritic
from src.chamelia.memory import EpisodeRecord, LatentMemory
from src.chamelia.plugins.base import DomainRegistry
from src.chamelia.world_model import ActionConditionedWorldModel
from src.models.hjepa import HJEPA
from tests.test_chamelia import DummyDomain, DummyHJEPA


def _build_chamelia(
    root: Path,
    *,
    embed_dim: int = 32,
    action_dim: int = 8,
    num_ctx_tokens: int = 4,
    world_model_backend: str = "transformer",
) -> tuple[Chamelia, DummyDomain, LatentMemory, ProceduralMemory]:
    domain = DummyDomain(embed_dim=embed_dim, action_dim=action_dim)
    DomainRegistry.register(domain)
    hjepa = DummyHJEPA(embed_dim=embed_dim)
    configurator = Configurator(
        embed_dim=embed_dim,
        num_ctx_tokens=num_ctx_tokens,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        memory_read_k=4,
    )
    actor = Actor(
        embed_dim=embed_dim,
        action_dim=action_dim,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_ctx_tokens=num_ctx_tokens,
        num_candidates=3,
        path_length=3,
        posture_dim=8,
    )
    intrinsic = IntrinsicCost(
        cost_fns=[lambda z, action, domain_state: action.pow(2).mean(dim=-1)],
        weights=[1.0],
    )
    critic = TrainableCritic(
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=1,
        mlp_ratio=2.0,
        dropout=0.0,
        num_ctx_tokens=num_ctx_tokens,
        horizon=4,
    )
    cost = CostModule(intrinsic_cost=intrinsic, trainable_critic=critic)
    memory = LatentMemory(embed_dim=embed_dim, max_episodes=32, retrieval_k=4, device="cpu")
    procedural = ProceduralMemory(
        root=root / "procedural",
        skill_dim=embed_dim,
        use_faiss=False,
        use_lancedb=False,
    )
    planner = HighLevelPlanner(embed_dim=embed_dim, skill_dim=embed_dim)
    world_model = None
    if world_model_backend != "mamba":
        world_model = ActionConditionedWorldModel(
            embed_dim=embed_dim,
            action_dim=action_dim,
            posture_dim=actor.posture_dim,
            num_heads=4,
            num_layers=1,
            mlp_ratio=2.0,
            dropout=0.0,
            max_horizon=4,
        )
    mcts = MCTSSearch(
        actor=actor,
        world_model=(
            world_model
            if world_model is not None
            else MambaActionConditionedWorldModel(
                embed_dim=embed_dim,
                action_dim=action_dim,
                posture_dim=actor.posture_dim,
                max_horizon=4,
                use_native_mamba=False,
            )
        ),
        cost_module=cost,
        high_level_planner=planner,
        simulations=4,
        max_depth=2,
        rollout_horizon=3,
    )
    model = Chamelia(
        hjepa=hjepa,
        configurator=configurator,
        actor=actor,
        cost=cost,
        memory=memory,
        procedural_memory=procedural,
        world_model=world_model,
        mcts_search=mcts,
        high_level_planner=planner,
        domain=domain,
        embed_dim=embed_dim,
        action_dim=action_dim,
        num_ctx_tokens=num_ctx_tokens,
        planner_backend="mcts",
        world_model_backend=world_model_backend,
    )
    return model, domain, memory, procedural


def test_representation_and_procedural_memory_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    vq = VectorQuantizer(embed_dim=8, codebook_size=16)
    quantized = vq(torch.randn(2, 4, 8))
    assert quantized.quantized.shape == (2, 4, 8)
    iob = InformationOrderedBottleneck(input_dim=8, bottleneck_dim=6)
    assert iob(torch.randn(3, 8)).shape == (3, 6)
    csr = ContrastiveSparseRepresentation(input_dim=8, output_dim=16, active_dims=4)
    sparse = csr(torch.randn(2, 8))
    assert sparse.shape == (2, 16)
    codec = IsotropicSkillCodec(embed_dim=8, num_tokens=4, codebook_size=16)
    coded = codec(torch.randn(2, 8))
    assert coded["reconstructed"].shape == (2, 8)

    memory = ProceduralMemory(root=tmp_path / "proc", skill_dim=16, use_faiss=False, csr_encoder=csr)
    record = memory.add_skill(
        embedding=torch.randn(8),
        action_path=torch.randn(3, 5),
        source_episodes=(1, 2),
        confidence=0.8,
        name="toy_skill",
    )
    retrieved = memory.retrieve(torch.randn(8), k=1)
    memory.save()
    reloaded = ProceduralMemory(root=tmp_path / "proc", skill_dim=16, use_faiss=False, csr_encoder=csr)

    assert record.skill_id == 1
    assert len(retrieved) == 1
    assert reloaded.get_skill(1) is not None

    episodic = LatentMemory(
        embed_dim=8,
        max_episodes=8,
        retrieval_k=1,
        device="cpu",
        iob_encoder=iob,
        iob_widths=(2, 4, 6),
    )
    episodic.store(
        EpisodeRecord(
            key=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            action=torch.zeros(5),
            ctx_tokens=torch.zeros(2, 8),
            ic_at_decision=0.2,
            ic_realized=0.1,
            tc_predicted=0.2,
            outcome_key=torch.zeros(8),
            step=0,
            domain_name="toy",
        )
    )
    episodic.store(
        EpisodeRecord(
            key=torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            action=torch.zeros(5),
            ctx_tokens=torch.zeros(2, 8),
            ic_at_decision=0.2,
            ic_realized=0.1,
            tc_predicted=0.2,
            outcome_key=torch.zeros(8),
            step=1,
            domain_name="toy",
        )
    )
    retrieved_keys, episode_lists, _ = episodic.retrieve_scored(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert retrieved_keys is not None
    assert episode_lists[0][0].step == 0


def test_lancedb_episode_archive_roundtrip(tmp_path: Path) -> None:
    if _import_lancedb() is None or _import_pyarrow() is None:
        pytest.skip("Requires lancedb and pyarrow")
    storage = CognitiveStorage(tmp_path / "archive")
    storage.archive_episode(
        7,
        {
            "key": torch.tensor([1.0, 2.0, 3.0]),
            "selected_path": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
            "candidate_total": torch.tensor([0.5, 0.25]),
            "domain_name": "toy",
            "metadata": {"source": "test"},
        },
    )
    fetched = storage.fetch_archived_episode(7)

    assert fetched is not None
    assert fetched["domain_name"] == "toy"
    assert fetched["metadata"]["source"] == "test"
    assert fetched["selected_path"] == [[0.10000000149011612, 0.20000000298023224], [0.30000001192092896, 0.4000000059604645]]


def test_domain_index_and_lora_bank_apply_cluster(tmp_path: Path) -> None:
    torch.manual_seed(0)

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(8, 2, batch_first=True)

    model = TinyModel()
    bank = LoRAAdapterBank(model, rank=2)
    index = DomainIndex(tmp_path / "domains", adapter_bank=bank)
    seed_latent = torch.randn(8)
    route = index.route(seed_latent, "dummy")
    index.record_skill_trigger(route.cluster_id, skill_id=3, weight=1.25)
    bank.ensure_cluster(route.cluster_id)
    for adapter in bank.adapters[route.cluster_id].values():
        adapter.out_proj_b.fill_(0.05)
        adapter.out_proj_a.fill_(0.05)
    second_route = index.route(seed_latent * 0.99, "dummy")

    inputs = torch.randn(1, 4, 8)
    baseline = model.attn(inputs, inputs, inputs)[0]
    bank.apply_mixture(second_route.mixture_cluster_ids, second_route.mixture_weights)
    adapted = model.attn(inputs, inputs, inputs)[0]

    assert route.spawned_new
    assert 3 in index.get_trigger_weights(route.cluster_id)
    assert second_route.primary_cluster_id == route.cluster_id
    assert abs(sum(second_route.mixture_weights) - 1.0) < 1.0e-6
    assert not torch.allclose(baseline, adapted)


def test_mcts_backpropagates_discounted_costs(tmp_path: Path) -> None:
    model, _domain, _memory, _procedural = _build_chamelia(tmp_path)
    assert model.mcts_search is not None
    root = MCTSNode(latent_state=torch.zeros(32), ctx_tokens=torch.zeros(4, 32), depth=0)
    child = MCTSNode(latent_state=torch.zeros(32), ctx_tokens=torch.zeros(4, 32), depth=1, immediate_cost=1.5)
    leaf = MCTSNode(latent_state=torch.zeros(32), ctx_tokens=torch.zeros(4, 32), depth=2, immediate_cost=2.0)
    model.mcts_search._backpropagate([root, child, leaf], leaf_cost=leaf.immediate_cost)

    gamma = float(model.cost.gamma)
    assert leaf.total_cost == torch.tensor(2.0).item()
    assert abs(child.total_cost - (1.5 + (gamma * 2.0))) < 1.0e-6
    assert abs(root.total_cost - child.total_cost) < 1.0e-6


def test_mcts_baseline_guard_rejects_tiny_predicted_edge(tmp_path: Path) -> None:
    model, _domain, _memory, _procedural = _build_chamelia(tmp_path)
    assert model.mcts_search is not None
    root = MCTSNode(latent_state=torch.zeros(32), ctx_tokens=torch.zeros(4, 32), depth=0)
    root.candidate_paths = torch.randn(3, 3, 8)
    root.candidate_costs = torch.tensor([0.50, 0.45, 0.47], dtype=torch.float32)
    root.children = [
        MCTSNode(
            latent_state=torch.zeros(32),
            ctx_tokens=torch.zeros(4, 32),
            depth=1,
            path_from_parent=root.candidate_paths[0],
            visit_count=4,
            total_cost=4.0,
        ),
        MCTSNode(
            latent_state=torch.zeros(32),
            ctx_tokens=torch.zeros(4, 32),
            depth=1,
            path_from_parent=root.candidate_paths[1],
            visit_count=4,
            total_cost=1.8,
        ),
        MCTSNode(
            latent_state=torch.zeros(32),
            ctx_tokens=torch.zeros(4, 32),
            depth=1,
            path_from_parent=root.candidate_paths[2],
            visit_count=4,
            total_cost=2.4,
        ),
    ]

    best_idx, _best_child, selection_debug = model.mcts_search._select_root_child(root)

    assert best_idx == 0
    assert selection_debug["reason"] == "baseline_uncertainty_guard"
    assert selection_debug["actual_predicted_improvement"] is not None
    assert selection_debug["required_predicted_improvement"] is not None
    assert (
        float(selection_debug["actual_predicted_improvement"])
        <= float(selection_debug["required_predicted_improvement"])
    )


def test_mcts_can_disable_baseline_guard(tmp_path: Path) -> None:
    model, _domain, _memory, _procedural = _build_chamelia(tmp_path)
    assert model.mcts_search is not None
    model.mcts_search.use_baseline_guard = False
    root = MCTSNode(latent_state=torch.zeros(32), ctx_tokens=torch.zeros(4, 32), depth=0)
    root.candidate_paths = torch.randn(3, 3, 8)
    root.candidate_costs = torch.tensor([0.50, 0.45, 0.47], dtype=torch.float32)
    root.children = [
        MCTSNode(
            latent_state=torch.zeros(32),
            ctx_tokens=torch.zeros(4, 32),
            depth=1,
            path_from_parent=root.candidate_paths[0],
            visit_count=4,
            total_cost=4.0,
        ),
        MCTSNode(
            latent_state=torch.zeros(32),
            ctx_tokens=torch.zeros(4, 32),
            depth=1,
            path_from_parent=root.candidate_paths[1],
            visit_count=4,
            total_cost=1.8,
        ),
        MCTSNode(
            latent_state=torch.zeros(32),
            ctx_tokens=torch.zeros(4, 32),
            depth=1,
            path_from_parent=root.candidate_paths[2],
            visit_count=4,
            total_cost=2.4,
        ),
    ]

    best_idx, _best_child, selection_debug = model.mcts_search._select_root_child(root)

    assert best_idx == 1
    assert selection_debug["reason"] == "lowest_mean_cost"
    assert selection_debug["use_baseline_guard"] is False


def test_mcts_reuses_selected_subtree_as_new_root(tmp_path: Path) -> None:
    model, domain, _memory, _procedural = _build_chamelia(tmp_path)
    assert model.mcts_search is not None
    observation = torch.randint(0, 5, (1, 16))
    domain_state = domain.get_domain_state(observation)
    tokens = observation.long()
    mask = torch.zeros(tokens.shape[0], tokens.shape[1])
    outputs = model(
        tokens=domain.get_tokenizer()(tokens).tokens,
        mask=mask,
        domain_state=domain_state,
        actor_mode="mode2",
        store_to_memory=False,
        input_kind="embedded_tokens",
    )
    selected_idx = int(outputs["selected_candidate_idx"][0].item())
    root_trace = outputs["mcts_traces"][0]
    assert root_trace is not None
    assert root_trace["children"][selected_idx]["visit_count"] > 0
    assert root_trace["selection_debug"]["selected_mean_cost"] <= max(
        root_trace["selection_debug"]["root_candidate_mean_costs"]
    )
    selected_terminal = outputs["rollout"]["terminal_latents"][0, selected_idx]
    result = model.mcts_search.search(
        z=selected_terminal.detach(),
        ctx_tokens=outputs["ctx_tokens"][0].detach(),
        domain_state=domain_state,
        goal_z=None,
        retrieved_postures=None,
        retrieved_posture_scores=None,
        retrieved_skills=[],
    )
    assert result.reused_tree
    assert result.root.depth == 0
    assert result.root.path_from_parent is None


def test_mcts_budget_multiplier_scales_simulation_budget(tmp_path: Path) -> None:
    model, _domain, _memory, _procedural = _build_chamelia(tmp_path)
    assert model.mcts_search is not None

    class ExpandCounter:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(
            self,
            node: MCTSNode,
            *,
            domain_state: dict[str, torch.Tensor],
            retrieved_postures: torch.Tensor | None = None,
            retrieved_posture_scores: torch.Tensor | None = None,
            skill_seed_path: torch.Tensor | None = None,
        ) -> None:
            _ = domain_state
            _ = retrieved_postures
            _ = retrieved_posture_scores
            _ = skill_seed_path
            self.calls += 1
            if node.expanded:
                return
            node.candidate_paths = torch.zeros(2, 3, model.actor.action_dim)
            node.candidate_costs = torch.tensor([0.5, 0.25], dtype=torch.float32)
            node.candidate_ic = torch.zeros(2, dtype=torch.float32)
            node.candidate_tc = torch.zeros(2, dtype=torch.float32)
            node.candidate_postures = torch.zeros(2, model.actor.posture_dim)
            node.candidate_reasoning_states = torch.zeros(2, model.actor.embed_dim)
            node.candidate_terminal_latents = torch.zeros(2, model.actor.embed_dim)
            node.candidate_trajectories = torch.zeros(2, 3, model.actor.embed_dim)
            node.expanded = True
            node.children = [
                MCTSNode(
                    latent_state=torch.zeros(model.actor.embed_dim),
                    ctx_tokens=torch.zeros(4, model.actor.embed_dim),
                    depth=node.depth + 1,
                    path_from_parent=node.candidate_paths[idx],
                    immediate_cost=float(node.candidate_costs[idx].item()),
                )
                for idx in range(2)
            ]

    counter = ExpandCounter()
    model.mcts_search._expand_node = counter  # type: ignore[assignment]
    result = model.mcts_search.search(
        z=torch.zeros(model.actor.embed_dim),
        ctx_tokens=torch.zeros(4, model.actor.embed_dim),
        domain_state={},
        goal_z=None,
        retrieved_postures=None,
        retrieved_posture_scores=None,
        retrieved_skills=[],
        budget_multiplier=1.75,
    )

    assert result.tree_trace["simulation_budget"] == 7
    assert result.tree_trace["budget_multiplier"] == 1.75
    assert counter.calls >= 1


def test_mcts_planner_and_talker_harness(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model, _domain, _memory, procedural = _build_chamelia(tmp_path)
    encoder = LatentActionEncoder(action_dim=8, skill_dim=32, num_heads=4, num_layers=1, max_path_length=4)
    action_path = torch.randn(3, 8)
    skill = encoder(action_path)
    procedural.add_skill(embedding=skill.squeeze(0), action_path=action_path, confidence=0.9)
    talker = Talker(latent_dim=32, vocab_size=24, num_heads=4, num_layers=1, max_tokens=5)

    domain = model.domain
    tokenizer = domain.get_tokenizer()
    raw_tokens = tokenizer.collate([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
    tokenized = tokenizer(raw_tokens)
    mask = torch.zeros(raw_tokens.shape[0], raw_tokens.shape[1], dtype=torch.float32)
    domain_state = domain.get_domain_state(raw_tokens)
    domain_state["goal_latent"] = torch.ones(1, 32)

    outputs = model(
        tokens=tokenized.tokens,
        mask=mask,
        domain_state=domain_state,
        actor_mode="mode2",
        store_to_memory=True,
    )
    assert outputs["planner_backend"] == "mcts"
    assert outputs["thinker_output"] is not None
    logits = talker(outputs["thinker_output"])
    assert outputs["selected_path"].shape == (1, 3, 8)
    assert logits.shape == (1, 5, 24)


def test_sleep_cycle_promotes_skills_from_toy_memory(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model, domain, memory, procedural = _build_chamelia(tmp_path)
    encoder = LatentActionEncoder(action_dim=8, skill_dim=32, num_heads=4, num_layers=1, max_path_length=4)
    sleep = SleepCoordinator(
        episodic_memory=memory,
        procedural_memory=procedural,
        latent_action_encoder=encoder,
        world_model=model.world_model,
        cost_module=model.cost,
        sleep_interval_steps=999,
    )
    tokenizer = domain.get_tokenizer()
    raw_tokens = tokenizer.collate([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
    tokenized = tokenizer(raw_tokens)
    mask = torch.zeros(raw_tokens.shape[0], raw_tokens.shape[1], dtype=torch.float32)
    domain_state = domain.get_domain_state(raw_tokens)
    domain_state["goal_latent"] = torch.ones(1, 32)

    for step in range(3):
        outputs = model(
            tokens=tokenized.tokens,
            mask=mask,
            domain_state=domain_state,
            actor_mode="mode2",
            store_to_memory=True,
        )
        model.fill_outcome(ic_realized=0.2 + (0.05 * step), outcome_z=outputs["z"])

    report = sleep.run_cycle()

    assert len(report.promotions) >= 1
    assert len(procedural.records) >= 1
    assert memory.size <= 3


def test_sleep_fasttrack_promotes_without_waiting_for_sleep_cycle(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model, _domain, memory, procedural = _build_chamelia(tmp_path)
    encoder = LatentActionEncoder(action_dim=8, skill_dim=32, num_heads=4, num_layers=1, max_path_length=4)
    sleep = SleepCoordinator(
        episodic_memory=memory,
        procedural_memory=procedural,
        latent_action_encoder=encoder,
        world_model=model.world_model,
        cost_module=model.cost,
        sleep_interval_steps=999,
        fasttrack_return_threshold=0.5,
        fasttrack_surprise_threshold=0.1,
    )
    record = EpisodeRecord(
        key=torch.randn(32),
        action=torch.tensor([0.0, 1.0] + [0.0] * 6),
        ctx_tokens=torch.randn(4, 32),
        ic_at_decision=0.1,
        ic_realized=-1.5,
        tc_predicted=0.0,
        outcome_key=torch.randn(32),
        step=0,
        domain_name="toy",
        selected_posture=torch.randn(8),
        selected_path=torch.randn(3, 8),
        record_id=7,
    )
    memory.store(record)

    promotion = sleep.maybe_fasttrack_record(record, utility_score=1.0, surprise_score=0.5)

    assert promotion is not None
    assert promotion.source == "online_fasttrack"
    assert len(procedural.records) == 1
    promoted_record = next(iter(procedural.records.values()))
    assert promoted_record.extras["promotion_source"] == "online_fasttrack"


class _ConstantWorldModel(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.embed_dim = embed_dim
        self.max_horizon = 4

    def forward(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        ctx_tokens: torch.Tensor,
        candidate_postures: torch.Tensor | None = None,
        reasoning_states: torch.Tensor | None = None,
        horizon: int = 1,
    ) -> dict[str, torch.Tensor]:
        _ = candidate_postures, reasoning_states, horizon
        batch_size, candidates, path_length, _ = actions.shape
        zeros = torch.zeros(
            batch_size,
            candidates,
            path_length,
            self.embed_dim,
            dtype=z.dtype,
            device=z.device,
        )
        return {
            "trajectory": zeros,
            "terminal_latents": zeros[:, :, -1, :],
        }


class _ActionSumCost:
    def score_candidates(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        ctx_tokens: torch.Tensor,
        domain_state: dict[str, object],
        future_z: torch.Tensor,
        future_trajectory: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _ = z, ctx_tokens, domain_state, future_z, future_trajectory
        total = -actions.sum(dim=-1).sum(dim=-1)
        return {
            "total": total,
            "ic": total,
            "tc": torch.zeros_like(total),
        }


def test_provisional_skill_is_persisted_but_hidden_from_default_retrieval(tmp_path: Path) -> None:
    procedural = ProceduralMemory(
        root=tmp_path / "procedural",
        skill_dim=8,
        use_faiss=False,
        use_lancedb=False,
    )
    embedding = torch.ones(8)
    record = procedural.add_skill(
        embedding=embedding,
        action_path=torch.ones(3, 4),
        name="candidate",
        lifecycle=SKILL_LIFECYCLE_PROVISIONAL,
    )

    hidden = procedural.retrieve(embedding, k=4)
    visible = procedural.retrieve(embedding, k=4, include_provisional=True)

    assert procedural.is_provisional(record)
    assert hidden == []
    assert len(visible) == 1
    assert visible[0].record.skill_id == record.skill_id


def test_sleep_cycle_registers_rejected_skill_as_provisional_and_activates_only_baseline_winner(
    tmp_path: Path,
) -> None:
    memory = LatentMemory(embed_dim=8, max_episodes=8, retrieval_k=1, device="cpu")
    procedural = ProceduralMemory(
        root=tmp_path / "procedural",
        skill_dim=8,
        use_faiss=False,
        use_lancedb=False,
    )
    encoder = LatentActionEncoder(
        action_dim=4,
        skill_dim=8,
        num_heads=2,
        num_layers=1,
        max_path_length=4,
    )
    sleep = SleepCoordinator(
        episodic_memory=memory,
        procedural_memory=procedural,
        latent_action_encoder=encoder,
        world_model=_ConstantWorldModel(embed_dim=8),
        cost_module=_ActionSumCost(),
        sleep_interval_steps=999,
    )
    record = EpisodeRecord(
        key=torch.zeros(8),
        action=torch.zeros(4),
        ctx_tokens=torch.zeros(2, 8),
        ic_at_decision=0.0,
        ic_realized=-1.0,
        tc_predicted=0.0,
        outcome_key=torch.zeros(8),
        step=0,
        domain_name="toy",
        selected_path=torch.ones(3, 4),
        record_id=1,
    )
    memory.store(record)
    good = LatentSkillCandidate(
        action_path=torch.full((3, 4), 2.0),
        symbolic_codes=None,
        target_delta=None,
        source_weight=0.5,
        source_episodes=(1,),
    )
    bad = LatentSkillCandidate(
        action_path=torch.full((3, 4), -2.0),
        symbolic_codes=None,
        target_delta=None,
        source_weight=3.0,
        source_episodes=(1,),
    )
    sleep.decomposer.decompose = lambda records: []
    sleep.compressor.compress = lambda segments: []
    sleep.dream.extract = lambda records: []
    sleep.rsd.propose = lambda **kwargs: [bad, good]
    sleep._bodegen_candidates = lambda **kwargs: []

    report = sleep.run_cycle()
    stored_records = list(procedural.records.values())
    active = [item for item in stored_records if procedural.is_active(item)]
    provisional = [item for item in stored_records if procedural.is_provisional(item)]
    active_record = active[0]
    provisional_record = provisional[0]

    assert len(report.promotions) == 1
    assert len(stored_records) == 2
    assert len(active) == 1
    assert len(provisional) == 1
    assert procedural.retrieve(active_record.embedding, k=4)
    assert not any(
        item.record.skill_id == provisional_record.skill_id
        for item in procedural.retrieve(provisional_record.embedding, k=4, include_provisional=False)
    )


def test_realized_usage_unlocks_internalizable_lifecycle(tmp_path: Path) -> None:
    procedural = ProceduralMemory(
        root=tmp_path / "procedural",
        skill_dim=8,
        use_faiss=False,
        use_lancedb=False,
        internalization_min_usage_count=2,
        internalization_min_success_count=1,
        internalization_min_avg_reward=0.0,
    )
    proven = procedural.add_skill(
        embedding=torch.randn(8),
        action_path=torch.randn(3, 4),
        name="proven",
        lifecycle=SKILL_LIFECYCLE_ACTIVE,
    )
    harmful = procedural.add_skill(
        embedding=torch.randn(8),
        action_path=torch.randn(3, 4),
        name="harmful",
        lifecycle=SKILL_LIFECYCLE_ACTIVE,
    )

    procedural.record_realized_usage((proven.skill_id,), realized_cost=-1.0)
    procedural.record_realized_usage((proven.skill_id,), realized_cost=-0.5)
    procedural.record_realized_usage((harmful.skill_id,), realized_cost=1.0)
    procedural.record_realized_usage((harmful.skill_id,), realized_cost=1.0)

    proven_record = procedural.get_skill(proven.skill_id)
    harmful_record = procedural.get_skill(harmful.skill_id)

    assert proven_record is not None
    assert harmful_record is not None
    assert procedural.lifecycle_for(proven_record) == SKILL_LIFECYCLE_INTERNALIZABLE
    assert procedural.lifecycle_for(harmful_record) == SKILL_LIFECYCLE_ACTIVE
    assert {record.skill_id for record in procedural.internalizable_records()} == {proven.skill_id}


def test_elite_restore_restores_procedural_state_and_replay(tmp_path: Path) -> None:
    from training.orchestrator import (
        DomainPhaseConfig,
        DomainRunConfig,
        OrchestratorConfig,
        UnifiedTrainingOrchestrator,
    )

    config = OrchestratorConfig(
        device="cpu",
        run_dir=str(tmp_path / "rollback"),
        replay_capacity=32,
        family_backbones={
            "state_vector_hjepa": {
                "encoder_type": "vit_tiny_patch16_224",
                "embed_dim": 192,
                "predictor_depth": 1,
                "predictor_num_heads": 4,
                "num_hierarchies": 2,
            }
        },
        memory={"max_episodes": 16, "retrieval_k": 2, "device": "cpu", "use_iob": False},
        procedural={"use_faiss": False, "use_lancedb": False, "use_csr": False, "use_isotropic_storage": False},
        sleep={"interval_steps": 2},
        logging={"representation_loss": "hjepa", "num_candidates": 3, "path_length": 2},
        domains=[
            DomainRunConfig(
                name="cartpole",
                family="state_vector_hjepa",
                bootstrap_random_episodes=1,
                bootstrap_simple_episodes=1,
                bootstrap_pretrain_steps=1,
                bootstrap_replay_warmup_steps=1,
                bootstrap_batch_size=1,
                mask_ratio=0.25,
                max_episode_steps=4,
                optimizer_interval=1,
                sleep_interval_episodes=1,
                checkpoint_interval_episodes=1,
                evaluation_episodes=1,
                world_model_backend="mamba",
                mcts_simulations=1,
                mcts_depth=1,
                mcts_rollout_horizon=1,
                baselines=("random",),
                phases={"core_control": DomainPhaseConfig(episodes=1)},
            )
        ],
    )
    orchestrator = UnifiedTrainingOrchestrator(config)
    domain_cfg = config.domains[0]
    adapter = orchestrator._build_adapter(domain_cfg)
    model = orchestrator._build_model(domain_cfg, adapter)
    optimizer = torch.optim.AdamW(orchestrator._parameter_groups(model, domain_cfg, DomainPhaseConfig(episodes=1)))
    observation, info = adapter.reset(seed=0)
    tokenized = adapter.tokenize_observation(observation)
    domain_state = adapter.build_domain_state(observation, info)
    model(
        tokens=tokenized.tokens.to(orchestrator.device),
        mask=torch.zeros(
            tokenized.tokens.shape[0],
            tokenized.tokens.shape[1],
            dtype=torch.float32,
            device=orchestrator.device,
        ),
        domain_state=domain_state,
        actor_mode="mode2",
        store_to_memory=False,
    )

    initial_param = next(model.parameters()).detach().clone()
    first_skill = model.procedural_memory.add_skill(
        embedding=torch.randn(model.embed_dim),
        action_path=torch.randn(model.actor.path_length, model.action_dim),
        name="elite_skill",
        lifecycle=SKILL_LIFECYCLE_ACTIVE,
    )
    assert model.domain_index is not None
    elite_route = model.domain_index.route(torch.zeros(model.embed_dim), adapter.domain_name)
    model.domain_index.record_skill_trigger(elite_route.cluster_id, first_skill.skill_id, weight=2.0)
    checkpoint_path = orchestrator._save_checkpoint(
        domain_cfg=domain_cfg,
        phase_name="core_control",
        episode_idx=1,
        model=model,
        optimizer=optimizer,
        evaluation={"full": {"episode_reward_mean": 100.0}},
    )

    with torch.no_grad():
        next(model.parameters()).add_(5.0)
    model.procedural_memory.add_skill(
        embedding=torch.randn(model.embed_dim),
        action_path=torch.randn(model.actor.path_length, model.action_dim),
        name="poison",
        lifecycle=SKILL_LIFECYCLE_ACTIVE,
    )
    model.domain_index.record_skill_trigger(elite_route.cluster_id, 999, weight=5.0)
    payload = torch.load(checkpoint_path, map_location=orchestrator.device, weights_only=False)
    orchestrator._restore_checkpoint_in_place(payload, model=model, optimizer=optimizer)

    restored_param = next(model.parameters()).detach().clone()
    restored_records = list(model.procedural_memory.records.values())
    restored_triggers = model.domain_index.get_trigger_weights(elite_route.cluster_id)

    assert torch.allclose(restored_param, initial_param)
    assert len(restored_records) == 1
    assert restored_records[0].skill_id == first_skill.skill_id
    assert restored_triggers == {first_skill.skill_id: 2.0}


def test_latent_action_encoder_moves_cpu_candidates_to_module_device(device: torch.device) -> None:
    if device.type == "cpu":
        pytest.skip("Requires a non-CPU device to verify device normalization.")

    encoder = LatentActionEncoder(
        action_dim=4,
        skill_dim=8,
        num_heads=2,
        num_layers=1,
        max_path_length=4,
    ).to(device)
    candidate = LatentSkillCandidate(
        action_path=torch.randn(3, 4, device="cpu"),
        symbolic_codes=None,
        target_delta=None,
        source_weight=1.0,
        source_episodes=(1,),
    )

    embedding = encoder.encode_candidate(candidate)

    assert embedding.device.type == device.type


def test_procedural_memory_retrieve_is_stable_during_concurrent_skill_adds(tmp_path: Path) -> None:
    procedural = ProceduralMemory(root=tmp_path / "procedural", skill_dim=8, use_faiss=False)
    procedural.add_skill(
        embedding=torch.randn(8),
        action_path=torch.randn(3, 4),
        name="seed",
    )
    query = torch.randn(8)
    failures: list[BaseException] = []
    stop_event = threading.Event()

    def reader() -> None:
        try:
            while not stop_event.is_set():
                procedural.retrieve(query, k=2)
        except BaseException as exc:  # pragma: no cover - regression capture
            failures.append(exc)

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        for idx in range(16):
            procedural.add_skill(
                embedding=torch.randn(8),
                action_path=torch.randn(3, 4),
                name=f"skill_{idx}",
            )
    finally:
        stop_event.set()
        thread.join(timeout=2.0)

    assert not failures


def test_love_decomposer_prefers_rare_high_quality_fragment() -> None:
    decomposer = LOVEDecomposer(min_frequency=1, max_segment_length=2)
    library: dict[tuple[int, ...], LOVEDecomposer._FragmentStats] = {
        (1, 1): LOVEDecomposer._FragmentStats(
            count=4,
            total_quality=0.4,
            total_utility=0.4,
            episode_ids={1, 2, 3, 4},
            paths=[torch.zeros(2, 2) for _ in range(4)],
        ),
        (9, 9): LOVEDecomposer._FragmentStats(
            count=1,
            total_quality=5.0,
            total_utility=5.0,
            episode_ids={8},
            paths=[torch.ones(2, 2)],
        ),
    }

    common_score = decomposer._segment_utility((1, 1), library[(1, 1)], num_traces=5)
    rare_high_value_score = decomposer._segment_utility((9, 9), library[(9, 9)], num_traces=5)

    assert rare_high_value_score > common_score


def test_phase6_scaffolds_smoke(tmp_path: Path) -> None:
    torch.manual_seed(0)
    reference = ActionConditionedWorldModel(
        embed_dim=16,
        action_dim=6,
        posture_dim=4,
        num_heads=4,
        num_layers=1,
        mlp_ratio=2.0,
        dropout=0.0,
        max_horizon=4,
    )
    candidate = MambaActionConditionedWorldModel(
        embed_dim=16,
        action_dim=6,
        posture_dim=4,
        max_horizon=4,
        use_native_mamba=False,
    )
    z_t = torch.randn(2, 16)
    actions = torch.randn(2, 6)
    z_tH = torch.randn(2, 16)
    ctx_tokens = torch.randn(2, 3, 16)
    postures = torch.randn(2, 4)
    benchmark = benchmark_world_models(
        reference_model=reference,
        candidate_model=candidate,
        z_t=z_t,
        actions=actions,
        z_tH=z_tH,
        ctx_tokens=ctx_tokens,
        candidate_postures=postures,
        horizon=1,
    )
    assessments = assess_vector_backends(
        skill_embeddings=torch.randn(8, 16),
        queries=torch.randn(3, 16),
        root=str(tmp_path / "lancedb"),
    )

    assert benchmark.backend in {"mamba2", "fallback_ssm"}
    assert len(assessments) >= 2


def test_hjepa_vq_path_exposes_codes_and_loss() -> None:
    torch.manual_seed(0)
    model = HJEPA(
        encoder_type="vit_tiny_patch16_224",
        img_size=224,
        embed_dim=192,
        predictor_depth=1,
        predictor_num_heads=3,
        predictor_mlp_ratio=2.0,
        num_hierarchies=2,
        use_vq=True,
        vq_codebook_size=32,
    )
    images = torch.randn(1, 3, 224, 224)
    mask = torch.zeros(1, model.get_num_patches(), dtype=torch.float32)
    mask[:, 0] = 1.0

    outputs = model(images, mask, return_all_levels=False)

    assert outputs["context_codes"] is not None
    assert outputs["target_codes"] is not None
    assert outputs["context_codes"].shape == outputs["context_features"].shape[:2]
    assert outputs["target_codes"].shape == outputs["target_features"].shape[:2]
    assert outputs["vq_commitment_loss"] is not None


def test_chamelia_can_select_mamba_world_model_backend(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model, domain, _memory, _procedural = _build_chamelia(
        tmp_path,
        world_model_backend="mamba",
    )
    tokenizer = domain.get_tokenizer()
    raw_tokens = tokenizer.collate([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
    tokenized = tokenizer(raw_tokens)
    mask = torch.zeros(raw_tokens.shape[0], raw_tokens.shape[1], dtype=torch.float32)
    domain_state = domain.get_domain_state(raw_tokens)
    domain_state["goal_latent"] = torch.ones(1, 32)

    outputs = model(
        tokens=tokenized.tokens,
        mask=mask,
        domain_state=domain_state,
        actor_mode="mode2",
        store_to_memory=False,
    )

    assert outputs["world_model_backend"] == "mamba"
    assert getattr(model.world_model, "backend", None) in {"mamba2", "fallback_ssm"}


def test_stitch_bodegen_and_gemma_worker_harness() -> None:
    if sleep_module.SingleTaskGP is None:
        pytest.skip("Requires botorch and gpytorch")
    segments = [
        LOVEDecomposer().decompose(
            [
                EpisodeRecord(
                    key=torch.zeros(8),
                    action=torch.zeros(5),
                    ctx_tokens=torch.zeros(2, 8),
                    ic_at_decision=0.1,
                    ic_realized=-1.0,
                    tc_predicted=0.1,
                    outcome_key=torch.zeros(8),
                    step=0,
                    domain_name="toy",
                    selected_path=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
                ),
                EpisodeRecord(
                    key=torch.zeros(8),
                    action=torch.zeros(5),
                    ctx_tokens=torch.zeros(2, 8),
                    ic_at_decision=0.1,
                    ic_realized=-1.0,
                    tc_predicted=0.1,
                    outcome_key=torch.zeros(8),
                    step=1,
                    domain_name="toy",
                    selected_path=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
                ),
            ]
        )
    ][0]
    compressed = StitchCompressor(iterations=2).compress(segments)
    assert len(compressed) >= 1

    encoder = LatentActionEncoder(action_dim=2, skill_dim=16, num_heads=2, num_layers=1, max_path_length=2)
    optimizer = BODEGenOptimizer(
        latent_prompt_dim=encoder.latent_prompt_dim,
        num_initial_points=3,
        num_iterations=1,
    )
    best_prompt, best_path, best_embedding, best_score = optimizer.optimize(
        lambda prompt, _path, _embedding: -float(((prompt - 0.5) ** 2).mean().item()),
        latent_action_encoder=encoder,
        path_length=2,
        seed_paths=[torch.zeros(2, 2)],
    )
    assert best_prompt.shape == (encoder.latent_prompt_dim,)
    assert best_path.shape == (2, 2)
    assert best_embedding.shape == (16,)
    assert isinstance(best_score, float)

    worker = GemmaAutoDocWorker(
        generator=lambda _prompt: "NAME: integrate_sequence\nDOC: Applies a learned sequence integration skill."
    )
    autodoc = LILOAutoDoc(worker=worker)
    name, description = autodoc.describe(
        candidate=type(
            "Candidate",
            (),
            {
                "symbolic_codes": torch.tensor([1, 2, 3]),
                "action_path": torch.zeros(2, 2),
                "source_episodes": (1, 2),
                "source_weight": 0.8,
            },
        )(),
        ordinal=1,
    )
    assert name == "integrate_sequence"
    assert "learned sequence integration" in description


def test_bodegen_optimizer_falls_back_from_mps_double() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    if sleep_module.SingleTaskGP is None:
        pytest.skip("Requires botorch and gpytorch")

    encoder = LatentActionEncoder(action_dim=2, skill_dim=16, num_heads=2, num_layers=1, max_path_length=2).to("mps")
    optimizer = BODEGenOptimizer(
        latent_prompt_dim=encoder.latent_prompt_dim,
        num_initial_points=3,
        num_iterations=1,
    )

    best_prompt, best_path, best_embedding, best_score = optimizer.optimize(
        lambda prompt, _path, _embedding: -float(((prompt - 0.5) ** 2).mean().item()),
        latent_action_encoder=encoder,
        path_length=2,
        seed_paths=[torch.zeros(2, 2)],
        device="mps",
    )

    assert best_prompt.shape == (encoder.latent_prompt_dim,)
    assert best_path.shape == (2, 2)
    assert best_embedding.shape == (16,)
    assert isinstance(best_score, float)

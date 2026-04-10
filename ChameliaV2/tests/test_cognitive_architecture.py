"""Harness tests for the additive cognitive architecture."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.chamelia.actor import Actor
from src.chamelia.chamelia import Chamelia
from src.chamelia.cognitive.clustering import DomainIndex, LoRAAdapterBank
from src.chamelia.cognitive.lancedb_assessment import assess_vector_backends
from src.chamelia.cognitive.latent_action import LatentActionEncoder
from src.chamelia.cognitive.mamba_world_model import (
    MambaActionConditionedWorldModel,
    benchmark_world_models,
)
from src.chamelia.cognitive.planning import HighLevelPlanner, MCTSNode, MCTSSearch, Talker
from src.chamelia.cognitive.procedural import ProceduralMemory
from src.chamelia.cognitive.representation import (
    ContrastiveSparseRepresentation,
    InformationOrderedBottleneck,
    IsotropicSkillCodec,
    VectorQuantizer,
)
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
    procedural = ProceduralMemory(root=root / "procedural", skill_dim=embed_dim, use_faiss=False)
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
    route = index.route(torch.randn(8), "dummy")
    index.record_skill_trigger(route.cluster_id, skill_id=3, weight=1.25)
    bank.ensure_cluster(route.cluster_id)
    for adapter in bank.adapters[route.cluster_id].values():
        adapter.out_proj_b.fill_(0.05)
        adapter.out_proj_a.fill_(0.05)

    inputs = torch.randn(1, 4, 8)
    baseline = model.attn(inputs, inputs, inputs)[0]
    bank.apply_cluster(route.cluster_id)
    adapted = model.attn(inputs, inputs, inputs)[0]

    assert route.spawned_new
    assert 3 in index.get_trigger_weights(route.cluster_id)
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
    reused_root_latent = outputs["mcts_traces"][0]
    assert reused_root_latent is not None
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

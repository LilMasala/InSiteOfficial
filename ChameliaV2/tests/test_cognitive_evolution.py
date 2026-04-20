"""Regression tests for the cognitive-evolution migration."""

from __future__ import annotations

from pathlib import Path

import torch

from src.chamelia.action_spec import ActionKind, ActionSpec
from src.chamelia.actor import Actor
from src.chamelia.cognitive.mamba_world_model import MambaActionConditionedWorldModel
from src.chamelia.cognitive.planning import (
    FrozenReasoningChain,
    ReasoningStep,
    Talker,
    ThinkerOutput,
    ThoughtTrace,
    apply_root_legal_action_mask,
)
from src.chamelia.cognitive.semantic import SemanticMemory
from src.chamelia.configurator import Configurator
from src.chamelia.cost import CostModule, IntrinsicCost, TrainableCritic
from src.chamelia.session_geometry import SessionGeometry
from src.chamelia.world_model import ActionConditionedWorldModel
from tests.test_chamelia import DummyDomain


def test_session_geometry_uses_action_spec_default() -> None:
    domain = DummyDomain(embed_dim=16, action_dim=7)
    geometry = SessionGeometry.from_domain(domain, D=16, P=4, K=3, H=2, T=4)

    assert geometry.action_spec == ActionSpec.continuous(7)
    assert geometry.A == 7


def test_configurator_accepts_thought_and_semantic_tokens() -> None:
    configurator = Configurator(
        embed_dim=16,
        num_ctx_tokens=4,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
    )
    hjepa_outputs = {
        "target_features_per_level": [
            torch.randn(2, 8, 16),
            torch.randn(2, 4, 16),
        ]
    }
    out = configurator(
        hjepa_outputs=hjepa_outputs,
        memory_tokens=torch.randn(2, 3, 16),
        memory_scores=torch.randn(2, 3),
        thought_tokens=torch.randn(2, 2, 16),
        semantic_tokens=torch.randn(2, 2, 16),
    )

    assert out.shape == (2, 4, 16)


def test_actor_supports_discrete_action_specs_and_thought_tokens() -> None:
    actor = Actor(
        embed_dim=16,
        action_dim=5,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        num_candidates=3,
        path_length=2,
        posture_dim=4,
    )
    geometry = SessionGeometry(
        D=16,
        action_spec=ActionSpec.discrete(5),
        P=4,
        K=3,
        H=2,
        T=4,
    )
    actor.bind_geometry(geometry)
    proposal = actor.propose(
        z=torch.randn(2, 16),
        ctx_tokens=torch.randn(2, 4, 16),
        thought_tokens=torch.randn(2, 2, 16),
    )

    assert actor.action_spec.kind == ActionKind.DISCRETE
    assert proposal["candidate_paths"].shape == (2, 3, 2, 1)
    assert proposal["candidate_action_paths"].action_spec.kind == ActionKind.DISCRETE
    assert proposal["thought_token"].shape == (2, 16)
    assert proposal["reflect_logits"].shape == (2,)


def test_root_legal_action_mask_constrains_only_executable_step() -> None:
    candidate_paths = torch.zeros(1, 3, 2, 5)
    candidate_paths[0, 0, 0, 0] = 5.0
    candidate_paths[0, 1, 0, 2] = 5.0
    candidate_paths[0, 2, 0, 4] = 5.0
    candidate_paths[0, :, 1, 0] = 3.0
    candidate_paths[0, :, 1, 4] = 7.0
    legal_mask = torch.tensor([[False, True, True, False, False]])

    constrained, diagnostics = apply_root_legal_action_mask(
        candidate_paths,
        {"legal_actions_mask": legal_mask},
    )

    assert diagnostics is not None
    assert diagnostics["root_illegal_candidate_count"] == 2
    assert constrained[0, :, 0, :].argmax(dim=-1).tolist() == [2, 2, 2]
    assert torch.equal(constrained[0, :, 1, :], candidate_paths[0, :, 1, :])


def test_world_models_emit_uncertainty_for_discrete_actions() -> None:
    geometry = SessionGeometry(
        D=16,
        action_spec=ActionSpec.discrete(5),
        P=4,
        K=3,
        H=2,
        T=4,
    )
    actions = torch.randint(0, 5, (2, 3, 2, 1)).float()
    for world_model in (
        ActionConditionedWorldModel(
            embed_dim=16,
            action_dim=5,
            posture_dim=4,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            max_horizon=4,
        ),
        MambaActionConditionedWorldModel(
            embed_dim=16,
            action_dim=5,
            posture_dim=4,
            num_layers=1,
            dropout=0.0,
            max_horizon=4,
            use_native_mamba=False,
        ),
    ):
        world_model.bind_geometry(geometry)
        outputs = world_model(
            z=torch.randn(2, 16),
            actions=actions,
            ctx_tokens=torch.randn(2, 4, 16),
            candidate_postures=torch.randn(2, 3, 4),
            reasoning_states=torch.randn(2, 3, 16),
            horizon=2,
        )
        assert outputs["trajectory"].shape[:3] == (2, 3, 2)
        assert outputs["uncertainty"].shape == (2, 3, 2)


def test_cost_module_accepts_uncertainty_penalty() -> None:
    intrinsic = IntrinsicCost(
        cost_fns=[lambda z, action, domain_state: action.float().mean(dim=-1)],
        weights=[1.0],
    )
    critic = TrainableCritic(
        embed_dim=16,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        num_ctx_tokens=4,
    )
    cost = CostModule(
        intrinsic_cost=intrinsic,
        trainable_critic=critic,
        uncertainty_penalty_weight=0.5,
    )
    outputs = cost.score_candidates(
        z=torch.randn(2, 16),
        actions=torch.randn(2, 3, 2, 4),
        ctx_tokens=torch.randn(2, 4, 16),
        domain_state={},
        future_z=torch.randn(2, 3, 16),
        future_trajectory=torch.randn(2, 3, 2, 16),
        uncertainty=torch.ones(2, 3, 2),
    )

    assert outputs["total"].shape == (2, 3)


def test_semantic_memory_is_domain_gated(tmp_path: Path) -> None:
    memory = SemanticMemory(tmp_path / "semantic", embed_dim=8, use_lancedb=False)
    memory.add_or_update_belief(
        embedding=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        domain_name="alpha",
        provenance={1, 2},
        description="alpha belief",
    )
    memory.add_or_update_belief(
        embedding=torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        domain_name="beta",
        provenance={3, 4},
        description="beta belief",
    )

    alpha = memory.retrieve(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), domain_name="alpha", k=1)
    beta = memory.retrieve(torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), domain_name="beta", k=1)

    assert alpha[0].domain_name == "alpha"
    assert beta[0].domain_name == "beta"


def test_talker_uses_thought_trace_when_present() -> None:
    talker = Talker(latent_dim=16, vocab_size=32, num_heads=4, num_layers=1, max_tokens=4)
    thought_trace = ThoughtTrace(tokens=(torch.randn(16), torch.randn(16)))
    thinker_output = ThinkerOutput(
        reasoning_chain=FrozenReasoningChain(
            steps=(
                ReasoningStep(
                    state=torch.randn(16),
                    candidate_paths=None,
                    candidate_costs=None,
                    selected_path=None,
                    source="unit",
                    depth=0,
                ),
            )
        ),
        action_vec=torch.randn(1, 16),
        selected_path=torch.randn(1, 2, 16),
        metadata={},
        thought_trace=thought_trace,
    )
    logits = talker(
        thinker_output=thinker_output,
        belief_tokens=torch.randn(1, 2, 16),
    )

    assert logits.shape == (1, 4, 32)

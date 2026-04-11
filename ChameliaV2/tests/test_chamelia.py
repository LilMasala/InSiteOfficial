"""Shape tests for the Chamelia V2 modules."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from src.chamelia.actor import Actor
from src.chamelia.chamelia import Chamelia
from src.chamelia.configurator import Configurator
from src.chamelia.cost import CostModule, IntrinsicCost, TrainableCritic
from src.chamelia.memory import EpisodeRecord, LatentMemory, RetrievalTraceStep
from src.chamelia.plugins import CartPoleDomain
from src.chamelia.plugins.base import AbstractDomain, DomainRegistry
from src.chamelia.retrieval import MemoryRelevanceScorer
from src.chamelia.tokenizers import SequenceTokenizer
from src.chamelia.world_model import ActionConditionedWorldModel
from src.models.hjepa import HJEPA


class DummyHJEPA(nn.Module):
    """Small HJEPA-compatible stub for Chamelia pipeline tests."""

    def __init__(self, embed_dim: int = 64) -> None:
        """Initialize the dummy HJEPA.

        Args:
            embed_dim: Shared latent dimension D.

        Returns:
            None.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("scale", torch.tensor(1.0), persistent=False)

    def _apply_fpn(
        self,
        features: torch.Tensor,
        is_prediction: bool = False,
    ) -> list[torch.Tensor]:
        """Create three pooled hierarchy levels.

        Args:
            features: Patch feature tensor [B, N, D].
            is_prediction: Unused compatibility flag.

        Returns:
            List of tensors [B, 16, D], [B, 4, D], [B, 1, D] for N=16.
        """
        _ = is_prediction
        level0 = features
        level1 = features.view(features.shape[0], 4, 4, features.shape[-1]).mean(dim=2)
        level2 = features.mean(dim=1, keepdim=True)
        return [level0, level1, level2]

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
        """Produce HJEPA-like outputs from pre-embedded tokens.

        Args:
            tokens: Embedded tokens [B, N, D].
            mask: Binary mask [B, N].

        Returns:
            Dict with target_features [B, N+1, D] and compatibility fields.
        """
        _ = mask
        cls = tokens.mean(dim=1, keepdim=True)
        target_features = torch.cat([cls, tokens], dim=1)
        return {
            "predictions": [tokens],
            "targets": [tokens],
            "mask_valid": torch.ones(tokens.shape[0], tokens.shape[1], dtype=torch.bool),
            "context_features": target_features,
            "target_features": target_features,
        }


class DummyDomain(AbstractDomain):
    """Minimal domain implementation for Chamelia pipeline tests."""

    def __init__(self, embed_dim: int = 64, action_dim: int = 8) -> None:
        """Initialize the dummy domain.

        Args:
            embed_dim: Token embedding dimension D.
            action_dim: Continuous action dimension A.

        Returns:
            None.
        """
        self._tokenizer = SequenceTokenizer(
            vocab_size=32,
            embed_dim=embed_dim,
            max_seq_len=16,
            domain_name="dummy",
            pad_token_id=0,
        )
        self._action_dim = action_dim

    def get_tokenizer(self) -> SequenceTokenizer:
        """Return the domain tokenizer.

        Returns:
            SequenceTokenizer whose forward() emits tokens [B, N, D].
        """
        return self._tokenizer

    def get_action_dim(self) -> int:
        """Return the actor output dimension.

        Returns:
            Integer A.
        """
        return self._action_dim

    def decode_action(self, action_vec: torch.Tensor) -> Any:
        """Decode the actor output.

        Args:
            action_vec: Action tensor [B, A].

        Returns:
            Tensor [B, A] for this dummy domain.
        """
        return action_vec

    def get_intrinsic_cost_fns(self):
        """Return simple dummy intrinsic cost functions.

        Returns:
            List of (cost_fn, weight) tuples; each cost_fn returns [B].
        """

        def action_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict) -> torch.Tensor:
            _ = z
            _ = domain_state
            return action.pow(2).mean(dim=-1)

        def state_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict) -> torch.Tensor:
            _ = action
            bonus = domain_state["bonus"]
            return z.abs().mean(dim=-1) + bonus

        return [(action_cost, 0.7), (state_cost, 0.3)]

    def get_domain_state(self, observation: Any) -> dict:
        """Construct an opaque domain state.

        Args:
            observation: Raw observation tensor [B, N].

        Returns:
            Dict with tensor field "bonus" of shape [B].
        """
        if not torch.is_tensor(observation):
            raise TypeError("DummyDomain expects tensor observations.")
        return {"bonus": observation.float().mean(dim=-1) / 100.0}

    def compute_regime_embedding(self, domain_state: dict) -> torch.Tensor | None:
        """Optionally compute a regime embedding.

        Args:
            domain_state: Domain-state dict with "bonus" [B].

        Returns:
            None for this dummy domain.
        """
        _ = domain_state
        return None

    @property
    def domain_name(self) -> str:
        """Return the domain name.

        Returns:
            String identifier.
        """
        return "dummy"

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size.

        Returns:
            Integer vocabulary size.
        """
        return 32


class ImaginedStateDomain(DummyDomain):
    """Domain whose intrinsic cost depends on imagined future state."""

    def get_intrinsic_cost_fns(self):
        def state_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict) -> torch.Tensor:
            _ = z
            _ = action
            return domain_state["state_scalar"].float().reshape(-1)

        return [(state_cost, 1.0)]

    def build_imagined_domain_state(
        self,
        current_domain_state: dict[str, Any],
        future_z: torch.Tensor,
        step_idx: int,
    ) -> dict[str, Any]:
        imagined = dict(current_domain_state)
        imagined["state_scalar"] = future_z[:, 0] + float(step_idx)
        return imagined


def test_chamelia_pipeline_shapes() -> None:
    """Instantiate each Chamelia V2 class and verify pipeline tensor shapes."""
    torch.manual_seed(0)

    embed_dim = 64
    num_ctx_tokens = 4
    action_dim = 8

    domain = DummyDomain(embed_dim=embed_dim, action_dim=action_dim)
    DomainRegistry.register(domain)

    tokenizer = domain.get_tokenizer()
    hjepa = DummyHJEPA(embed_dim=embed_dim)
    configurator = Configurator(
        embed_dim=embed_dim,
        num_ctx_tokens=num_ctx_tokens,
        num_heads=8,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.1,
        memory_read_k=4,
    )
    actor = Actor(
        embed_dim=embed_dim,
        action_dim=action_dim,
        num_heads=8,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.1,
        num_ctx_tokens=num_ctx_tokens,
    )
    cost_fns, weights = zip(*domain.get_intrinsic_cost_fns(), strict=False)
    intrinsic_cost = IntrinsicCost(list(cost_fns), list(weights))
    critic = TrainableCritic(
        embed_dim=embed_dim,
        num_heads=8,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.1,
        num_ctx_tokens=num_ctx_tokens,
        horizon=5,
    )
    cost_module = CostModule(intrinsic_cost=intrinsic_cost, trainable_critic=critic)
    memory = LatentMemory(embed_dim=embed_dim, max_episodes=32, retrieval_k=4, device="cpu")

    record = EpisodeRecord(
        key=torch.randn(embed_dim),
        action=torch.randn(action_dim),
        ctx_tokens=torch.randn(num_ctx_tokens, embed_dim),
        ic_at_decision=0.1,
        ic_realized=0.2,
        tc_predicted=0.3,
        outcome_key=torch.randn(embed_dim),
        step=0,
        domain_name=domain.domain_name,
        selected_posture=torch.randn(actor.posture_dim),
    )
    memory_idx = memory.store(record)
    memory.fill_outcome(memory_idx, ic_realized=0.25, outcome_key=torch.randn(embed_dim))

    model = Chamelia(
        hjepa=hjepa,
        configurator=configurator,
        actor=actor,
        cost=cost_module,
        memory=memory,
        domain=domain,
        embed_dim=embed_dim,
        action_dim=action_dim,
        num_ctx_tokens=num_ctx_tokens,
    )

    raw_tokens = tokenizer.collate([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
    tokenized = tokenizer(raw_tokens)
    mask = torch.zeros(raw_tokens.shape[0], raw_tokens.shape[1], dtype=torch.float32)
    domain_state = domain.get_domain_state(raw_tokens)

    assert tokenized.tokens.shape == (1, 16, embed_dim)
    assert tokenized.position_ids.shape == (1, 16)
    assert tokenized.padding_mask.shape == (1, 16)

    outputs = model(
        tokens=tokenized.tokens,
        mask=mask,
        domain_state=domain_state,
        actor_mode="mode2",
        store_to_memory=True,
    )

    assert outputs["z"].shape == (1, embed_dim)
    assert outputs["ctx_tokens"].shape == (1, num_ctx_tokens, embed_dim)
    assert outputs["action_vec"].shape == (1, action_dim)
    assert outputs["candidate_postures"].shape == (1, actor.num_candidates, actor.posture_dim)
    assert outputs["retrieved_episode_summaries"] is not None
    assert outputs["retrieved_episode_summaries"].shape[-1] == embed_dim
    assert outputs["retrieved_episode_scores"] is not None
    assert outputs["retrieval_base_scores"] is not None
    assert outputs["retrieval_base_quality_scores"] is not None
    assert outputs["retrieval_relevance_scores"] is not None
    assert outputs["retrieval_relevance_weights"] is not None
    assert outputs["retrieval_relevance_features"] is not None
    assert outputs["retrieved_postures"] is not None
    assert outputs["retrieved_postures"].shape[-1] == actor.posture_dim
    assert outputs["reasoning_states"].shape == (1, actor.num_candidates, embed_dim)
    assert outputs["candidate_paths"].shape == (
        1,
        actor.num_candidates,
        actor.path_length,
        action_dim,
    )
    assert outputs["candidate_actions"].shape == (1, actor.num_candidates, action_dim)
    assert outputs["selected_path"].shape == (1, actor.path_length, action_dim)
    assert torch.allclose(
        outputs["candidate_paths"][:, 0, :, :],
        torch.zeros_like(outputs["candidate_paths"][:, 0, :, :]),
    )
    assert outputs["candidate_costs"]["total"].shape == (1, actor.num_candidates)
    assert outputs["selected_candidate_idx"].shape == (1,)
    assert outputs["selected_posture"].shape == (1, actor.posture_dim)
    assert outputs["cost"]["ic"].shape == (1,)
    assert outputs["cost"]["tc"].shape == (1,)
    assert outputs["cost"]["total"].shape == (1,)
    assert outputs["hjepa_out"]["target_features"].shape == (1, 17, embed_dim)

    mode1_actions = actor(outputs["z"], outputs["ctx_tokens"], mode="mode1")
    assert mode1_actions.shape == (1, action_dim)

    distill_loss = actor.distill_from_mode2(
        states=outputs["z"],
        ctx_tokens=outputs["ctx_tokens"],
        mode2_actions=outputs["action_vec"],
    )
    assert distill_loss.dim() == 0
    assert torch.allclose(
        outputs["action_vec"],
        outputs["selected_path"][:, 0, :],
    )
    stored_record = memory.get_record_by_id(model._pending_record_indices[0])
    assert stored_record.candidate_postures is not None
    assert stored_record.selected_posture is not None
    assert stored_record.candidate_reasoning_states is not None
    assert stored_record.candidate_paths is not None
    assert stored_record.selected_path is not None
    assert stored_record.retrieval_trace is not None
    assert len(stored_record.retrieval_trace) >= 1

    model.fill_outcome(ic_realized=0.4, outcome_observation=raw_tokens)
    critic_loss = model.train_critic_from_memory()
    assert critic_loss is not None
    assert critic_loss.dim() == 0


def test_trainable_critic_supports_signed_future_costs() -> None:
    """TC should represent signed cost-to-go, including good futures with negative cost."""
    torch.manual_seed(0)

    critic = TrainableCritic(
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        mlp_ratio=2.0,
        dropout=0.0,
        num_ctx_tokens=4,
        horizon=5,
    )
    z = torch.randn(6, 32)
    ctx = torch.randn(6, 4, 32)

    with torch.no_grad():
        for param in critic.parameters():
            param.zero_()
        critic.value_head[2].bias.fill_(-0.75)

    predicted = critic(z, ctx)
    target = torch.full((6,), -1.25)
    loss = critic.compute_critic_loss(predicted, target)

    assert predicted.shape == (6,)
    assert torch.allclose(predicted, torch.full((6,), -0.75), atol=1.0e-6)
    assert torch.isclose(loss, torch.tensor(0.125), atol=1.0e-6)


def test_cartpole_imagined_state_calibration_penalizes_safety_miscalibration() -> None:
    cartpole = CartPoleDomain(embed_dim=4)
    cartpole.state_decoder = nn.Identity()
    action = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    target_state = torch.tensor([[0.0, 0.0, 0.05, 0.0]], dtype=torch.float32)
    target_domain_state = cartpole.build_domain_state(target_state, None)

    matched_loss = cartpole.compute_imagined_state_calibration_loss(
        target_state,
        action,
        target_domain_state,
        step_idx=0,
    )
    risky_prediction = torch.tensor([[0.0, 0.0, 0.4, 0.0]], dtype=torch.float32)
    risky_loss = cartpole.compute_imagined_state_calibration_loss(
        risky_prediction,
        action,
        target_domain_state,
        step_idx=0,
    )

    assert matched_loss is not None
    assert risky_loss is not None
    assert torch.isclose(matched_loss, torch.tensor(0.0), atol=1.0e-7)
    assert float(risky_loss.item()) > 1.0


def test_cost_module_requires_future_latents_for_candidate_scoring() -> None:
    """Candidate scoring over [B, K, A] must use future latents, not current z fallback."""
    cost_module = CostModule(
        intrinsic_cost=IntrinsicCost(
            cost_fns=[lambda z, action, domain_state: action.pow(2).mean(dim=-1)],
            weights=[1.0],
        ),
        trainable_critic=TrainableCritic(
            embed_dim=8,
            num_heads=2,
            num_layers=1,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=2,
        ),
    )
    z = torch.randn(2, 8)
    actions = torch.randn(2, 3, 4)
    ctx = torch.randn(2, 2, 8)

    with pytest.raises(ValueError, match="future_z is required"):
        cost_module.score_candidates(
            z=z,
            actions=actions,
            ctx_tokens=ctx,
            domain_state={},
        )


def test_cost_module_path_scoring_uses_discounted_sum_and_tail_discount() -> None:
    """Path scoring should use discounted IC accumulation plus discounted terminal TC."""

    class ConstantCritic(TrainableCritic):
        def forward(self, z: torch.Tensor, ctx_tokens: torch.Tensor) -> torch.Tensor:
            _ = ctx_tokens
            return torch.full((z.shape[0],), 2.0, device=z.device, dtype=z.dtype)

    gamma = 0.5
    cost_module = CostModule(
        intrinsic_cost=IntrinsicCost(
            cost_fns=[lambda z, action, domain_state: torch.ones(z.shape[0], device=z.device, dtype=z.dtype)],
            weights=[1.0],
        ),
        trainable_critic=ConstantCritic(
            embed_dim=8,
            num_heads=2,
            num_layers=1,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=2,
        ),
        gamma=gamma,
    )
    z = torch.randn(1, 8)
    actions = torch.randn(1, 2, 3, 4)
    ctx = torch.randn(1, 2, 8)
    future_trajectory = torch.randn(1, 2, 3, 8)
    future_z = future_trajectory[:, :, -1, :]

    scored = cost_module.score_candidates(
        z=z,
        actions=actions,
        ctx_tokens=ctx,
        domain_state={},
        future_z=future_z,
        future_trajectory=future_trajectory,
    )

    expected_ic = 1.0 + gamma + (gamma**2)
    expected_total = expected_ic + ((gamma**3) * 2.0)
    assert torch.allclose(scored["ic"], torch.full((1, 2), expected_ic))
    assert torch.allclose(scored["tc"], torch.full((1, 2), 2.0))
    assert torch.allclose(scored["total"], torch.full((1, 2), expected_total))


def test_latent_memory_summarizes_retrieved_selected_postures() -> None:
    """Retrieved episodes should expose selected postures with better-outcome scores."""
    memory = LatentMemory(embed_dim=8, max_episodes=8, retrieval_k=4, device="cpu")
    query_key = torch.tensor([1.0] + [0.0] * 7)
    better_posture = torch.tensor([1.0, 0.0, 0.0])
    worse_posture = torch.tensor([0.0, 1.0, 0.0])
    memory.store(
        EpisodeRecord(
            key=query_key,
            action=torch.zeros(4),
            ctx_tokens=torch.zeros(2, 8),
            ic_at_decision=0.5,
            ic_realized=0.1,
            tc_predicted=0.2,
            outcome_key=torch.zeros(8),
            step=0,
            domain_name="dummy",
            selected_posture=better_posture,
        )
    )
    memory.store(
        EpisodeRecord(
            key=query_key * 0.9,
            action=torch.zeros(4),
            ctx_tokens=torch.zeros(2, 8),
            ic_at_decision=0.5,
            ic_realized=0.8,
            tc_predicted=0.2,
            outcome_key=torch.zeros(8),
            step=1,
            domain_name="dummy",
            selected_posture=worse_posture,
        )
    )

    _, episodes = memory.retrieve(query_key.unsqueeze(0))
    retrieved_postures, retrieved_scores = memory.summarize_retrieved_postures(
        episodes,
        posture_dim=3,
    )

    assert retrieved_postures is not None
    assert retrieved_scores is not None
    assert retrieved_postures.shape == (1, 2, 3)
    assert torch.allclose(retrieved_postures[0, 0], better_posture)
    assert torch.allclose(retrieved_postures[0, 1], worse_posture)
    assert float(retrieved_scores[0, 0].item()) > float(retrieved_scores[0, 1].item())


def test_latent_memory_retrieve_scored_blends_quality_and_posture_relevance() -> None:
    """Scored retrieval should support both quality-aware and posture-aware reranking."""
    memory = LatentMemory(embed_dim=8, max_episodes=8, retrieval_k=4, device="cpu")
    shared_key = torch.tensor([1.0] + [0.0] * 7)
    quality_posture = torch.tensor([1.0, 0.0, 0.0])
    posture_match = torch.tensor([0.0, 1.0, 0.0])
    memory.store(
        EpisodeRecord(
            key=shared_key,
            action=torch.zeros(4),
            ctx_tokens=torch.zeros(2, 8),
            ic_at_decision=0.3,
            ic_realized=0.05,
            tc_predicted=0.1,
            outcome_key=torch.zeros(8),
            step=0,
            domain_name="dummy",
            selected_posture=quality_posture,
        )
    )
    memory.store(
        EpisodeRecord(
            key=shared_key,
            action=torch.zeros(4),
            ctx_tokens=torch.zeros(2, 8),
            ic_at_decision=0.3,
            ic_realized=0.10,
            tc_predicted=0.1,
            outcome_key=torch.zeros(8),
            step=1,
            domain_name="dummy",
            selected_posture=posture_match,
        )
    )

    _, quality_ranked, _ = memory.retrieve_scored(
        shared_key.unsqueeze(0),
        quality_weight=1.0,
        posture_weight=0.0,
    )
    _, posture_ranked, _ = memory.retrieve_scored(
        shared_key.unsqueeze(0),
        query_posture=posture_match.unsqueeze(0),
        quality_weight=0.0,
        posture_weight=1.0,
    )

    assert quality_ranked[0][0].selected_posture is not None
    assert posture_ranked[0][0].selected_posture is not None
    assert torch.allclose(quality_ranked[0][0].selected_posture, quality_posture)
    assert torch.allclose(posture_ranked[0][0].selected_posture, posture_match)


def test_latent_memory_summarizes_retrieved_episodes_with_outcomes() -> None:
    """Retrieved episode summaries should include outcome/context information, not just keys."""
    memory = LatentMemory(embed_dim=8, max_episodes=8, retrieval_k=4, device="cpu")
    shared_key = torch.tensor([1.0] + [0.0] * 7)
    memory.store(
        EpisodeRecord(
            key=shared_key,
            action=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            ctx_tokens=torch.ones(2, 8),
            ic_at_decision=0.2,
            ic_realized=0.1,
            tc_predicted=0.2,
            outcome_key=torch.tensor([0.0, 1.0] + [0.0] * 6),
            step=0,
            domain_name="dummy",
            selected_posture=torch.tensor([1.0, 0.0, 0.0]),
        )
    )
    memory.store(
        EpisodeRecord(
            key=shared_key,
            action=torch.tensor([0.0, 1.0, 0.0, 0.0]),
            ctx_tokens=torch.full((2, 8), 0.5),
            ic_at_decision=0.2,
            ic_realized=0.7,
            tc_predicted=0.2,
            outcome_key=torch.tensor([0.0, 0.0, 1.0] + [0.0] * 5),
            step=1,
            domain_name="dummy",
            selected_posture=torch.tensor([0.0, 1.0, 0.0]),
        )
    )

    _, episodes = memory.retrieve(shared_key.unsqueeze(0))
    summary_tokens, summary_scores = memory.summarize_retrieved_episodes(episodes)

    assert summary_tokens is not None
    assert summary_scores is not None
    assert summary_tokens.shape == (1, 2, 8)
    assert not torch.allclose(summary_tokens[0, 0], summary_tokens[0, 1])
    assert float(summary_scores[0, 0].item()) > float(summary_scores[0, 1].item())


def test_chamelia_with_real_hjepa_backbone() -> None:
    """Run a lightweight integration pass using the real HJEPA backbone.

    This smoke test uses the backbone at its natural tiny-ViT dimension to verify that
    Chamelia can assemble around the real model without relying on the lightweight stub.

    Returns:
        None.
    """
    torch.manual_seed(0)

    embed_dim = 192
    num_ctx_tokens = 4
    action_dim = 8

    domain = DummyDomain(embed_dim=64, action_dim=action_dim)
    DomainRegistry.register(domain)

    hjepa = HJEPA(
        encoder_type="vit_tiny_patch16_224",
        img_size=224,
        embed_dim=embed_dim,
        predictor_depth=1,
        predictor_num_heads=3,
        predictor_mlp_ratio=2.0,
        num_hierarchies=3,
        pretrained=False,
        drop_path_rate=0.0,
        use_fpn=True,
        fpn_feature_dim=embed_dim,
        use_gradient_checkpointing=False,
        use_layerscale=False,
        use_flash_attention=False,
    )
    hjepa.eval()

    configurator = Configurator(
        embed_dim=embed_dim,
        num_ctx_tokens=num_ctx_tokens,
        num_heads=3,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        memory_read_k=4,
    )
    actor = Actor(
        embed_dim=embed_dim,
        action_dim=action_dim,
        num_heads=3,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_ctx_tokens=num_ctx_tokens,
    )
    cost_fns, weights = zip(*domain.get_intrinsic_cost_fns(), strict=False)
    intrinsic_cost = IntrinsicCost(list(cost_fns), list(weights))
    critic = TrainableCritic(
        embed_dim=embed_dim,
        num_heads=3,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_ctx_tokens=num_ctx_tokens,
        horizon=5,
    )
    cost_module = CostModule(intrinsic_cost=intrinsic_cost, trainable_critic=critic)
    memory = LatentMemory(embed_dim=embed_dim, max_episodes=8, retrieval_k=4, device="cpu")

    model = Chamelia(
        hjepa=hjepa,
        configurator=configurator,
        actor=actor,
        cost=cost_module,
        memory=memory,
        domain=domain,
        embed_dim=embed_dim,
        action_dim=action_dim,
        num_ctx_tokens=num_ctx_tokens,
    )
    model.eval()

    images = torch.randn(1, 3, 224, 224)
    mask = torch.zeros(1, 196, dtype=torch.float32)
    mask[:, :8] = 1.0
    domain_state = {"bonus": torch.zeros(1)}

    with torch.no_grad():
        outputs = model(
            tokens=images,
            mask=mask,
            domain_state=domain_state,
            actor_mode="mode2",
            store_to_memory=False,
        )

    level_feats = model._extract_level_features(outputs["hjepa_out"])

    assert outputs["z"].shape == (1, embed_dim)
    assert outputs["ctx_tokens"].shape == (1, num_ctx_tokens, embed_dim)
    assert outputs["action_vec"].shape == (1, action_dim)
    assert outputs["reasoning_states"].shape == (1, actor.num_candidates, embed_dim)
    assert outputs["candidate_paths"].shape == (1, actor.num_candidates, actor.path_length, action_dim)
    assert outputs["candidate_actions"].shape == (1, actor.num_candidates, action_dim)
    assert outputs["cost"]["total"].shape == (1,)
    assert outputs["hjepa_out"]["target_features"].shape == (1, 197, embed_dim)
    assert len(level_feats) == 3
    assert level_feats[0].shape == (1, 196, embed_dim)
    assert level_feats[1].shape == (1, 98, embed_dim)
    assert level_feats[2].shape == (1, 49, embed_dim)


def test_chamelia_refreshes_memory_retrieval_with_candidate_posture_query() -> None:
    """Later reasoning rounds should refresh memory using the current candidate postures."""

    class CaptureScorer(MemoryRelevanceScorer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.query_postures: list[torch.Tensor | None] = []

        def forward(
            self,
            query_key: torch.Tensor,
            memory_keys: torch.Tensor,
            memory_summaries: torch.Tensor,
            memory_quality: torch.Tensor | None = None,
            query_posture: torch.Tensor | None = None,
            memory_postures: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            self.query_postures.append(
                query_posture.detach().clone() if query_posture is not None else None
            )
            return super().forward(
                query_key=query_key,
                memory_keys=memory_keys,
                memory_summaries=memory_summaries,
                memory_quality=memory_quality,
                query_posture=query_posture,
                memory_postures=memory_postures,
            )

    torch.manual_seed(0)

    embed_dim = 32
    num_ctx_tokens = 4
    action_dim = 8
    domain = DummyDomain(embed_dim=embed_dim, action_dim=action_dim)
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
    )
    cost_fns, weights = zip(*domain.get_intrinsic_cost_fns(), strict=False)
    cost_module = CostModule(
        intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
        trainable_critic=TrainableCritic(
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=num_ctx_tokens,
        ),
    )
    memory = LatentMemory(embed_dim=embed_dim, max_episodes=8, retrieval_k=4, device="cpu")
    memory.store(
        EpisodeRecord(
            key=torch.randn(embed_dim),
            action=torch.randn(action_dim),
            ctx_tokens=torch.randn(num_ctx_tokens, embed_dim),
            ic_at_decision=0.1,
            ic_realized=0.2,
            tc_predicted=0.3,
            outcome_key=torch.randn(embed_dim),
            step=0,
            domain_name=domain.domain_name,
            selected_posture=torch.randn(actor.posture_dim),
        )
    )
    retrieval_scorer = CaptureScorer(
        embed_dim=embed_dim,
        posture_dim=actor.posture_dim,
        hidden_dim=32,
    )
    model = Chamelia(
        hjepa=hjepa,
        configurator=configurator,
        actor=actor,
        cost=cost_module,
        memory=memory,
        domain=domain,
        retrieval_scorer=retrieval_scorer,
        embed_dim=embed_dim,
        action_dim=action_dim,
        num_ctx_tokens=num_ctx_tokens,
        reasoning_steps=2,
    )

    tokenizer = domain.get_tokenizer()
    raw_tokens = tokenizer.collate([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]])
    tokenized = tokenizer(raw_tokens)
    mask = torch.zeros(raw_tokens.shape[0], raw_tokens.shape[1], dtype=torch.float32)
    domain_state = domain.get_domain_state(raw_tokens)

    _ = model(
        tokens=tokenized.tokens,
        mask=mask,
        domain_state=domain_state,
        actor_mode="mode2",
        store_to_memory=False,
    )

    assert len(retrieval_scorer.query_postures) >= 2
    assert retrieval_scorer.query_postures[0] is None
    assert retrieval_scorer.query_postures[1] is not None
    assert retrieval_scorer.query_postures[1].shape == (1, actor.posture_dim)


def test_action_conditioned_world_model_changes_with_action() -> None:
    """Ensure different candidate actions produce different imagined futures."""
    torch.manual_seed(0)

    model = ActionConditionedWorldModel(
        embed_dim=32,
        action_dim=8,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        max_horizon=4,
    )
    z = torch.randn(2, 32)
    ctx = torch.randn(2, 4, 32)
    actions = torch.zeros(2, 2, 3, 8)
    actions[:, 1, 1:, :] = 1.0

    outputs = model(z=z, actions=actions, ctx_tokens=ctx, horizon=3)

    assert outputs["trajectory"].shape == (2, 2, 3, 32)
    assert outputs["terminal_latents"].shape == (2, 2, 32)
    assert outputs["summary_tokens"].shape == (2, 2, 32)
    action_difference = (
        outputs["terminal_latents"][:, 0, :] - outputs["terminal_latents"][:, 1, :]
    ).abs().sum()
    assert float(action_difference.item()) > 0.0


def test_configurator_memory_scores_change_context() -> None:
    """Outcome-weighted memory summaries should change configurator output."""
    torch.manual_seed(0)

    configurator = Configurator(
        embed_dim=16,
        num_ctx_tokens=4,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        memory_read_k=4,
    )
    levels = [
        torch.randn(1, 4, 16),
        torch.randn(1, 2, 16),
        torch.randn(1, 1, 16),
    ]
    memory_tokens = torch.randn(1, 2, 16)
    preferred_first = torch.tensor([[2.0, -2.0]])
    preferred_second = torch.tensor([[-2.0, 2.0]])

    out_first = configurator(
        hjepa_outputs={"target_features_per_level": levels},
        memory_tokens=memory_tokens,
        memory_scores=preferred_first,
    )
    out_second = configurator(
        hjepa_outputs={"target_features_per_level": levels},
        memory_tokens=memory_tokens,
        memory_scores=preferred_second,
    )

    assert out_first.shape == (1, 4, 16)
    assert out_second.shape == (1, 4, 16)
    assert float((out_first - out_second).abs().sum().item()) > 0.0


def test_action_conditioned_world_model_changes_with_posture() -> None:
    """Different candidate postures should alter imagined futures even for the same path."""
    torch.manual_seed(0)

    model = ActionConditionedWorldModel(
        embed_dim=32,
        action_dim=8,
        posture_dim=6,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        max_horizon=4,
    )
    z = torch.randn(1, 32)
    ctx = torch.randn(1, 4, 32)
    actions = torch.zeros(1, 2, 3, 8)
    candidate_postures = torch.zeros(1, 2, 6)
    candidate_postures[0, 1, 2] = 1.0

    outputs = model(
        z=z,
        actions=actions,
        ctx_tokens=ctx,
        candidate_postures=candidate_postures,
        horizon=3,
    )

    posture_difference = (
        outputs["terminal_latents"][:, 0, :] - outputs["terminal_latents"][:, 1, :]
    ).abs().sum()
    assert float(posture_difference.item()) > 0.0


def test_path_level_intrinsic_cost_accumulates_over_future_trajectory() -> None:
    """Candidate path IC should reflect the whole imagined path, not just the first action."""
    torch.manual_seed(0)

    domain = DummyDomain(embed_dim=32, action_dim=8)
    cost_fns, weights = zip(*domain.get_intrinsic_cost_fns(), strict=False)
    cost_module = CostModule(
        intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
        trainable_critic=TrainableCritic(
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=4,
        ),
    )
    z = torch.zeros(1, 32)
    ctx = torch.zeros(1, 4, 32)
    domain_state = {"bonus": torch.zeros(1)}
    candidate_paths = torch.zeros(1, 2, 3, 8)
    candidate_paths[:, 1, 1:, :] = 2.0
    future_trajectory = torch.zeros(1, 2, 3, 32)
    future_trajectory[:, 1, 1:, :] = 0.5

    scored = cost_module.score_candidates(
        z=z,
        actions=candidate_paths,
        ctx_tokens=ctx,
        domain_state=domain_state,
        future_z=future_trajectory[:, :, -1, :],
        future_trajectory=future_trajectory,
    )

    assert scored["ic"].shape == (1, 2)
    assert float(scored["ic"][0, 1].item()) > float(scored["ic"][0, 0].item())


def test_path_level_intrinsic_cost_can_use_imagined_domain_state() -> None:
    """Path scoring should be able to rebuild domain state from imagined future latents."""
    torch.manual_seed(0)

    domain = ImaginedStateDomain(embed_dim=32, action_dim=8)
    cost_fns, weights = zip(*domain.get_intrinsic_cost_fns(), strict=False)
    cost_module = CostModule(
        intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
        trainable_critic=TrainableCritic(
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=4,
        ),
    )
    z = torch.zeros(1, 32)
    ctx = torch.zeros(1, 4, 32)
    domain_state = {"state_scalar": torch.zeros(1)}
    candidate_paths = torch.zeros(1, 2, 3, 8)
    future_trajectory = torch.zeros(1, 2, 3, 32)
    future_trajectory[0, 0, :, 0] = torch.tensor([0.0, 0.5, 1.0])
    future_trajectory[0, 1, :, 0] = torch.tensor([1.0, 1.5, 2.0])

    scored = cost_module.score_candidates(
        z=z,
        actions=candidate_paths,
        ctx_tokens=ctx,
        domain_state=domain_state,
        future_z=future_trajectory[:, :, -1, :],
        future_trajectory=future_trajectory,
        imagined_domain_state_builder=domain.build_imagined_domain_state,
    )

    assert float(scored["ic"][0, 1].item()) > float(scored["ic"][0, 0].item())


def test_actor_propose_exposes_reasoning_states_and_null_baseline() -> None:
    """The planner should expose per-candidate reasoning states and a null baseline."""
    torch.manual_seed(0)

    actor = Actor(
        embed_dim=32,
        action_dim=8,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_ctx_tokens=4,
        num_candidates=4,
    )
    z = torch.randn(2, 32)
    ctx = torch.randn(2, 4, 32)
    proposal = actor.propose(z, ctx)

    assert proposal["candidate_postures"].shape == (2, 4, actor.posture_dim)
    assert proposal["reasoning_states"].shape == (2, 4, 32)
    assert proposal["candidate_paths"].shape == (2, 4, actor.path_length, 8)
    assert proposal["candidate_actions"].shape == (2, 4, 8)
    assert torch.allclose(
        proposal["candidate_postures"][:, 0, :],
        torch.zeros_like(proposal["candidate_postures"][:, 0, :]),
    )
    assert torch.allclose(
        proposal["candidate_paths"][:, 0, :, :],
        torch.zeros_like(proposal["candidate_paths"][:, 0, :, :]),
    )


def test_actor_propose_uses_retrieved_postures_to_seed_candidates() -> None:
    """Retrieved successful postures should change non-baseline candidate postures and paths."""
    torch.manual_seed(0)

    actor = Actor(
        embed_dim=32,
        action_dim=8,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_ctx_tokens=4,
        num_candidates=4,
        posture_dim=6,
    )
    z = torch.randn(1, 32)
    ctx = torch.randn(1, 4, 32)
    base_proposal = actor.propose(z, ctx)
    retrieved_postures = torch.zeros(1, 2, 6)
    retrieved_postures[0, 0, 1] = 1.0
    retrieved_postures[0, 1, 3] = 1.0
    retrieved_scores = torch.tensor([[1.5, 0.3]], dtype=torch.float32)
    memory_seeded_proposal = actor.propose(
        z,
        ctx,
        retrieved_postures=retrieved_postures,
        retrieved_posture_scores=retrieved_scores,
    )

    posture_delta = (
        memory_seeded_proposal["candidate_postures"][:, 1:, :]
        - base_proposal["candidate_postures"][:, 1:, :]
    ).abs().sum()
    path_delta = (
        memory_seeded_proposal["candidate_paths"][:, 1:, :, :]
        - base_proposal["candidate_paths"][:, 1:, :, :]
    ).abs().sum()

    assert float(posture_delta.item()) > 0.0
    assert float(path_delta.item()) > 0.0
    assert torch.allclose(
        memory_seeded_proposal["candidate_postures"][:, 0, :],
        torch.zeros_like(memory_seeded_proposal["candidate_postures"][:, 0, :]),
    )


def test_actor_posture_directly_biases_candidate_paths() -> None:
    """Candidate postures should shape the path beyond the reasoning-state action head alone."""
    torch.manual_seed(0)

    actor = Actor(
        embed_dim=32,
        action_dim=8,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_ctx_tokens=4,
        num_candidates=4,
        posture_dim=6,
    )
    proposal = actor.propose(torch.randn(1, 32), torch.randn(1, 4, 32))
    reasoning_only_paths = actor.action_head(proposal["reasoning_states"]).view(
        1,
        actor.num_candidates,
        actor.path_length,
        actor.action_dim,
    )
    nonbaseline_delta = (
        proposal["candidate_paths"][:, 1:, :, :] - reasoning_only_paths[:, 1:, :, :]
    ).abs().sum()

    assert float(nonbaseline_delta.item()) > 0.0


def test_actor_refine_uses_retrieved_memory_to_shift_later_rounds() -> None:
    """Retrieved episode memory should be able to change later refinement rounds, not just proposal."""
    torch.manual_seed(0)

    actor = Actor(
        embed_dim=32,
        action_dim=8,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_ctx_tokens=4,
        num_candidates=4,
        posture_dim=6,
    )
    z = torch.randn(1, 32)
    ctx = torch.randn(1, 4, 32)
    proposal = actor.propose(z, ctx)
    rollout_summary = torch.randn(1, actor.num_candidates, 32)
    candidate_scores = torch.zeros(1, actor.num_candidates)

    base_paths, base_postures, _ = actor.refine(
        z=z,
        ctx_tokens=ctx,
        candidate_paths=proposal["candidate_paths"].clone(),
        candidate_postures=proposal["candidate_postures"].clone(),
        reasoning_states=proposal["reasoning_states"].clone(),
        rollout_summary=rollout_summary,
        candidate_scores=candidate_scores,
    )
    memory_paths, memory_postures, _ = actor.refine(
        z=z,
        ctx_tokens=ctx,
        candidate_paths=proposal["candidate_paths"].clone(),
        candidate_postures=proposal["candidate_postures"].clone(),
        reasoning_states=proposal["reasoning_states"].clone(),
        rollout_summary=rollout_summary,
        candidate_scores=candidate_scores,
        retrieved_postures=torch.tensor(
            [[[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        retrieved_posture_scores=torch.tensor([[1.2, 0.4]], dtype=torch.float32),
        retrieved_episode_summaries=torch.randn(1, 2, 32),
        retrieved_episode_scores=torch.tensor([[1.4, 0.2]], dtype=torch.float32),
    )

    posture_delta = (memory_postures[:, 1:, :] - base_postures[:, 1:, :]).abs().sum()
    path_delta = (memory_paths[:, 1:, :, :] - base_paths[:, 1:, :, :]).abs().sum()

    assert float(posture_delta.item()) > 0.0
    assert float(path_delta.item()) > 0.0
    assert torch.allclose(
        memory_postures[:, 0, :],
        torch.zeros_like(memory_postures[:, 0, :]),
    )


def test_actor_posture_diversity_loss_penalizes_collapsed_nonbaseline_postures() -> None:
    """Non-baseline candidate postures should not collapse to the same region/path."""
    actor = Actor(
        embed_dim=16,
        action_dim=4,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_ctx_tokens=4,
        num_candidates=4,
        path_length=2,
    )
    collapsed_postures = torch.zeros(1, 4, actor.posture_dim)
    collapsed_paths = torch.zeros(1, 4, 2, 4)
    collapsed_paths[:, 1:, :, :] = 1.0
    collapsed_loss = actor.compute_posture_diversity_loss(
        candidate_postures=collapsed_postures,
        candidate_paths=collapsed_paths,
        max_posture_similarity=0.5,
        max_path_similarity=0.5,
    )

    diverse_postures = torch.zeros(1, 4, actor.posture_dim)
    diverse_postures[0, 1, 0] = 1.0
    diverse_postures[0, 2, 1] = 1.0
    diverse_postures[0, 3, 2] = 1.0
    diverse_paths = torch.zeros(1, 4, 2, 4)
    diverse_paths[0, 1, :, 0] = 1.0
    diverse_paths[0, 2, :, 1] = 1.0
    diverse_paths[0, 3, :, 2] = 1.0
    diverse_loss = actor.compute_posture_diversity_loss(
        candidate_postures=diverse_postures,
        candidate_paths=diverse_paths,
        max_posture_similarity=0.5,
        max_path_similarity=0.5,
    )

    assert collapsed_loss.dim() == 0
    assert diverse_loss.dim() == 0
    assert float(collapsed_loss.item()) > float(diverse_loss.item())


def test_train_critic_from_memory_uses_stored_context() -> None:
    """Critic training should consume the stored episode context, not dummy zeros."""

    class CaptureCritic(TrainableCritic):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.last_ctx: torch.Tensor | None = None

        def forward(self, z: torch.Tensor, ctx_tokens: torch.Tensor) -> torch.Tensor:
            self.last_ctx = ctx_tokens.detach().clone()
            return super().forward(z, ctx_tokens)

    torch.manual_seed(0)
    embed_dim = 32
    num_ctx_tokens = 4
    action_dim = 8
    domain = DummyDomain(embed_dim=embed_dim, action_dim=action_dim)

    critic = CaptureCritic(
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_ctx_tokens=num_ctx_tokens,
        horizon=5,
    )
    cost_fns, weights = zip(*domain.get_intrinsic_cost_fns(), strict=False)
    model = Chamelia(
        hjepa=DummyHJEPA(embed_dim=embed_dim),
        configurator=Configurator(
            embed_dim=embed_dim,
            num_ctx_tokens=num_ctx_tokens,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            memory_read_k=4,
        ),
        actor=Actor(
            embed_dim=embed_dim,
            action_dim=action_dim,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=num_ctx_tokens,
        ),
        cost=CostModule(
            intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
            trainable_critic=critic,
        ),
        memory=LatentMemory(embed_dim=embed_dim, max_episodes=8, retrieval_k=4, device="cpu"),
        domain=domain,
        embed_dim=embed_dim,
        action_dim=action_dim,
        num_ctx_tokens=num_ctx_tokens,
    )

    stored_ctx = torch.randn(num_ctx_tokens, embed_dim)
    idx = model.memory.store(
        EpisodeRecord(
            key=torch.randn(embed_dim),
            action=torch.randn(action_dim),
            ctx_tokens=stored_ctx,
            ic_at_decision=0.1,
            ic_realized=None,
            tc_predicted=0.2,
            outcome_key=None,
            step=0,
            domain_name=domain.domain_name,
        )
    )
    outcome_key = torch.randn(embed_dim)
    model.memory.fill_outcome(idx, ic_realized=0.4, outcome_key=outcome_key)

    loss = model.train_critic_from_memory()
    assert loss is not None
    assert critic.last_ctx is not None
    assert critic.last_ctx.shape == (1, num_ctx_tokens, embed_dim)
    assert torch.allclose(critic.last_ctx[0], stored_ctx, atol=1e-6)


def test_train_world_model_from_memory_uses_selected_posture() -> None:
    """World-model memory training should consume the stored selected posture."""

    class CaptureWorldModel(ActionConditionedWorldModel):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.last_postures: torch.Tensor | None = None

        def compute_transition_loss(
            self,
            z_t: torch.Tensor,
            actions: torch.Tensor,
            z_tH: torch.Tensor,
            ctx_tokens: torch.Tensor,
            candidate_postures: torch.Tensor | None = None,
            horizon: int = 1,
        ) -> torch.Tensor:
            self.last_postures = (
                candidate_postures.detach().clone()
                if candidate_postures is not None
                else None
            )
            return super().compute_transition_loss(
                z_t=z_t,
                actions=actions,
                z_tH=z_tH,
                ctx_tokens=ctx_tokens,
                candidate_postures=candidate_postures,
                horizon=horizon,
            )

    torch.manual_seed(0)
    embed_dim = 32
    num_ctx_tokens = 4
    action_dim = 8
    posture_dim = 6
    domain = DummyDomain(embed_dim=embed_dim, action_dim=action_dim)

    world_model = CaptureWorldModel(
        embed_dim=embed_dim,
        action_dim=action_dim,
        posture_dim=posture_dim,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        max_horizon=4,
    )
    cost_fns, weights = zip(*domain.get_intrinsic_cost_fns(), strict=False)
    model = Chamelia(
        hjepa=DummyHJEPA(embed_dim=embed_dim),
        configurator=Configurator(
            embed_dim=embed_dim,
            num_ctx_tokens=num_ctx_tokens,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            memory_read_k=4,
        ),
        actor=Actor(
            embed_dim=embed_dim,
            action_dim=action_dim,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=num_ctx_tokens,
            posture_dim=posture_dim,
        ),
        cost=CostModule(
            intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
            trainable_critic=TrainableCritic(
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=2,
                mlp_ratio=2.0,
                dropout=0.0,
                num_ctx_tokens=num_ctx_tokens,
            ),
        ),
        memory=LatentMemory(embed_dim=embed_dim, max_episodes=8, retrieval_k=4, device="cpu"),
        domain=domain,
        world_model=world_model,
        embed_dim=embed_dim,
        action_dim=action_dim,
        num_ctx_tokens=num_ctx_tokens,
        rollout_horizon=3,
    )

    stored_posture = torch.randn(posture_dim)
    idx = model.memory.store(
        EpisodeRecord(
            key=torch.randn(embed_dim),
            action=torch.randn(action_dim),
            ctx_tokens=torch.randn(num_ctx_tokens, embed_dim),
            ic_at_decision=0.1,
            ic_realized=0.2,
            tc_predicted=0.3,
            outcome_key=torch.randn(embed_dim),
            step=0,
            domain_name=domain.domain_name,
            selected_posture=stored_posture,
        )
    )
    model.memory.fill_outcome(idx, ic_realized=0.25, outcome_key=torch.randn(embed_dim))

    loss = model.train_world_model_from_memory()
    assert loss is not None
    assert world_model.last_postures is not None
    assert world_model.last_postures.shape == (1, posture_dim)
    assert torch.allclose(world_model.last_postures[0], stored_posture, atol=1e-6)


def test_train_retrieval_from_memory_uses_stored_retrieval_trace() -> None:
    """Retrieval replay should consume stored shortlist traces from memory."""

    class CaptureScorer(MemoryRelevanceScorer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.last_query_key: torch.Tensor | None = None
            self.last_query_posture: torch.Tensor | None = None
            self.last_memory_postures: torch.Tensor | None = None

        def forward(
            self,
            query_key: torch.Tensor,
            memory_keys: torch.Tensor,
            memory_summaries: torch.Tensor,
            memory_quality: torch.Tensor | None = None,
            query_posture: torch.Tensor | None = None,
            memory_postures: torch.Tensor | None = None,
        ) -> dict[str, torch.Tensor]:
            self.last_query_key = query_key.detach().clone()
            self.last_query_posture = (
                query_posture.detach().clone() if query_posture is not None else None
            )
            self.last_memory_postures = (
                memory_postures.detach().clone() if memory_postures is not None else None
            )
            return super().forward(
                query_key=query_key,
                memory_keys=memory_keys,
                memory_summaries=memory_summaries,
                memory_quality=memory_quality,
                query_posture=query_posture,
                memory_postures=memory_postures,
            )

    torch.manual_seed(0)
    embed_dim = 32
    num_ctx_tokens = 4
    action_dim = 8
    posture_dim = 6
    domain = DummyDomain(embed_dim=embed_dim, action_dim=action_dim)
    scorer = CaptureScorer(embed_dim=embed_dim, posture_dim=posture_dim)
    cost_fns, weights = zip(*domain.get_intrinsic_cost_fns(), strict=False)
    model = Chamelia(
        hjepa=DummyHJEPA(embed_dim=embed_dim),
        configurator=Configurator(
            embed_dim=embed_dim,
            num_ctx_tokens=num_ctx_tokens,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            memory_read_k=4,
        ),
        actor=Actor(
            embed_dim=embed_dim,
            action_dim=action_dim,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=num_ctx_tokens,
            posture_dim=posture_dim,
        ),
        cost=CostModule(
            intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
            trainable_critic=TrainableCritic(
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=2,
                mlp_ratio=2.0,
                dropout=0.0,
                num_ctx_tokens=num_ctx_tokens,
            ),
        ),
        memory=LatentMemory(embed_dim=embed_dim, max_episodes=8, retrieval_k=4, device="cpu"),
        domain=domain,
        retrieval_scorer=scorer,
        embed_dim=embed_dim,
        action_dim=action_dim,
        num_ctx_tokens=num_ctx_tokens,
    )

    query_key = torch.randn(embed_dim)
    query_posture = torch.randn(posture_dim)
    memory_keys = torch.randn(2, embed_dim)
    memory_summaries = torch.randn(2, embed_dim)
    memory_postures = torch.randn(2, posture_dim)
    stored_trace = (
        RetrievalTraceStep(
            query_key=query_key,
            query_posture=query_posture,
            memory_keys=memory_keys,
            memory_summaries=memory_summaries,
            memory_postures=memory_postures,
            base_scores=torch.tensor([0.7, 0.5], dtype=torch.float32),
            base_quality_scores=torch.tensor([-0.2, -0.8], dtype=torch.float32),
            relevance_scores=torch.tensor([0.1, -0.1], dtype=torch.float32),
            relevance_weights=torch.tensor([0.6, 0.4], dtype=torch.float32),
        ),
    )
    idx = model.memory.store(
        EpisodeRecord(
            key=torch.randn(embed_dim),
            action=torch.randn(action_dim),
            ctx_tokens=torch.randn(num_ctx_tokens, embed_dim),
            ic_at_decision=0.1,
            ic_realized=0.2,
            tc_predicted=0.3,
            outcome_key=torch.randn(embed_dim),
            step=0,
            domain_name=domain.domain_name,
            selected_posture=query_posture,
            retrieval_trace=stored_trace,
        )
    )
    model.memory.fill_outcome(idx, ic_realized=0.35, outcome_key=torch.randn(embed_dim))

    loss = model.train_retrieval_from_memory(temperature=0.25)
    assert loss is not None
    assert scorer.last_query_key is not None
    assert scorer.last_query_posture is not None
    assert scorer.last_memory_postures is not None
    assert torch.allclose(scorer.last_query_key[0], query_key, atol=1e-6)
    assert torch.allclose(scorer.last_query_posture[0], query_posture, atol=1e-6)
    assert torch.allclose(scorer.last_memory_postures[0], memory_postures, atol=1e-6)


def test_episode_records_capture_model_version() -> None:
    """Stored episode traces should carry the producing model version."""
    torch.manual_seed(0)

    embed_dim = 32
    num_ctx_tokens = 4
    action_dim = 8
    domain = DummyDomain(embed_dim=embed_dim, action_dim=action_dim)
    tokenizer = domain.get_tokenizer()

    model = Chamelia(
        hjepa=DummyHJEPA(embed_dim=embed_dim),
        configurator=Configurator(
            embed_dim=embed_dim,
            num_ctx_tokens=num_ctx_tokens,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            memory_read_k=4,
        ),
        actor=Actor(
            embed_dim=embed_dim,
            action_dim=action_dim,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=num_ctx_tokens,
        ),
        cost=CostModule(
            intrinsic_cost=IntrinsicCost(
                [pair[0] for pair in domain.get_intrinsic_cost_fns()],
                [pair[1] for pair in domain.get_intrinsic_cost_fns()],
            ),
            trainable_critic=TrainableCritic(
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=2,
                mlp_ratio=2.0,
                dropout=0.0,
                num_ctx_tokens=num_ctx_tokens,
            ),
        ),
        memory=LatentMemory(embed_dim=embed_dim, max_episodes=8, retrieval_k=4, device="cpu"),
        domain=domain,
        embed_dim=embed_dim,
        action_dim=action_dim,
        num_ctx_tokens=num_ctx_tokens,
        model_version="bridge-test-model-v1",
    )

    observation = torch.randint(0, domain.vocab_size, (1, 16))
    tokenized = tokenizer(observation.long())
    outputs = model(
        tokens=tokenized.tokens,
        mask=torch.zeros(1, tokenized.tokens.shape[1], dtype=torch.float32),
        domain_state=domain.get_domain_state(observation),
        input_kind="embedded_tokens",
        store_to_memory=True,
    )

    assert outputs["action_vec"].shape == (1, action_dim)
    assert model.memory.records
    assert model.memory.records[-1].model_version == "bridge-test-model-v1"


def test_latent_memory_fill_outcome_safe_after_buffer_wrap() -> None:
    """fill_outcome must not corrupt a newer record when the buffer wraps.

    This is a regression test for the circular-buffer index corruption bug:
    store() used to return the raw slot position, so if the buffer wrapped
    between store() and fill_outcome(), the fill would write to a different
    (newer) record and silently corrupt critic/world-model training pairs.

    After the fix, store() returns a stable record_id; fill_outcome() looks
    up the slot via _id_to_slot and is a no-op when the record was evicted.
    """
    embed_dim = 4
    max_episodes = 3  # tiny buffer so we can wrap quickly
    memory = LatentMemory(embed_dim=embed_dim, max_episodes=max_episodes, retrieval_k=2)

    def _make_record(step: int) -> EpisodeRecord:
        return EpisodeRecord(
            key=torch.randn(embed_dim),
            action=torch.randn(2),
            ctx_tokens=torch.randn(2, embed_dim),
            ic_at_decision=float(step),
            ic_realized=None,
            tc_predicted=0.0,
            outcome_key=None,
            step=step,
            domain_name="test",
        )

    # Store 3 records to fill the buffer.
    rid0 = memory.store(_make_record(0))
    rid1 = memory.store(_make_record(1))
    rid2 = memory.store(_make_record(2))

    # All three should be retrievable.
    assert memory.get_record_by_id(rid0) is not None
    assert memory.get_record_by_id(rid1) is not None
    assert memory.get_record_by_id(rid2) is not None

    # Fill outcome for rid0 before it is evicted.
    filled = memory.fill_outcome(rid0, ic_realized=0.5, outcome_key=torch.randn(embed_dim))
    assert filled
    assert memory.get_record_by_id(rid0).ic_realized == 0.5  # type: ignore[union-attr]

    # Store one more record — this overwrites slot 0, evicting rid0.
    rid3 = memory.store(_make_record(3))
    assert memory.get_record_by_id(rid0) is None  # evicted

    # Attempting to fill the evicted record must be a silent no-op,
    # not corrupt the new record that now occupies that slot.
    filled_after_eviction = memory.fill_outcome(
        rid0, ic_realized=99.0, outcome_key=torch.randn(embed_dim)
    )
    assert not filled_after_eviction

    new_record = memory.get_record_by_id(rid3)
    assert new_record is not None
    assert new_record.ic_realized is None  # untouched — the bug would have set this to 99.0

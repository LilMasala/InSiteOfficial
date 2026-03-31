"""Shape tests for the Chamelia V2 modules."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.chamelia.actor import Actor
from src.chamelia.chamelia import Chamelia
from src.chamelia.configurator import Configurator
from src.chamelia.cost import CostModule, IntrinsicCost, TrainableCritic
from src.chamelia.memory import EpisodeRecord, LatentMemory
from src.chamelia.plugins.base import AbstractDomain, DomainRegistry
from src.chamelia.tokenizers import SequenceTokenizer
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

    model.fill_outcome(ic_realized=0.4, outcome_observation=raw_tokens)
    critic_loss = model.train_critic_from_memory()
    assert critic_loss is not None
    assert critic_loss.dim() == 0


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
    assert outputs["cost"]["total"].shape == (1,)
    assert outputs["hjepa_out"]["target_features"].shape == (1, 197, embed_dim)
    assert len(level_feats) == 3
    assert level_feats[0].shape == (1, 196, embed_dim)
    assert level_feats[1].shape == (1, 98, embed_dim)
    assert level_feats[2].shape == (1, 49, embed_dim)

"""Contract tests for Chamelia domain plugins."""

from __future__ import annotations

from typing import Any

import torch

from src.chamelia.cost import IntrinsicCost
from src.chamelia.plugins.base import AbstractDomain
from src.chamelia.tokenizers import SequenceTokenizer
from training.curriculum.domains.base import BoardRuntimeDomain, DomainSpec


class CaptureDomain(AbstractDomain):
    """Small plugin used to validate default hook behavior."""

    def __init__(self, embed_dim: int = 16, action_dim: int = 4) -> None:
        self._tokenizer = SequenceTokenizer(
            vocab_size=32,
            embed_dim=embed_dim,
            max_seq_len=8,
            domain_name="capture",
            pad_token_id=0,
        )
        self._action_dim = action_dim
        self.last_decoded_action: torch.Tensor | None = None
        self.last_simulated_action: torch.Tensor | None = None

    def get_tokenizer(self) -> SequenceTokenizer:
        return self._tokenizer

    def get_action_dim(self) -> int:
        return self._action_dim

    def decode_action(self, action_vec: torch.Tensor) -> Any:
        self.last_decoded_action = action_vec.detach().clone()
        return action_vec

    def get_intrinsic_cost_fns(self) -> list[tuple[Any, float]]:
        return []

    def get_domain_state(self, observation: Any) -> dict:
        return {"raw": observation}

    def compute_regime_embedding(self, domain_state: dict) -> torch.Tensor | None:
        _ = domain_state
        return None

    def simulate_delayed_outcome(
        self,
        action_vec: torch.Tensor,
        domain_state: dict,
    ) -> dict[str, torch.Tensor] | None:
        _ = domain_state
        self.last_simulated_action = action_vec.detach().clone()
        return {
            "outcome_observation": torch.zeros(action_vec.shape[0], 2, dtype=torch.long),
            "realized_intrinsic_cost": torch.zeros(action_vec.shape[0], dtype=torch.float32),
        }

    @property
    def domain_name(self) -> str:
        return "capture"

    @property
    def vocab_size(self) -> int:
        return 32


class PresentingDomain(CaptureDomain):
    """Plugin overriding optional persistence and rendering hooks."""

    def get_persistable_domain_state(self, domain_state: dict) -> dict[str, Any] | None:
        return {"safe": domain_state.get("safe")}

    def render_recommendation(
        self,
        action: Any,
        action_path: torch.Tensor | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> Any | None:
        return {
            "action": action,
            "has_path": action_path is not None,
            "diagnostic_keys": sorted((diagnostics or {}).keys()),
        }


class OpaqueCostDomain(CaptureDomain):
    """Plugin whose intrinsic cost terms depend on plugin-only state keys."""

    def get_intrinsic_cost_fns(self) -> list[tuple[Any, float]]:
        def plugin_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict) -> torch.Tensor:
            _ = action
            return z.mean(dim=-1) + domain_state["plugin_only_bias"].float()

        return [(plugin_cost, 1.0)]


class _ToyBoardOwner:
    """Minimal owner surface for BoardRuntimeDomain contract tests."""

    def __init__(self) -> None:
        self.spec = DomainSpec(
            name="toy_board",
            stage_idx=0,
            action_dim=9,
            vocab_size=16,
            batch_size=2,
            seq_len=9,
        )

    @property
    def vocab_size(self) -> int:
        return self.spec.vocab_size

    @property
    def action_dim(self) -> int:
        return self.spec.action_dim

    def domain_name(self) -> str:
        return self.spec.name

    def _current_cost_terms(self) -> list[tuple[Any, float]]:
        return []


def test_default_decode_action_path_uses_first_step() -> None:
    """Default path decoding should remain plugin-owned and based on the first step."""
    domain = CaptureDomain()
    path = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.8, 0.7, 0.6],
        ],
        dtype=torch.float32,
    )
    decoded = domain.decode_action_path(path)

    assert domain.last_decoded_action is not None
    assert domain.last_decoded_action.shape == (1, 4)
    assert torch.allclose(domain.last_decoded_action[0], path[0], atol=1.0e-6)
    assert torch.allclose(decoded[0], path[0], atol=1.0e-6)


def test_default_simulate_path_outcome_uses_first_step() -> None:
    """Default path simulation should delegate through the plugin's single-action hook."""
    domain = CaptureDomain()
    action_path = torch.tensor(
        [
            [
                [0.1, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    outcome = domain.simulate_path_outcome(action_path, {"raw": torch.tensor([1.0])})

    assert outcome is not None
    assert domain.last_simulated_action is not None
    assert domain.last_simulated_action.shape == (1, 4)
    assert torch.allclose(domain.last_simulated_action[0], action_path[0, 0], atol=1.0e-6)


def test_optional_persistence_and_rendering_hooks_are_plugin_owned() -> None:
    """Presentation and persistence should be optional plugin hooks, not core requirements."""
    base_domain = CaptureDomain()
    assert base_domain.get_persistable_domain_state({"safe": 1, "secret": 2}) is None
    assert base_domain.render_recommendation(action="noop") is None

    presenting = PresentingDomain()
    persistable = presenting.get_persistable_domain_state({"safe": 1, "secret": 2})
    rendered = presenting.render_recommendation(
        action="support",
        action_path=torch.zeros(2, 4),
        diagnostics={"score": 0.1},
    )

    assert persistable == {"safe": 1}
    assert rendered == {
        "action": "support",
        "has_path": True,
        "diagnostic_keys": ["score"],
    }


def test_board_runtime_domain_decodes_paths_without_core_grid_logic() -> None:
    """Board/grid semantics should stay in the plugin/runtime adapter, not the actor."""
    runtime_domain = BoardRuntimeDomain(owner=_ToyBoardOwner(), embed_dim=16)
    action_path = torch.tensor(
        [
            [0.0, 0.0, 1.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    decoded = runtime_domain.decode_action_path(action_path)
    assert torch.equal(decoded, torch.tensor([2]))


def test_intrinsic_cost_terms_can_depend_on_plugin_owned_state() -> None:
    """Core cost aggregation should treat plugin domain-state fields opaquely."""
    domain = OpaqueCostDomain()
    intrinsic_cost = IntrinsicCost(
        [pair[0] for pair in domain.get_intrinsic_cost_fns()],
        [pair[1] for pair in domain.get_intrinsic_cost_fns()],
    )
    z = torch.tensor([[1.0, 3.0, 5.0, 7.0]], dtype=torch.float32)
    action = torch.zeros(1, domain.get_action_dim(), dtype=torch.float32)
    domain_state = {"plugin_only_bias": torch.tensor([0.25], dtype=torch.float32)}

    value = intrinsic_cost(z, action, domain_state)
    assert torch.allclose(value, torch.tensor([4.25], dtype=torch.float32), atol=1.0e-6)

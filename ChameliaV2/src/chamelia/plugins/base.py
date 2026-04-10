"""Base plugin interface for Chamelia domains."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from src.chamelia.tokenizers import AbstractTokenizer


class AbstractDomain(ABC):
    """Abstract plugin interface for Chamelia domains."""

    @abstractmethod
    def get_tokenizer(self) -> AbstractTokenizer:
        """Return the domain tokenizer.

        Returns:
            AbstractTokenizer instance. Its forward() must produce tokens [B, N, D].
        """

    @abstractmethod
    def get_action_dim(self) -> int:
        """Return the continuous actor output dimension.

        Returns:
            Integer action dimension A.
        """

    @abstractmethod
    def decode_action(self, action_vec: torch.Tensor) -> Any:
        """Decode a continuous action vector.

        Args:
            action_vec: Tensor of shape [B, A].

        Returns:
            Domain-specific action object(s).
        """

    def decode_action_path(self, action_path: torch.Tensor) -> Any:
        """Decode a whole candidate path.

        Args:
            action_path: Tensor of shape [B, P, A], [P, A], or [A].

        Returns:
            Domain-specific path object(s).

        Default behavior delegates to ``decode_action`` on the first action of the path.
        Plugins may override this when path-level semantics matter.
        """
        if action_path.dim() == 3:
            return self.decode_action(action_path[:, 0, :])
        if action_path.dim() == 2:
            return self.decode_action(action_path[0, :].unsqueeze(0))
        return self.decode_action(action_path)

    @abstractmethod
    def get_intrinsic_cost_fns(self) -> list[tuple[Callable, float]]:
        """Return domain intrinsic cost functions and fixed weights.

        Returns:
            List of (cost_fn, weight) tuples where each cost_fn consumes z [B, D],
            action [B, A], and domain_state dict, and returns [B].
        """

    @abstractmethod
    def get_domain_state(self, observation: Any) -> dict:
        """Build an opaque domain-state payload from an observation.

        Args:
            observation: Raw domain observation.

        Returns:
            Domain-state dict consumed opaquely by cost functions.
        """

    def prepare_bridge_observation(self, observation: Any) -> Any:
        """Convert a bridge observation payload into tokenizer input.

        Args:
            observation: Plugin-owned bridge observation payload.

        Returns:
            Tokenizer input object compatible with ``get_tokenizer()``.

        Default behavior passes the payload through unchanged.
        Plugins should override this when the bridge observation format differs from the
        tokenizer's native input format.
        """
        return observation

    def get_persistable_domain_state(self, domain_state: dict) -> dict[str, Any] | None:
        """Optionally filter domain state for persistence outside the plugin.

        Args:
            domain_state: Plugin-owned opaque domain-state payload.

        Returns:
            Optional filtered payload safe to persist or transport as plugin-owned state.

        Default behavior returns ``None`` so core does not assume the full domain state
        is portable or persistence-safe.
        """
        _ = domain_state
        return None

    def get_trainable_modules(self) -> dict[str, nn.Module]:
        """Optionally expose plugin-owned trainable modules to core.

        Returns:
            Mapping of stable module names to ``nn.Module`` instances.

        Default behavior exposes no additional trainable modules.
        """
        return {}

    @abstractmethod
    def compute_regime_embedding(self, domain_state: dict) -> torch.Tensor | None:
        """Optionally return a regime embedding.

        Args:
            domain_state: Opaque domain-state dict.

        Returns:
            Tensor of shape [D] or [B, D], or None if unavailable.
        """

    def build_imagined_domain_state(
        self,
        current_domain_state: dict[str, Any],
        future_z: torch.Tensor,
        step_idx: int,
    ) -> dict[str, Any]:
        """Build domain state for imagined future steps during planning.

        Args:
            current_domain_state: Current batched domain state payload.
            future_z: Predicted latent for the imagined future step.
            step_idx: Zero-based rollout step index.

        Returns:
            Domain-state payload aligned with ``future_z``.

        Default behavior preserves the current domain state unchanged.
        """
        _ = future_z
        _ = step_idx
        return current_domain_state

    def compute_latent_state_decoder_loss(
        self,
        predicted_future_z: torch.Tensor,
        target_domain_state: dict[str, Any],
    ) -> torch.Tensor | None:
        """Optionally supervise plugin-owned latent-to-state decoders.

        Args:
            predicted_future_z: Predicted future latent tensor [B, D].
            target_domain_state: Target future domain-state payload aligned to ``predicted_future_z``.

        Returns:
            Optional scalar loss tensor, or ``None`` when unsupported.
        """
        _ = predicted_future_z
        _ = target_domain_state
        return None

    def simulate_delayed_outcome(
        self,
        action_vec: torch.Tensor,
        domain_state: dict,
    ) -> dict[str, torch.Tensor] | None:
        """Optionally simulate a delayed outcome for training-time memory filling.

        Args:
            action_vec: Selected action tensor [B, A].
            domain_state: Opaque domain-state dict from the current batch.

        Returns:
            Optional dict with:
                - ``outcome_observation``: raw observation payload for ``fill_outcome``
                - ``realized_intrinsic_cost``: tensor [B]
            Domains that cannot provide a delayed outcome should return ``None``.
        """
        _ = action_vec
        _ = domain_state
        return None

    def simulate_path_outcome(
        self,
        action_path: torch.Tensor,
        domain_state: dict,
    ) -> dict[str, torch.Tensor] | None:
        """Optionally simulate the delayed outcome of a whole candidate path.

        Args:
            action_path: Candidate path tensor [B, P, A] or [B, A].
            domain_state: Opaque domain-state dict from the current batch.

        Returns:
            Optional dict with:
                - ``outcome_observation``: raw observation payload for ``fill_outcome``
                - ``realized_intrinsic_cost``: tensor [B]

        Default behavior falls back to the first action of the path when possible.
        """
        if action_path.dim() == 3:
            return self.simulate_delayed_outcome(action_path[:, 0, :], domain_state)
        return self.simulate_delayed_outcome(action_path, domain_state)

    def render_recommendation(
        self,
        action: Any,
        action_path: torch.Tensor | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> Any | None:
        """Optionally render a human-facing recommendation package.

        Args:
            action: Plugin-decoded selected action.
            action_path: Optional raw selected path tensor.
            diagnostics: Optional planner diagnostics.

        Returns:
            Optional plugin-owned presentation object. Core should treat this as opaque.

        Default behavior returns ``None`` because presentation belongs outside core.
        """
        _ = action
        _ = action_path
        _ = diagnostics
        return None

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Return the human-readable domain name.

        Returns:
            String identifier.
        """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return tokenizer vocabulary size if relevant.

        Returns:
            Integer vocabulary size, or 0 for non-sequence domains.
        """


class DomainRegistry:
    """Singleton-style registry for domain plugins."""

    _registry: dict[str, AbstractDomain] = {}

    @staticmethod
    def register(domain: AbstractDomain) -> None:
        """Register a domain plugin.

        Args:
            domain: Domain plugin instance.

        Returns:
            None.
        """
        DomainRegistry._registry[domain.domain_name] = domain

    @staticmethod
    def get(domain_name: str) -> AbstractDomain:
        """Return a previously registered domain plugin.

        Args:
            domain_name: Registry key.

        Returns:
            AbstractDomain instance.
        """
        if domain_name not in DomainRegistry._registry:
            raise KeyError(f"Domain '{domain_name}' not registered.")
        return DomainRegistry._registry[domain_name]

    @staticmethod
    def list_domains() -> list[str]:
        """List registered domain names.

        Returns:
            Python list of registry keys.
        """
        return list(DomainRegistry._registry.keys())


class InteractiveDomainAdapter(AbstractDomain):
    """Interactive environment contract used by the unified training orchestrator."""

    modality_family: str = "unknown"
    action_space_type: str = "unknown"

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and return ``(observation, info)``."""

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Apply one environment action and return Gym-style step outputs."""

    def tokenize_observation(self, observation: Any) -> Any:
        """Tokenize one observation via the domain tokenizer."""
        tokenizer = self.get_tokenizer()
        tokenizer_input = self.prepare_bridge_observation(observation)
        batched_input = (
            tokenizer.collate([tokenizer_input])
            if hasattr(tokenizer, "collate")
            else tokenizer_input
        )
        if isinstance(tokenizer, torch.nn.Module) and torch.is_tensor(batched_input):
            try:
                tokenizer_device = next(tokenizer.parameters()).device
                batched_input = batched_input.to(tokenizer_device)
            except StopIteration:
                pass
        return tokenizer(batched_input)

    def build_domain_state(self, observation: Any, info: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build the training-time domain state payload."""
        _ = info
        return self.get_domain_state(observation)

    def legal_action_mask(self, observation: Any, info: dict[str, Any] | None = None) -> torch.Tensor | None:
        """Return a legal-action mask when the environment exposes one."""
        _ = observation
        _ = info
        return None

    def compute_realized_cost(
        self,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> float:
        """Convert environment feedback into the cost scalar used by Chamelia."""
        _ = observation
        _ = terminated
        _ = truncated
        _ = info
        return -float(reward)

    def compute_metrics(self, episode_records: list[dict[str, Any]]) -> dict[str, float]:
        """Aggregate episode-level metrics for one evaluation pass."""
        if not episode_records:
            return {"episode_reward_mean": 0.0}
        rewards = [float(record.get("episode_reward", 0.0)) for record in episode_records]
        return {"episode_reward_mean": sum(rewards) / len(rewards)}

    def baseline_action(
        self,
        kind: str,
        observation: Any,
        info: dict[str, Any] | None = None,
    ) -> Any:
        """Return a baseline action for evaluation.

        The default implementation supports ``random`` for discrete domains.
        """
        _ = observation
        legal_mask = self.legal_action_mask(observation, info)
        if kind != "random":
            raise ValueError(f"Unsupported baseline '{kind}' for domain '{self.domain_name}'.")
        if self.action_space_type != "discrete":
            raise ValueError("Default random baseline only supports discrete action spaces.")
        if legal_mask is None:
            return torch.randint(self.get_action_dim(), (1,), dtype=torch.long)
        legal_indices = torch.nonzero(legal_mask.reshape(-1), as_tuple=False).squeeze(-1)
        if legal_indices.numel() == 0:
            return torch.zeros(1, dtype=torch.long)
        choice = legal_indices[torch.randint(legal_indices.numel(), (1,))]
        return choice

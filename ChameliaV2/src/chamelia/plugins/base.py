"""Base plugin interface for Chamelia domains."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch

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

    @abstractmethod
    def compute_regime_embedding(self, domain_state: dict) -> torch.Tensor | None:
        """Optionally return a regime embedding.

        Args:
            domain_state: Opaque domain-state dict.

        Returns:
            Tensor of shape [D] or [B, D], or None if unavailable.
        """

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

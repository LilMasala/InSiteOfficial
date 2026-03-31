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

    @abstractmethod
    def compute_regime_embedding(self, domain_state: dict) -> torch.Tensor | None:
        """Optionally return a regime embedding.

        Args:
            domain_state: Opaque domain-state dict.

        Returns:
            Tensor of shape [D] or [B, D], or None if unavailable.
        """

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


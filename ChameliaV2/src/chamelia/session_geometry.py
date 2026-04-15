"""Session-level geometry contract for Chamelia.

A ``SessionGeometry`` instance is created once per session (typically inside
``Chamelia.set_domain``) and propagated through the BridgeRuntime so that every
sub-module shares a consistent view of {D, A, O, P, K, H}.

Rules
-----
* ``D`` is immutable for the lifetime of a model.  All Transformer blocks,
  Mamba layers, and LatentMemory keys operate in this space.
* ``P == D`` always.  Postures and Skills are full D-dimensional *intent*
  vectors stored directly in LatentMemory — they are **not** projected to a
  narrower posture space.
* ``A``, ``O`` are domain-specific and bound at the start of each session via
  ``SessionGeometry.from_domain()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SessionGeometry:
    """Immutable geometry snapshot for a single Chamelia session.

    Attributes:
        D: Global latent width (default 512).  All internal Transformer /
           Mamba blocks and memory keys use this dimension.  Immutable.
        A: Action dimension for the active domain.
        O: Observation token dimension (0 = unknown / not provided).
        P: Posture / intent dimension.  Must equal D.
        K: Number of candidate action proposals per forward pass.
        H: Rollout horizon (path length) per candidate.
    """

    D: int = 512
    A: int = 64
    O: int = 0
    P: int = 512
    K: int = 6
    H: int = 3

    def __post_init__(self) -> None:
        if self.P != self.D:
            raise ValueError(
                f"SessionGeometry: P={self.P} must equal D={self.D}.  "
                "Postures live in the full D-dimensional intent space."
            )
        if self.D <= 0:
            raise ValueError("D must be a positive integer.")
        if self.A <= 0:
            raise ValueError("A must be a positive integer.")
        if self.K <= 0:
            raise ValueError("K must be a positive integer.")
        if self.H <= 0:
            raise ValueError("H must be a positive integer.")

    @classmethod
    def from_domain(
        cls,
        domain: object,
        *,
        D: int = 512,
        K: int = 6,
        H: int = 3,
    ) -> "SessionGeometry":
        """Construct geometry by querying a live AbstractDomain.

        Args:
            domain: AbstractDomain instance exposing ``get_action_dim()``.
            D: Global latent width (default 512).
            K: Number of candidates (default 6).
            H: Rollout horizon / path length (default 3).

        Returns:
            SessionGeometry with A bound to ``domain.get_action_dim()``.
        """
        A: int = domain.get_action_dim()  # type: ignore[attr-defined]
        return cls(D=D, A=A, O=0, P=D, K=K, H=H)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def num_heads(self) -> int:
        """Largest power-of-two divisor of D, capped at 8.

        Used to configure MultiheadAttention so that ``D % num_heads == 0``
        is always satisfied regardless of D.
        """
        for h in (8, 4, 2, 1):
            if self.D % h == 0:
                return h
        return 1  # pragma: no cover

    @property
    def num_ctx_tokens(self) -> int:
        """Context-token budget scaled with session complexity.

        Computed as ``ceil(sqrt(K * H) * 4)`` rounded up to the next
        multiple of 8, with a floor of 8.
        """
        raw = max(8, math.ceil(math.sqrt(self.K * self.H) * 4))
        return ((raw + 7) // 8) * 8

"""Session-level geometry contract for Chamelia.

A ``SessionGeometry`` is created once per session inside ``Chamelia.set_domain``
and propagated to every sub-module that has geometry-dependent weights.

Design rules
------------
* ``D`` is immutable for the lifetime of a model.  All Transformer / Mamba
  blocks and LatentMemory keys operate in this space.
* ``P`` (posture dim) is intentionally *smaller* than ``D`` — a behavioural-
  intent bottleneck analogous to a style code.  ``MemoryRelevanceScorer``
  already projects P → D internally before comparing postures, so no extra
  projection layer is needed elsewhere.
* ``H`` is the Actor's per-candidate *path length* (planning horizon).
* ``T`` is the World Model's *rollout horizon* used for value estimation.
  H and T are distinct concepts and may differ.
* ``A`` is domain-specific and bound at session start via ``from_domain()``.
* The dataclass is frozen so that geometry objects cannot be mutated after
  construction, keeping the contract between the geometry record and the
  model's internal state consistent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionGeometry:
    """Immutable geometry snapshot for a single Chamelia session.

    Attributes:
        D: Global latent width (default 512).  Immutable for model lifetime.
        A: Action dimension for the active domain.
        P: Posture bottleneck dimension (default 16).  Kept small to preserve
           the behavioural-intent compression role of the posture vector.
        K: Number of candidate action proposals per deliberate forward pass.
        H: Per-candidate path length (Actor planning horizon).
        T: World-model rollout horizon for value estimation.  May differ from H.
    """

    D: int = 512
    A: int = 64
    P: int = 16
    K: int = 6
    H: int = 3
    T: int = 8

    def __post_init__(self) -> None:
        if self.D <= 0:
            raise ValueError("D must be a positive integer.")
        if self.A <= 0:
            raise ValueError("A must be a positive integer.")
        if self.P <= 0:
            raise ValueError("P must be a positive integer.")
        if self.K <= 0:
            raise ValueError("K must be a positive integer.")
        if self.H <= 0:
            raise ValueError("H must be a positive integer.")
        if self.T <= 0:
            raise ValueError("T must be a positive integer.")
        if self.P > self.D:
            raise ValueError(
                f"P={self.P} must be <= D={self.D}.  "
                "The posture dim is a bottleneck inside D, not a superset."
            )

    @classmethod
    def from_domain(
        cls,
        domain: object,
        *,
        D: int = 512,
        P: int = 16,
        K: int = 6,
        H: int = 3,
        T: int = 8,
    ) -> "SessionGeometry":
        """Construct geometry by querying a live AbstractDomain.

        Args:
            domain: AbstractDomain instance exposing ``get_action_dim()``.
            D: Global latent width (default 512).
            P: Posture bottleneck dimension (default 16).
            K: Number of candidates (default 6).
            H: Actor path length / planning horizon (default 3).
            T: World-model rollout horizon (default 8).

        Returns:
            SessionGeometry with A bound to ``domain.get_action_dim()``.
        """
        A: int = domain.get_action_dim()  # type: ignore[attr-defined]
        return cls(D=D, A=A, P=P, K=K, H=H, T=T)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def num_heads(self) -> int:
        """Largest power-of-two divisor of D, capped at 8.

        Guarantees ``D % num_heads == 0`` for any D so MultiheadAttention
        never raises on misaligned head counts.
        """
        for h in (8, 4, 2, 1):
            if self.D % h == 0:
                return h
        return 1  # pragma: no cover

    @property
    def num_ctx_tokens(self) -> int:
        """Suggested context-token budget scaled with session complexity.

        Computed as ``ceil(sqrt(K * H) * 4)`` rounded up to the next
        multiple of 8, with a floor of 8.  Passed to ``Configurator.bind_geometry``
        so the token budget can adapt when K or H changes between sessions.
        """
        raw = max(8, math.ceil(math.sqrt(self.K * self.H) * 4))
        return ((raw + 7) // 8) * 8

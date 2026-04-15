"""Session-level geometry contract for Chamelia.

A ``SessionGeometry`` instance is created once per session (typically inside
``Chamelia.set_domain``) and propagated through the BridgeRuntime so that every
sub-module shares a consistent view of {D, A, O, P, K, H, T}.

Design rules
------------
* ``D`` is immutable for the lifetime of a model.  All Transformer blocks,
  Mamba layers, and LatentMemory keys operate in this space.
* ``P`` (posture dim) is intentionally *smaller* than D.  The posture vector is a
  behavioural-intent bottleneck inside the Actor — a compressed "strategy code"
  analogous to a style vector.  Before storage in LatentMemory, postures are
  projected to D via ``Actor.posture_to_intent`` so that retrieval and novelty
  comparisons happen in the shared D-dimensional space.
* ``H`` is the Actor's per-candidate *path length* (planning horizon).
* ``T`` is the World Model's *rollout horizon* used for value estimation.
  H and T are separate concepts and may differ.
* ``A``, ``O`` are domain-specific and bound at session start via
  ``SessionGeometry.from_domain()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class SessionGeometry:
    """Immutable geometry snapshot for a single Chamelia session.

    Attributes:
        D: Global latent width (default 512).  All internal Transformer /
           Mamba blocks and memory keys live in this space.  Immutable.
        A: Action dimension for the active domain.
        O: Observation token dimension (0 = unknown / unset).
        P: Posture bottleneck dimension (default 16).  Kept small to preserve
           the behavioural-intent compression role of the posture vector.
           Postures are projected to D via ``Actor.posture_to_intent`` before
           being written to LatentMemory.
        K: Number of candidate action proposals per deliberate forward pass.
        H: Per-candidate path length (Actor planning horizon).
        T: World-model rollout horizon for value estimation.  May differ from H.
    """

    D: int = 512
    A: int = 64
    O: int = 0
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
                "The posture dim is a bottleneck inside D."
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
        return cls(D=D, A=A, O=0, P=P, K=K, H=H, T=T)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def num_heads(self) -> int:
        """Largest power-of-two divisor of D, capped at 8.

        Guarantees ``D % num_heads == 0`` for any D so MultiheadAttention
        never raises a shape error.
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

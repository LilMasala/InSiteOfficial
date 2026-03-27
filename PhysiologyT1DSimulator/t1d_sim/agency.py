"""User agency profiles for closed-loop simulation (Chamelia Block 8)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class UserAgencyProfile:
    """Decision profile governing how a simulated user interacts with recommendations.

    These traits interact with existing persona characteristics — a high-stress,
    low-logging-quality user typically has low initial trust and high engagement decay.

    Attributes:
        aggressiveness:    0.0–1.0. Willingness to accept large setting changes;
                           maps to optimizer constraint width.
        initial_trust:     0.0–1.0. Starting confidence in the system; affects
                           acceptance probability early in Phase 3.
        trust_growth_rate: How quickly trust builds as shadow predictions prove
                           accurate (per accepted recommendation).
        compliance_noise:  How precisely the user implements recommendations.
                           Applied as std of N(1.0, noise) multiplied onto the
                           recommended action; 0.0 = perfect, 0.5 = heavy noise.
        revert_threshold:  Absolute TIR drop (e.g. 0.08 = 8 pp) that triggers
                           rollback to previous settings.
        engagement_decay:  Per-day probability of not checking recommendations,
                           scaled by days elapsed in Phase 3.
    """
    aggressiveness: float
    initial_trust: float
    trust_growth_rate: float
    compliance_noise: float
    revert_threshold: float
    engagement_decay: float


def sample_agency(cfg: "PatientConfig", rng: np.random.Generator) -> UserAgencyProfile:  # type: ignore[name-defined]
    """Derive a UserAgencyProfile from an existing PatientConfig.

    Derivation is grounded in the same persona trait relationships used elsewhere
    in the simulator:
    - High stress_reactivity → lower initial_trust (diabetes distress → suspicion of AI)
    - High mood_stability → faster trust growth (emotional resilience → openness)
    - Poor logging quality → faster engagement decay (already disengaged)
    - High activity_propensity → higher aggressiveness (action-oriented)
    - Good logging quality → lower compliance_noise (attentive implementer)
    """

    def _c(val: float, lo: float, hi: float) -> float:
        return float(np.clip(val, lo, hi))

    initial_trust = _c(
        rng.normal(0.5 - 0.3 * cfg.stress_reactivity, 0.10),
        0.05, 0.95,
    )
    trust_growth_rate = _c(
        rng.normal(0.02 * (1.0 + cfg.mood_stability), 0.005),
        0.001, 0.10,
    )
    engagement_decay = _c(
        rng.normal(0.01 + 0.03 * (1.0 - cfg.logging_quality_raw), 0.005),
        0.001, 0.20,
    )
    aggressiveness = _c(
        rng.normal(0.5 * cfg.activity_propensity + 0.2, 0.15),
        0.0, 1.0,
    )
    compliance_noise = _c(
        rng.normal(0.3 - 0.2 * cfg.logging_quality_raw, 0.05),
        0.0, 0.5,
    )
    revert_threshold = _c(
        rng.normal(0.05 + 0.10 * (1.0 - aggressiveness), 0.02),
        0.02, 0.30,
    )

    return UserAgencyProfile(
        aggressiveness=aggressiveness,
        initial_trust=initial_trust,
        trust_growth_rate=trust_growth_rate,
        compliance_noise=compliance_noise,
        revert_threshold=revert_threshold,
        engagement_decay=engagement_decay,
    )

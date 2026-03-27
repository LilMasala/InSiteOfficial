"""Personality traits, recommendation budget, and framing selection.

Adapts system behavior to the person's personality and emotional state.
Same optimizer, same models, same safety gates — but different recommendation
cadence, framing, detail level, and emotional tone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Personality Traits
# ---------------------------------------------------------------------------

@dataclass
class UserPersonality:
    """Personality traits that modulate system behavior.

    All traits are 0–1 floats. Defaults represent a moderate user.
    """
    explanation_appetite: float = 0.5
    notification_tolerance: float = 0.5
    autonomy_preference: float = 0.3
    change_anxiety: float = 0.4
    celebration_receptiveness: float = 0.5
    education_need: float = 0.3

    # Psychological response traits (simulation-only)
    mood_boost_from_success: float = 0.08
    mood_hit_from_failure: float = -0.12
    mood_hit_from_overload: float = -0.06
    mood_boost_from_celebration: float = 0.04
    recommendation_fatigue_rate: float = 0.3
    burnout_threshold: float = -0.5

    def to_dict(self) -> dict:
        return {
            "explanation_appetite": self.explanation_appetite,
            "notification_tolerance": self.notification_tolerance,
            "autonomy_preference": self.autonomy_preference,
            "change_anxiety": self.change_anxiety,
            "celebration_receptiveness": self.celebration_receptiveness,
            "education_need": self.education_need,
            "mood_boost_from_success": self.mood_boost_from_success,
            "mood_hit_from_failure": self.mood_hit_from_failure,
            "mood_hit_from_overload": self.mood_hit_from_overload,
            "mood_boost_from_celebration": self.mood_boost_from_celebration,
            "recommendation_fatigue_rate": self.recommendation_fatigue_rate,
            "burnout_threshold": self.burnout_threshold,
        }


# ---------------------------------------------------------------------------
# Personality Archetypes
# ---------------------------------------------------------------------------

def anxious_optimizer() -> UserPersonality:
    """High transparency, high change anxiety."""
    return UserPersonality(
        explanation_appetite=0.9,
        notification_tolerance=0.85,
        autonomy_preference=0.15,
        change_anxiety=0.8,
        celebration_receptiveness=0.6,
        education_need=0.3,
        mood_boost_from_success=0.10,
        mood_hit_from_failure=-0.18,
        mood_hit_from_overload=-0.10,
        mood_boost_from_celebration=0.06,
        recommendation_fatigue_rate=0.5,
        burnout_threshold=-0.35,
    )


def hands_off_delegator() -> UserPersonality:
    """Minimal contact, high autonomy."""
    return UserPersonality(
        explanation_appetite=0.15,
        notification_tolerance=0.2,
        autonomy_preference=0.85,
        change_anxiety=0.15,
        celebration_receptiveness=0.3,
        education_need=0.1,
        mood_boost_from_success=0.05,
        mood_hit_from_failure=-0.08,
        mood_hit_from_overload=-0.04,
        mood_boost_from_celebration=0.02,
        recommendation_fatigue_rate=0.15,
        burnout_threshold=-0.60,
    )


def skeptic() -> UserPersonality:
    """Needs to be proven wrong slowly."""
    return UserPersonality(
        explanation_appetite=0.8,
        notification_tolerance=0.4,
        autonomy_preference=0.1,
        change_anxiety=0.7,
        celebration_receptiveness=0.4,
        education_need=0.2,
        mood_boost_from_success=0.06,
        mood_hit_from_failure=-0.15,
        mood_hit_from_overload=-0.08,
        mood_boost_from_celebration=0.03,
        recommendation_fatigue_rate=0.4,
        burnout_threshold=-0.40,
    )


def newly_diagnosed() -> UserPersonality:
    """High education need, moderate anxiety."""
    return UserPersonality(
        explanation_appetite=0.6,
        notification_tolerance=0.6,
        autonomy_preference=0.2,
        change_anxiety=0.5,
        celebration_receptiveness=0.7,
        education_need=0.9,
        mood_boost_from_success=0.12,
        mood_hit_from_failure=-0.10,
        mood_hit_from_overload=-0.08,
        mood_boost_from_celebration=0.08,
        recommendation_fatigue_rate=0.3,
        burnout_threshold=-0.40,
    )


ARCHETYPE_FACTORIES = {
    "anxious_optimizer": anxious_optimizer,
    "hands_off_delegator": hands_off_delegator,
    "skeptic": skeptic,
    "newly_diagnosed": newly_diagnosed,
}


def sample_personality(rng: np.random.Generator, archetype: str | None = None) -> UserPersonality:
    """Sample a personality, optionally biased toward an archetype."""
    if archetype and archetype in ARCHETYPE_FACTORIES:
        base = ARCHETYPE_FACTORIES[archetype]()
        # Add noise
        return UserPersonality(
            explanation_appetite=float(np.clip(base.explanation_appetite + rng.normal(0, 0.08), 0, 1)),
            notification_tolerance=float(np.clip(base.notification_tolerance + rng.normal(0, 0.08), 0, 1)),
            autonomy_preference=float(np.clip(base.autonomy_preference + rng.normal(0, 0.08), 0, 1)),
            change_anxiety=float(np.clip(base.change_anxiety + rng.normal(0, 0.08), 0, 1)),
            celebration_receptiveness=float(np.clip(base.celebration_receptiveness + rng.normal(0, 0.08), 0, 1)),
            education_need=float(np.clip(base.education_need + rng.normal(0, 0.08), 0, 1)),
            mood_boost_from_success=float(np.clip(base.mood_boost_from_success + rng.normal(0, 0.02), 0.01, 0.25)),
            mood_hit_from_failure=float(np.clip(base.mood_hit_from_failure + rng.normal(0, 0.03), -0.30, -0.02)),
            mood_hit_from_overload=float(np.clip(base.mood_hit_from_overload + rng.normal(0, 0.02), -0.20, -0.01)),
            mood_boost_from_celebration=float(np.clip(base.mood_boost_from_celebration + rng.normal(0, 0.015), 0.005, 0.15)),
            recommendation_fatigue_rate=float(np.clip(base.recommendation_fatigue_rate + rng.normal(0, 0.05), 0.05, 0.8)),
            burnout_threshold=float(np.clip(base.burnout_threshold + rng.normal(0, 0.05), -0.70, -0.20)),
        )
    # Random personality
    return UserPersonality(
        explanation_appetite=float(rng.uniform(0.1, 0.9)),
        notification_tolerance=float(rng.uniform(0.1, 0.9)),
        autonomy_preference=float(rng.uniform(0.05, 0.85)),
        change_anxiety=float(rng.uniform(0.1, 0.8)),
        celebration_receptiveness=float(rng.uniform(0.2, 0.8)),
        education_need=float(rng.uniform(0.1, 0.8)),
        mood_boost_from_success=float(rng.uniform(0.03, 0.15)),
        mood_hit_from_failure=float(rng.uniform(-0.20, -0.05)),
        mood_hit_from_overload=float(rng.uniform(-0.12, -0.02)),
        mood_boost_from_celebration=float(rng.uniform(0.01, 0.10)),
        recommendation_fatigue_rate=float(rng.uniform(0.1, 0.6)),
        burnout_threshold=float(rng.uniform(-0.60, -0.25)),
    )


# ---------------------------------------------------------------------------
# Recommendation Budget
# ---------------------------------------------------------------------------

@dataclass
class RecommendationBudget:
    """Per-patient recommendation budget that modulates cadence.

    Budget fills when mood is stable/positive and acceptance rate is healthy.
    Budget drains when mood is negative, acceptance drops, or recs fail.
    Empty budget → system goes quiet.
    """
    budget: float = 1.0           # 0–2 range; 1.0 = normal cadence
    max_budget: float = 2.0
    fill_rate: float = 0.15       # Per day when conditions are good
    drain_rate_mood: float = 0.20  # Per day when mood is negative
    drain_rate_failure: float = 0.30  # Per failed recommendation
    drain_rate_disengagement: float = 0.25  # Per day of low acceptance
    recs_since_last_fill: int = 0

    def can_recommend(self) -> bool:
        """Whether the budget allows surfacing a recommendation."""
        return self.budget > 0.3

    def is_overflow(self) -> bool:
        """Budget has been accumulating — may surface a bolder rec."""
        return self.budget > 1.5

    def consume(self, cost: float = 0.5) -> None:
        """Consume budget when a recommendation is surfaced."""
        self.budget = max(0.0, self.budget - cost)
        self.recs_since_last_fill += 1

    def daily_update(
        self,
        mood_valence: float,
        acceptance_rate_7d: float,
        last_rec_succeeded: bool | None,
        personality: UserPersonality,
    ) -> None:
        """Update budget based on daily signals."""
        # Fill when mood is stable/positive and acceptance is healthy
        if mood_valence > -0.1 and acceptance_rate_7d > 0.4:
            self.budget += self.fill_rate
            self.recs_since_last_fill = 0

        # Drain from negative mood
        if mood_valence < -0.2:
            drain = self.drain_rate_mood * abs(mood_valence)
            self.budget -= drain

        # Drain from low acceptance (disengagement signal)
        if acceptance_rate_7d < 0.3:
            self.budget -= self.drain_rate_disengagement

        # Drain from failed recommendations
        if last_rec_succeeded is False:
            self.budget -= self.drain_rate_failure

        # Personality modulation: high notification_tolerance → slower drain
        tolerance_factor = 0.7 + 0.6 * personality.notification_tolerance
        if self.budget < 1.0:
            self.budget = 1.0 - (1.0 - self.budget) / tolerance_factor

        self.budget = float(np.clip(self.budget, 0.0, self.max_budget))


# ---------------------------------------------------------------------------
# Recommendation Framing
# ---------------------------------------------------------------------------

class RecommendationFraming(Enum):
    TENTATIVE = "tentative"
    REINFORCING = "reinforcing"
    GENTLE_REMINDER = "gentle_reminder"
    CELEBRATING = "celebrating"


def select_framing(
    n_prior_recs: int,
    last_rec_accepted: bool | None,
    last_rec_succeeded: bool | None,
    tir_improved: bool,
    personality: UserPersonality,
) -> RecommendationFraming:
    """Select recommendation framing based on history and personality."""
    # First recommendation ever → tentative
    if n_prior_recs == 0:
        return RecommendationFraming.TENTATIVE

    # They accepted and it worked → celebrating
    if last_rec_accepted and last_rec_succeeded and tir_improved:
        return RecommendationFraming.CELEBRATING

    # They accepted before and it helped → reinforcing
    if last_rec_accepted and last_rec_succeeded:
        return RecommendationFraming.REINFORCING

    # They accepted but haven't followed through recently
    if last_rec_accepted is False and n_prior_recs > 2:
        return RecommendationFraming.GENTLE_REMINDER

    # Default: tentative for anxious users, reinforcing for confident ones
    if personality.change_anxiety > 0.5:
        return RecommendationFraming.TENTATIVE
    return RecommendationFraming.REINFORCING

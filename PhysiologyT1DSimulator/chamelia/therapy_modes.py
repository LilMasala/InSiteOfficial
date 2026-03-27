"""Therapy mode unlock ladder — state machine and level-specific constraints.

Each level is a small trust increment earned through demonstrated performance.
The system never leaps — it earns the right to make progressively more
sophisticated changes to the user's therapy.

Levels:
    0 — Shadow: watch, learn, log
    1 — Suggest Values: ISF/CR/basal within existing blocks
    2 — Shift Boundaries: move block start/end times
    3 — Create/Merge Blocks: split or merge blocks
    4 — Context Profiles: named profiles based on context
    5 — Autonomous Curves: B-spline continuous functions (stub)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Therapy Mode Levels
# ---------------------------------------------------------------------------

class TherapyLevel(IntEnum):
    SHADOW = 0
    SUGGEST_VALUES = 1
    SHIFT_BOUNDARIES = 2
    CREATE_MERGE_BLOCKS = 3
    CONTEXT_PROFILES = 4
    AUTONOMOUS_CURVES = 5


LEVEL_NAMES = {
    TherapyLevel.SHADOW: "Shadow",
    TherapyLevel.SUGGEST_VALUES: "Suggest Values",
    TherapyLevel.SHIFT_BOUNDARIES: "Shift Boundaries",
    TherapyLevel.CREATE_MERGE_BLOCKS: "Create/Merge Blocks",
    TherapyLevel.CONTEXT_PROFILES: "Context Profiles",
    TherapyLevel.AUTONOMOUS_CURVES: "Autonomous Curves",
}


# ---------------------------------------------------------------------------
# Graduation Criteria
# ---------------------------------------------------------------------------

@dataclass
class LevelGraduationCriteria:
    """Conditions that must all hold simultaneously for level transition."""
    min_days_at_level: int = 21
    positive_tir_delta: bool = True     # Sustained positive TIR improvement
    min_acceptance_rate: float = 0.5
    zero_safety_violations: bool = True
    mood_stable_or_improving: bool = True
    user_opt_in: bool = True            # In simulation: auto-approved if criteria met


# Default criteria per level transition (from level N to N+1)
DEFAULT_GRADUATION_CRITERIA: dict[TherapyLevel, LevelGraduationCriteria] = {
    TherapyLevel.SHADOW: LevelGraduationCriteria(
        min_days_at_level=21,
        min_acceptance_rate=0.0,  # No recommendations in shadow
    ),
    TherapyLevel.SUGGEST_VALUES: LevelGraduationCriteria(
        min_days_at_level=21,
        min_acceptance_rate=0.5,
    ),
    TherapyLevel.SHIFT_BOUNDARIES: LevelGraduationCriteria(
        min_days_at_level=21,
        min_acceptance_rate=0.5,
    ),
    TherapyLevel.CREATE_MERGE_BLOCKS: LevelGraduationCriteria(
        min_days_at_level=21,
        min_acceptance_rate=0.5,
    ),
    TherapyLevel.CONTEXT_PROFILES: LevelGraduationCriteria(
        min_days_at_level=21,
        min_acceptance_rate=0.5,
    ),
}


# ---------------------------------------------------------------------------
# Therapy Mode State
# ---------------------------------------------------------------------------

@dataclass
class TherapyModeState:
    """Per-patient therapy mode state machine."""
    current_level: TherapyLevel = TherapyLevel.SHADOW
    days_at_current_level: int = 0
    consecutive_criteria_met_days: int = 0
    level_history: list[tuple[int, int]] = field(default_factory=list)  # [(day, level), ...]

    # Tracking for graduation criteria
    rolling_tir_delta: float = 0.0
    rolling_acceptance_rate: float = 0.0
    safety_violations_at_level: int = 0
    mood_trend: float = 0.0  # Positive = improving

    def advance_day(self) -> None:
        """Called each simulated day."""
        self.days_at_current_level += 1

    def check_graduation(
        self,
        tir_delta: float,
        acceptance_rate: float,
        safety_violations: int,
        mood_trend: float,
        day_index: int,
    ) -> bool:
        """Check if all graduation criteria are met for current level.

        Returns True if the patient should be promoted to the next level.
        """
        if self.current_level >= TherapyLevel.AUTONOMOUS_CURVES:
            return False  # Already at max level

        criteria = DEFAULT_GRADUATION_CRITERIA.get(
            self.current_level,
            LevelGraduationCriteria(),
        )

        self.rolling_tir_delta = tir_delta
        self.rolling_acceptance_rate = acceptance_rate
        self.safety_violations_at_level = safety_violations
        self.mood_trend = mood_trend

        conditions_met = all([
            self.days_at_current_level >= criteria.min_days_at_level,
            tir_delta > 0 if criteria.positive_tir_delta else True,
            acceptance_rate >= criteria.min_acceptance_rate,
            safety_violations == 0 if criteria.zero_safety_violations else True,
            mood_trend >= -0.05 if criteria.mood_stable_or_improving else True,
        ])

        if conditions_met:
            self.consecutive_criteria_met_days += 1
        else:
            self.consecutive_criteria_met_days = 0

        # Need 7 consecutive days of all criteria met
        if self.consecutive_criteria_met_days >= 7:
            return True
        return False

    def promote(self, day_index: int) -> TherapyLevel:
        """Promote to the next therapy level."""
        if self.current_level < TherapyLevel.AUTONOMOUS_CURVES:
            self.level_history.append((day_index, int(self.current_level)))
            self.current_level = TherapyLevel(self.current_level + 1)
            self.days_at_current_level = 0
            self.consecutive_criteria_met_days = 0
            self.safety_violations_at_level = 0
        return self.current_level

    def demote(self, day_index: int) -> TherapyLevel:
        """Demote to the previous therapy level (regression)."""
        if self.current_level > TherapyLevel.SHADOW:
            self.level_history.append((day_index, int(self.current_level)))
            self.current_level = TherapyLevel(self.current_level - 1)
            self.days_at_current_level = 0
            self.consecutive_criteria_met_days = 0
            self.safety_violations_at_level = 0
        return self.current_level

    def check_regression(
        self,
        tir_delta: float,
        safety_violations: int,
        mood_trend: float,
    ) -> bool:
        """Check if performance has degraded enough to warrant demotion."""
        if self.current_level <= TherapyLevel.SHADOW:
            return False

        # Demote if TIR is consistently negative or safety violations
        if safety_violations > 0:
            return True
        if tir_delta < -0.03 and self.days_at_current_level > 7:
            return True
        if mood_trend < -0.15:
            return True
        return False


# ---------------------------------------------------------------------------
# Level-Specific Optimizer Constraints
# ---------------------------------------------------------------------------

@dataclass
class LevelConstraints:
    """Constraints applied to the optimizer based on therapy level."""
    can_change_values: bool = False
    can_shift_boundaries: bool = False
    can_create_merge_blocks: bool = False
    can_use_profiles: bool = False
    can_use_curves: bool = False
    max_change_magnitude: float = 0.0  # Maximum multiplier deviation


def get_level_constraints(level: TherapyLevel) -> LevelConstraints:
    """Return optimizer constraints for a given therapy level."""
    if level == TherapyLevel.SHADOW:
        return LevelConstraints()
    elif level == TherapyLevel.SUGGEST_VALUES:
        return LevelConstraints(
            can_change_values=True,
            max_change_magnitude=0.15,
        )
    elif level == TherapyLevel.SHIFT_BOUNDARIES:
        return LevelConstraints(
            can_change_values=True,
            can_shift_boundaries=True,
            max_change_magnitude=0.15,
        )
    elif level == TherapyLevel.CREATE_MERGE_BLOCKS:
        return LevelConstraints(
            can_change_values=True,
            can_shift_boundaries=True,
            can_create_merge_blocks=True,
            max_change_magnitude=0.20,
        )
    elif level == TherapyLevel.CONTEXT_PROFILES:
        return LevelConstraints(
            can_change_values=True,
            can_shift_boundaries=True,
            can_create_merge_blocks=True,
            can_use_profiles=True,
            max_change_magnitude=0.20,
        )
    elif level == TherapyLevel.AUTONOMOUS_CURVES:
        return LevelConstraints(
            can_change_values=True,
            can_shift_boundaries=True,
            can_create_merge_blocks=True,
            can_use_profiles=True,
            can_use_curves=True,
            max_change_magnitude=0.25,
        )
    return LevelConstraints()


# ---------------------------------------------------------------------------
# Personalization Weight (Hybrid Learning Mode)
# ---------------------------------------------------------------------------

def compute_personalization_weight(days_of_data: int) -> float:
    """Compute the individual vs community blend weight.

    Days 0-14:  0.0 (pure community)
    Days 14-60: 0.0 → 0.5 (linear ramp)
    Days 60+:   0.5 → 0.85 (slow approach, never reaches 1.0)
    """
    if days_of_data <= 14:
        return 0.0
    elif days_of_data <= 60:
        # Linear ramp from 0 to 0.5
        return 0.5 * (days_of_data - 14) / (60 - 14)
    else:
        # Asymptotic approach toward 0.85
        excess = days_of_data - 60
        return 0.5 + 0.35 * (1.0 - np.exp(-excess / 90.0))

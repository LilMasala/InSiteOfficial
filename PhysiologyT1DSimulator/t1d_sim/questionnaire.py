"""Questionnaire-derived priors for personalized twin initialization."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from math import sqrt
from typing import Optional

import numpy as np

from t1d_sim.agency import UserAgencyProfile, sample_agency
from t1d_sim.constants import PERSONAS
from t1d_sim.feedback import EventSchedule
from t1d_sim.missingness import make_missingness_profile
from t1d_sim.population import PatientConfig, apply_cross_parameter_interactions


@dataclass
class ColdStartTargets:
    """Self-reported glycemic summary metrics for cold-start twin calibration.

    All fields optional — calibration is graceful when some are missing.
    Values should reflect the user's recent experience (last 14–30 days).
    """
    recent_tir:      float | None = None   # fraction in [0, 1]
    recent_pct_low:  float | None = None   # fraction in [0, 1]
    recent_pct_high: float | None = None   # fraction in [0, 1]
    window_days:     int          = 14     # reporting window in days

    def to_chamelia_targets(self) -> dict[str, float]:
        """Return dict suitable for Chamelia calibration_targets preference field."""
        out = {}
        if self.recent_tir is not None:
            out["recent_tir"] = float(np.clip(self.recent_tir, 0.0, 1.0))
        if self.recent_pct_low is not None:
            out["recent_pct_low"] = float(np.clip(self.recent_pct_low, 0.0, 1.0))
        if self.recent_pct_high is not None:
            out["recent_pct_high"] = float(np.clip(self.recent_pct_high, 0.0, 1.0))
        return out

    @classmethod
    def from_patient_config(cls, cfg: "PatientConfig", rng: np.random.Generator | None = None) -> "ColdStartTargets":
        """Synthesize plausible self-reported targets from a known patient config.

        Used in simulator experiments to simulate what a user would report.
        Adds realistic noise to the ground-truth TIR estimate.
        """
        if rng is None:
            rng = np.random.default_rng(cfg.seed + 77777)

        # Linearized T1D heuristic (mirrors InSiteDomainAdapter.calibrate_posterior!)
        isf = getattr(cfg, "isf_multiplier", 1.0)
        basal = getattr(cfg, "basal_multiplier", 1.0)
        tir_true = float(np.clip(0.50 + 0.35*(isf - 1.0) + 0.15*(basal - 1.0), 0.05, 0.98))
        pct_low_true = float(np.clip(0.08 - 0.12*(isf - 1.0) - 0.04*(basal - 1.0), 0.0, 0.40))
        pct_high_true = float(np.clip(1.0 - tir_true - pct_low_true, 0.0, 0.95))

        # Add realistic self-report noise (~3-5 percentage points)
        noise_scale = 0.04
        tir      = float(np.clip(tir_true      + rng.normal(0, noise_scale), 0.0, 1.0))
        pct_low  = float(np.clip(pct_low_true  + rng.normal(0, noise_scale * 0.5), 0.0, 0.5))
        pct_high = float(np.clip(pct_high_true + rng.normal(0, noise_scale), 0.0, 1.0))

        return cls(
            recent_tir=round(tir, 3),
            recent_pct_low=round(pct_low, 3),
            recent_pct_high=round(pct_high, 3),
            window_days=14,
        )


class ExerciseFreq(Enum):
    NEVER = "never"
    LIGHT_WEEK = "1_2x_week"
    MOD_WEEK = "3_5x_week"
    DAILY = "daily"


class ExerciseType(Enum):
    CARDIO = "cardio"
    STRENGTH = "strength"
    MIXED = "mixed"
    LIGHT = "light"
    NONE = "none"


class ExerciseIntensity(Enum):
    CASUAL = "casual"
    MODERATE = "moderate"
    HARD = "hard"
    INTENSE = "intense"


class FitnessLevel(Enum):
    LOW = "low"
    AVERAGE = "average"
    FIT = "fit"
    VERY_FIT = "very_fit"


class BedtimeCategory(Enum):
    EARLY = "before_10pm"
    NORMAL = "10pm_midnight"
    LATE = "midnight_2am"
    VERY_LATE = "after_2am"


class SleepHours(Enum):
    UNDER_6 = "under_6"
    SIX_SEVEN = "6_7"
    SEVEN_EIGHT = "7_8"
    OVER_8 = "over_8"


class SleepConsistency(Enum):
    VERY = "very_consistent"
    FAIRLY = "fairly_consistent"
    VARIABLE = "pretty_variable"
    IRREGULAR = "completely_irregular"


class RestedFeeling(Enum):
    USUALLY = "usually"
    SOMETIMES = "sometimes"
    RARELY = "rarely"


class FirstMealTime(Enum):
    EARLY = "before_7am"
    NORMAL = "7am_9am"
    LATE = "9am_11am"
    SKIP = "after_11am_or_skip"


class BreakfastSkip(Enum):
    NEVER = "almost_never"
    SOMETIMES = "sometimes"
    OFTEN = "often"
    ALWAYS = "almost_always"


class LunchSkip(Enum):
    NEVER = "almost_never"
    SOMETIMES = "sometimes"
    OFTEN = "often"
    ALWAYS = "almost_always"


class MealFrequency(Enum):
    ONE_TWO = "1_2"
    THREE = "3"
    FOUR_FIVE = "4_5"
    SIX_PLUS = "6_plus"


class MealConsistency(Enum):
    VERY = "very_consistent"
    FAIRLY = "fairly_consistent"
    VARIABLE = "pretty_variable"
    CHAOTIC = "very_irregular"


class PortionSize(Enum):
    SMALL = "small"
    AVERAGE = "average"
    GENEROUS = "generous"
    LARGE = "large"


class DietType(Enum):
    LOW_CARB = "low_carb"
    MODERATE = "moderate"
    HIGH_CARB = "high_carb"
    VERY_VARIABLE = "very_variable"


class LastMealTime(Enum):
    EARLY = "before_7pm"
    NORMAL = "7pm_9pm"
    LATE = "after_9pm"
    VERY_LATE = "midnight_snacks"


class StressLevel(Enum):
    RARELY = "rarely"
    SOMETIMES = "sometimes"
    OFTEN = "often"
    ALWAYS = "almost_always"


class StressBgEffect(Enum):
    NOTICEABLY = "yes_noticeably"
    LITTLE = "a_little"
    NOT_REALLY = "not_really"
    UNSURE = "not_sure"


class MoodVariability(Enum):
    STABLE = "very_stable"
    SOME = "some_variation"
    VARIABLE = "quite_variable"
    VERY = "very_variable"


class ScheduleType(Enum):
    REGULAR = "regular_9_5"
    SHIFT = "shift_work"
    VARIABLE = "very_variable"


class CyclePresence(Enum):
    REGULAR = "yes_regular"
    IRREGULAR = "yes_irregular"
    NO = "no"


class CycleBgEffect(Enum):
    NOTICEABLY = "yes_noticeably"
    LITTLE = "a_little"
    NOT_REALLY = "not_really"
    UNSURE = "not_sure"


class CycleHunger(Enum):
    NOTICEABLY = "yes_noticeably"
    LITTLE = "a_little"
    NOT_REALLY = "not_really"


class CycleMood(Enum):
    NOTICEABLY = "yes_noticeably"
    LITTLE = "a_little"
    NOT_REALLY = "not_really"


class InsulinSensitivity(Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"
    UNSURE = "not_sure"


class CarbSpike(Enum):
    LOT = "yes_a_lot"
    AVERAGE = "average"
    NOT_MUCH = "not_much"
    UNSURE = "not_sure"


class Aggressiveness(Enum):
    VERY_CAUTIOUS = "very_cautious"
    MODERATE = "moderate"
    WILLING = "willing"
    VERY_WILLING = "very_willing"


class ComplianceLevel(Enum):
    EXACT = "exactly"
    CLOSE = "pretty_closely"
    ROUGH = "roughly"
    FORGET = "sometimes_forget"


class CheckFrequency(Enum):
    EVERY_DAY = "every_day"
    MOST_DAYS = "most_days"
    SOMETIMES = "whenever_i_remember"
    RARELY = "probably_rarely"


class TrustLevel(Enum):
    SKEPTICAL = "very_skeptical"
    CAUTIOUS = "cautiously_open"
    TRUSTING = "fairly_trusting"
    VERY_TRUSTING = "very_trusting"


@dataclass
class QuestionnaireAnswers:
    bedtime_category: Optional[BedtimeCategory] = None
    sleep_hours: Optional[SleepHours] = None
    sleep_consistency: Optional[SleepConsistency] = None
    rested_feeling: Optional[RestedFeeling] = None
    exercise_freq: Optional[ExerciseFreq] = None
    exercise_type: Optional[ExerciseType] = None
    exercise_intensity: Optional[ExerciseIntensity] = None
    fitness_level: Optional[FitnessLevel] = None
    first_meal_time: Optional[FirstMealTime] = None
    breakfast_skip: Optional[BreakfastSkip] = None
    lunch_skip: Optional[LunchSkip] = None
    meal_frequency: Optional[MealFrequency] = None
    meal_consistency: Optional[MealConsistency] = None
    portion_size: Optional[PortionSize] = None
    diet_type: Optional[DietType] = None
    last_meal_time: Optional[LastMealTime] = None
    stress_level: Optional[StressLevel] = None
    stress_bg_effect: Optional[StressBgEffect] = None
    mood_variability: Optional[MoodVariability] = None
    schedule_type: Optional[ScheduleType] = None
    cycle_presence: Optional[CyclePresence] = None
    cycle_bg_effect: Optional[CycleBgEffect] = None
    cycle_hunger: Optional[CycleHunger] = None
    cycle_mood: Optional[CycleMood] = None
    insulin_sensitivity: Optional[InsulinSensitivity] = None
    carb_spike: Optional[CarbSpike] = None
    aggressiveness: Optional[Aggressiveness] = None
    compliance_level: Optional[ComplianceLevel] = None
    check_frequency: Optional[CheckFrequency] = None
    trust_level: Optional[TrustLevel] = None


@dataclass(frozen=True)
class NumericFieldSpec:
    persona_key: str
    fallback: tuple[float, float]
    bounds: tuple[float, float]


_FIELD_SPECS: dict[str, NumericFieldSpec] = {
    "activity_propensity": NumericFieldSpec("activity_propensity", (0.5, 0.2), (0.0, 1.0)),
    "sleep_regularity": NumericFieldSpec("sleep_regularity", (0.5, 0.2), (0.0, 1.0)),
    "sleep_total_min_mean": NumericFieldSpec("sleep_total_min_mean", (400.0, 65.0), (240.0, 570.0)),
    "sleep_efficiency": NumericFieldSpec("sleep_efficiency", (0.82, 0.10), (0.55, 0.97)),
    "sleep_schedule_offset_h": NumericFieldSpec("sleep_schedule_offset_h", (0.0, 0.3), (-4.0, 6.0)),
    "stress_reactivity": NumericFieldSpec("stress_reactivity", (0.5, 0.2), (0.0, 1.0)),
    "stress_baseline": NumericFieldSpec("stress_baseline", (0.15, 0.10), (0.0, 0.8)),
    "cycle_sensitivity": NumericFieldSpec("cycle_sensitivity", (0.0, 0.0), (0.0, 1.0)),
    "mood_stability": NumericFieldSpec("mood_stability", (0.5, 0.2), (0.0, 1.0)),
    "meal_regularity": NumericFieldSpec("meal_regularity", (0.5, 0.2), (0.0, 1.0)),
    "meal_schedule_offset_h": NumericFieldSpec("meal_schedule_offset_h", (0.0, 0.2), (-3.0, 5.0)),
    "meal_size_multiplier": NumericFieldSpec("meal_size_multiplier", (1.0, 0.12), (0.4, 2.0)),
    "n_meals_per_day_mean": NumericFieldSpec("n_meals_per_day", (3.0, 0.5), (1.0, 7.0)),
    "skips_breakfast_p": NumericFieldSpec("skips_breakfast_p", (0.15, 0.10), (0.0, 1.0)),
    "skips_lunch_p": NumericFieldSpec("skips_lunch_p", (0.08, 0.07), (0.0, 1.0)),
    "luteal_meal_size_boost": NumericFieldSpec("luteal_meal_size_boost", (0.08, 0.03), (0.0, 0.4)),
    "luteal_mood_drop": NumericFieldSpec("luteal_mood_drop", (0.15, 0.05), (0.0, 0.5)),
    "exercise_intensity_mean": NumericFieldSpec("exercise_intensity_mean", (0.5, 0.15), (0.1, 1.0)),
    "logging_quality_raw": NumericFieldSpec("logging_quality", (0.75, 0.08), (0.0, 1.0)),
    "fitness_level": NumericFieldSpec("fitness_level", (0.5, 0.2), (0.0, 1.0)),
    "base_rhr": NumericFieldSpec("base_rhr", (63.0, 6.0), (45.0, 85.0)),
    "isf_multiplier": NumericFieldSpec("isf_base_multiplier", (1.0, 0.12), (0.70, 1.35)),
    "cr_multiplier": NumericFieldSpec("cr_base_multiplier", (1.0, 0.12), (0.70, 1.35)),
    "basal_multiplier": NumericFieldSpec("basal_multiplier", (1.0, 0.10), (0.75, 1.25)),
}

_DEFAULT_LOGGING_QUALITY = (0.75, 0.08)
_MIN_STD = 0.01


def _clip_mean(field: str, mean: float) -> float:
    lo, hi = _FIELD_SPECS[field].bounds
    return float(np.clip(mean, lo, hi))


def _normalize_prior(field: str, mean: float, std: float) -> tuple[float, float]:
    mean = _clip_mean(field, mean)
    std = max(float(std), 0.0)
    return mean, std


def _set(priors: dict[str, tuple[float, float]], field: str, mean: float, std: float | None = None) -> None:
    _, current_std = priors[field]
    priors[field] = _normalize_prior(field, mean, current_std if std is None else std)


def _adjust_mean(priors: dict[str, tuple[float, float]], field: str, delta: float) -> None:
    mean, std = priors[field]
    priors[field] = _normalize_prior(field, mean + delta, std)


def _adjust_std(priors: dict[str, tuple[float, float]], field: str, delta: float, minimum: float = 0.0) -> None:
    mean, std = priors[field]
    priors[field] = _normalize_prior(field, mean, max(std + delta, minimum))


def _bounded_normal(rng: np.random.Generator, mean: float, std: float, bounds: tuple[float, float]) -> float:
    value = float(rng.normal(mean, std))
    return float(np.clip(value, bounds[0], bounds[1]))


@lru_cache(maxsize=1)
def population_defaults() -> dict[str, tuple[float, float]]:
    defaults: dict[str, tuple[float, float]] = {}
    total_weight = sum(float(persona.get("weight", 1.0)) for persona in PERSONAS.values())

    for field, spec in _FIELD_SPECS.items():
        weighted_mean = 0.0
        second_moment = 0.0
        for persona in PERSONAS.values():
            weight = float(persona.get("weight", 1.0)) / total_weight
            raw = persona.get(spec.persona_key, spec.fallback)
            if isinstance(raw, tuple):
                mean, std = float(raw[0]), float(raw[1])
            else:
                mean, std = float(raw), spec.fallback[1]
            weighted_mean += weight * mean
            second_moment += weight * (std * std + mean * mean)

        variance = max(second_moment - weighted_mean * weighted_mean, 0.0)
        defaults[field] = _normalize_prior(field, weighted_mean, sqrt(variance))

    # These are runtime onboarding assumptions, not persona-derived truths.
    defaults["logging_quality_raw"] = _DEFAULT_LOGGING_QUALITY
    defaults["basal_multiplier"] = _FIELD_SPECS["basal_multiplier"].fallback
    return defaults


def questionnaire_to_patientconfig_priors(
    answers: QuestionnaireAnswers,
) -> dict[str, tuple[float, float]]:
    priors = dict(population_defaults())

    if answers.bedtime_category == BedtimeCategory.EARLY:
        _set(priors, "sleep_schedule_offset_h", -1.5, 0.25)
    elif answers.bedtime_category == BedtimeCategory.NORMAL:
        _set(priors, "sleep_schedule_offset_h", 0.0, 0.25)
    elif answers.bedtime_category == BedtimeCategory.LATE:
        _set(priors, "sleep_schedule_offset_h", 1.5, 0.35)
    elif answers.bedtime_category == BedtimeCategory.VERY_LATE:
        _set(priors, "sleep_schedule_offset_h", 3.0, 0.45)

    if answers.sleep_hours == SleepHours.UNDER_6:
        _set(priors, "sleep_total_min_mean", 330.0, 25.0)
        _set(priors, "sleep_efficiency", 0.76, 0.08)
    elif answers.sleep_hours == SleepHours.SIX_SEVEN:
        _set(priors, "sleep_total_min_mean", 390.0, 25.0)
        _set(priors, "sleep_efficiency", 0.81, 0.07)
    elif answers.sleep_hours == SleepHours.SEVEN_EIGHT:
        _set(priors, "sleep_total_min_mean", 450.0, 25.0)
        _set(priors, "sleep_efficiency", 0.85, 0.06)
    elif answers.sleep_hours == SleepHours.OVER_8:
        _set(priors, "sleep_total_min_mean", 510.0, 30.0)
        _set(priors, "sleep_efficiency", 0.87, 0.06)

    if answers.sleep_consistency == SleepConsistency.VERY:
        _set(priors, "sleep_regularity", 0.90, 0.05)
    elif answers.sleep_consistency == SleepConsistency.FAIRLY:
        _set(priors, "sleep_regularity", 0.70, 0.08)
    elif answers.sleep_consistency == SleepConsistency.VARIABLE:
        _set(priors, "sleep_regularity", 0.45, 0.10)
    elif answers.sleep_consistency == SleepConsistency.IRREGULAR:
        _set(priors, "sleep_regularity", 0.18, 0.10)

    if answers.rested_feeling == RestedFeeling.USUALLY:
        _adjust_mean(priors, "sleep_efficiency", 0.04)
    elif answers.rested_feeling == RestedFeeling.RARELY:
        _adjust_mean(priors, "sleep_efficiency", -0.05)

    if answers.exercise_freq == ExerciseFreq.NEVER:
        _set(priors, "activity_propensity", 0.08, 0.06)
    elif answers.exercise_freq == ExerciseFreq.LIGHT_WEEK:
        _set(priors, "activity_propensity", 0.32, 0.10)
    elif answers.exercise_freq == ExerciseFreq.MOD_WEEK:
        _set(priors, "activity_propensity", 0.65, 0.10)
    elif answers.exercise_freq == ExerciseFreq.DAILY:
        _set(priors, "activity_propensity", 0.88, 0.07)

    activity_mean = priors["activity_propensity"][0]
    if answers.exercise_type == ExerciseType.CARDIO:
        _set(priors, "exercise_intensity_mean", 0.72, 0.10)
        _set(priors, "fitness_level", activity_mean * 0.95, 0.10)
        _set(priors, "base_rhr", 51.0, 6.0)
        _set(priors, "isf_multiplier", 1.16, 0.10)
    elif answers.exercise_type == ExerciseType.STRENGTH:
        _set(priors, "exercise_intensity_mean", 0.68, 0.10)
        _set(priors, "fitness_level", activity_mean * 0.88, 0.10)
        _set(priors, "base_rhr", 57.0, 7.0)
        _set(priors, "isf_multiplier", 1.08, 0.10)
    elif answers.exercise_type == ExerciseType.MIXED:
        _set(priors, "exercise_intensity_mean", 0.64, 0.10)
        _set(priors, "fitness_level", activity_mean * 0.90, 0.10)
        _set(priors, "base_rhr", 58.0, 7.0)
        _set(priors, "isf_multiplier", 1.11, 0.10)
    elif answers.exercise_type == ExerciseType.LIGHT:
        _set(priors, "exercise_intensity_mean", 0.32, 0.10)
        _set(priors, "fitness_level", activity_mean * 0.70, 0.10)
        _set(priors, "base_rhr", 64.0, 8.0)
        _set(priors, "isf_multiplier", 1.02, 0.10)

    if answers.exercise_intensity == ExerciseIntensity.CASUAL:
        _adjust_mean(priors, "exercise_intensity_mean", -0.10)
    elif answers.exercise_intensity == ExerciseIntensity.HARD:
        _adjust_mean(priors, "exercise_intensity_mean", 0.08)
    elif answers.exercise_intensity == ExerciseIntensity.INTENSE:
        _adjust_mean(priors, "exercise_intensity_mean", 0.15)
        _adjust_mean(priors, "base_rhr", -3.0)

    if answers.fitness_level == FitnessLevel.LOW:
        _adjust_mean(priors, "fitness_level", -0.10)
    elif answers.fitness_level == FitnessLevel.FIT:
        _adjust_mean(priors, "fitness_level", 0.08)
    elif answers.fitness_level == FitnessLevel.VERY_FIT:
        _adjust_mean(priors, "fitness_level", 0.15)
        _adjust_mean(priors, "base_rhr", -4.0)

    if answers.first_meal_time == FirstMealTime.EARLY:
        _set(priors, "meal_schedule_offset_h", -1.5, 0.25)
    elif answers.first_meal_time == FirstMealTime.NORMAL:
        _set(priors, "meal_schedule_offset_h", 0.0, 0.25)
    elif answers.first_meal_time == FirstMealTime.LATE:
        _set(priors, "meal_schedule_offset_h", 1.5, 0.30)
    elif answers.first_meal_time == FirstMealTime.SKIP:
        _set(priors, "meal_schedule_offset_h", 2.5, 0.40)
        _set(priors, "skips_breakfast_p", 0.75, 0.15)

    if answers.breakfast_skip == BreakfastSkip.NEVER:
        _set(priors, "skips_breakfast_p", 0.05, 0.04)
    elif answers.breakfast_skip == BreakfastSkip.SOMETIMES:
        _set(priors, "skips_breakfast_p", 0.25, 0.10)
    elif answers.breakfast_skip == BreakfastSkip.OFTEN:
        _set(priors, "skips_breakfast_p", 0.55, 0.12)
    elif answers.breakfast_skip == BreakfastSkip.ALWAYS:
        _set(priors, "skips_breakfast_p", 0.85, 0.08)

    if answers.lunch_skip == LunchSkip.NEVER:
        _set(priors, "skips_lunch_p", 0.04, 0.03)
    elif answers.lunch_skip == LunchSkip.SOMETIMES:
        _set(priors, "skips_lunch_p", 0.20, 0.10)
    elif answers.lunch_skip == LunchSkip.OFTEN:
        _set(priors, "skips_lunch_p", 0.45, 0.12)
    elif answers.lunch_skip == LunchSkip.ALWAYS:
        _set(priors, "skips_lunch_p", 0.75, 0.10)

    if answers.meal_frequency == MealFrequency.ONE_TWO:
        _set(priors, "n_meals_per_day_mean", 1.8, 0.4)
    elif answers.meal_frequency == MealFrequency.THREE:
        _set(priors, "n_meals_per_day_mean", 3.0, 0.4)
    elif answers.meal_frequency == MealFrequency.FOUR_FIVE:
        _set(priors, "n_meals_per_day_mean", 4.5, 0.4)
    elif answers.meal_frequency == MealFrequency.SIX_PLUS:
        _set(priors, "n_meals_per_day_mean", 6.5, 0.6)

    if answers.meal_consistency == MealConsistency.VERY:
        _set(priors, "meal_regularity", 0.90, 0.05)
    elif answers.meal_consistency == MealConsistency.FAIRLY:
        _set(priors, "meal_regularity", 0.65, 0.09)
    elif answers.meal_consistency == MealConsistency.VARIABLE:
        _set(priors, "meal_regularity", 0.38, 0.10)
    elif answers.meal_consistency == MealConsistency.CHAOTIC:
        _set(priors, "meal_regularity", 0.15, 0.08)

    if answers.portion_size == PortionSize.SMALL:
        _set(priors, "meal_size_multiplier", 0.78, 0.10)
    elif answers.portion_size == PortionSize.AVERAGE:
        _set(priors, "meal_size_multiplier", 1.00, 0.10)
    elif answers.portion_size == PortionSize.GENEROUS:
        _set(priors, "meal_size_multiplier", 1.18, 0.10)
    elif answers.portion_size == PortionSize.LARGE:
        _set(priors, "meal_size_multiplier", 1.35, 0.12)

    if answers.diet_type == DietType.LOW_CARB:
        _set(priors, "cr_multiplier", 0.86, 0.09)
        _adjust_mean(priors, "meal_size_multiplier", -0.08)
    elif answers.diet_type == DietType.MODERATE:
        _set(priors, "cr_multiplier", 1.00, 0.10)
    elif answers.diet_type == DietType.HIGH_CARB:
        _set(priors, "cr_multiplier", 1.14, 0.10)
        _adjust_mean(priors, "meal_size_multiplier", 0.08)
    elif answers.diet_type == DietType.VERY_VARIABLE:
        _set(priors, "cr_multiplier", 1.00, 0.18)

    if answers.last_meal_time == LastMealTime.EARLY:
        _adjust_std(priors, "meal_schedule_offset_h", -0.10, minimum=0.05)
    elif answers.last_meal_time == LastMealTime.LATE:
        _adjust_std(priors, "meal_schedule_offset_h", 0.20)
    elif answers.last_meal_time == LastMealTime.VERY_LATE:
        _adjust_mean(priors, "meal_schedule_offset_h", 0.50)
        _adjust_std(priors, "meal_schedule_offset_h", 0.30)

    if answers.stress_level == StressLevel.RARELY:
        _set(priors, "stress_baseline", 0.07, 0.05)
        _set(priors, "stress_reactivity", 0.28, 0.08)
    elif answers.stress_level == StressLevel.SOMETIMES:
        _set(priors, "stress_baseline", 0.22, 0.08)
        _set(priors, "stress_reactivity", 0.48, 0.10)
    elif answers.stress_level == StressLevel.OFTEN:
        _set(priors, "stress_baseline", 0.48, 0.10)
        _set(priors, "stress_reactivity", 0.68, 0.10)
    elif answers.stress_level == StressLevel.ALWAYS:
        _set(priors, "stress_baseline", 0.72, 0.09)
        _set(priors, "stress_reactivity", 0.84, 0.08)

    if answers.stress_bg_effect == StressBgEffect.NOTICEABLY:
        _adjust_mean(priors, "stress_reactivity", 0.15)
    elif answers.stress_bg_effect == StressBgEffect.LITTLE:
        _adjust_mean(priors, "stress_reactivity", 0.05)
    elif answers.stress_bg_effect == StressBgEffect.NOT_REALLY:
        _adjust_mean(priors, "stress_reactivity", -0.10)

    if answers.mood_variability == MoodVariability.STABLE:
        _set(priors, "mood_stability", 0.88, 0.06)
    elif answers.mood_variability == MoodVariability.SOME:
        _set(priors, "mood_stability", 0.68, 0.09)
    elif answers.mood_variability == MoodVariability.VARIABLE:
        _set(priors, "mood_stability", 0.42, 0.10)
    elif answers.mood_variability == MoodVariability.VERY:
        _set(priors, "mood_stability", 0.22, 0.10)

    if answers.schedule_type == ScheduleType.SHIFT:
        _adjust_std(priors, "sleep_schedule_offset_h", 0.50)
        _adjust_mean(priors, "stress_baseline", 0.08)
        _adjust_mean(priors, "sleep_regularity", -0.12)
    elif answers.schedule_type == ScheduleType.VARIABLE:
        _adjust_std(priors, "sleep_regularity", 0.08)
        _adjust_std(priors, "meal_regularity", 0.08)

    if answers.cycle_presence == CyclePresence.REGULAR:
        _set(priors, "cycle_sensitivity", 0.50, 0.14)
    elif answers.cycle_presence == CyclePresence.IRREGULAR:
        _set(priors, "cycle_sensitivity", 0.40, 0.18)
    elif answers.cycle_presence == CyclePresence.NO:
        _set(priors, "cycle_sensitivity", 0.0, 0.0)
        _set(priors, "luteal_meal_size_boost", 0.0, 0.0)
        _set(priors, "luteal_mood_drop", 0.0, 0.0)

    if answers.cycle_bg_effect == CycleBgEffect.NOTICEABLY:
        _set(priors, "cycle_sensitivity", priors["cycle_sensitivity"][0] + 0.28, 0.09)
    elif answers.cycle_bg_effect == CycleBgEffect.LITTLE:
        _adjust_mean(priors, "cycle_sensitivity", 0.08)
    elif answers.cycle_bg_effect == CycleBgEffect.NOT_REALLY:
        _set(priors, "cycle_sensitivity", priors["cycle_sensitivity"][0] - 0.18, 0.09)

    if answers.cycle_hunger == CycleHunger.NOTICEABLY:
        _set(priors, "luteal_meal_size_boost", 0.14, 0.04)
    elif answers.cycle_hunger == CycleHunger.LITTLE:
        _set(priors, "luteal_meal_size_boost", 0.07, 0.03)
    elif answers.cycle_hunger == CycleHunger.NOT_REALLY:
        _set(priors, "luteal_meal_size_boost", 0.02, 0.02)

    if answers.cycle_mood == CycleMood.NOTICEABLY:
        _set(priors, "luteal_mood_drop", 0.22, 0.05)
    elif answers.cycle_mood == CycleMood.LITTLE:
        _set(priors, "luteal_mood_drop", 0.10, 0.04)
    elif answers.cycle_mood == CycleMood.NOT_REALLY:
        _set(priors, "luteal_mood_drop", 0.03, 0.02)

    if answers.insulin_sensitivity == InsulinSensitivity.HIGH:
        if answers.exercise_type is None:
            _set(priors, "isf_multiplier", 1.18, 0.10)
        else:
            _adjust_mean(priors, "isf_multiplier", 0.12)
    elif answers.insulin_sensitivity == InsulinSensitivity.LOW:
        _adjust_mean(priors, "isf_multiplier", -0.12)

    if answers.carb_spike == CarbSpike.LOT:
        _adjust_mean(priors, "cr_multiplier", 0.10)
    elif answers.carb_spike == CarbSpike.NOT_MUCH:
        _adjust_mean(priors, "cr_multiplier", -0.10)

    if priors["activity_propensity"][0] > 0.60 and priors["sleep_regularity"][0] < 0.50:
        _adjust_mean(priors, "stress_reactivity", 0.10)

    if (
        priors["sleep_schedule_offset_h"][0] > 1.5
        and answers.schedule_type == ScheduleType.SHIFT
    ):
        _adjust_mean(priors, "sleep_regularity", -0.08)
        _adjust_std(priors, "sleep_regularity", 0.06)

    if answers.diet_type == DietType.VERY_VARIABLE:
        mean, std = priors["cr_multiplier"]
        priors["cr_multiplier"] = _normalize_prior("cr_multiplier", mean, max(std, 0.16))

    if (
        priors["activity_propensity"][0] > 0.70
        and answers.exercise_type == ExerciseType.CARDIO
        and priors["sleep_regularity"][0] > 0.70
    ):
        _adjust_mean(priors, "isf_multiplier", 0.05)

    if (
        answers.diet_type == DietType.HIGH_CARB
        and (
            priors["skips_breakfast_p"][0] > 0.40
            or priors["skips_lunch_p"][0] > 0.40
        )
    ):
        _adjust_std(priors, "meal_size_multiplier", 0.06)

    return priors


def questionnaire_to_agency_priors(
    answers: QuestionnaireAnswers,
) -> dict[str, tuple[float, float]]:
    physical = questionnaire_to_patientconfig_priors(answers)
    agency = {
        "aggressiveness": (0.50, 0.15),
        "compliance_noise": (0.18, 0.08),
        "engagement_decay": (0.08, 0.04),
        "initial_trust": (0.40, 0.10),
    }

    if answers.aggressiveness == Aggressiveness.VERY_CAUTIOUS:
        agency["aggressiveness"] = (0.20, 0.05)
    elif answers.aggressiveness == Aggressiveness.MODERATE:
        agency["aggressiveness"] = (0.50, 0.08)
    elif answers.aggressiveness == Aggressiveness.WILLING:
        agency["aggressiveness"] = (0.70, 0.08)
    elif answers.aggressiveness == Aggressiveness.VERY_WILLING:
        agency["aggressiveness"] = (0.85, 0.06)

    if answers.compliance_level == ComplianceLevel.EXACT:
        agency["compliance_noise"] = (0.05, 0.02)
    elif answers.compliance_level == ComplianceLevel.CLOSE:
        agency["compliance_noise"] = (0.15, 0.05)
    elif answers.compliance_level == ComplianceLevel.ROUGH:
        agency["compliance_noise"] = (0.28, 0.08)
    elif answers.compliance_level == ComplianceLevel.FORGET:
        agency["compliance_noise"] = (0.42, 0.10)

    if answers.check_frequency == CheckFrequency.EVERY_DAY:
        agency["engagement_decay"] = (0.02, 0.01)
    elif answers.check_frequency == CheckFrequency.MOST_DAYS:
        agency["engagement_decay"] = (0.06, 0.02)
    elif answers.check_frequency == CheckFrequency.SOMETIMES:
        agency["engagement_decay"] = (0.14, 0.04)
    elif answers.check_frequency == CheckFrequency.RARELY:
        agency["engagement_decay"] = (0.25, 0.06)

    if answers.trust_level == TrustLevel.SKEPTICAL:
        agency["initial_trust"] = (0.15, 0.05)
    elif answers.trust_level == TrustLevel.CAUTIOUS:
        agency["initial_trust"] = (0.35, 0.08)
    elif answers.trust_level == TrustLevel.TRUSTING:
        agency["initial_trust"] = (0.60, 0.08)
    elif answers.trust_level == TrustLevel.VERY_TRUSTING:
        agency["initial_trust"] = (0.80, 0.07)

    if physical["stress_baseline"][0] > 0.45 and physical["mood_stability"][0] < 0.40:
        mean, std = agency["initial_trust"]
        agency["initial_trust"] = (max(mean - 0.08, 0.10), std)

    agg_mean = agency["aggressiveness"][0]
    trust_mean = agency["initial_trust"][0]
    mood_mean = physical["mood_stability"][0]
    agency["revert_threshold"] = (
        float(np.clip(0.05 + 0.10 * (1.0 - agg_mean), 0.02, 0.30)),
        0.02,
    )
    agency["trust_growth_rate"] = (
        float(np.clip(0.01 + 0.02 * trust_mean + 0.01 * mood_mean, 0.001, 0.10)),
        0.008,
    )

    return agency


def sample_twins_from_priors(
    priors: dict[str, tuple[float, float]],
    n: int = 50,
    seed: int = 42,
    is_female: bool = True,
) -> list[PatientConfig]:
    base = dict(population_defaults())
    base.update(priors)
    base["logging_quality_raw"] = priors.get("logging_quality_raw", _DEFAULT_LOGGING_QUALITY)
    base["basal_multiplier"] = priors.get("basal_multiplier", _FIELD_SPECS["basal_multiplier"].fallback)

    if not is_female:
        base["cycle_sensitivity"] = (0.0, 0.0)
        base["luteal_meal_size_boost"] = (0.0, 0.0)
        base["luteal_mood_drop"] = (0.0, 0.0)

    rng = np.random.default_rng(seed)
    twins: list[PatientConfig] = []

    for i in range(n):
        sampled: dict[str, float] = {}
        for field, spec in _FIELD_SPECS.items():
            mean, std = base[field]
            sampled[field] = _bounded_normal(rng, mean, std, spec.bounds)

        cfg = PatientConfig(
            patient_id=f"real_twin_{i:03d}",
            persona="questionnaire_derived",
            is_female=is_female,
            activity_propensity=sampled["activity_propensity"],
            sleep_regularity=sampled["sleep_regularity"],
            sleep_total_min_mean=sampled["sleep_total_min_mean"],
            sleep_efficiency=sampled["sleep_efficiency"],
            sleep_schedule_offset_h=sampled["sleep_schedule_offset_h"],
            stress_reactivity=sampled["stress_reactivity"],
            stress_baseline=sampled["stress_baseline"],
            cycle_sensitivity=sampled["cycle_sensitivity"] if is_female else 0.0,
            mood_stability=sampled["mood_stability"],
            meal_regularity=sampled["meal_regularity"],
            meal_schedule_offset_h=sampled["meal_schedule_offset_h"],
            meal_size_multiplier=sampled["meal_size_multiplier"],
            n_meals_per_day_mean=sampled["n_meals_per_day_mean"],
            skips_breakfast_p=sampled["skips_breakfast_p"],
            skips_lunch_p=sampled["skips_lunch_p"],
            luteal_meal_size_boost=sampled["luteal_meal_size_boost"] if is_female else 0.0,
            luteal_mood_drop=sampled["luteal_mood_drop"] if is_female else 0.0,
            exercise_intensity_mean=sampled["exercise_intensity_mean"],
            logging_quality_raw=sampled["logging_quality_raw"],
            fitness_level=sampled["fitness_level"],
            base_rhr=sampled["base_rhr"],
            isf_multiplier=sampled["isf_multiplier"],
            cr_multiplier=sampled["cr_multiplier"],
            basal_multiplier=sampled["basal_multiplier"],
            base_patient_name="adult#001",
            seed=int(rng.integers(0, 1_000_000)),
            n_days=180,
            split="real",
        )

        cfg.uses_aid = False
        miss_rng = np.random.default_rng(int(rng.integers(0, 1_000_000)))
        cfg.missingness_profile = make_missingness_profile(cfg.logging_quality_raw, miss_rng)
        cfg = apply_cross_parameter_interactions(cfg)
        agency_rng = np.random.default_rng(int(rng.integers(0, 1_000_000)))
        cfg.agency_profile = sample_agency(cfg, agency_rng)
        cfg.event_schedule = EventSchedule()
        twins.append(cfg)

    return twins


def physical_priors_from_twins(
    twins: list[PatientConfig],
) -> dict[str, tuple[float, float]]:
    if not twins:
        return {}

    fields = (
        "isf_multiplier",
        "cr_multiplier",
        "basal_multiplier",
        "base_rhr",
        "activity_propensity",
        "sleep_regularity",
        "sleep_total_min_mean",
        "stress_reactivity",
        "stress_baseline",
        "cycle_sensitivity",
        "meal_regularity",
        "exercise_intensity_mean",
        "fitness_level",
    )

    result: dict[str, tuple[float, float]] = {}
    for field in fields:
        values = np.array([float(getattr(twin, field)) for twin in twins], dtype=float)
        result[field] = (float(values.mean()), max(float(values.std(ddof=0)), 0.02))
    return result


def agency_profile_from_priors(
    priors: dict[str, tuple[float, float]],
    rng: np.random.Generator,
) -> UserAgencyProfile:
    def sample(key: str, lo: float, hi: float) -> float:
        mean, std = priors[key]
        return float(np.clip(rng.normal(mean, max(std, _MIN_STD)), lo, hi))

    return UserAgencyProfile(
        aggressiveness=sample("aggressiveness", 0.0, 1.0),
        initial_trust=sample("initial_trust", 0.0, 1.0),
        trust_growth_rate=sample("trust_growth_rate", 0.001, 0.10),
        compliance_noise=sample("compliance_noise", 0.0, 0.5),
        revert_threshold=sample("revert_threshold", 0.02, 0.30),
        engagement_decay=sample("engagement_decay", 0.001, 0.30),
    )

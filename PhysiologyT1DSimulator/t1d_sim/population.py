"""Population sampling."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import numpy as np

from t1d_sim.agency import UserAgencyProfile, sample_agency
from t1d_sim.constants import PERSONAS, PERSONA_SITE_INTERVALS
from t1d_sim.feedback import EventSchedule, sample_life_events
from t1d_sim.missingness import MissingnessProfile, make_missingness_profile
from t1d_sim.therapy import TherapySchedule, make_default_schedule

try:
    from chamelia.personality import UserPersonality, sample_personality
except ImportError:
    UserPersonality = None  # type: ignore[assignment,misc]
    sample_personality = None  # type: ignore[assignment]


@dataclass
class PatientConfig:
    patient_id: str
    persona: str
    is_female: bool
    activity_propensity: float
    sleep_regularity: float
    sleep_total_min_mean: float
    sleep_efficiency: float
    sleep_schedule_offset_h: float
    stress_reactivity: float
    stress_baseline: float
    cycle_sensitivity: float
    mood_stability: float
    meal_regularity: float
    meal_schedule_offset_h: float
    meal_size_multiplier: float
    n_meals_per_day_mean: float
    skips_breakfast_p: float
    skips_lunch_p: float
    luteal_meal_size_boost: float
    luteal_mood_drop: float
    exercise_intensity_mean: float
    logging_quality_raw: float
    fitness_level: float
    base_rhr: float
    isf_multiplier: float
    cr_multiplier: float
    basal_multiplier: float
    base_patient_name: str
    seed: int
    site_change_interval_days: int = 3
    exercise_time_preference: str = "random"
    dawn_sensitivity: float = 0.0
    n_days: int = 180
    split: str = "train"
    uses_aid: bool = False
    missingness_profile: MissingnessProfile | None = None
    agency_profile: UserAgencyProfile | None = None
    event_schedule: EventSchedule | None = None
    baseline_therapy_schedule: TherapySchedule | None = None
    therapy_schedule: TherapySchedule | None = None
    personality: "UserPersonality | None" = None

    @property
    def logging_quality(self) -> str:
        if self.logging_quality_raw > 0.9:
            return "great"
        if self.logging_quality_raw > 0.7:
            return "good"
        if self.logging_quality_raw >= 0.35:
            return "mediocre"
        return "poor"

    def to_record(self) -> dict:
        d = {k: v for k, v in asdict(self).items()
             if k not in (
                 "missingness_profile",
                 "agency_profile",
                 "event_schedule",
                 "baseline_therapy_schedule",
                 "therapy_schedule",
                 "personality",
             )}
        d["logging_quality"] = self.logging_quality
        return d


def sample_population(n_patients: int, seed: int = 42, male_fraction: float = 0.45, aid_fraction: float = 0.35) -> list[PatientConfig]:
    """Sample synthetic patient configurations."""
    rng = np.random.default_rng(seed)
    personas = list(PERSONAS.keys())
    patients: list[PatientConfig] = []

    def trait(persona: dict, key: str, lo: float = 0.0, hi: float = 1.0) -> float:
        mean, std = persona.get(key, (0.5, 0.2))
        return float(np.clip(rng.normal(mean, std), lo, hi))

    for i in range(n_patients):
        is_female = rng.random() > male_fraction
        eligible = [(p, PERSONAS[p]["weight"]) for p in personas if is_female or p not in ("cycle_sensitive", "cycle_insensitive")]
        p_names, p_weights = zip(*eligible)
        weights = np.array(p_weights, dtype=float)
        weights /= weights.sum()
        persona_name = str(rng.choice(p_names, p=weights))
        persona = PERSONAS[persona_name]
        cfg = PatientConfig(
            patient_id=f"sim_{'f' if is_female else 'm'}_{i:03d}",
            persona=persona_name,
            is_female=is_female,
            activity_propensity=trait(persona, "activity_propensity"),
            sleep_regularity=trait(persona, "sleep_regularity"),
            # Research: T1D actigraphy mean 358±48 min; cross-sectional mean ~440 min
            # Only 50.3% of T1D adults meet 7-9h NSF target. Default reflects real population.
            sleep_total_min_mean=float(np.clip(rng.normal(*persona.get("sleep_total_min_mean", (400, 65))), 240, 570)),
            # Research: 40-63% of T1D adults are "bad sleepers" (PSQI>5). Pop mean ~0.82 not 0.84.
            sleep_efficiency=float(np.clip(rng.normal(*persona.get("sleep_efficiency", (0.82, 0.10))), 0.55, 0.97)),
            sleep_schedule_offset_h=float(rng.normal(*persona.get("sleep_schedule_offset_h", (0.0, 0.3)))),
            stress_reactivity=trait(persona, "stress_reactivity"),
            stress_baseline=trait(persona, "stress_baseline") if "stress_baseline" in persona else float(np.clip(rng.normal(0.15, 0.10), 0, 0.6)),
            cycle_sensitivity=trait(persona, "cycle_sensitivity") if is_female else 0.0,
            mood_stability=trait(persona, "mood_stability"),
            meal_regularity=trait(persona, "meal_regularity"),
            meal_schedule_offset_h=float(rng.normal(*persona.get("meal_schedule_offset_h", (0.0, 0.2)))),
            meal_size_multiplier=float(np.clip(rng.normal(*persona.get("meal_size_multiplier", (1.0, 0.12))), 0.4, 2.0)),
            n_meals_per_day_mean=float(np.clip(rng.normal(*persona.get("n_meals_per_day", (3.0, 0.5))), 1.0, 7.0)),
            skips_breakfast_p=float(persona.get("skips_breakfast_p", 0.15)),
            skips_lunch_p=float(persona.get("skips_lunch_p", 0.08)),
            luteal_meal_size_boost=float(persona.get("luteal_meal_size_boost", (0.08, 0.03))[0]),
            luteal_mood_drop=float(persona.get("luteal_mood_drop", (0.15, 0.05))[0]),
            exercise_intensity_mean=float(np.clip(rng.normal(*persona.get("exercise_intensity_mean", (0.5, 0.15))), 0.1, 1.0)),
            logging_quality_raw=trait(persona, "logging_quality"),
            fitness_level=trait(persona, "fitness_level"),
            base_rhr=float(np.clip(rng.normal(*persona.get("base_rhr", (63, 6))), 45, 85)),
            isf_multiplier=float(np.clip(rng.normal(*persona.get("isf_base_multiplier", (1.0, 0.12))), 0.70, 1.35)),
            cr_multiplier=float(np.clip(rng.normal(*persona.get("cr_base_multiplier", (1.0, 0.12))), 0.70, 1.35)),
            basal_multiplier=float(np.clip(rng.normal(1.0, 0.10), 0.75, 1.25)),
            base_patient_name=f"adult#0{rng.integers(1,10):02d}",
            seed=int(rng.integers(0, 1_000_000)),
            exercise_time_preference=(
                "morning" if persona_name == "athlete"
                else "evening" if persona_name == "sedentary"
                else str(rng.choice(["morning", "midday", "evening", "random"]))
            ),
            dawn_sensitivity=float(rng.beta(2, 3)) if rng.random() < 0.5 else 0.0,
            n_days=180,
            split="train",
        )
        cfg.site_change_interval_days = PERSONA_SITE_INTERVALS.get(persona_name, 3)
        cfg.uses_aid = rng.random() < aid_fraction
        miss_rng = np.random.default_rng(int(rng.integers(0, 1_000_000)))
        cfg.missingness_profile = make_missingness_profile(cfg.logging_quality_raw, miss_rng)
        cfg = apply_cross_parameter_interactions(cfg)
        cfg.baseline_therapy_schedule = make_default_schedule(cfg)
        cfg.therapy_schedule = copy.deepcopy(cfg.baseline_therapy_schedule)
        agency_rng = np.random.default_rng(int(rng.integers(0, 1_000_000)))
        cfg.agency_profile = sample_agency(cfg, agency_rng)
        event_rng = np.random.default_rng(int(rng.integers(0, 1_000_000)))
        cfg.event_schedule = sample_life_events(cfg, cfg.n_days, event_rng)
        if sample_personality is not None:
            pers_rng = np.random.default_rng(int(rng.integers(0, 1_000_000)))
            cfg.personality = sample_personality(pers_rng)
        patients.append(cfg)
    return patients


def apply_cross_parameter_interactions(config: PatientConfig) -> PatientConfig:
    """Apply research-documented cross-trait interactions.

    Research basis:
    - Low mood_stability degrades meal_regularity and activity_propensity
      (diabetes distress → behavioural entropy; 40% T1D chronic fatigue rate)
    - Sleep irregularity causes eating jetlag (β=0.285, p=0.023 for wake-to-meal lag)
    - High stress reactivity degrades sleep (bidirectional cortisol loop)
    - High activity improves mood and sleep (CAN protective: OR=0.131)
    - AID users: fewer sleep interruptions, higher activity, better mood
    """
    c = copy.deepcopy(config)

    # Mood stability degrading meal regularity and activity
    if c.mood_stability < 0.5:
        deficit = 0.5 - c.mood_stability
        c.meal_regularity = float(np.clip(c.meal_regularity * (1.0 - 0.25 * deficit), 0, 1))
        c.activity_propensity = float(np.clip(c.activity_propensity * (1.0 - 0.30 * deficit), 0, 1))
        c.logging_quality_raw = float(np.clip(c.logging_quality_raw - 0.10 * deficit, 0, 1))

    # Sleep irregularity → eating jetlag
    if c.sleep_regularity < 0.6:
        deficit = 0.6 - c.sleep_regularity
        c.meal_regularity = float(np.clip(c.meal_regularity * (1.0 - 0.20 * deficit), 0, 1))
        c.stress_reactivity = float(np.clip(c.stress_reactivity + 0.08 * deficit, 0, 1))

    # High stress → degrades sleep
    if c.stress_reactivity > 0.6:
        excess = c.stress_reactivity - 0.6
        c.sleep_regularity = float(np.clip(c.sleep_regularity - 0.15 * excess, 0.1, 1))
        c.sleep_total_min_mean = float(np.clip(c.sleep_total_min_mean - 20 * excess, 240, 570))
        c.sleep_efficiency = float(np.clip(c.sleep_efficiency - 0.08 * excess, 0.55, 0.97))

    # High activity → improves mood, sleep, lowers stress reactivity
    if c.activity_propensity > 0.65:
        boost = c.activity_propensity - 0.65
        c.mood_stability = float(np.clip(c.mood_stability + 0.12 * boost, 0, 1))
        c.sleep_efficiency = float(np.clip(c.sleep_efficiency + 0.06 * boost, 0.55, 0.97))
        c.stress_reactivity = float(np.clip(c.stress_reactivity - 0.10 * boost, 0, 1))

    # AID technology tier
    if c.uses_aid:
        c.sleep_efficiency = float(np.clip(c.sleep_efficiency + 0.04, 0.55, 0.97))
        c.activity_propensity = float(np.clip(c.activity_propensity + 0.08, 0, 1))
        c.mood_stability = float(np.clip(c.mood_stability + 0.06, 0, 1))

    return c

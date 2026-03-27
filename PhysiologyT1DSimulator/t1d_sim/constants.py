"""Constants for t1d synthetic simulation."""

from __future__ import annotations

THERAPY_PROFILE_ID = "profile_001"
THERAPY_PROFILE_NAME = "Default"
SNAPSHOT_TIMESTAMP = "2025-03-01T00:00:00Z"

CGM_LOW = 70.0
CGM_HIGH = 180.0

SITE_LOCATIONS = [
    "abdomen_left",
    "abdomen_right",
    "thigh_left",
    "thigh_right",
    "arm_left",
    "arm_right",
]

# How many days a persona typically wears each infusion site before changing.
# ADA recommendation: ≤3 days, real-world median ~3–4 days.
# Values here represent the mean of a ±1 day jitter applied in behavior.py.
PERSONA_SITE_INTERVALS: dict[str, int] = {
    "night_owl":         3,
    "insomniac":         3,
    "solid_sleeper":     4,
    "athlete":           2,   # sweat/movement loosens adhesive faster
    "casual_mover":      3,
    "sedentary":         5,   # less active, site stays longer
    "grazer":            3,
    "meal_skipper":      4,
    "structured_eater":  4,
    "cycle_sensitive":   3,
    "cycle_insensitive": 4,
    "high_stress":       3,
    "low_stress":        4,
}

PERSONAS = {

    # ── SLEEP ─────────────────────────────────────────────────────────────

    "night_owl": {
        # Consistent but late schedule. Sleeps 1-3am, wakes 9-11am.
        # Skips breakfast most days. Meals and exercise pushed late.
        "weight": 0.08,
        "sleep_regularity":         (0.75, 0.10),
        "sleep_total_min_mean":     (420, 50),
        "sleep_efficiency":         (0.83, 0.08),
        "sleep_schedule_offset_h":  (3.0, 0.5),
        "meal_schedule_offset_h":   (2.5, 0.5),
        "skips_breakfast_p":        0.80,
        "activity_propensity":      (0.35, 0.15),
        "meal_regularity":          (0.50, 0.15),
        "stress_reactivity":        (0.48, 0.15),
        "mood_stability":           (0.50, 0.15),
        "logging_quality":          (0.45, 0.15),
        "fitness_level":            (0.35, 0.15),
        "base_rhr":                 (65, 5),
    },

    "insomniac": {
        # Fragmented sleep, poor efficiency, chronically fatigued.
        # Research: fragmentation → ~25% SI reduction. 5h sleep → ~12% EGP elevation.
        "weight": 0.07,
        "sleep_regularity":         (0.20, 0.10),
        "sleep_total_min_mean":     (300, 70),
        "sleep_efficiency":         (0.65, 0.10),
        "activity_propensity":      (0.25, 0.15),
        "meal_regularity":          (0.30, 0.15),
        "stress_reactivity":        (0.75, 0.15),
        "mood_stability":           (0.25, 0.10),
        "logging_quality":          (0.35, 0.15),
        "fitness_level":            (0.25, 0.15),
        "base_rhr":                 (68, 6),
    },

    "solid_sleeper": {
        # Research: T1D cross-sectional mean ~440 min, only 50.3% meet NSF target.
        # Pop mean efficiency 0.82 (not 0.88) due to nocturia/alarms.
        "weight": 0.15,
        "sleep_regularity":         (0.85, 0.08),
        "sleep_total_min_mean":     (440, 45),
        "sleep_efficiency":         (0.86, 0.07),
        "activity_propensity":      (0.55, 0.18),
        "meal_regularity":          (0.75, 0.10),
        "stress_reactivity":        (0.45, 0.15),
        "mood_stability":           (0.70, 0.10),
        "logging_quality":          (0.62, 0.14),
        "fitness_level":            (0.55, 0.18),
        "base_rhr":                 (60, 6),
    },

    # ── EXERCISE ──────────────────────────────────────────────────────────

    "athlete": {
        # Research: high activity propensity protects against CAN (OR=0.131).
        # Post-exercise SI improvement 24-48h (λ=0.035/hr decay).
        "weight": 0.08,
        "activity_propensity":      (0.92, 0.05),
        "exercise_intensity_mean":  (0.75, 0.10),
        "sleep_regularity":         (0.82, 0.08),
        "sleep_total_min_mean":     (475, 35),
        "sleep_efficiency":         (0.90, 0.05),
        "meal_regularity":          (0.80, 0.08),
        "meal_size_multiplier":     (1.35, 0.10),
        "stress_reactivity":        (0.28, 0.12),
        "mood_stability":           (0.78, 0.08),
        "logging_quality":          (0.68, 0.12),
        "fitness_level":            (0.90, 0.07),
        "base_rhr":                 (50, 5),
        "isf_base_multiplier":      (1.22, 0.08),
    },

    "casual_mover": {
        "weight": 0.15,
        "activity_propensity":      (0.45, 0.15),
        "exercise_intensity_mean":  (0.40, 0.12),
        "sleep_regularity":         (0.60, 0.15),
        "meal_regularity":          (0.55, 0.15),
        "stress_reactivity":        (0.45, 0.15),
        "mood_stability":           (0.60, 0.12),
        "logging_quality":          (0.55, 0.15),
        "fitness_level":            (0.45, 0.15),
        "base_rhr":                 (64, 6),
    },

    "sedentary": {
        # Research: 72% T1D adults don't meet MVPA. CAN risk elevated.
        # Eating jetlag common. Chronically lower SI.
        "weight": 0.10,
        "activity_propensity":      (0.15, 0.08),
        "sleep_regularity":         (0.48, 0.18),
        "sleep_total_min_mean":     (395, 60),
        "sleep_efficiency":         (0.80, 0.10),
        "meal_regularity":          (0.42, 0.18),
        "stress_reactivity":        (0.58, 0.15),
        "mood_stability":           (0.48, 0.14),
        "logging_quality":          (0.42, 0.15),
        "fitness_level":            (0.18, 0.10),
        "base_rhr":                 (72, 6),
        "isf_base_multiplier":      (0.83, 0.08),
        "cr_base_multiplier":       (0.87, 0.08),
    },

    # ── EATING ────────────────────────────────────────────────────────────

    "grazer": {
        # 5-6 small meals throughout day. Low carb per event.
        # Hardest for meal detector — many small diffuse CGM events.
        "weight": 0.07,
        "n_meals_per_day":          (5.0, 1.0),
        "meal_size_multiplier":     (0.55, 0.10),
        "meal_regularity":          (0.30, 0.15),
        "skips_breakfast_p":        0.10,
        "activity_propensity":      (0.45, 0.20),
        "sleep_regularity":         (0.55, 0.20),
        "stress_reactivity":        (0.45, 0.15),
        "mood_stability":           (0.55, 0.15),
        "logging_quality":          (0.45, 0.20),
        "fitness_level":            (0.45, 0.20),
        "base_rhr":                 (63, 6),
    },

    "meal_skipper": {
        # Skips breakfast most days, often skips lunch, compensates with
        # large dinner. BG runs low mid-day, spikes hard in evening.
        "weight": 0.08,
        "skips_breakfast_p":        0.85,
        "skips_lunch_p":            0.40,
        "meal_size_multiplier":     (1.40, 0.15),
        "n_meals_per_day":          (1.8, 0.6),
        "meal_regularity":          (0.45, 0.15),
        "activity_propensity":      (0.40, 0.20),
        "sleep_regularity":         (0.55, 0.20),
        "stress_reactivity":        (0.55, 0.15),
        "mood_stability":           (0.50, 0.15),
        "logging_quality":          (0.40, 0.20),
        "fitness_level":            (0.40, 0.20),
        "base_rhr":                 (65, 6),
    },

    "structured_eater": {
        # 3 meals at very consistent times. Meal prepper.
        # Easiest for meal detector and therapy recommender.
        "weight": 0.10,
        "meal_regularity":          (0.90, 0.06),
        "n_meals_per_day":          (3.0, 0.3),
        "meal_size_multiplier":     (1.00, 0.08),
        "skips_breakfast_p":        0.05,
        "activity_propensity":      (0.60, 0.20),
        "sleep_regularity":         (0.75, 0.12),
        "stress_reactivity":        (0.35, 0.15),
        "mood_stability":           (0.70, 0.12),
        "logging_quality":          (0.75, 0.12),
        "fitness_level":            (0.55, 0.20),
        "base_rhr":                 (61, 5),
    },

    # ── HORMONAL (females only) ────────────────────────────────────────────

    "cycle_sensitive": {
        # Strong hormonal ISF shifts between phases.
        # Research (Apple AWHS 2023, 1,982 cycles): follicular TIR 68.5%
        # vs luteal 66.8% at population level. Sensitive individuals show
        # up to 15-20% ISF change — the therapy recommender's hardest test.
        "weight": 0.06,
        "cycle_sensitivity":        (0.90, 0.07),
        "luteal_meal_size_boost":   (0.18, 0.05),
        "luteal_mood_drop":         (0.35, 0.10),
        "activity_propensity":      (0.55, 0.20),
        "sleep_regularity":         (0.60, 0.15),
        "meal_regularity":          (0.55, 0.15),
        "stress_reactivity":        (0.60, 0.15),
        "mood_stability":           (0.45, 0.12),
        "logging_quality":          (0.55, 0.15),
        "fitness_level":            (0.50, 0.20),
        "base_rhr":                 (63, 6),
    },

    "cycle_insensitive": {
        # Female but minimal hormonal effect on BG. Control case.
        "weight": 0.04,
        "cycle_sensitivity":        (0.10, 0.06),
        "activity_propensity":      (0.50, 0.20),
        "sleep_regularity":         (0.65, 0.15),
        "meal_regularity":          (0.60, 0.15),
        "stress_reactivity":        (0.40, 0.15),
        "mood_stability":           (0.65, 0.12),
        "logging_quality":          (0.60, 0.15),
        "fitness_level":            (0.50, 0.20),
        "base_rhr":                 (62, 5),
    },

    # ── STRESS / MOOD ──────────────────────────────────────────────────────

    "high_stress": {
        # Research: pharmacological stress (prednisone 60mg/3d) → TDI +70-80%.
        # Chronic distress → 10-20% persistent TDI elevation.
        # Bidirectional: high GV → diabetes distress → more stress.
        # Sleep efficiency below pop mean from fragmentation.
        "weight": 0.07,
        "stress_reactivity":        (0.85, 0.08),
        "stress_baseline":          (0.55, 0.15),
        "mood_stability":           (0.22, 0.08),
        "sleep_regularity":         (0.32, 0.14),
        "sleep_total_min_mean":     (355, 65),
        "sleep_efficiency":         (0.74, 0.10),
        "meal_regularity":          (0.32, 0.14),
        "activity_propensity":      (0.28, 0.14),
        "logging_quality":          (0.33, 0.14),
        "fitness_level":            (0.28, 0.14),
        "base_rhr":                 (72, 6),
        "isf_base_multiplier":      (0.87, 0.08),
    },

    "low_stress": {
        # Relaxed lifestyle. Minimal stress confounders. Clean BG signal.
        "weight": 0.10,
        "stress_reactivity":        (0.20, 0.10),
        "stress_baseline":          (0.10, 0.07),
        "mood_stability":           (0.82, 0.08),
        "sleep_regularity":         (0.78, 0.10),
        "sleep_total_min_mean":     (445, 40),
        "sleep_efficiency":         (0.87, 0.06),
        "meal_regularity":          (0.68, 0.12),
        "activity_propensity":      (0.55, 0.20),
        "logging_quality":          (0.65, 0.15),
        "fitness_level":            (0.55, 0.20),
        "base_rhr":                 (60, 5),
    },
}

"""Behavioral generator."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
import numpy as np

from t1d_sim.population import PatientConfig
from t1d_sim.constants import SITE_LOCATIONS

if TYPE_CHECKING:
    from t1d_sim.feedback import PatientState, YesterdayOutcome


@dataclass
class ContextState:
    cycle_phase: str | None
    cycle_day: int
    cycle_sensitivity: float
    sleep_min_last_night: float
    sleep_efficiency: float
    hours_since_exercise: float
    exercise_intensity: float
    stress: float
    stress_reactivity: float
    stress_baseline: float
    mood_valence: float
    mood_arousal: float
    is_ill: bool
    day_of_week: int
    is_weekend: bool
    exercise_hour: int | None = None
    post_exercise_hypo_window: bool = False


def _cycle_phase(is_female: bool, day_idx: int, cycle_len: int = 28) -> tuple[str | None, int]:
    if not is_female:
        return None, -1
    d = day_idx % cycle_len
    f = d / cycle_len
    if f < 0.18:
        return "menstrual", d
    if f < 0.45:
        return "follicular", d
    if f < 0.55:
        return "ovulation", d
    return "luteal", d


def generate_day_behavior(
    cfg: PatientConfig,
    start_day: datetime,
    day_index: int,
    prev_mood: tuple[float, float],
    yesterday: "YesterdayOutcome | None" = None,
    event_modifiers: dict | None = None,
    patient_state: "PatientState | None" = None,
) -> dict:
    """Generate one day of behavior and context.

    Optional feedback parameters (all backward-compatible, default None):
        yesterday: previous day's outcome for daily feedback modifiers.
        event_modifiers: dict from apply_event_modifiers() for life events.
        patient_state: drifted physiological state from biweekly updates.
    """
    from t1d_sim.feedback import apply_daily_feedback

    rng = np.random.default_rng(cfg.seed * 10000 + day_index)
    phase, cycle_day = _cycle_phase(cfg.is_female, day_index)
    is_weekend = start_day.weekday() >= 5
    ev = event_modifiers or {}

    # ── Resolve effective baselines (PatientState > cfg) ──
    eff_stress_baseline = (
        patient_state.effective_stress_baseline if patient_state else cfg.stress_baseline
    )
    eff_sleep_regularity = (
        patient_state.effective_sleep_regularity if patient_state else cfg.sleep_regularity
    )

    # ── Daily feedback modifiers ──
    fb: dict = {}
    if yesterday is not None:
        fb = apply_daily_feedback(cfg, yesterday, rng)

    # ── Sleep ──
    sleep_min = float(np.clip(
        rng.normal(cfg.sleep_total_min_mean, 50 * (1 - eff_sleep_regularity)),
        240, 600,
    ))
    # Event: insomnia / travel multiply total sleep
    sleep_min *= ev.get("sleep_minutes_mult", 1.0)
    # Feedback: hyper/hypo sleep penalty
    sleep_min -= fb.get("sleep_penalty_min", 0.0) + ev.get("sleep_penalty_min", 0.0)
    sleep_min = max(120.0, sleep_min)

    sleep_eff = cfg.sleep_efficiency
    sleep_eff -= fb.get("sleep_efficiency_penalty", 0.0) + ev.get("sleep_efficiency_penalty", 0.0)
    sleep_eff = float(np.clip(sleep_eff, 0.40, 0.97))

    # ── Stress ──
    stress = float(np.clip(
        rng.normal(eff_stress_baseline + 0.12 * (1 - cfg.mood_stability), 0.15), 0, 1,
    ))
    stress += fb.get("stress_add", 0.0) + ev.get("stress_add", 0.0)
    stress = float(np.clip(stress, 0.0, 1.0))

    # ── Exercise ──
    ex_prob = cfg.activity_propensity
    ex_prob *= fb.get("exercise_prob_mult", 1.0) * ev.get("exercise_prob_mult", 1.0)
    exercise_today = rng.random() < ex_prob
    ex_minutes = float(rng.uniform(20, 75)) if exercise_today else 0.0
    ex_intensity = float(np.clip(rng.normal(cfg.exercise_intensity_mean, 0.1), 0.1, 1.0)) if exercise_today else 0.0
    if exercise_today:
        pref = getattr(cfg, "exercise_time_preference", "random")
        if pref == "morning":
            exercise_hour: int | None = int(rng.integers(6, 10))
        elif pref == "midday":
            exercise_hour = int(rng.integers(11, 15))
        elif pref == "evening":
            exercise_hour = int(rng.integers(16, 21))
        else:
            exercise_hour = int(rng.integers(6, 20))
    else:
        exercise_hour = None

    # ── Mood ──
    prev_v, prev_a = prev_mood
    v = float(np.clip(
        0.7 * prev_v + rng.normal(0, 0.4 * (1 - cfg.mood_stability)) - 0.25 * max(0, stress - 0.4),
        -1, 1,
    ))
    a = float(np.clip(
        0.7 * prev_a + rng.normal(0, 0.4 * (1 - cfg.mood_stability)) + 0.25 * stress,
        -1, 1,
    ))
    if sleep_min >= 420:
        v += min(0.12, (sleep_min - 420) / 1500.0)
        a -= min(0.08, (sleep_min - 420) / 1800.0)
    if exercise_today and ex_minutes > 0:
        v += min(0.14, ex_minutes / 500.0)
        a += min(0.06, ex_intensity * 0.08)
    if phase == "luteal":
        v -= cfg.luteal_mood_drop
    v += fb.get("mood_bias", 0.0) + ev.get("mood_offset", 0.0)
    v = float(np.clip(v, -1.0, 1.0))

    # ── Meals ──
    meal_regularity = cfg.meal_regularity + ev.get("meal_regularity_add", 0.0)
    meal_regularity = float(np.clip(meal_regularity, 0.0, 1.0))
    meal_size_mult = cfg.meal_size_multiplier
    meal_size_mult *= fb.get("meal_size_mult", 1.0) * ev.get("meal_size_mult", 1.0)

    n_meals = int(np.clip(round(rng.normal(cfg.n_meals_per_day_mean, 0.8)), 1, 6))
    windows = [(8, (30, 60), cfg.skips_breakfast_p), (13, (40, 80), cfg.skips_lunch_p), (19, (50, 90), 0.05)]
    meals: list[tuple[datetime, float, str | None]] = []
    schedule_shift = ev.get("sleep_schedule_shift_h", 0.0)
    for h, (low, high), skip_p in windows:
        if rng.random() < skip_p + 0.15 * stress:
            continue
        minute_std = 15 + 75 * (1 - meal_regularity)
        hh = h + cfg.meal_schedule_offset_h + schedule_shift + (0.7 if (is_weekend and h == 8) else 0)
        t = start_day + timedelta(hours=float(rng.normal(hh, minute_std / 60)), minutes=float(rng.normal(0, minute_std)))
        carb = float(rng.uniform(low, high) * meal_size_mult * (1.0 + (cfg.luteal_meal_size_boost if phase == "luteal" else 0.0)))
        profile_tag = "grazer" if cfg.persona == "grazer" else None
        meals.append((t, carb, profile_tag))
    while len(meals) < n_meals:
        h = float(rng.uniform(10, 22))
        t = start_day + timedelta(hours=h)
        profile_tag = "grazer" if cfg.persona == "grazer" else None
        meals.append((t, float(rng.uniform(10, 30) * meal_size_mult), profile_tag))

    # Mood events — frequency modulated by logging quality
    lq = cfg.logging_quality_raw * ev.get("logging_quality_mult", 1.0)
    if lq > 0.7:
        n_mood_events = int(rng.integers(2, 5))
    elif lq >= 0.35:
        n_mood_events = 1
    else:
        n_mood_events = 1 if rng.random() < 0.4 else 0
    mood_event_hours = sorted(rng.uniform(7, 22, size=n_mood_events).tolist())
    mood_events_list = [
        {
            "timestamp": (start_day + timedelta(hours=h)).astimezone(timezone.utc),
            "valence": v,
            "arousal": a,
        }
        for h in mood_event_hours
    ]

    # Illness is driven by the sampled event schedule. Avoid layering an extra
    # unscheduled severe illness day on top of the event system.
    is_ill = bool(ev.get("is_ill_override", False))

    # Persona-driven site change interval with per-patient ±1 day jitter so
    # patients don't all change on the same cycle offset.
    _base_interval = getattr(cfg, "site_change_interval_days", 3)
    _jitter_rng = np.random.default_rng(cfg.seed + 99999)
    _site_interval = max(2, _base_interval + int(_jitter_rng.integers(-1, 2)))
    site_days = day_index % _site_interval
    site_loc = SITE_LOCATIONS[(day_index // _site_interval) % len(SITE_LOCATIONS)]
    ctx = ContextState(
        cycle_phase=phase,
        cycle_day=cycle_day,
        cycle_sensitivity=cfg.cycle_sensitivity,
        sleep_min_last_night=float(sleep_min),
        sleep_efficiency=sleep_eff,
        hours_since_exercise=0.0 if exercise_today else 24.0,
        exercise_intensity=ex_intensity,
        stress=stress,
        stress_reactivity=cfg.stress_reactivity,
        stress_baseline=eff_stress_baseline,
        mood_valence=v,
        mood_arousal=a,
        is_ill=is_ill,
        day_of_week=start_day.weekday(),
        is_weekend=is_weekend,
        exercise_hour=exercise_hour,
    )
    return {
        "context": ctx,
        "meals": sorted(meals, key=lambda x: x[0]),
        "exercise_minutes": ex_minutes,
        "exercise_intensity": ex_intensity,
        "exercise_hour": exercise_hour,
        "sleep_minutes": float(sleep_min),
        "site_days_since_change": site_days,
        "site_location": site_loc,
        "mood_events": mood_events_list,
    }

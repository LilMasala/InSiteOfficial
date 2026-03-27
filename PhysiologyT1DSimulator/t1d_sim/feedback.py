"""
Compounding feedback and slow-drift state for t1d_sim.

Three timescales of feedback:
  1. Daily — yesterday's BG outcomes shape today's behavior
  2. Biweekly — sustained patterns shift baseline physiology
  3. Stochastic — random life events create cascading disruptions

All dataclasses and pure computation functions live here to avoid
circular imports between behavior.py, physiology.py, and patient.py.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from t1d_sim.population import PatientConfig


# ───────────────────────────────────────────────────────────────────
# Part 1: Daily Feedback — YesterdayOutcome
# ───────────────────────────────────────────────────────────────────

@dataclass
class YesterdayOutcome:
    """Summary of the previous day, fed into today's behavior generation."""
    mean_bg: float
    tir: float                    # time in range 70-180 (fraction)
    percent_low: float            # fraction < 70
    percent_high: float           # fraction > 180
    sleep_minutes: float
    exercise_minutes: float
    stress: float
    mood_valence: float
    total_carbs: float
    consecutive_bad_days: int     # streak of days with TIR < 0.55


def compute_yesterday_outcome(
    true_bg: np.ndarray,
    beh: dict,
    prev_outcome: YesterdayOutcome | None,
) -> YesterdayOutcome:
    """Compute outcome summary from a simulated day.

    Args:
        true_bg: (288,) ground truth BG trace.
        beh: dict returned by generate_day_behavior.
        prev_outcome: previous day's outcome (for streak tracking).
    """
    finite = true_bg[np.isfinite(true_bg)]
    if finite.size == 0:
        mean_bg = 120.0
        tir = 0.70
        pct_low = 0.0
        pct_high = 0.0
    else:
        mean_bg = float(np.mean(finite))
        pct_low = float(np.mean(finite < 70))
        pct_high = float(np.mean(finite > 180))
        tir = float(np.mean((finite >= 70) & (finite <= 180)))

    # Streak tracking
    prev_streak = prev_outcome.consecutive_bad_days if prev_outcome else 0
    if tir < 0.55:
        streak = prev_streak + 1
    elif tir > 0.65:
        streak = 0
    else:
        streak = prev_streak  # in the 0.55-0.65 band, hold

    # Extract carbs from meals
    total_carbs = sum(m[1] for m in beh.get("meals", []))

    return YesterdayOutcome(
        mean_bg=mean_bg,
        tir=tir,
        percent_low=pct_low,
        percent_high=pct_high,
        sleep_minutes=beh.get("sleep_minutes", 420.0),
        exercise_minutes=beh.get("exercise_minutes", 0.0),
        stress=beh.get("context", None) and beh["context"].stress or 0.2,
        mood_valence=beh.get("context", None) and beh["context"].mood_valence or 0.0,
        total_carbs=total_carbs,
        consecutive_bad_days=streak,
    )


def apply_daily_feedback(
    cfg: "PatientConfig",
    outcome: YesterdayOutcome,
    rng: np.random.Generator,
) -> dict:
    """Compute behavior modifiers from yesterday's outcome.

    Returns a dict of modifier keys consumed by generate_day_behavior().
    """
    mods: dict = {}

    # ── Sleep disruption from hyperglycemia ──
    if outcome.mean_bg > 180:
        penalty = min(60.0, (outcome.mean_bg - 180) * 0.4)
        mods["sleep_penalty_min"] = mods.get("sleep_penalty_min", 0.0) + penalty
        mods["sleep_efficiency_penalty"] = mods.get("sleep_efficiency_penalty", 0.0) + (
            0.02 * (outcome.mean_bg - 180) / 100.0
        )

    # ── Sleep disruption from hypoglycemia ──
    if outcome.percent_low > 0.05:
        penalty = min(45.0, outcome.percent_low * 500.0)
        mods["sleep_penalty_min"] = mods.get("sleep_penalty_min", 0.0) + penalty
        # Next-day fatigue → stress
        mods["stress_add"] = mods.get("stress_add", 0.0) + float(
            rng.uniform(0.05, 0.10)
        )

    # ── Stress amplification from bad BG days ──
    if outcome.tir < 0.50:
        stress_boost = 0.08 * (1.0 - outcome.tir) * cfg.stress_reactivity
        mods["stress_add"] = mods.get("stress_add", 0.0) + stress_boost

    # ── Good glycemic days allow real recovery ──
    if outcome.tir > 0.72 and outcome.percent_low < 0.03:
        relief = min(0.08, 0.10 * (outcome.tir - 0.72))
        mods["stress_add"] = mods.get("stress_add", 0.0) - relief
        mods["mood_bias"] = mods.get("mood_bias", 0.0) + min(0.18, 0.30 * (outcome.tir - 0.72))

    # ── Restorative sleep improves next-day regulation ──
    if outcome.sleep_minutes > 450:
        mods["stress_add"] = mods.get("stress_add", 0.0) - min(0.05, (outcome.sleep_minutes - 450) / 2400.0)
        mods["exercise_prob_mult"] = max(mods.get("exercise_prob_mult", 1.0), 1.05)

    # ── Exercise avoidance from fatigue ──
    bad_sleep = outcome.sleep_minutes < 300  # < 5 hours
    bad_bg = outcome.mean_bg > 200
    if bad_sleep and bad_bg:
        mods["exercise_prob_mult"] = 0.5
    elif bad_sleep or bad_bg:
        mods["exercise_prob_mult"] = 0.7

    # ── Mood carry-forward ──
    if outcome.mood_valence < -0.3:
        mods["mood_bias"] = 0.3 * outcome.mood_valence

    # ── Meal compensation from lows ──
    if outcome.percent_low > 0.08:
        mods["meal_size_mult"] = 1.0 + 0.15 * (outcome.percent_low / 0.20)

    # ── Consecutive bad day escalation ──
    if outcome.consecutive_bad_days >= 3:
        mods["stress_add"] = mods.get("stress_add", 0.0) + min(
            0.15, 0.03 * outcome.consecutive_bad_days
        )

    # ── Stable stretches make future good days more likely ──
    if outcome.consecutive_bad_days == 0 and outcome.tir > 0.80 and outcome.percent_low < 0.02:
        mods["mood_bias"] = mods.get("mood_bias", 0.0) + 0.08
        mods["stress_add"] = mods.get("stress_add", 0.0) - 0.03

    return mods


# ───────────────────────────────────────────────────────────────────
# Part 2: Biweekly Slow-Drift — PatientState
# ───────────────────────────────────────────────────────────────────

@dataclass
class PatientState:
    """Evolving physiological state that drifts from PatientConfig baselines."""
    effective_isf_mult: float
    effective_fitness: float
    effective_rhr: float
    effective_stress_baseline: float
    effective_sleep_regularity: float
    last_drift_day: int = 0

    @classmethod
    def from_config(cls, cfg: "PatientConfig") -> "PatientState":
        return cls(
            effective_isf_mult=cfg.isf_multiplier,
            effective_fitness=cfg.fitness_level,
            effective_rhr=cfg.base_rhr,
            effective_stress_baseline=cfg.stress_baseline,
            effective_sleep_regularity=cfg.sleep_regularity,
        )

    def effective_isf_with_fitness(self, cfg: "PatientConfig") -> float:
        """ISF including the fitness bonus/penalty stacked on energy-balance drift."""
        fitness_bonus = 0.15 * (self.effective_fitness - cfg.fitness_level)
        return self.effective_isf_mult * (1.0 + fitness_bonus)


def update_patient_state(
    state: PatientState,
    cfg: "PatientConfig",
    outcomes: list[YesterdayOutcome],
) -> PatientState:
    """Biweekly drift update from sustained patterns.

    Called every 14 simulated days with the outcomes from that window.
    Returns a new PatientState with updated effective values.
    """
    if not outcomes:
        return state

    n = len(outcomes)

    # ── Energy balance → ISF drift ──
    daily_balances = []
    for o in outcomes:
        # Proxy: carbs in vs exercise-equivalent burn
        burn_equiv = o.exercise_minutes * 0.5 * 8.0  # avg intensity ~0.5
        daily_balances.append(o.total_carbs - burn_equiv)
    avg_balance = float(np.mean(daily_balances))

    isf = state.effective_isf_mult
    if avg_balance > 30:
        # Sustained surplus → ISF degrades
        delta = min(0.012, 0.004 * (avg_balance / 50.0))
        isf -= delta
    elif avg_balance < -10:
        # Sustained deficit → ISF improves (slower)
        delta = min(0.006, 0.002 * (abs(avg_balance) / 50.0))
        isf += delta
    # Clamp to [0.65, 1.15] of genetic baseline
    isf = float(np.clip(isf, cfg.isf_multiplier * 0.65, cfg.isf_multiplier * 1.15))

    # ── Fitness adaptation ──
    exercise_days = sum(1 for o in outcomes if o.exercise_minutes > 15)
    fitness = state.effective_fitness

    if exercise_days >= 5:
        fitness += 0.012
    elif exercise_days >= 3:
        fitness += 0.005
    elif exercise_days <= 1:
        fitness -= 0.018
    elif exercise_days == 2:
        fitness -= 0.005
    fitness = float(np.clip(fitness, 0.08, 0.95))

    # ── RHR derived from fitness ──
    rhr = cfg.base_rhr * (1.25 - 0.30 * fitness)
    rhr = float(np.clip(rhr, 40.0, 90.0))

    # ── Stress baseline drift ──
    avg_stress = float(np.mean([o.stress for o in outcomes]))
    stress_bl = state.effective_stress_baseline
    if avg_stress > 0.50:
        stress_bl += 0.010
    elif avg_stress > 0.35:
        stress_bl += 0.004
    elif avg_stress < 0.15:
        stress_bl -= 0.010
    elif avg_stress < 0.25:
        stress_bl -= 0.006
    stress_bl = float(np.clip(stress_bl, 0.03, 0.65))

    # ── Sleep regularity drift ──
    sleep_durations = [o.sleep_minutes for o in outcomes]
    sleep_std = float(np.std(sleep_durations)) if len(sleep_durations) > 1 else 0.0
    sleep_reg = state.effective_sleep_regularity
    if sleep_std > 90:
        sleep_reg -= 0.008
    elif sleep_std < 45:
        sleep_reg += 0.005
    sleep_reg = float(np.clip(sleep_reg, 0.15, 0.92))

    return PatientState(
        effective_isf_mult=isf,
        effective_fitness=fitness,
        effective_rhr=rhr,
        effective_stress_baseline=stress_bl,
        effective_sleep_regularity=sleep_reg,
        last_drift_day=state.last_drift_day + n,
    )


# ───────────────────────────────────────────────────────────────────
# Part 3: Random Life Events
# ───────────────────────────────────────────────────────────────────

class EventType(Enum):
    INSOMNIA_BOUT = "insomnia_bout"
    ILLNESS = "illness"
    ACUTE_STRESS = "acute_stress"
    POSITIVE_LIFE_EVENT = "positive_life_event"
    TRAVEL = "travel"
    MENSTRUAL_IRREGULARITY = "menstrual_irregularity"
    INJURY = "injury"
    DEVICE_HIATUS = "device_hiatus"
    VACATION = "vacation"
    MEDICATION_CHANGE = "medication_change"
    SEASONAL_SHIFT = "seasonal_shift"


# Which events are "major" (max 2 simultaneous)
_MAJOR_EVENTS = {
    EventType.ILLNESS,
    EventType.ACUTE_STRESS,
    EventType.INJURY,
    EventType.MEDICATION_CHANGE,
}


@dataclass
class LifeEvent:
    """A single life event placed at a specific day."""
    event_type: EventType
    start_day: int
    duration_days: int
    severity: float           # 0.0-1.0
    taper_days: int           # gradual recovery after end
    is_major: bool
    params: dict = field(default_factory=dict)


@dataclass
class ActiveEvent:
    """A LifeEvent resolved for a specific day, including taper."""
    event: LifeEvent
    day_offset: int           # how far into event+taper we are
    intensity: float          # severity × taper curve


@dataclass
class EventSchedule:
    """All life events for a patient, sampled at creation time."""
    events: list[LifeEvent] = field(default_factory=list)


# Event catalog: (probability_per_180d, duration_range, taper_frac, is_major)
_EVENT_CATALOG = {
    EventType.INSOMNIA_BOUT:         (0.25, (3, 10),  0.30, False),
    EventType.ILLNESS:               (0.40, (3, 7),   0.40, True),
    EventType.ACUTE_STRESS:          (0.15, (14, 45),  0.50, True),
    EventType.POSITIVE_LIFE_EVENT:   (0.28, (4, 12),  0.25, False),
    EventType.TRAVEL:                (0.30, (5, 14),  0.30, False),
    EventType.MENSTRUAL_IRREGULARITY:(0.20, (28, 56), 0.20, False),
    EventType.INJURY:                (0.10, (14, 42),  0.40, True),
    EventType.DEVICE_HIATUS:         (0.08, (3, 14),  0.00, False),
    EventType.VACATION:              (0.35, (5, 10),  0.20, False),
    EventType.MEDICATION_CHANGE:     (0.10, (999, 999), 0.00, True),  # permanent
}


def sample_life_events(
    cfg: "PatientConfig",
    n_days: int,
    rng: np.random.Generator,
) -> EventSchedule:
    """Sample life events for a patient's simulation timeline.

    Enforces max-2-major constraint: if a 3rd major event would overlap
    2 existing majors, defer it until the earliest one ends.
    """
    events: list[LifeEvent] = []

    for etype, (prob_180, dur_range, taper_frac, is_major) in _EVENT_CATALOG.items():
        # Skip menstrual events for males
        if etype == EventType.MENSTRUAL_IRREGULARITY and not cfg.is_female:
            continue

        # Scale probability to simulation length
        prob = 1.0 - (1.0 - prob_180) ** (n_days / 180.0)
        if rng.random() >= prob:
            continue

        # Duration
        if etype == EventType.MEDICATION_CHANGE:
            duration = n_days  # permanent from onset
        else:
            duration = int(rng.integers(dur_range[0], dur_range[1] + 1))

        taper = max(0, int(duration * taper_frac))
        severity = float(rng.uniform(0.4, 1.0))

        # Place at a random day (leave room for the event)
        latest_start = max(1, n_days - duration)
        start = int(rng.integers(1, latest_start + 1))

        # Event-specific params
        params = _sample_event_params(etype, cfg, severity, rng)

        event = LifeEvent(
            event_type=etype,
            start_day=start,
            duration_days=duration,
            severity=severity,
            taper_days=taper,
            is_major=is_major,
            params=params,
        )
        events.append(event)

    # Enforce max-2-major constraint by deferring events
    events = _enforce_major_constraint(events, n_days)

    # Add seasonal shift for everyone (continuous sinusoidal)
    phase_offset = float(rng.uniform(0, 2 * math.pi))
    events.append(LifeEvent(
        event_type=EventType.SEASONAL_SHIFT,
        start_day=0,
        duration_days=n_days,
        severity=1.0,
        taper_days=0,
        is_major=False,
        params={"phase_offset": phase_offset},
    ))

    return EventSchedule(events=events)


def _sample_event_params(
    etype: EventType,
    cfg: "PatientConfig",
    severity: float,
    rng: np.random.Generator,
) -> dict:
    """Sample event-specific parameters."""
    if etype == EventType.INSOMNIA_BOUT:
        return {
            "sleep_reduction_frac": float(rng.uniform(0.30, 0.50)),
            "efficiency_penalty": float(rng.uniform(0.15, 0.25)),
        }
    elif etype == EventType.POSITIVE_LIFE_EVENT:
        return {
            "stress_relief": float(rng.uniform(0.05, 0.14)),
            "mood_offset": float(rng.uniform(0.12, 0.28)),
            "exercise_prob_mult": float(rng.uniform(1.05, 1.20)),
            "meal_regularity_boost": float(rng.uniform(0.05, 0.12)),
            "logging_quality_mult": float(rng.uniform(1.05, 1.20)),
        }
    elif etype == EventType.ILLNESS:
        return {
            "egp_mult": float(rng.uniform(1.30, 1.50)),
            "isf_factor": float(rng.uniform(0.70, 0.80)),
            "appetite_direction": float(rng.choice([-1, 1])),  # smaller or larger meals
        }
    elif etype == EventType.ACUTE_STRESS:
        return {
            "stress_jump": float(rng.uniform(0.15, 0.30)),
            "mood_offset": float(rng.uniform(-0.40, -0.20)),
            "exercise_prob_mult": 0.6,
            "meal_regularity_drop": float(rng.uniform(0.15, 0.25)),
        }
    elif etype == EventType.TRAVEL:
        return {
            "timezone_shift_h": float(rng.uniform(2, 6)) * rng.choice([-1, 1]),
            "efficiency_penalty": float(rng.uniform(0.10, 0.20)),
            "exercise_change": float(rng.choice([-0.3, 0.2])),  # less or more
        }
    elif etype == EventType.MENSTRUAL_IRREGULARITY:
        return {
            "cycle_length_days": int(rng.choice([
                rng.integers(21, 26),   # shortened
                rng.integers(32, 41),   # lengthened
            ])),
            "sensitivity_amplifier": float(rng.uniform(1.3, 1.5)),
        }
    elif etype == EventType.INJURY:
        return {
            "ramp_back_days": int(rng.integers(14, 21)),
            "mood_hit": float(rng.uniform(-0.15, -0.05)),
            "meal_size_boost": float(rng.uniform(1.05, 1.15)),
        }
    elif etype == EventType.DEVICE_HIATUS:
        # Which device(s) go offline
        if cfg.logging_quality_raw < 0.35:
            # Poor users: could be CGM too
            device = str(rng.choice(["watch", "watch", "cgm", "both"]))
        else:
            device = "watch"
        return {"device": device}
    elif etype == EventType.VACATION:
        return {
            "meal_size_mult": float(rng.uniform(1.10, 1.30)),
            "meal_regularity_drop": float(rng.uniform(0.20, 0.35)),
            "sleep_shift_h": float(rng.uniform(0.5, 2.0)),
            "exercise_change": float(rng.choice([-0.4, 0.1])),
            "logging_quality_mult": float(rng.uniform(0.50, 0.75)),
        }
    elif etype == EventType.MEDICATION_CHANGE:
        # Step change in ISF — could be improvement or worsening
        med_type = str(rng.choice(["metformin", "corticosteroid", "beta_blocker"]))
        if med_type == "metformin":
            return {"med_type": med_type, "isf_factor": float(rng.uniform(1.10, 1.15))}
        elif med_type == "corticosteroid":
            return {"med_type": med_type, "isf_factor": float(rng.uniform(0.70, 0.80))}
        else:  # beta_blocker
            return {"med_type": med_type, "rhr_offset": float(rng.uniform(-8, -4))}
    return {}


def _enforce_major_constraint(
    events: list[LifeEvent],
    n_days: int,
) -> list[LifeEvent]:
    """Defer major events so at most 2 are active on any given day."""
    majors = [e for e in events if e.is_major]
    non_majors = [e for e in events if not e.is_major]

    # Sort majors by start day
    majors.sort(key=lambda e: e.start_day)

    accepted: list[LifeEvent] = []
    for event in majors:
        # Count how many accepted majors overlap with this event's window
        ev_start = event.start_day
        ev_end = event.start_day + event.duration_days + event.taper_days

        overlap_count = 0
        earliest_end = n_days
        for acc in accepted:
            acc_end = acc.start_day + acc.duration_days + acc.taper_days
            if acc.start_day < ev_end and acc_end > ev_start:
                overlap_count += 1
                earliest_end = min(earliest_end, acc_end)

        if overlap_count < 2:
            accepted.append(event)
        else:
            # Defer to after the earliest overlapping event ends
            new_start = earliest_end + 1
            if new_start + event.duration_days <= n_days:
                event.start_day = new_start
                accepted.append(event)
            # else: drop event (no room)

    return accepted + non_majors


# ───────────────────────────────────────────────────────────────────
# Part 3c: Psychological Response Model (Section 7.1)
# ───────────────────────────────────────────────────────────────────

@dataclass
class PsychState:
    """Per-patient psychological state for mood feedback from system interactions."""
    mood_valence: float = 0.0           # Current mood valence
    recommendation_fatigue: float = 0.0  # Accumulated fatigue from frequent recs
    recent_rec_count: int = 0           # Recs received in last 7 days
    burned_out: bool = False            # Below burnout threshold

    def reset_weekly_counter(self) -> None:
        self.recent_rec_count = 0
        self.recommendation_fatigue = max(0.0, self.recommendation_fatigue - 0.1)


def apply_psychological_feedback(
    psych: PsychState,
    personality: "UserPersonality",  # from chamelia.personality
    rec_accepted: bool | None,
    rec_succeeded: bool | None,
    received_recommendation: bool,
    received_celebration: bool,
    rng: np.random.Generator,
) -> PsychState:
    """Update psychological state from system interactions.

    Models how the system's own behavior affects patient engagement.
    An over-aggressive recommender causes burnout. A well-calibrated one
    builds a sustained upward trajectory.

    Args:
        psych: Current psychological state.
        personality: UserPersonality with response traits.
        rec_accepted: Did the user accept the most recent recommendation?
        rec_succeeded: Did the accepted recommendation improve outcomes?
        received_recommendation: Was a recommendation surfaced today?
        received_celebration: Was a celebration/positive observation surfaced?
        rng: Random number generator for stochastic responses.

    Returns:
        Updated PsychState.
    """
    new = PsychState(
        mood_valence=psych.mood_valence,
        recommendation_fatigue=psych.recommendation_fatigue,
        recent_rec_count=psych.recent_rec_count,
        burned_out=psych.burned_out,
    )

    # Mood boost from successful recommendations
    if rec_accepted and rec_succeeded:
        boost = personality.mood_boost_from_success * (1.0 + rng.normal(0, 0.02))
        new.mood_valence += boost

    # Mood hit from failed recommendations
    if rec_accepted and rec_succeeded is False:
        hit = personality.mood_hit_from_failure * (1.0 + rng.normal(0, 0.02))
        new.mood_valence += hit  # hit is negative

    # Recommendation fatigue and overload
    if received_recommendation:
        new.recent_rec_count += 1
        new.recommendation_fatigue += personality.recommendation_fatigue_rate * 0.15

        # Mood hit from overload (too many recs in a short period)
        if new.recent_rec_count > 3:
            overload_factor = (new.recent_rec_count - 3) / 5.0
            hit = personality.mood_hit_from_overload * overload_factor
            new.mood_valence += hit

    # Celebration boost
    if received_celebration:
        boost = personality.mood_boost_from_celebration * personality.celebration_receptiveness
        new.mood_valence += boost

    # Natural mood recovery toward 0 (mean reversion)
    new.mood_valence *= 0.95

    # Fatigue decay
    new.recommendation_fatigue *= 0.90

    # Clamp mood
    new.mood_valence = float(np.clip(new.mood_valence, -1.0, 1.0))

    # Burnout check
    if new.mood_valence < personality.burnout_threshold:
        new.burned_out = True
    elif new.mood_valence > personality.burnout_threshold + 0.15:
        new.burned_out = False

    return new


# ───────────────────────────────────────────────────────────────────
# Part 3b: Active Event Resolution
# ───────────────────────────────────────────────────────────────────

def get_active_events(
    schedule: EventSchedule,
    day_index: int,
) -> list[ActiveEvent]:
    """Return events active on the given day, with taper intensity."""
    active = []
    for event in schedule.events:
        total_window = event.duration_days + event.taper_days
        offset = day_index - event.start_day
        if offset < 0 or offset >= total_window:
            continue

        if offset < event.duration_days:
            # During the event — full intensity
            intensity = event.severity

            # Insomnia: gradual onset/offset
            if event.event_type == EventType.INSOMNIA_BOUT:
                # First night ramps in, last 2-3 nights ramp out
                if offset == 0:
                    intensity *= 0.6
                elif offset >= event.duration_days - 2:
                    days_left = event.duration_days - offset
                    intensity *= 0.4 + 0.3 * days_left
        else:
            # Taper period — linear decay
            taper_offset = offset - event.duration_days
            if event.taper_days > 0:
                intensity = event.severity * max(
                    0.0, 1.0 - taper_offset / event.taper_days
                )
            else:
                intensity = 0.0

        active.append(ActiveEvent(
            event=event,
            day_offset=offset,
            intensity=intensity,
        ))
    return active


def apply_event_modifiers(
    active_events: list[ActiveEvent],
    day_index: int,
) -> dict:
    """Aggregate modifiers from all active events into a single dict.

    Additive for stress/sleep penalties, multiplicative for ISF/exercise.
    """
    mods: dict = {
        "stress_add": 0.0,
        "sleep_penalty_min": 0.0,
        "sleep_efficiency_penalty": 0.0,
        "sleep_minutes_mult": 1.0,
        "exercise_prob_mult": 1.0,
        "meal_size_mult": 1.0,
        "meal_regularity_add": 0.0,
        "mood_offset": 0.0,
        "isf_mult_factor": 1.0,
        "egp_mult": 1.0,
        "is_ill_override": False,
        "sleep_schedule_shift_h": 0.0,
        "cgm_blackout": False,
        "watch_blackout": False,
        "logging_quality_mult": 1.0,
    }

    for ae in active_events:
        ev = ae.event
        intensity = ae.intensity
        p = ev.params

        if ev.event_type == EventType.INSOMNIA_BOUT:
            reduction = p.get("sleep_reduction_frac", 0.4) * intensity
            mods["sleep_minutes_mult"] *= (1.0 - reduction)
            mods["sleep_efficiency_penalty"] += p.get("efficiency_penalty", 0.20) * intensity

        elif ev.event_type == EventType.ILLNESS:
            mods["is_ill_override"] = True
            mods["egp_mult"] *= p.get("egp_mult", 1.40) ** intensity
            isf_penalty = 1.0 - (1.0 - p.get("isf_factor", 0.75)) * intensity
            mods["isf_mult_factor"] *= isf_penalty
            mods["exercise_prob_mult"] *= 0.0  # no exercise when sick
            mods["stress_add"] += 0.15 * intensity
            direction = p.get("appetite_direction", -1)
            mods["meal_size_mult"] *= 1.0 + direction * 0.15 * intensity

        elif ev.event_type == EventType.ACUTE_STRESS:
            mods["stress_add"] += p.get("stress_jump", 0.20) * intensity
            mods["mood_offset"] += p.get("mood_offset", -0.30) * intensity
            mods["exercise_prob_mult"] *= p.get("exercise_prob_mult", 0.6)
            mods["meal_regularity_add"] -= p.get("meal_regularity_drop", 0.20) * intensity

        elif ev.event_type == EventType.POSITIVE_LIFE_EVENT:
            mods["stress_add"] -= p.get("stress_relief", 0.10) * intensity
            mods["mood_offset"] += p.get("mood_offset", 0.18) * intensity
            mods["exercise_prob_mult"] *= p.get("exercise_prob_mult", 1.10)
            mods["meal_regularity_add"] += p.get("meal_regularity_boost", 0.08) * intensity
            mods["logging_quality_mult"] *= p.get("logging_quality_mult", 1.10)

        elif ev.event_type == EventType.TRAVEL:
            shift = p.get("timezone_shift_h", 3.0)
            # Jet lag recovery: 1 day per hour of offset
            hours_remaining = max(0, abs(shift) - ae.day_offset)
            mods["sleep_schedule_shift_h"] += math.copysign(hours_remaining, shift)
            mods["sleep_efficiency_penalty"] += p.get("efficiency_penalty", 0.15) * intensity
            ex_change = p.get("exercise_change", -0.3)
            mods["exercise_prob_mult"] *= max(0.1, 1.0 + ex_change * intensity)

        elif ev.event_type == EventType.MENSTRUAL_IRREGULARITY:
            # Amplified cycle sensitivity handled in behavior.py via params
            pass

        elif ev.event_type == EventType.INJURY:
            if ae.day_offset < ev.duration_days:
                # During injury: no exercise
                if ae.day_offset < 7:
                    mods["exercise_prob_mult"] *= 0.0
                else:
                    # After first week: light walks
                    mods["exercise_prob_mult"] *= 0.15
            else:
                # Recovery ramp
                ramp_days = p.get("ramp_back_days", 14)
                taper_into_ramp = min(1.0, ae.day_offset - ev.duration_days) / max(1, ramp_days)
                mods["exercise_prob_mult"] *= 0.5 + 0.5 * taper_into_ramp
            mods["mood_offset"] += p.get("mood_hit", -0.10) * intensity
            mods["meal_size_mult"] *= p.get("meal_size_boost", 1.10)

        elif ev.event_type == EventType.DEVICE_HIATUS:
            device = p.get("device", "watch")
            if device in ("watch", "both"):
                mods["watch_blackout"] = True
            if device in ("cgm", "both"):
                mods["cgm_blackout"] = True

        elif ev.event_type == EventType.VACATION:
            mods["meal_size_mult"] *= p.get("meal_size_mult", 1.20) * intensity
            mods["meal_regularity_add"] -= p.get("meal_regularity_drop", 0.25) * intensity
            mods["sleep_schedule_shift_h"] += p.get("sleep_shift_h", 1.0) * intensity
            ex_change = p.get("exercise_change", -0.3)
            mods["exercise_prob_mult"] *= max(0.1, 1.0 + ex_change * intensity)
            mods["logging_quality_mult"] *= p.get("logging_quality_mult", 0.65)

        elif ev.event_type == EventType.MEDICATION_CHANGE:
            mods["isf_mult_factor"] *= p.get("isf_factor", 1.0)
            # Beta-blocker masks HR (handled in observation layer)

        elif ev.event_type == EventType.SEASONAL_SHIFT:
            # Smooth sinusoidal modifier on exercise probability
            phase = p.get("phase_offset", 0.0)
            seasonal = 1.0 + 0.15 * math.sin(2 * math.pi * day_index / 365.0 + phase)
            mods["exercise_prob_mult"] *= seasonal

    # Clamp to prevent impossible values
    mods["sleep_minutes_mult"] = max(0.2, mods["sleep_minutes_mult"])
    mods["exercise_prob_mult"] = float(np.clip(mods["exercise_prob_mult"], 0.0, 2.0))
    mods["meal_size_mult"] = float(np.clip(mods["meal_size_mult"], 0.5, 2.0))
    mods["isf_mult_factor"] = float(np.clip(mods["isf_mult_factor"], 0.5, 1.5))
    mods["egp_mult"] = float(np.clip(mods["egp_mult"], 0.8, 2.0))
    mods["stress_add"] = float(np.clip(mods["stress_add"], -0.20, 0.50))
    mods["mood_offset"] = float(np.clip(mods["mood_offset"], -0.60, 0.20))

    return mods

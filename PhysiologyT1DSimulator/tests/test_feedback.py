"""Tests for the compounding feedback and slow-drift state system."""
import numpy as np
import pytest

from t1d_sim.feedback import (
    EventSchedule,
    EventType,
    LifeEvent,
    PatientState,
    YesterdayOutcome,
    apply_daily_feedback,
    apply_event_modifiers,
    compute_yesterday_outcome,
    get_active_events,
    sample_life_events,
    update_patient_state,
)
from t1d_sim.population import sample_population
from t1d_sim.simulate import SimulationCarryState, simulate_day


def _make_outcome(**overrides) -> YesterdayOutcome:
    defaults = dict(
        mean_bg=140.0, tir=0.65, percent_low=0.03, percent_high=0.10,
        sleep_minutes=420.0, exercise_minutes=30.0, stress=0.2,
        mood_valence=0.1, total_carbs=200.0, consecutive_bad_days=0,
    )
    defaults.update(overrides)
    return YesterdayOutcome(**defaults)


# ── compute_yesterday_outcome ──

def test_compute_yesterday_outcome_normal():
    bg = np.full(288, 120.0)
    beh = {"meals": [(None, 50.0, None)], "sleep_minutes": 400.0,
           "exercise_minutes": 30.0, "context": type("C", (), {"stress": 0.2, "mood_valence": 0.1})()}
    out = compute_yesterday_outcome(bg, beh, None)
    assert out.mean_bg == pytest.approx(120.0)
    assert out.tir == pytest.approx(1.0)
    assert out.percent_low == pytest.approx(0.0)
    assert out.consecutive_bad_days == 0


def test_compute_yesterday_outcome_streak():
    bg = np.full(288, 220.0)  # all high → TIR ~0
    beh = {"meals": [], "sleep_minutes": 400.0, "exercise_minutes": 0.0,
           "context": type("C", (), {"stress": 0.3, "mood_valence": -0.2})()}
    prev = _make_outcome(consecutive_bad_days=2)
    out = compute_yesterday_outcome(bg, beh, prev)
    assert out.tir < 0.55
    assert out.consecutive_bad_days == 3


def test_compute_yesterday_outcome_streak_reset():
    bg = np.full(288, 120.0)  # all in range → TIR=1.0
    beh = {"meals": [], "sleep_minutes": 400.0, "exercise_minutes": 0.0,
           "context": type("C", (), {"stress": 0.1, "mood_valence": 0.2})()}
    prev = _make_outcome(consecutive_bad_days=5)
    out = compute_yesterday_outcome(bg, beh, prev)
    assert out.tir > 0.65
    assert out.consecutive_bad_days == 0


# ── apply_daily_feedback ──

def test_daily_feedback_hyper():
    cfg = sample_population(1, seed=1)[0]
    outcome = _make_outcome(mean_bg=250.0, tir=0.30)
    rng = np.random.default_rng(42)
    mods = apply_daily_feedback(cfg, outcome, rng)
    assert mods.get("sleep_penalty_min", 0) > 0
    assert mods.get("stress_add", 0) > 0


def test_daily_feedback_hypo():
    cfg = sample_population(1, seed=1)[0]
    outcome = _make_outcome(percent_low=0.15)
    rng = np.random.default_rng(42)
    mods = apply_daily_feedback(cfg, outcome, rng)
    assert mods.get("sleep_penalty_min", 0) > 0
    assert mods.get("meal_size_mult", 1.0) > 1.0


def test_daily_feedback_consecutive_bad():
    cfg = sample_population(1, seed=1)[0]
    outcome = _make_outcome(consecutive_bad_days=5, tir=0.40)
    rng = np.random.default_rng(42)
    mods = apply_daily_feedback(cfg, outcome, rng)
    assert mods.get("stress_add", 0) >= 0.08 * (1.0 - 0.40) * cfg.stress_reactivity


# ── update_patient_state (biweekly drift) ──

def test_drift_isf_clamp():
    cfg = sample_population(1, seed=1)[0]
    state = PatientState.from_config(cfg)
    # 14 days of extreme surplus → ISF should degrade but stay in bounds
    outcomes = [_make_outcome(total_carbs=500.0, exercise_minutes=0.0) for _ in range(14)]
    new_state = update_patient_state(state, cfg, outcomes)
    assert new_state.effective_isf_mult >= cfg.isf_multiplier * 0.65
    assert new_state.effective_isf_mult <= cfg.isf_multiplier * 1.15


def test_drift_fitness_improves():
    cfg = sample_population(1, seed=1)[0]
    state = PatientState.from_config(cfg)
    # 14 days of daily exercise
    outcomes = [_make_outcome(exercise_minutes=45.0) for _ in range(14)]
    new_state = update_patient_state(state, cfg, outcomes)
    assert new_state.effective_fitness >= state.effective_fitness


def test_drift_fitness_degrades():
    cfg = sample_population(1, seed=1)[0]
    state = PatientState.from_config(cfg)
    # 14 days of no exercise
    outcomes = [_make_outcome(exercise_minutes=0.0) for _ in range(14)]
    new_state = update_patient_state(state, cfg, outcomes)
    assert new_state.effective_fitness <= state.effective_fitness


# ── sample_life_events ──

def test_life_events_max_two_major():
    """Ensure no more than 2 major events overlap on any given day."""
    cfg = sample_population(1, seed=1)[0]
    # Run many seeds to exercise the constraint
    for seed in range(20):
        rng = np.random.default_rng(seed)
        schedule = sample_life_events(cfg, 180, rng)
        for day in range(180):
            actives = get_active_events(schedule, day)
            major_count = sum(1 for ae in actives if ae.event.is_major)
            assert major_count <= 2, f"seed={seed}, day={day}: {major_count} major events"


def test_life_events_seasonal_always_present():
    cfg = sample_population(1, seed=1)[0]
    rng = np.random.default_rng(99)
    schedule = sample_life_events(cfg, 180, rng)
    seasonal = [e for e in schedule.events if e.event_type == EventType.SEASONAL_SHIFT]
    assert len(seasonal) == 1


def test_illness_duration_is_short_and_front_loaded():
    cfg = sample_population(1, seed=2)[0]
    durations: list[int] = []
    for seed in range(300):
        schedule = sample_life_events(cfg, 180, np.random.default_rng(seed))
        durations.extend(
            event.duration_days
            for event in schedule.events
            if event.event_type == EventType.ILLNESS
        )

    assert durations
    assert min(durations) >= 3
    assert max(durations) <= 10
    assert float(np.mean(durations)) < 5.2
    short = sum(1 for duration in durations if duration <= 4)
    long = sum(1 for duration in durations if duration >= 7)
    assert short > long


def test_life_events_menstrual_male_excluded():
    cfgs = sample_population(5, seed=10)
    for cfg in cfgs:
        if not cfg.is_female:
            rng = np.random.default_rng(42)
            schedule = sample_life_events(cfg, 180, rng)
            menstrual = [e for e in schedule.events
                         if e.event_type == EventType.MENSTRUAL_IRREGULARITY]
            assert len(menstrual) == 0


# ── Event taper ──

def test_event_taper_decay():
    event = LifeEvent(
        event_type=EventType.ILLNESS, start_day=10, duration_days=5,
        severity=1.0, taper_days=5, is_major=True,
        params={"egp_mult": 1.4, "isf_factor": 0.75, "appetite_direction": -1},
    )
    schedule = EventSchedule(events=[event])
    # During event: full intensity
    day12 = get_active_events(schedule, 12)
    assert len(day12) == 1
    assert day12[0].intensity == pytest.approx(1.0)
    # During taper: decaying intensity
    day17 = get_active_events(schedule, 17)
    assert len(day17) == 1
    assert 0 < day17[0].intensity < 1.0
    # After taper: no event
    day20 = get_active_events(schedule, 20)
    assert len(day20) == 0


# ── apply_event_modifiers ──

def test_illness_modifiers():
    event = LifeEvent(
        event_type=EventType.ILLNESS, start_day=0, duration_days=5,
        severity=0.8, taper_days=2, is_major=True,
        params={
            "egp_mult": 1.16,
            "isf_factor": 0.90,
            "exercise_prob_mult": 0.25,
            "stress_bump": 0.08,
            "appetite_swing": 0.08,
            "appetite_direction": -1,
        },
    )
    schedule = EventSchedule(events=[event])
    actives = get_active_events(schedule, 2)
    mods = apply_event_modifiers(actives, 2)
    assert mods["is_ill_override"] is True
    assert 0.05 <= mods["exercise_prob_mult"] < 1.0
    assert mods["egp_mult"] > 1.0
    assert mods["isf_mult_factor"] < 1.0
    assert mods["egp_mult"] < 1.35
    assert mods["isf_mult_factor"] >= 0.75


def test_illness_recovery_returns_close_to_baseline():
    from datetime import datetime, timedelta, timezone
    import copy

    cfg = next(c for c in sample_population(300, seed=11) if c.persona == "solid_sleeper")
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    base_cfg = copy.deepcopy(cfg)
    base_cfg.event_schedule = EventSchedule(events=[])

    illness_cfg = copy.deepcopy(cfg)
    illness_cfg.event_schedule = EventSchedule(events=[
        LifeEvent(
            event_type=EventType.ILLNESS,
            start_day=10,
            duration_days=3,
            severity=0.6,
            taper_days=0,
            is_major=True,
            params={
                "egp_mult": 1.14,
                "isf_factor": 0.92,
                "exercise_prob_mult": 0.35,
                "stress_bump": 0.06,
                "appetite_swing": 0.06,
                "appetite_direction": -1,
            },
        )
    ])

    def _run(run_cfg):
        carry = SimulationCarryState()
        rows = []
        for day in range(20):
            result = simulate_day(
                run_cfg,
                run_cfg.therapy_schedule,
                start + timedelta(days=day),
                day_index=day,
                carry_state=carry,
            )
            carry = result.carry_state
            bg = result.true_bg
            rows.append((
                float(np.mean(bg)),
                float(np.mean((bg >= 70) & (bg <= 180))),
            ))
        return rows

    base_rows = _run(base_cfg)
    illness_rows = _run(illness_cfg)

    illness_bg = float(np.mean([mean_bg for mean_bg, _ in illness_rows[10:13]]))
    recovery_bg = float(np.mean([mean_bg for mean_bg, _ in illness_rows[13:16]]))
    base_recovery_bg = float(np.mean([mean_bg for mean_bg, _ in base_rows[13:16]]))
    recovery_tir = float(np.mean([tir for _, tir in illness_rows[13:16]]))
    base_recovery_tir = float(np.mean([tir for _, tir in base_rows[13:16]]))

    assert recovery_bg < illness_bg - 25.0
    assert abs(recovery_bg - base_recovery_bg) < 5.0
    assert abs(recovery_tir - base_recovery_tir) < 0.05


# ── Integration: patient simulation with feedback ──

def test_simulate_patient_with_feedback():
    """Full integration test: simulate 30 days with feedback active."""
    from datetime import datetime, timezone
    from t1d_sim.patient import simulate_patient

    cfg = sample_population(1, seed=42)[0]
    cfg.n_days = 30
    payload = simulate_patient(cfg, 30, datetime(2025, 1, 1, tzinfo=timezone.utc))
    assert len(payload["bg_hourly"]) == 30 * 24
    assert len(payload["ground_truth"]) == 30
    # Ground truth should have 19 columns (including feedback cols)
    gt0 = payload["ground_truth"][0]
    assert len(gt0) == 19
    # effective_isf and effective_fitness should be populated
    assert gt0[16] is not None  # effective_isf
    assert gt0[17] is not None  # effective_fitness
    # BG values should be in valid range
    vals = [r[4] for r in payload["bg_hourly"] if r[4] is not None]
    assert len(vals) > 0
    assert min(vals) > 30

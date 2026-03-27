"""Daily closed-loop simulation primitives for Week 4.9."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
import uuid

import numpy as np

from t1d_sim.behavior import generate_day_behavior
from t1d_sim.feedback import (
    EventSchedule,
    PatientState,
    YesterdayOutcome,
    apply_event_modifiers,
    compute_yesterday_outcome,
    get_active_events,
    update_patient_state,
)
from t1d_sim.feature_frame import FeatureFrameHourly
from t1d_sim.missingness import generate_day_missingness, menstrual_is_missing, mood_event_count
from t1d_sim.observation import observe_cgm, synthesize_energy, synthesize_hr
from t1d_sim.physiology import apply_context_effectors, simulate_day_cgm
from t1d_sim.population import PatientConfig
from t1d_sim.therapy import TherapySchedule


def _iso_hour(dt: datetime) -> str:
    return dt.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")


def _iso_day(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")


@dataclass
class SimulationCarryState:
    """Mutable carry state threaded across historical days."""

    prev_mood: tuple[float, float] = (0.0, 0.0)
    last_outcome: YesterdayOutcome | None = None
    patient_state: PatientState | None = None
    drift_outcomes: list[YesterdayOutcome] = field(default_factory=list)
    sleep_totals_7d: list[float] = field(default_factory=list)
    last_exercise_hour: int | None = None
    bg_avgs_7d: list[float] = field(default_factory=list)
    tir_daily_7d: list[float] = field(default_factory=list)
    pct_low_daily_7d: list[float] = field(default_factory=list)
    pct_high_daily_7d: list[float] = field(default_factory=list)
    site_history: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DailySimResult:
    """Full per-day result bundle used by the closed-loop harness."""

    feature_frames: list[FeatureFrameHourly]
    decision_frame: FeatureFrameHourly
    mood_events: list[dict[str, Any]]
    site_change_event: dict[str, Any] | None
    bg_hourly: list[dict[str, Any]]
    bg_average_hourly: list[dict[str, Any]]
    bg_percent_hourly: list[dict[str, Any]]
    bg_uroc_hourly: list[dict[str, Any]]
    hr_hourly: list[dict[str, Any]]
    hr_daily_average: dict[str, Any]
    energy_hourly: list[dict[str, Any]]
    energy_daily_average: dict[str, Any]
    exercise_hourly: list[dict[str, Any]]
    exercise_daily_average: dict[str, Any]
    sleep_daily: dict[str, Any] | None
    menstrual_daily: dict[str, Any] | None
    site_daily: dict[str, Any]
    therapy_hourly: list[dict[str, Any]]
    mood_hourly: list[dict[str, Any]]
    carry_state: SimulationCarryState
    true_bg: np.ndarray
    observed_cgm: np.ndarray


def simulate_day(
    cfg: PatientConfig,
    schedule: TherapySchedule,
    date: datetime,
    rng_seed: int = 0,
    day_index: int = 0,
    carry_state: SimulationCarryState | None = None,
) -> DailySimResult:
    """Simulate one historical day under the current therapy schedule."""
    state = carry_state or SimulationCarryState()
    patient_state = state.patient_state or PatientState.from_config(cfg)
    event_schedule = cfg.event_schedule or EventSchedule()
    active_events = get_active_events(event_schedule, day_index)
    event_mods = apply_event_modifiers(active_events, day_index)

    behavior = generate_day_behavior(
        cfg,
        date,
        day_index,
        state.prev_mood,
        yesterday=state.last_outcome,
        event_modifiers=event_mods,
        patient_state=patient_state,
    )
    ctx = behavior["context"]
    next_prev_mood = (ctx.mood_valence, ctx.mood_arousal)

    # Gap 2: post-exercise late hypoglycemia window
    post_ex_hypo_window = False
    if state.last_exercise_hour is not None:
        if 14 <= state.last_exercise_hour <= 22:
            post_ex_hypo_window = True
    ctx.post_exercise_hypo_window = post_ex_hypo_window

    base = {"k1": 1.0, "k2": 1.0, "EGP0": 1.0}
    mod = apply_context_effectors(base, ctx, patient_state=patient_state, event_modifiers=event_mods)
    true_bg = simulate_day_cgm(
        base,
        mod,
        behavior["meals"],
        (cfg.seed or rng_seed or 1) * 10000 + day_index,
        therapy_schedule=schedule,
        post_exercise_hypo_window=ctx.post_exercise_hypo_window,
        dawn_sensitivity=getattr(cfg, "dawn_sensitivity", 0.0),
    )

    miss = cfg.missingness_profile
    miss_rng = np.random.default_rng((cfg.seed or rng_seed or 1) * 20000 + day_index)
    exercise_hour = behavior.get("exercise_hour", 17) or 17
    ex_bins = np.zeros(288, dtype=bool)
    if behavior["exercise_minutes"] > 0:
        ex_bins[exercise_hour * 12:(exercise_hour + 1) * 12] = True

    dm = generate_day_missingness(
        miss,
        day_index,
        ctx.is_weekend,
        exercise_hour if behavior["exercise_minutes"] > 0 else None,
        ex_bins,
        miss_rng,
    )
    if event_mods.get("watch_blackout", False):
        dm.watch_hourly_mask[:] = False
        dm.sleep_missing = True
        dm.exercise_captured = False
    if event_mods.get("cgm_blackout", False):
        dm.cgm_mask[:] = False

    sleep_total = float(behavior["sleep_minutes"])
    prior_sleep = state.sleep_totals_7d[-6:] + [sleep_total]
    sleep_debt = max(0.0, 8 * 60 * len(prior_sleep) - float(sum(prior_sleep)))

    cgm = observe_cgm(true_bg, dm, cfg.seed or rng_seed or 1, day_index)
    hr = synthesize_hr(
        cfg.base_rhr,
        ctx.stress,
        behavior["exercise_minutes"],
        behavior["exercise_intensity"],
        dm,
        day_index,
        cfg.seed or rng_seed or 1,
        exercise_hour=exercise_hour if behavior["exercise_minutes"] > 0 else None,
        sleep_debt_min=sleep_debt,
        stress_reactivity=cfg.stress_reactivity,
    )
    basal_arr, active_arr = synthesize_energy(
        cfg.base_rhr,
        schedule.weighted_mean("basal") / 0.85 if schedule.segments else cfg.basal_multiplier,
        behavior["exercise_minutes"],
        behavior["exercise_intensity"],
        dm,
        day_index,
        cfg.seed or rng_seed or 1,
        exercise_hour=exercise_hour if behavior["exercise_minutes"] > 0 else None,
    )

    skip_menstrual = menstrual_is_missing(miss, day_index, ctx.is_weekend, miss_rng)
    n_mood = mood_event_count(miss, day_index, ctx.is_weekend, miss_rng)

    mood_events = _build_mood_events(cfg, date, day_index, n_mood, ctx.mood_valence, ctx.mood_arousal)
    mood_hourly = _build_mood_hourly(date, mood_events)

    cgm_finite = cgm[np.isfinite(cgm)]
    if cgm_finite.size:
        daily_pct_low = float(np.mean(cgm_finite < 70) * 100.0)
        daily_pct_high = float(np.mean(cgm_finite > 180) * 100.0)
        daily_tir = max(0.0, 100.0 - daily_pct_low - daily_pct_high)
    else:
        daily_pct_low = daily_pct_high = daily_tir = None

    rolling_tir = _rolling_daily_mean(state.tir_daily_7d, daily_tir)
    rolling_pct_low = _rolling_daily_mean(state.pct_low_daily_7d, daily_pct_low)
    rolling_pct_high = _rolling_daily_mean(state.pct_high_daily_7d, daily_pct_high)
    wake_minute = int(max(0, min(1439, round((7.0 + cfg.sleep_schedule_offset_h) * 60))))

    bg_hourly: list[dict[str, Any]] = []
    bg_avg_hourly: list[dict[str, Any]] = []
    bg_pct_hourly: list[dict[str, Any]] = []
    bg_uroc_hourly: list[dict[str, Any]] = []
    hr_hourly: list[dict[str, Any]] = []
    energy_hourly: list[dict[str, Any]] = []
    exercise_hourly: list[dict[str, Any]] = []
    therapy_hourly: list[dict[str, Any]] = []
    frames: list[FeatureFrameHourly] = []

    prior_bg_series: list[float] = []
    prior_hr_series: list[float] = []
    prior_energy_series: list[float] = []
    prior_ex_series: list[float] = []

    for hour in range(24):
        hour_start = date + timedelta(hours=hour)
        hour_id = _iso_hour(hour_start)
        s = hour * 12
        e = s + 12
        vals = cgm[s:e]
        finite = vals[np.isfinite(vals)]

        if finite.size:
            start_bg = float(np.nan_to_num(vals[0], nan=float(finite[0])))
            end_bg = float(np.nan_to_num(vals[-1], nan=float(finite[-1])))
            avg_bg = float(np.nanmean(vals))
            pct_low = float(np.mean(finite < 70) * 100.0)
            pct_high = float(np.mean(finite > 180) * 100.0)
            uroc = float((end_bg - start_bg) / 60.0)
            predicted_bg = avg_bg + uroc * 60.0
        else:
            start_bg = end_bg = avg_bg = pct_low = pct_high = uroc = predicted_bg = None

        bg_hourly.append({
            "hourUtc": hour_id,
            "startUtc": hour_id,
            "endUtc": _iso_hour(hour_start + timedelta(hours=1)),
            "startBg": start_bg,
            "endBg": end_bg,
            "therapyProfileId": "profile_default",
        })
        bg_avg_hourly.append({
            "hourUtc": hour_id,
            "startUtc": hour_id,
            "endUtc": _iso_hour(hour_start + timedelta(hours=1)),
            "averageBg": avg_bg,
            "therapyProfileId": "profile_default",
        })
        bg_pct_hourly.append({
            "hourUtc": hour_id,
            "startUtc": hour_id,
            "endUtc": _iso_hour(hour_start + timedelta(hours=1)),
            "percentLow": pct_low,
            "percentHigh": pct_high,
            "therapyProfileId": "profile_default",
        })
        bg_uroc_hourly.append({
            "hourUtc": hour_id,
            "startUtc": hour_id,
            "endUtc": _iso_hour(hour_start + timedelta(hours=1)),
            "uRoc": uroc,
            "expectedEndBg": predicted_bg,
            "therapyProfileId": "profile_default",
        })

        hr_value = float(hr[hour]) if np.isfinite(hr[hour]) else None
        hr_hourly.append({"hourUtc": hour_id, "heartRate": hr_value, "therapyProfileId": "profile_default"})

        basal_energy = float(basal_arr[hour]) if np.isfinite(basal_arr[hour]) else None
        active_energy = float(active_arr[hour]) if np.isfinite(active_arr[hour]) else None
        total_energy = None if basal_energy is None or active_energy is None else basal_energy + active_energy
        energy_hourly.append({
            "hourUtc": hour_id,
            "basalEnergy": basal_energy,
            "activeEnergy": active_energy,
            "totalEnergy": total_energy,
            "therapyProfileId": "profile_default",
        })

        if not bool(dm.watch_hourly_mask[hour]):
            move_minutes = exercise_minutes = total_minutes = None
        elif dm.exercise_captured and abs(hour - exercise_hour) < 1:
            move_minutes = float(behavior["exercise_minutes"])
            exercise_minutes = float(behavior["exercise_minutes"])
            total_minutes = float(behavior["exercise_minutes"])
        else:
            move_minutes = exercise_minutes = total_minutes = 0.0
        exercise_hourly.append({
            "hourUtc": hour_id,
            "moveMinutes": move_minutes,
            "exerciseMinutes": exercise_minutes,
            "totalMinutes": total_minutes,
            "therapyProfileId": "profile_default",
        })

        seg = schedule.value_at_minute(hour * 60)
        therapy_hourly.append({
            "hourStartUtc": hour_id,
            "profileId": "profile_default",
            "profileName": "Default",
            "snapshotTimestamp": date.isoformat(),
            "carbRatio": seg.cr,
            "basalRate": seg.basal,
            "insulinSensitivity": seg.isf,
            "localTz": schedule.tz_identifier,
            "localHour": hour,
        })

        mood_ctx = mood_hourly.get(hour_id, {})
        bg_window = prior_bg_series[-7:] + ([avg_bg] if avg_bg is not None else [])
        hr_window = prior_hr_series[-7:] + ([hr_value] if hr_value is not None else [])
        energy_window = prior_energy_series[-7:] + ([active_energy] if active_energy is not None else [])
        ex_window = prior_ex_series[-3:] + ([exercise_minutes] if exercise_minutes is not None else [])

        frame = FeatureFrameHourly(
            hour_start_utc=hour_start.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc),
            bg_avg=avg_bg,
            bg_tir=rolling_tir,
            bg_percent_low=rolling_pct_low,
            bg_percent_high=rolling_pct_high,
            bg_u_roc=uroc,
            bg_delta_avg7h=_delta(bg_window),
            bg_z_avg7h=_zscore(bg_window),
            hr_mean=hr_value,
            hr_delta7h=_delta(hr_window),
            hr_z7h=_zscore(hr_window),
            rhr_daily=float(cfg.base_rhr),
            kcal_active=active_energy,
            kcal_active_last3h=_rolling_sum(prior_energy_series + ([active_energy] if active_energy is not None else []), 3),
            kcal_active_last6h=_rolling_sum(prior_energy_series + ([active_energy] if active_energy is not None else []), 6),
            kcal_active_delta7h=_delta(energy_window),
            kcal_active_z7h=_zscore(energy_window),
            sleep_prev_total_min=sleep_total if not dm.sleep_missing else None,
            sleep_debt_7d_min=sleep_debt,
            minutes_since_wake=max(0, hour * 60 - wake_minute),
            ex_move_min=move_minutes,
            ex_exercise_min=exercise_minutes,
            ex_min_last3h=_rolling_sum(ex_window, 3),
            ex_hours_since=_hours_since_exercise(hour, exercise_hour, behavior["exercise_minutes"]),
            days_since_period_start=ctx.cycle_day if (cfg.is_female and not skip_menstrual) else None,
            cycle_follicular=1 if ctx.cycle_phase == "follicular" else 0 if cfg.is_female else None,
            cycle_ovulation=1 if ctx.cycle_phase == "ovulation" else 0 if cfg.is_female else None,
            cycle_luteal=1 if ctx.cycle_phase == "luteal" else 0 if cfg.is_female else None,
            days_since_site_change=int(behavior["site_days_since_change"]),
            site_loc_current=behavior["site_location"],
            site_loc_same_as_last=_site_repeat(state.site_history, behavior["site_location"]),
            mood_valence=mood_ctx.get("valence"),
            mood_arousal=mood_ctx.get("arousal"),
            mood_quad_pos_pos=mood_ctx.get("quad_posPos"),
            mood_quad_pos_neg=mood_ctx.get("quad_posNeg"),
            mood_quad_neg_pos=mood_ctx.get("quad_negPos"),
            mood_quad_neg_neg=mood_ctx.get("quad_negNeg"),
            mood_hours_since=mood_ctx.get("hoursSinceMood"),
            day_of_week=(date.weekday() + 1) % 7,
        )
        frames.append(frame)

        if avg_bg is not None:
            prior_bg_series.append(avg_bg)
        if hr_value is not None:
            prior_hr_series.append(hr_value)
        if active_energy is not None:
            prior_energy_series.append(active_energy)
        if exercise_minutes is not None:
            prior_ex_series.append(exercise_minutes)

    sleep_daily = None
    if not dm.sleep_missing:
        multiplier = 0.6 if dm.sleep_partial else 1.0
        sleep_daily = {
            "dateUtc": _iso_day(date),
            "awake": (1 - cfg.sleep_efficiency) * behavior["sleep_minutes"] * multiplier,
            "asleepCore": 0.5 * behavior["sleep_minutes"] * multiplier,
            "asleepDeep": 0.2 * behavior["sleep_minutes"] * multiplier,
            "asleepREM": 0.23 * behavior["sleep_minutes"] * multiplier,
            "asleepUnspecified": 0.0,
        }

    menstrual_daily = None
    if cfg.is_female and not skip_menstrual:
        menstrual_daily = {"dateUtc": _iso_day(date), "daysSincePeriodStart": int(ctx.cycle_day)}

    site_daily = {
        "dateUtc": _iso_day(date),
        "daysSinceChange": int(behavior["site_days_since_change"]),
        "location": behavior["site_location"],
    }
    site_change_event = None
    if int(behavior["site_days_since_change"]) == 0:
        site_change_event = {
            "id": str(uuid.uuid4()),
            "location": behavior["site_location"],
            "localTz": schedule.tz_identifier,
            "timestamp": (date + timedelta(hours=12)).astimezone(timezone.utc),
        }

    bg_valid = [frame.bg_avg for frame in frames if frame.bg_avg is not None]
    day_outcome = compute_yesterday_outcome(true_bg, behavior, state.last_outcome)
    next_state = SimulationCarryState(
        prev_mood=next_prev_mood,
        last_outcome=day_outcome,
        patient_state=_updated_patient_state(patient_state, cfg, state.drift_outcomes, day_outcome),
        drift_outcomes=_updated_drift_outcomes(state.drift_outcomes, day_outcome),
        sleep_totals_7d=(state.sleep_totals_7d + [sleep_total])[-7:],
        last_exercise_hour=behavior.get("exercise_hour"),
        bg_avgs_7d=(state.bg_avgs_7d + ([float(np.mean(bg_valid))] if bg_valid else []))[-7:],
        tir_daily_7d=_append_daily_metric(state.tir_daily_7d, daily_tir),
        pct_low_daily_7d=_append_daily_metric(state.pct_low_daily_7d, daily_pct_low),
        pct_high_daily_7d=_append_daily_metric(state.pct_high_daily_7d, daily_pct_high),
        site_history=(state.site_history + [behavior["site_location"]])[-14:],
    )

    hr_daily_avg = np.mean([row["heartRate"] for row in hr_hourly if row["heartRate"] is not None]) if hr_hourly else None
    active_daily_avg = np.mean([row["activeEnergy"] for row in energy_hourly if row["activeEnergy"] is not None]) if energy_hourly else None
    exercise_daily_avg = np.mean([row["exerciseMinutes"] for row in exercise_hourly if row["exerciseMinutes"] is not None]) if exercise_hourly else None
    total_ex_daily_avg = np.mean([row["totalMinutes"] for row in exercise_hourly if row["totalMinutes"] is not None]) if exercise_hourly else None

    return DailySimResult(
        feature_frames=frames,
        decision_frame=frames[-1],
        mood_events=mood_events,
        site_change_event=site_change_event,
        bg_hourly=bg_hourly,
        bg_average_hourly=bg_avg_hourly,
        bg_percent_hourly=bg_pct_hourly,
        bg_uroc_hourly=bg_uroc_hourly,
        hr_hourly=hr_hourly,
        hr_daily_average={"dateUtc": _iso_day(date), "averageHeartRate": hr_daily_avg},
        energy_hourly=energy_hourly,
        energy_daily_average={"dateUtc": _iso_day(date), "averageActiveEnergy": active_daily_avg},
        exercise_hourly=exercise_hourly,
        exercise_daily_average={
            "dateUtc": _iso_day(date),
            "averageMoveMinutes": exercise_daily_avg,
            "averageExerciseMinutes": exercise_daily_avg,
            "averageTotalMinutes": total_ex_daily_avg,
        },
        sleep_daily=sleep_daily,
        menstrual_daily=menstrual_daily,
        site_daily=site_daily,
        therapy_hourly=therapy_hourly,
        mood_hourly=list(mood_hourly.values()),
        carry_state=next_state,
        true_bg=true_bg,
        observed_cgm=cgm,
    )


def _rolling_sum(values: list[float | None], window: int) -> float | None:
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return None
    return float(sum(finite[-window:]))


def _append_daily_metric(history: list[float], current: float | None, window: int = 7) -> list[float]:
    if current is None:
        return history[-window:]
    return (history + [float(current)])[-window:]


def _rolling_daily_mean(history: list[float], current: float | None, window: int = 7) -> float | None:
    values = _append_daily_metric(history, current, window)
    if not values:
        return None
    return float(np.mean(values))


def _delta(values: list[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None]
    if len(finite) < 2:
        return None
    return float(finite[-1] - finite[0])


def _zscore(values: list[float | None]) -> float | None:
    finite = np.asarray([float(v) for v in values if v is not None], dtype=float)
    if finite.size < 2:
        return None
    std = float(np.std(finite))
    if std <= 1e-6:
        return 0.0
    return float((finite[-1] - float(np.mean(finite))) / std)


def _hours_since_exercise(hour: int, exercise_hour: int, exercise_minutes: float) -> float | None:
    if exercise_minutes <= 0:
        return 24.0
    if hour < exercise_hour:
        return float(24 + hour - exercise_hour)
    return float(hour - exercise_hour)


def _site_repeat(site_history: list[str], current_location: str) -> int:
    if not site_history:
        return 0
    return 1 if site_history[-1] == current_location else 0


def _build_mood_events(
    cfg: PatientConfig,
    day_start: datetime,
    day_index: int,
    n_events: int,
    valence: float,
    arousal: float,
) -> list[dict[str, Any]]:
    if n_events <= 0:
        return []
    rng = np.random.default_rng((cfg.seed or 1) * 30000 + day_index)
    event_hours = sorted(rng.uniform(8, 22, size=n_events).tolist())
    return [
        {
            "id": str(uuid.uuid4()),
            "timestamp": (day_start + timedelta(hours=h)).astimezone(timezone.utc),
            "valence": float(valence),
            "arousal": float(arousal),
        }
        for h in event_hours
    ]


def _build_mood_hourly(day_start: datetime, mood_events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not mood_events:
        return {}
    mood_events = sorted(mood_events, key=lambda item: item["timestamp"])
    event_index = 0
    latest_event: dict[str, Any] | None = None
    hourly: dict[str, dict[str, Any]] = {}
    for hour in range(24):
        hour_start = day_start + timedelta(hours=hour)
        while event_index < len(mood_events) and mood_events[event_index]["timestamp"] <= hour_start:
            latest_event = mood_events[event_index]
            event_index += 1
        if latest_event is None:
            continue
        hours_since = max(0.0, (hour_start - latest_event["timestamp"]).total_seconds() / 3600.0)
        valence = float(latest_event["valence"])
        arousal = float(latest_event["arousal"])
        hourly[_iso_hour(hour_start)] = {
            "hourUtc": _iso_hour(hour_start),
            "valence": valence,
            "arousal": arousal,
            "quad_posPos": int(valence >= 0 and arousal >= 0),
            "quad_posNeg": int(valence >= 0 and arousal < 0),
            "quad_negPos": int(valence < 0 and arousal >= 0),
            "quad_negNeg": int(valence < 0 and arousal < 0),
            "hoursSinceMood": hours_since,
        }
    return hourly


def _updated_drift_outcomes(
    drift_outcomes: list[YesterdayOutcome],
    day_outcome: YesterdayOutcome,
) -> list[YesterdayOutcome]:
    updated = [*drift_outcomes, day_outcome]
    if len(updated) >= 14:
        return []
    return updated


def _updated_patient_state(
    patient_state: PatientState,
    cfg: PatientConfig,
    drift_outcomes: list[YesterdayOutcome],
    day_outcome: YesterdayOutcome,
) -> PatientState:
    updated = [*drift_outcomes, day_outcome]
    if len(updated) < 14:
        return patient_state
    return update_patient_state(patient_state, cfg, updated)

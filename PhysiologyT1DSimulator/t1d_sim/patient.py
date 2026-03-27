"""Per-patient simulation orchestrator."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import uuid
import numpy as np

from t1d_sim.behavior import generate_day_behavior
from t1d_sim.constants import THERAPY_PROFILE_ID, THERAPY_PROFILE_NAME
from t1d_sim.feedback import (
    EventSchedule,
    PatientState,
    YesterdayOutcome,
    apply_event_modifiers,
    compute_yesterday_outcome,
    get_active_events,
    update_patient_state,
)
from t1d_sim.missingness import (
    generate_day_missingness,
    menstrual_is_missing,
    mood_event_count,
)
from t1d_sim.observation import observe_cgm, synthesize_hr, synthesize_energy
from t1d_sim.physiology import apply_context_effectors, simulate_day_cgm
from t1d_sim.population import PatientConfig
from t1d_sim.therapy import make_default_schedule


def _iso_hour(dt: datetime) -> str:
    return dt.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")


def simulate_patient(cfg: PatientConfig, days: int, start_utc: datetime) -> dict:
    miss = cfg.missingness_profile
    bg_rows = []
    hr_rows = []
    energy_rows = []
    exercise_rows = []
    sleep_rows = []
    menstrual_rows = []
    site_rows = []
    therapy_rows = []
    mood_hourly_rows = []
    mood_events = []
    gt_rows = []
    prev_mood = (0.0, 0.0)

    # Feedback state
    state = PatientState.from_config(cfg)
    schedule = cfg.event_schedule or EventSchedule()
    outcome: YesterdayOutcome | None = None
    drift_outcomes: list[YesterdayOutcome] = []
    therapy_schedule = cfg.therapy_schedule or cfg.baseline_therapy_schedule or make_default_schedule(cfg)

    for d in range(days):
        day_start = start_utc + timedelta(days=d)

        # 1. Active events and modifiers
        active = get_active_events(schedule, d)
        event_mods = apply_event_modifiers(active, d)

        # 2. Build base with drifted ISF
        isf = state.effective_isf_with_fitness(cfg)
        base = {"k1": isf, "k2": isf, "EGP0": 1.0}

        # 3. Generate behavior with feedback + events
        beh = generate_day_behavior(
            cfg, day_start, d, prev_mood,
            yesterday=outcome,
            event_modifiers=event_mods,
            patient_state=state,
        )
        ctx = beh["context"]
        prev_mood = (ctx.mood_valence, ctx.mood_arousal)

        # 4. Apply physiology with events
        mod = apply_context_effectors(base, ctx, patient_state=state, event_modifiers=event_mods)
        true_bg = simulate_day_cgm(
            base,
            mod,
            beh["meals"],
            cfg.seed * 10000 + d,
            therapy_schedule=therapy_schedule,
        )

        # 5. Compute outcome for tomorrow's feedback
        outcome = compute_yesterday_outcome(true_bg, beh, outcome)
        drift_outcomes.append(outcome)

        # 6. Biweekly drift update
        if len(drift_outcomes) >= 14:
            state = update_patient_state(state, cfg, drift_outcomes)
            drift_outcomes = []

        miss_rng = np.random.default_rng(cfg.seed * 20000 + d)

        # Exercise active bins for CGM compression (mark bins around exercise window)
        exercise_hour = 17
        ex_bins = np.zeros(288, dtype=bool)
        if beh["exercise_minutes"] > 0:
            ex_bins[exercise_hour * 12:(exercise_hour + 1) * 12] = True

        # --- Block-structured missingness: single causal root ---
        dm = generate_day_missingness(
            miss, d, ctx.is_weekend,
            exercise_hour if beh["exercise_minutes"] > 0 else None,
            ex_bins, miss_rng,
        )

        # Device hiatus event overrides
        if event_mods.get("watch_blackout", False):
            dm.watch_hourly_mask[:] = False
            dm.sleep_missing = True
            dm.exercise_captured = False
        if event_mods.get("cgm_blackout", False):
            dm.cgm_mask[:] = False

        # CGM with block-structured gaps
        cgm = observe_cgm(true_bg, dm, cfg.seed, d)

        # HR with watch-derived missingness
        hr = synthesize_hr(cfg.base_rhr, ctx.stress,
                           beh["exercise_minutes"], beh["exercise_intensity"],
                           dm, d, cfg.seed)

        # Energy with watch-derived missingness
        basal_arr, active_arr = synthesize_energy(
            cfg.base_rhr, cfg.basal_multiplier,
            beh["exercise_minutes"], beh["exercise_intensity"],
            dm, d, cfg.seed)

        # Menstrual and mood — not block-structured, keep independent
        skip_menstrual = menstrual_is_missing(miss, d, ctx.is_weekend, miss_rng)
        n_mood = mood_event_count(miss, d, ctx.is_weekend, miss_rng)

        for h in range(24):
            s = h * 12
            e = s + 12
            ts = _iso_hour(day_start + timedelta(hours=h))

            vals = cgm[s:e]
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                start_bg = end_bg = avg_bg = None
                pct_low = pct_high = uroc = predicted_bg = None
            else:
                start_bg = float(np.nan_to_num(vals[0], nan=float(finite[0])))
                end_bg = float(np.nan_to_num(vals[-1], nan=float(finite[-1])))
                avg_bg = float(np.nanmean(vals))
                pct_low = float(np.mean(finite < 70))
                pct_high = float(np.mean(finite > 180))
                uroc = float((end_bg - start_bg) / 60.0)
                predicted_bg = avg_bg + uroc * 60.0
            bg_rows.append((cfg.patient_id, ts, start_bg, end_bg, avg_bg, pct_low, pct_high, uroc, predicted_bg, THERAPY_PROFILE_ID))

            hr_rows.append((cfg.patient_id, ts, float(hr[h]) if np.isfinite(hr[h]) else None))

            b_val = basal_arr[h]
            a_val = active_arr[h]
            b_out = float(b_val) if np.isfinite(b_val) else None
            a_out = float(a_val) if np.isfinite(a_val) else None
            t_out = (b_out + a_out) if (b_out is not None and a_out is not None) else None
            energy_rows.append((cfg.patient_id, ts, b_out, a_out, t_out))

            # Exercise: use DayMissingness for watch mask + capture status
            worn_h = bool(dm.watch_hourly_mask[h])
            if not worn_h:
                ex = None
            elif dm.exercise_captured and abs(h - exercise_hour) < 1:
                ex = beh["exercise_minutes"]
            else:
                ex = 0.0
            exercise_rows.append((cfg.patient_id, ts, ex, ex, ex))

            seg = therapy_schedule.value_at_minute(h * 60)
            therapy_rows.append((
                cfg.patient_id,
                ts,
                THERAPY_PROFILE_ID,
                THERAPY_PROFILE_NAME,
                seg.cr,
                seg.basal,
                seg.isf,
            ))
            mood_hourly_rows.append((cfg.patient_id, ts, ctx.mood_valence, ctx.mood_arousal, int(ctx.mood_valence >= 0 and ctx.mood_arousal >= 0), int(ctx.mood_valence >= 0 and ctx.mood_arousal < 0), int(ctx.mood_valence < 0 and ctx.mood_arousal >= 0), int(ctx.mood_valence < 0 and ctx.mood_arousal < 0), float(h)))

        dstr = day_start.strftime("%Y-%m-%d")

        # Sleep: derived from watch overnight status in DayMissingness
        if not dm.sleep_missing:
            if dm.sleep_partial:
                # Partial sleep data — reduced duration accuracy
                sleep_rows.append((cfg.patient_id, dstr, (1 - cfg.sleep_efficiency) * beh["sleep_minutes"] * 0.6, 0.5 * beh["sleep_minutes"] * 0.6, 0.2 * beh["sleep_minutes"] * 0.6, 0.23 * beh["sleep_minutes"] * 0.6, 0.0))
            else:
                sleep_rows.append((cfg.patient_id, dstr, (1 - cfg.sleep_efficiency) * beh["sleep_minutes"], 0.5 * beh["sleep_minutes"], 0.2 * beh["sleep_minutes"], 0.23 * beh["sleep_minutes"], 0.0))

        if cfg.is_female and not skip_menstrual:
            menstrual_rows.append((cfg.patient_id, dstr, ctx.cycle_day))
        site_rows.append((cfg.patient_id, dstr, beh["site_days_since_change"], beh["site_location"]))

        # Mood events — count driven by missingness model (logging quality + temporal decay)
        if n_mood > 0:
            mood_event_rng = np.random.default_rng(cfg.seed * 30000 + d)
            event_hours = sorted(mood_event_rng.uniform(8, 22, size=n_mood).tolist())
            for h in event_hours:
                ts_me = (day_start + timedelta(hours=h)).astimezone(timezone.utc)
                mood_events.append((cfg.patient_id, str(uuid.uuid4()), ts_me.strftime("%Y-%m-%dT%H:%M:%SZ"), ctx.mood_valence, ctx.mood_arousal))

        active_event_names = json.dumps([ae.event.event_type.value for ae in active])
        gt_rows.append((
            cfg.patient_id,
            dstr,
            45.0 * (mod["k1"] / base["k1"]),
            therapy_schedule.weighted_mean("cr"),
            therapy_schedule.weighted_mean("basal"),
            json.dumps([m[0].strftime("%H:%M") for m in beh["meals"]]),
            json.dumps([round(m[1], 1) for m in beh["meals"]]),
            int(beh["exercise_minutes"]),
            beh["exercise_intensity"],
            int(beh["sleep_minutes"]),
            str(ctx.cycle_phase),
            ctx.mood_valence,
            ctx.mood_arousal,
            ctx.stress,
            0,
            "",
            state.effective_isf_mult,
            state.effective_fitness,
            active_event_names,
        ))
    return {
        "patient": cfg.to_record(), "bg_hourly": bg_rows, "hr_hourly": hr_rows,
        "energy_hourly": energy_rows, "exercise_hourly": exercise_rows, "sleep_daily": sleep_rows,
        "menstrual_daily": menstrual_rows, "site_daily": site_rows, "therapy": therapy_rows,
        "mood_hourly": mood_hourly_rows, "mood_events": mood_events, "ground_truth": gt_rows,
    }

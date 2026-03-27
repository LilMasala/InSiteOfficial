"""Three-phase patient simulation with fork-of-forks timeline branching (Chamelia Block 2)."""
from __future__ import annotations

import copy
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from typing import TYPE_CHECKING

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

try:
    from chamelia.optimizer import (
        GridSearchOptimizer,
        TherapyAction,
        ObjectiveWeights,
        ConstraintConfig,
        RecommendationDecision,
    )
    from chamelia.shadow import ShadowModule
    from chamelia.confidence import ConfidenceModule
except ImportError:
    pass

if TYPE_CHECKING:
    from chamelia.models.base import PredictorCard


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------

class SimPhase(IntEnum):
    OBSERVATION = 0
    SHADOW = 1
    INTERVENTION = 2


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class PhaseConfig:
    """Controls the three-phase timeline and fork branching parameters.

    Attributes:
        obs_days:              Length of Phase 1 (observation only).
        shadow_days:           Length of Phase 2 (shadow mode — recommendations
                               generated but not shown).
        total_days:            Total simulation length.  Phase 3 = total -
                               obs_days - shadow_days.
        decision_interval:     Days between recommendation decision points in
                               Phase 3.
        fork_probability:      P(stochastic fork) at each decision point.
                               Expected branching factor ≈ 1 + fork_probability.
        max_depth:             Maximum bitstring length.  2^max_depth terminal
                               paths possible; default 8 → 256 max.
        convergence_threshold: Mean BG difference (mg/dL) below which a
                               sibling pair is considered converged and the
                               reject branch is pruned.
    """
    obs_days: int = 30
    shadow_days: int = 30
    total_days: int = 180
    decision_interval: int = 5
    fork_probability: float = 0.3
    max_depth: int = 8
    convergence_threshold: float = 5.0


# ---------------------------------------------------------------------------
# Branch state
# ---------------------------------------------------------------------------

@dataclass
class BranchState:
    """Mutable state carried by a single timeline branch through Phase 3.

    Attributes:
        path_id:           Bitstring of accept(1)/reject(0) decisions so far.
                           A trailing '+' denotes a branch that survived a
                           convergence prune (the sibling was dropped).
        depth:             Number of fork decisions made so far.
        isf_mult:          Current ISF multiplier for this branch.
        cr_mult:           Current CR multiplier.
        basal_mult:        Current basal multiplier.
        pre_int_isf_mult:  ISF multiplier at Phase 3 start (for revert).
        pre_int_cr_mult:   CR multiplier at Phase 3 start (for revert).
        pre_int_basal_mult:Basal multiplier at Phase 3 start (for revert).
        trust:             Current running trust level.
        prev_mood:         Last (valence, arousal) for mood AR(1) continuity.
        rows:              Shallow-copied row-lists from the Phase 1+2 checkpoint
                           extended with Phase 3 rows.  Tuples are immutable
                           so shallow copy is safe.
        accepted_recs:     Log of (day, proposed_action, accepted) tuples.
        pruned_siblings:   Log of (day, reason) for siblings pruned at this
                           branch's decision points.
    """
    path_id: str
    depth: int
    isf_mult: float
    cr_mult: float
    basal_mult: float
    pre_int_isf_mult: float
    pre_int_cr_mult: float
    pre_int_basal_mult: float
    trust: float
    prev_mood: tuple[float, float]
    rows: dict = field(default_factory=dict)
    accepted_recs: list = field(default_factory=list)
    pruned_siblings: list = field(default_factory=list)
    last_outcome: YesterdayOutcome | None = None
    patient_state: PatientState | None = None
    drift_outcomes: list = field(default_factory=list)
    shadow_records: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ROW_KEYS = (
    "bg_hourly", "hr_hourly", "energy_hourly", "exercise_hourly",
    "sleep_daily", "menstrual_daily", "site_daily", "therapy",
    "mood_hourly", "mood_events", "ground_truth",
)


def _iso_hour(dt: datetime) -> str:
    return dt.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc).strftime(
        "%Y-%m-%dT%H:00:00Z"
    )


def _build_daily_features(
    rows: dict,
    recommender: "PredictorCard",
    cfg: PatientConfig,
) -> tuple[np.ndarray, dict]:
    """Build a daily-aggregated feature vector from the last 24 h of branch data.

    Returns ``(feature_array, feature_dict)`` where *feature_array* follows
    the column order stored in ``recommender.feature_schema``.
    """
    bg_24 = rows.get("bg_hourly", [])[-24:]
    hr_24 = rows.get("hr_hourly", [])[-24:]
    energy_24 = rows.get("energy_hourly", [])[-24:]
    exercise_24 = rows.get("exercise_hourly", [])[-24:]
    mood_24 = rows.get("mood_hourly", [])[-24:]
    site_last = rows.get("site_daily", [])[-1:]
    menstrual_last = rows.get("menstrual_daily", [])[-1:]
    sleep_last = rows.get("sleep_daily", [])[-1:]

    # -- BG aggregates --
    bg_vals = [r[4] for r in bg_24 if r[4] is not None]
    pct_low = [r[5] for r in bg_24 if r[5] is not None]
    pct_high = [r[6] for r in bg_24 if r[6] is not None]
    uroc_vals = [r[7] for r in bg_24 if r[7] is not None]

    bg_avg = float(np.mean(bg_vals)) if bg_vals else 120.0
    bg_pct_low = float(np.mean(pct_low)) if pct_low else 0.05
    bg_pct_high = float(np.mean(pct_high)) if pct_high else 0.30
    bg_tir = max(0.0, 1.0 - bg_pct_low - bg_pct_high)
    bg_uroc = float(np.mean(uroc_vals)) if uroc_vals else 0.0

    if len(bg_vals) >= 7:
        tail7 = bg_vals[-7:]
        bg_delta = bg_vals[-1] - bg_vals[-7]
        bg_z = (bg_vals[-1] - float(np.mean(tail7))) / max(float(np.std(tail7)), 1e-6)
    else:
        bg_delta = 0.0
        bg_z = 0.0

    # -- HR aggregates --
    hr_vals = [r[2] for r in hr_24 if r[2] is not None]
    hr_mean = float(np.mean(hr_vals)) if hr_vals else cfg.base_rhr
    if len(hr_vals) >= 7:
        hr_tail = hr_vals[-7:]
        hr_delta = hr_vals[-1] - hr_vals[-7]
        hr_z = (hr_vals[-1] - float(np.mean(hr_tail))) / max(float(np.std(hr_tail)), 1e-6)
    else:
        hr_delta = 0.0
        hr_z = 0.0

    # -- Active energy --
    ae_vals = [r[3] for r in energy_24 if r[3] is not None]
    kcal_active = float(np.mean(ae_vals)) if ae_vals else 0.0
    kcal_last3 = float(np.sum(ae_vals[-3:])) if len(ae_vals) >= 3 else kcal_active * 3
    kcal_last6 = float(np.sum(ae_vals[-6:])) if len(ae_vals) >= 6 else kcal_active * 6

    # -- Sleep (previous night) --
    sleep_total = 0.0
    if sleep_last:
        sleep_total = sum(v for v in sleep_last[0][2:7] if v is not None and v > 0)

    # -- Exercise --
    ex_vals = [r[3] for r in exercise_24 if r[3] is not None]
    ex_exercise = float(np.mean(ex_vals)) if ex_vals else 0.0
    ex_last3 = float(np.sum(ex_vals[-3:])) if len(ex_vals) >= 3 else 0.0

    # -- Menstrual / cycle phase --
    days_period = 0
    cyc_f, cyc_o, cyc_l = 0, 0, 0
    if menstrual_last:
        days_period = menstrual_last[0][2] or 0
        if days_period <= 14:
            cyc_f = 1
        elif days_period <= 16:
            cyc_o = 1
        else:
            cyc_l = 1

    # -- Site --
    days_site = 0
    site_loc = "unknown"
    if site_last:
        days_site = site_last[0][2] or 0
        site_loc = site_last[0][3] or "unknown"

    site_enc = 0
    if hasattr(recommender, "_site_encoder") and recommender._site_encoder is not None:
        try:
            site_enc = int(recommender._site_encoder.transform([site_loc])[0])
        except (ValueError, KeyError):
            pass

    # -- Mood --
    mood_v = [r[2] for r in mood_24 if r[2] is not None]
    mood_a = [r[3] for r in mood_24 if r[3] is not None]
    m_val = float(np.mean(mood_v)) if mood_v else 0.0
    m_aro = float(np.mean(mood_a)) if mood_a else 0.0
    mood_hrs = [r[8] for r in mood_24 if r[8] is not None]
    m_hrs = float(np.mean(mood_hrs)) if mood_hrs else 1.0

    features = {
        "bg_avg": bg_avg,
        "bg_tir": bg_tir,
        "bg_percent_low": bg_pct_low,
        "bg_percent_high": bg_pct_high,
        "bg_uroc": bg_uroc,
        "bg_delta_avg_7h": bg_delta,
        "bg_z_avg_7h": bg_z,
        "hr_mean": hr_mean,
        "hr_delta_7h": hr_delta,
        "hr_z_7h": hr_z,
        "kcal_active": kcal_active,
        "kcal_active_last3h": kcal_last3,
        "kcal_active_last6h": kcal_last6,
        "sleep_prev_total_min": sleep_total,
        "ex_exercise_min": ex_exercise,
        "ex_min_last3h": ex_last3,
        "days_since_period_start": days_period,
        "cycle_follicular": cyc_f,
        "cycle_ovulation": cyc_o,
        "cycle_luteal": cyc_l,
        "days_since_site_change": days_site,
        "site_loc_same_as_last": 1,
        "mood_valence": m_val,
        "mood_arousal": m_aro,
        "mood_quad_pos_pos": int(m_val >= 0 and m_aro >= 0),
        "mood_quad_pos_neg": int(m_val >= 0 and m_aro < 0),
        "mood_quad_neg_pos": int(m_val < 0 and m_aro >= 0),
        "mood_quad_neg_neg": int(m_val < 0 and m_aro < 0),
        "mood_hours_since": m_hrs,
        "site_loc_current_enc": site_enc,
    }

    schema = getattr(recommender, "feature_schema", list(features.keys()))
    vec = np.array([features.get(col, 0.0) for col in schema], dtype=float)
    return vec, features


def _run_recommendation_cycle(
    rows: dict,
    recommender: "PredictorCard",
    cfg: PatientConfig,
    day_idx: int,
    isf_mult: float,
    cr_mult: float,
    basal_mult: float,
    optimizer: "GridSearchOptimizer",
    confidence_gate: "ConfidenceModule",
    shadow_mod: "ShadowModule",
    conservativeness: float = 0.5,
) -> tuple:
    """Run one recommendation cycle.

    Returns ``(shadow_record, recommendation_package)``.
    """
    features_vec, features_dict = _build_daily_features(rows, recommender, cfg)

    baseline = TherapyAction(
        isf_multiplier=isf_mult,
        cr_multiplier=cr_mult,
        basal_multiplier=basal_mult,
    )
    models = {recommender.model_id: recommender}
    aggressiveness = cfg.agency_profile.aggressiveness if cfg.agency_profile else 0.5

    rec_pkg = optimizer.search(
        features=features_vec,
        baseline_action=baseline,
        models=models,
        weights=ObjectiveWeights(conservativeness=conservativeness),
        constraints=ConstraintConfig(),
        aggressiveness=aggressiveness,
    )

    proposed_action = rec_pkg.primary if rec_pkg.primary else baseline
    proposed_env = recommender.predict(features_vec, action=proposed_action.to_array())
    baseline_env = recommender.predict(features_vec, action=baseline.to_array())

    proposed_envs = {recommender.model_id: proposed_env}
    baseline_envs = {recommender.model_id: baseline_env}

    familiarity = proposed_env.confidence
    calibration = shadow_mod.get_calibration_scores()

    gate = confidence_gate.evaluate(
        proposed_envelopes=proposed_envs,
        baseline_envelopes=baseline_envs,
        familiarity_score=familiarity,
        calibration_scores=calibration,
        user_aggressiveness=aggressiveness,
    )

    def _env_dict(env):
        return {
            "point": np.asarray(env.point).tolist(),
            "lower": np.asarray(env.lower).tolist(),
            "upper": np.asarray(env.upper).tolist(),
            "confidence": env.confidence,
        }

    record = shadow_mod.create_record(
        patient_id=cfg.patient_id,
        day_index=day_idx,
        feature_snapshot=features_dict,
        proposed_action=proposed_action.to_array().tolist(),
        baseline_action=baseline.to_array().tolist(),
        proposed_predictions={recommender.model_id: _env_dict(proposed_env)},
        baseline_predictions={recommender.model_id: _env_dict(baseline_env)},
        gate_passed=gate.passed,
        gate_composite_score=gate.composite_score,
        gate_layer_scores=gate.layer_scores,
        gate_blocked_by=gate.blocked_by,
        familiarity_score=familiarity,
        calibration_scores=calibration,
    )
    shadow_mod.add_record(record)

    return record, rec_pkg


def _run_day(
    cfg: PatientConfig,
    day_idx: int,
    day_start: datetime,
    prev_mood: tuple[float, float],
    isf_mult: float,
    cr_mult: float,
    basal_mult: float,
    phase: SimPhase,
    path_id: str = "",
    yesterday: YesterdayOutcome | None = None,
    patient_state: PatientState | None = None,
    event_modifiers: dict | None = None,
) -> tuple[dict, tuple[float, float], YesterdayOutcome]:
    """Simulate one calendar day for a branch.

    Unlike the original ``patient.py`` this function uses a fixed reference
    base of {k1=1.0, k2=1.0, EGP0=1.0} as the denominator in the
    sensitivity ratio so that changes to *isf_mult* produce actual BG
    differences between accept and reject branches.

    Returns:
        (row_dict, new_prev_mood, yesterday_outcome)
    """
    ev = event_modifiers or {}

    beh = generate_day_behavior(
        cfg, day_start, day_idx, prev_mood,
        yesterday=yesterday,
        event_modifiers=ev,
        patient_state=patient_state,
    )
    ctx = beh["context"]
    new_prev_mood = (ctx.mood_valence, ctx.mood_arousal)

    # Effective base carries the branch multipliers; reference base is the
    # denominator for the sensitivity ratio in simulate_day_cgm.
    effective_base = {"k1": isf_mult, "k2": isf_mult, "EGP0": 1.0}
    ref_base = {"k1": 1.0, "k2": 1.0, "EGP0": 1.0}
    mod = apply_context_effectors(effective_base, ctx, patient_state=patient_state, event_modifiers=ev)
    true_bg = simulate_day_cgm(ref_base, mod, beh["meals"], cfg.seed * 10000 + day_idx)

    miss = cfg.missingness_profile
    miss_rng = np.random.default_rng(cfg.seed * 20000 + day_idx)

    exercise_hour = 17
    ex_bins = np.zeros(288, dtype=bool)
    if beh["exercise_minutes"] > 0:
        ex_bins[exercise_hour * 12:(exercise_hour + 1) * 12] = True

    # Block-structured missingness: single causal root
    dm = generate_day_missingness(
        miss, day_idx, ctx.is_weekend,
        exercise_hour if beh["exercise_minutes"] > 0 else None,
        ex_bins, miss_rng,
    )

    # Device hiatus event overrides
    if ev.get("watch_blackout", False):
        dm.watch_hourly_mask[:] = False
        dm.sleep_missing = True
        dm.exercise_captured = False
    if ev.get("cgm_blackout", False):
        dm.cgm_mask[:] = False

    cgm = observe_cgm(true_bg, dm, cfg.seed, day_idx)
    hr = synthesize_hr(
        cfg.base_rhr, ctx.stress,
        beh["exercise_minutes"], beh["exercise_intensity"],
        dm, day_idx, cfg.seed,
    )
    basal_arr, active_arr = synthesize_energy(
        cfg.base_rhr, basal_mult,
        beh["exercise_minutes"], beh["exercise_intensity"],
        dm, day_idx, cfg.seed,
    )

    skip_menstrual = menstrual_is_missing(miss, day_idx, ctx.is_weekend, miss_rng)
    n_mood = mood_event_count(miss, day_idx, ctx.is_weekend, miss_rng)

    bg_rows: list = []
    hr_rows: list = []
    energy_rows: list = []
    exercise_rows: list = []
    therapy_rows: list = []
    mood_hourly_rows: list = []

    for h in range(24):
        s, e = h * 12, h * 12 + 12
        ts = _iso_hour(day_start + timedelta(hours=h))
        vals = cgm[s:e]
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            start_bg = end_bg = avg_bg = pct_low = pct_high = uroc = predicted_bg = None
        else:
            start_bg = float(np.nan_to_num(vals[0], nan=float(finite[0])))
            end_bg = float(np.nan_to_num(vals[-1], nan=float(finite[-1])))
            avg_bg = float(np.nanmean(vals))
            pct_low = float(np.mean(finite < 70))
            pct_high = float(np.mean(finite > 180))
            uroc = float((end_bg - start_bg) / 60.0)
            predicted_bg = avg_bg + uroc * 60.0
        bg_rows.append((cfg.patient_id, ts, start_bg, end_bg, avg_bg,
                         pct_low, pct_high, uroc, predicted_bg, THERAPY_PROFILE_ID))
        hr_rows.append((cfg.patient_id, ts, float(hr[h]) if np.isfinite(hr[h]) else None))

        b_val = basal_arr[h]
        a_val = active_arr[h]
        b_out = float(b_val) if np.isfinite(b_val) else None
        a_out = float(a_val) if np.isfinite(a_val) else None
        t_out = (b_out + a_out) if (b_out is not None and a_out is not None) else None
        energy_rows.append((cfg.patient_id, ts, b_out, a_out, t_out))

        worn_h = bool(dm.watch_hourly_mask[h])
        if not worn_h:
            ex = None
        elif dm.exercise_captured and abs(h - exercise_hour) < 1:
            ex = beh["exercise_minutes"]
        else:
            ex = 0.0
        exercise_rows.append((cfg.patient_id, ts, ex, ex, ex))

        therapy_rows.append((
            cfg.patient_id, ts, THERAPY_PROFILE_ID, THERAPY_PROFILE_NAME,
            12.0 * cr_mult, 0.85 * basal_mult, 45.0 / isf_mult,
        ))
        mood_hourly_rows.append((
            cfg.patient_id, ts, ctx.mood_valence, ctx.mood_arousal,
            int(ctx.mood_valence >= 0 and ctx.mood_arousal >= 0),
            int(ctx.mood_valence >= 0 and ctx.mood_arousal < 0),
            int(ctx.mood_valence < 0 and ctx.mood_arousal >= 0),
            int(ctx.mood_valence < 0 and ctx.mood_arousal < 0),
            float(h),
        ))

    dstr = day_start.strftime("%Y-%m-%d")
    sleep_rows: list = []
    if not dm.sleep_missing:
        if dm.sleep_partial:
            # Partial sleep data — reduced duration accuracy
            sleep_rows.append((
                cfg.patient_id, dstr,
                (1 - cfg.sleep_efficiency) * beh["sleep_minutes"] * 0.6,
                0.5 * beh["sleep_minutes"] * 0.6, 0.2 * beh["sleep_minutes"] * 0.6,
                0.23 * beh["sleep_minutes"] * 0.6, 0.0,
            ))
        else:
            sleep_rows.append((
                cfg.patient_id, dstr,
                (1 - cfg.sleep_efficiency) * beh["sleep_minutes"],
                0.5 * beh["sleep_minutes"], 0.2 * beh["sleep_minutes"],
                0.23 * beh["sleep_minutes"], 0.0,
            ))

    menstrual_rows: list = []
    if cfg.is_female and not skip_menstrual:
        menstrual_rows.append((cfg.patient_id, dstr, ctx.cycle_day))

    site_rows = [(cfg.patient_id, dstr, beh["site_days_since_change"], beh["site_location"])]

    mood_events: list = []
    if n_mood > 0:
        event_rng = np.random.default_rng(cfg.seed * 30000 + day_idx)
        for h in sorted(event_rng.uniform(8, 22, size=n_mood).tolist()):
            ts_me = (day_start + timedelta(hours=h)).astimezone(timezone.utc)
            mood_events.append((
                cfg.patient_id, str(uuid.uuid4()),
                ts_me.strftime("%Y-%m-%dT%H:%M:%SZ"),
                ctx.mood_valence, ctx.mood_arousal,
            ))

    eff_isf = patient_state.effective_isf_mult if patient_state else isf_mult
    eff_fit = patient_state.effective_fitness if patient_state else 0.0
    gt_rows = [(
        cfg.patient_id, dstr,
        45.0 * (mod["k1"] / effective_base["k1"]),  # context-adjusted ISF
        12.0 * cr_mult,
        0.85 * basal_mult,
        json.dumps([m[0].strftime("%H:%M") for m in beh["meals"]]),
        json.dumps([round(m[1], 1) for m in beh["meals"]]),
        int(beh["exercise_minutes"]),
        beh["exercise_intensity"],
        int(beh["sleep_minutes"]),
        str(ctx.cycle_phase),
        ctx.mood_valence,
        ctx.mood_arousal,
        ctx.stress,
        int(phase),
        path_id,
        eff_isf,
        eff_fit,
        json.dumps([]),  # active_events filled by caller if needed
    )]

    # Compute outcome for feedback
    day_outcome = compute_yesterday_outcome(true_bg, beh, yesterday)

    return {
        "bg_hourly": bg_rows,
        "hr_hourly": hr_rows,
        "energy_hourly": energy_rows,
        "exercise_hourly": exercise_rows,
        "sleep_daily": sleep_rows,
        "menstrual_daily": menstrual_rows,
        "site_daily": site_rows,
        "therapy": therapy_rows,
        "mood_hourly": mood_hourly_rows,
        "mood_events": mood_events,
        "ground_truth": gt_rows,
    }, new_prev_mood, day_outcome


def _enrich_shadow_record(
    record,
    day_rows: dict,
    user_action: str,
    settings: list[float],
) -> None:
    """Enrich a shadow record with today's actual BG outcomes and user decision.

    Fills Stage 2 (outcome) and computes shadow_score_delta (Stage 3) inline
    so that off-policy and surrogate evaluation methods have the data they need.
    """
    bg_today = day_rows["bg_hourly"]
    pl = [r[5] for r in bg_today if r[5] is not None]
    ph = [r[6] for r in bg_today if r[6] is not None]
    ab = [r[4] for r in bg_today if r[4] is not None]
    if not pl or not ph:
        return
    record.actual_outcomes = {
        "percent_low": float(np.mean(pl)),
        "percent_high": float(np.mean(ph)),
        "tir": max(0.0, 1.0 - float(np.mean(pl)) - float(np.mean(ph))),
        "mean_bg": float(np.mean(ab)) if ab else 120.0,
    }
    record.actual_user_action = user_action
    record.actual_settings = settings
    # Compute shadow_score_delta: did recommendation outperform baseline?
    if record.baseline_predictions:
        actual_tir = record.actual_outcomes["tir"]
        baseline_tirs = []
        for model_preds in record.baseline_predictions.values():
            if "point" in model_preds:
                p = model_preds["point"]
                if isinstance(p, (list, np.ndarray)) and len(p) > 2:
                    baseline_tirs.append(float(p[2]))
        if baseline_tirs:
            record.shadow_score_delta = actual_tir - float(np.mean(baseline_tirs))


def _branch_mean_bg(branch: BranchState, last_n_days: int = 3) -> float | None:
    """Return the mean avg_bg over the last *last_n_days* of bg_hourly rows."""
    rows = branch.rows.get("bg_hourly", [])
    if not rows:
        return None
    # Each day contributes 24 rows; grab last last_n_days * 24.
    tail = rows[-(last_n_days * 24):]
    vals = [r[4] for r in tail if r[4] is not None]
    return float(np.mean(vals)) if vals else None


def _assemble_payload(cfg: PatientConfig, branch: BranchState) -> dict:
    """Build the final return dict for a terminal branch."""
    payload = {
        "patient": cfg.to_record(),
        **branch.rows,
        "branch_meta": {
            "path_id": branch.path_id,
            "final_therapy": {
                "isf_multiplier": branch.isf_mult,
                "cr_multiplier": branch.cr_mult,
                "basal_multiplier": branch.basal_mult,
            },
            "pre_intervention_therapy": {
                "isf_multiplier": branch.pre_int_isf_mult,
                "cr_multiplier": branch.pre_int_cr_mult,
                "basal_multiplier": branch.pre_int_basal_mult,
            },
            "accepted_recs": branch.accepted_recs,
            "pruned_siblings": branch.pruned_siblings,
        },
    }
    # Attach shadow records if present.
    if hasattr(branch, "shadow_records"):
        payload["shadow_records"] = branch.shadow_records
    return payload


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_patient_threephase(
    cfg: PatientConfig,
    phase_cfg: PhaseConfig,
    start_utc: datetime,
    recommender: "PredictorCard | None" = None,
) -> list[dict]:
    """Run a three-phase simulation returning one payload per terminal branch.

    Phase 1 — Observation Only (days 0 .. obs_days):
        Normal simulation; data collected but no recommendations generated.

    Phase 2 — Shadow Active (days obs_days .. obs_days+shadow_days):
        Recommendations generated (if *recommender* provided) and logged
        silently.  Therapy settings are NOT altered.

    Phase 3 — Intervention (days obs_days+shadow_days .. total_days):
        Recommendations surface.  The user agency profile determines
        acceptance / partial compliance / rejection.  ``fork_timeline``
        branches the trajectory at each decision point.

    Returns:
        List of payload dicts, one per surviving terminal branch.  Each dict
        has the same keys as the original ``simulate_patient`` return value
        plus a ``"branch_meta"`` entry.
    """
    rng = np.random.default_rng(cfg.seed)
    p1p2_days = phase_cfg.obs_days + phase_cfg.shadow_days

    # --- Phase 1 + 2 linear pass -------------------------------------------
    rows: dict[str, list] = {k: [] for k in _ROW_KEYS}
    prev_mood: tuple[float, float] = (0.0, 0.0)
    state = PatientState.from_config(cfg)
    schedule = cfg.event_schedule or EventSchedule()
    outcome: YesterdayOutcome | None = None
    drift_outcomes: list[YesterdayOutcome] = []

    # Initialize Chamelia modeling stack if recommender is available.
    if recommender is not None:
        _optimizer = GridSearchOptimizer()
        _confidence_gate = ConfidenceModule()
        _shadow_mod = ShadowModule(window_size=phase_cfg.shadow_days)
        _shadow_records_p1: list = []
    else:
        _optimizer = _confidence_gate = _shadow_mod = None
        _shadow_records_p1 = []

    for d in range(p1p2_days):
        phase = SimPhase.OBSERVATION if d < phase_cfg.obs_days else SimPhase.SHADOW
        day_start = start_utc + timedelta(days=d)

        active_events = get_active_events(schedule, d)
        event_mods = apply_event_modifiers(active_events, d)

        day_rows, prev_mood, outcome = _run_day(
            cfg, d, day_start, prev_mood,
            cfg.isf_multiplier, cfg.cr_multiplier, cfg.basal_multiplier,
            phase, path_id="",
            yesterday=outcome,
            patient_state=state,
            event_modifiers=event_mods,
        )
        drift_outcomes.append(outcome)
        if len(drift_outcomes) >= 14:
            state = update_patient_state(state, cfg, drift_outcomes)
            drift_outcomes = []

        for k in _ROW_KEYS:
            rows[k].extend(day_rows[k])

        # Shadow phase: run recommendation cycle (log only, no action).
        if phase == SimPhase.SHADOW and _shadow_mod is not None:
            record, _ = _run_recommendation_cycle(
                rows, recommender, cfg, d,
                cfg.isf_multiplier, cfg.cr_multiplier, cfg.basal_multiplier,
                _optimizer, _confidence_gate, _shadow_mod,
            )
            _shadow_records_p1.append(record)
            # Enrich with today's actual outcomes.
            bg_today = day_rows["bg_hourly"]
            pl = [r[5] for r in bg_today if r[5] is not None]
            ph = [r[6] for r in bg_today if r[6] is not None]
            ab = [r[4] for r in bg_today if r[4] is not None]
            if pl and ph:
                actual = {
                    "percent_low": float(np.mean(pl)),
                    "percent_high": float(np.mean(ph)),
                    "tir": max(0.0, 1.0 - float(np.mean(pl)) - float(np.mean(ph))),
                    "mean_bg": float(np.mean(ab)) if ab else 120.0,
                }
                _shadow_mod.enrich_outcome(
                    record.record_id, actual, "not_shown",
                    [cfg.isf_multiplier, cfg.cr_multiplier, cfg.basal_multiplier],
                )
                _shadow_mod.evaluate_record(record.record_id)

    # Checkpoint: shallow copy — tuples are immutable so this is safe.
    checkpoint = {k: list(v) for k, v in rows.items()}
    checkpoint_prev_mood = prev_mood

    # Phase 2 baseline TIR (used for revert threshold in fork_timeline).
    shadow_bg_rows = [
        r for r in checkpoint["bg_hourly"]
        if r[4] is not None
    ]
    if shadow_bg_rows:
        shadow_pct_low = float(np.mean([r[5] for r in shadow_bg_rows if r[5] is not None]))
        shadow_pct_high = float(np.mean([r[6] for r in shadow_bg_rows if r[6] is not None]))
        phase2_baseline_tir = max(0.0, 1.0 - shadow_pct_low - shadow_pct_high)
    else:
        phase2_baseline_tir = 0.65  # population prior

    # Checkpoint feedback state for Phase 3 branches
    checkpoint_state = copy.deepcopy(state)
    checkpoint_outcome = outcome
    checkpoint_drift_outcomes = list(drift_outcomes)

    # Compute Phase 1 scorecard for graduation check.
    _scorecard = None
    _graduated = False
    if _shadow_mod is not None and _shadow_records_p1:
        _scorecard = _shadow_mod.compute_scorecard()
        _graduated = _shadow_mod.check_graduation(_scorecard)

    # --- Phase 3 forking ----------------------------------------------------
    if phase_cfg.total_days <= p1p2_days:
        # No Phase 3: return single payload with shared data only.
        branch = BranchState(
            path_id="",
            depth=0,
            isf_mult=cfg.isf_multiplier,
            cr_mult=cfg.cr_multiplier,
            basal_mult=cfg.basal_multiplier,
            pre_int_isf_mult=cfg.isf_multiplier,
            pre_int_cr_mult=cfg.cr_multiplier,
            pre_int_basal_mult=cfg.basal_multiplier,
            trust=cfg.agency_profile.initial_trust if cfg.agency_profile else 0.5,
            prev_mood=checkpoint_prev_mood,
            rows=checkpoint,
            last_outcome=checkpoint_outcome,
            patient_state=checkpoint_state,
            drift_outcomes=checkpoint_drift_outcomes,
        )
        branch.shadow_records = list(_shadow_records_p1)
        payloads = [_assemble_payload(cfg, branch)]
        if _scorecard is not None:
            payloads[0]["scorecard"] = _scorecard
        return payloads

    branches = fork_timeline(
        cfg=cfg,
        phase_cfg=phase_cfg,
        checkpoint=checkpoint,
        checkpoint_prev_mood=checkpoint_prev_mood,
        phase2_baseline_tir=phase2_baseline_tir,
        start_utc=start_utc,
        recommender=recommender,
        rng=rng,
        checkpoint_state=checkpoint_state,
        checkpoint_outcome=checkpoint_outcome,
        checkpoint_drift_outcomes=checkpoint_drift_outcomes,
        event_schedule=schedule,
        optimizer=_optimizer,
        confidence_gate=_confidence_gate,
        shadow_mod=_shadow_mod,
        shadow_records_phase1=_shadow_records_p1,
        graduated=_graduated,
    )
    # Attach scorecard to first payload for DB persistence.
    if _scorecard is not None and branches:
        branches[0]["scorecard"] = _scorecard
    return branches


def fork_timeline(
    cfg: PatientConfig,
    phase_cfg: PhaseConfig,
    checkpoint: dict,
    checkpoint_prev_mood: tuple[float, float],
    phase2_baseline_tir: float,
    start_utc: datetime,
    recommender: "PredictorCard | None",
    rng: np.random.Generator,
    checkpoint_state: PatientState | None = None,
    checkpoint_outcome: YesterdayOutcome | None = None,
    checkpoint_drift_outcomes: list | None = None,
    event_schedule: EventSchedule | None = None,
    optimizer=None,
    confidence_gate=None,
    shadow_mod=None,
    shadow_records_phase1: list | None = None,
    graduated: bool = False,
) -> list[dict]:
    """Branch the Phase 3 trajectory at stochastic decision points.

    At each *decision_interval* days, with probability *fork_probability*:
    - Two child branches are created: accept (path_id += "1") and reject (path_id += "0").
    - Accept branch applies the recommended action with compliance noise.
    - Reject branch keeps the current therapy settings unchanged.

    Pruning:
    - If the mean BG of an accept/reject sibling pair differs by less than
      *convergence_threshold* mg/dL over the last 3 days, the reject branch is
      dropped and the accept branch's path_id receives a trailing "+" to
      document the prune.  This is recorded in branch_meta["pruned_siblings"]
      so that training pipelines can filter or flag pruned paths.

    Revert logic:
    - If a branch's rolling TIR drops more than *agency.revert_threshold* below
      *phase2_baseline_tir*, the multipliers are restored to their Phase 3
      start values (pre_int_*).  This is the "pre-intervention baseline",
      i.e. the settings at the moment Phase 3 began — not the per-interval
      previous settings.

    Returns:
        List of assembled payload dicts, one per surviving terminal branch.
    """
    p1p2_days = phase_cfg.obs_days + phase_cfg.shadow_days
    phase3_len = phase_cfg.total_days - p1p2_days
    agency = cfg.agency_profile

    schedule = event_schedule or EventSchedule()

    # Initialise single root branch.
    root = BranchState(
        path_id="",
        depth=0,
        isf_mult=cfg.isf_multiplier,
        cr_mult=cfg.cr_multiplier,
        basal_mult=cfg.basal_multiplier,
        pre_int_isf_mult=cfg.isf_multiplier,
        pre_int_cr_mult=cfg.cr_multiplier,
        pre_int_basal_mult=cfg.basal_multiplier,
        trust=agency.initial_trust if agency else 0.5,
        prev_mood=checkpoint_prev_mood,
        rows={k: list(v) for k, v in checkpoint.items()},
        last_outcome=checkpoint_outcome,
        patient_state=copy.deepcopy(checkpoint_state) if checkpoint_state else PatientState.from_config(cfg),
        drift_outcomes=list(checkpoint_drift_outcomes or []),
    )
    active: list[BranchState] = [root]
    root.shadow_records = list(shadow_records_phase1 or [])
    _conserv = 0.3 if graduated else 0.5

    # Pre-compute set of decision days (relative to global day index).
    n_decisions = math.ceil(phase3_len / phase_cfg.decision_interval)
    decision_day_set = {
        p1p2_days + i * phase_cfg.decision_interval
        for i in range(n_decisions)
    }

    for d in range(p1p2_days, phase_cfg.total_days):
        day_start = start_utc + timedelta(days=d)
        next_active: list[BranchState] = []

        # Resolve events once per day (shared across branches)
        active_events = get_active_events(schedule, d)
        event_mods = apply_event_modifiers(active_events, d)

        for branch in active:
            # --- Simulate one day for this branch ---------------------------
            day_rows, new_prev_mood, day_outcome = _run_day(
                cfg, d, day_start, branch.prev_mood,
                branch.isf_mult, branch.cr_mult, branch.basal_mult,
                SimPhase.INTERVENTION, path_id=branch.path_id,
                yesterday=branch.last_outcome,
                patient_state=branch.patient_state,
                event_modifiers=event_mods,
            )
            for k in _ROW_KEYS:
                branch.rows[k].extend(day_rows[k])
            branch.prev_mood = new_prev_mood
            branch.last_outcome = day_outcome
            branch.drift_outcomes.append(day_outcome)

            # Biweekly drift (per-branch)
            if len(branch.drift_outcomes) >= 14:
                branch.patient_state = update_patient_state(
                    branch.patient_state, cfg, branch.drift_outcomes,
                )
                branch.drift_outcomes = []

            # --- Decision point? --------------------------------------------
            if d not in decision_day_set:
                next_active.append(branch)
                continue

            # Run the full recommendation cycle through the modeling stack.
            current_action = np.array([branch.isf_mult, branch.cr_mult, branch.basal_mult])
            has_recommendation = False
            try:
                if recommender is not None and optimizer is not None:
                    record, rec_pkg = _run_recommendation_cycle(
                        branch.rows, recommender, cfg, d,
                        branch.isf_mult, branch.cr_mult, branch.basal_mult,
                        optimizer, confidence_gate, shadow_mod,
                        conservativeness=_conserv,
                    )
                    branch.shadow_records.append(record)
                    if (rec_pkg.decision == RecommendationDecision.RECOMMEND
                            and rec_pkg.primary is not None):
                        proposed_action = rec_pkg.primary.to_array()
                    else:
                        # Exploration: small perturbation for data generation.
                        proposed_action = np.clip(
                            current_action + rng.normal(0, 0.05, 3),
                            0.70, 1.35,
                        )
                    has_recommendation = True
                else:
                    proposed_action = current_action.copy()
            except Exception:
                proposed_action = current_action.copy()

            if not has_recommendation:
                next_active.append(branch)
                continue

            # Stochastic fork decision.
            if rng.random() > phase_cfg.fork_probability or branch.depth >= phase_cfg.max_depth:
                # No fork: single outcome determined by agency profile.
                if agency is not None:
                    days_in_p3 = d - p1p2_days + 1
                    engaged = rng.random() >= agency.engagement_decay * days_in_p3 / max(1, phase3_len)
                    prob_accept = branch.trust * agency.aggressiveness if engaged else 0.0
                    accepted = rng.random() < prob_accept
                else:
                    accepted = False

                if accepted:
                    noise = np.clip(rng.normal(1.0, (agency.compliance_noise if agency else 0.1), 3), 0.5, 1.5)
                    applied = np.clip(proposed_action * noise, 0.70, 1.35)
                    branch.isf_mult, branch.cr_mult, branch.basal_mult = float(applied[0]), float(applied[1]), float(applied[2])
                    if agency:
                        branch.trust = min(1.0, branch.trust + agency.trust_growth_rate)
                    branch.accepted_recs.append((d, proposed_action.tolist(), True))
                else:
                    branch.accepted_recs.append((d, proposed_action.tolist(), False))

                # Enrich Phase 3 shadow record with accept/reject outcome.
                if shadow_mod is not None:
                    _enrich_shadow_record(
                        record, day_rows,
                        "accept" if accepted else "reject",
                        [branch.isf_mult, branch.cr_mult, branch.basal_mult],
                    )

                next_active.append(branch)
            else:
                # Fork into accept (1) and reject (0) child branches.
                # Accept branch: apply recommendation with compliance noise.
                accept_b = BranchState(
                    path_id=branch.path_id + "1",
                    depth=branch.depth + 1,
                    isf_mult=branch.isf_mult,
                    cr_mult=branch.cr_mult,
                    basal_mult=branch.basal_mult,
                    pre_int_isf_mult=branch.pre_int_isf_mult,
                    pre_int_cr_mult=branch.pre_int_cr_mult,
                    pre_int_basal_mult=branch.pre_int_basal_mult,
                    trust=branch.trust,
                    prev_mood=branch.prev_mood,
                    rows={k: list(v) for k, v in branch.rows.items()},
                    accepted_recs=list(branch.accepted_recs),
                    pruned_siblings=list(branch.pruned_siblings),
                    last_outcome=branch.last_outcome,
                    patient_state=copy.deepcopy(branch.patient_state),
                    drift_outcomes=list(branch.drift_outcomes),
                    shadow_records=list(branch.shadow_records),
                )
                if agency is not None:
                    noise = np.clip(rng.normal(1.0, agency.compliance_noise, 3), 0.5, 1.5)
                    applied = np.clip(proposed_action * noise, 0.70, 1.35)
                    accept_b.isf_mult = float(applied[0])
                    accept_b.cr_mult = float(applied[1])
                    accept_b.basal_mult = float(applied[2])
                    accept_b.trust = min(1.0, accept_b.trust + agency.trust_growth_rate)
                else:
                    accept_b.isf_mult = float(proposed_action[0])
                    accept_b.cr_mult = float(proposed_action[1])
                    accept_b.basal_mult = float(proposed_action[2])
                accept_b.accepted_recs.append((d, proposed_action.tolist(), True))

                # Reject branch: keep current multipliers unchanged.
                reject_b = BranchState(
                    path_id=branch.path_id + "0",
                    depth=branch.depth + 1,
                    isf_mult=branch.isf_mult,
                    cr_mult=branch.cr_mult,
                    basal_mult=branch.basal_mult,
                    pre_int_isf_mult=branch.pre_int_isf_mult,
                    pre_int_cr_mult=branch.pre_int_cr_mult,
                    pre_int_basal_mult=branch.pre_int_basal_mult,
                    trust=branch.trust,
                    prev_mood=branch.prev_mood,
                    rows={k: list(v) for k, v in branch.rows.items()},
                    accepted_recs=list(branch.accepted_recs),
                    pruned_siblings=list(branch.pruned_siblings),
                    last_outcome=branch.last_outcome,
                    patient_state=copy.deepcopy(branch.patient_state),
                    drift_outcomes=list(branch.drift_outcomes),
                    shadow_records=list(branch.shadow_records),
                )
                reject_b.accepted_recs.append((d, proposed_action.tolist(), False))

                # Enrich forked shadow records with accept/reject outcomes.
                if shadow_mod is not None:
                    rec_accept = copy.deepcopy(record)
                    _enrich_shadow_record(
                        rec_accept, day_rows, "accept",
                        [accept_b.isf_mult, accept_b.cr_mult, accept_b.basal_mult],
                    )
                    accept_b.shadow_records[-1] = rec_accept

                    rec_reject = copy.deepcopy(record)
                    _enrich_shadow_record(
                        rec_reject, day_rows, "reject",
                        [reject_b.isf_mult, reject_b.cr_mult, reject_b.basal_mult],
                    )
                    reject_b.shadow_records[-1] = rec_reject

                next_active.extend([accept_b, reject_b])

        active = next_active

        # --- Convergence pruning (every 3rd decision day so branches diverge) -
        steps_into_p3 = (d - p1p2_days) // phase_cfg.decision_interval
        if (d in decision_day_set and len(active) > 1
                and steps_into_p3 > 0 and steps_into_p3 % 3 == 0):
            _prune_converged_siblings(active, phase_cfg.convergence_threshold)

        # --- Revert check ---------------------------------------------------
        if agency is not None:
            for branch in active:
                _check_revert(branch, phase2_baseline_tir, agency.revert_threshold)

    return [_assemble_payload(cfg, b) for b in active]


def _prune_converged_siblings(
    active: list[BranchState],
    threshold: float,
) -> None:
    """Drop reject branches whose mean BG has converged with their accept sibling.

    Pruning is done *in-place* on *active*.  Surviving accept branches receive
    a trailing "+" on their path_id, and the prune event is recorded in
    ``branch.pruned_siblings`` so training pipelines can identify and handle
    these paths (e.g. exclude from causal pair analysis).

    "+" suffix convention: a path_id of "110+" means decisions 1+2 accepted,
    decision 3 rejected, but at decision 4 the reject sibling was pruned.
    Training pipelines should filter out paths containing "+" when building
    strict counterfactual pairs.
    """
    # Build sibling pairs: paths differing only in the last character.
    path_map: dict[str, BranchState] = {b.path_id: b for b in active}
    to_remove: set[str] = set()

    for path_id, branch in list(path_map.items()):
        if not path_id or path_id[-1] != "1":
            continue
        reject_id = path_id[:-1] + "0"
        if reject_id not in path_map or reject_id in to_remove:
            continue
        accept_bg = _branch_mean_bg(branch, last_n_days=3)
        reject_bg = _branch_mean_bg(path_map[reject_id], last_n_days=3)
        if accept_bg is None or reject_bg is None:
            continue
        if abs(accept_bg - reject_bg) < threshold:
            to_remove.add(reject_id)
            branch.path_id += "+"
            branch.pruned_siblings.append({
                "day": len(branch.rows.get("ground_truth", [])),
                "reason": "convergence",
                "accept_mean_bg": round(accept_bg, 2),
                "reject_mean_bg": round(reject_bg, 2),
                "pruned_path_id": reject_id,
            })

    for path_id in to_remove:
        idx = next(i for i, b in enumerate(active) if b.path_id == path_id)
        active.pop(idx)


def _check_revert(
    branch: BranchState,
    phase2_baseline_tir: float,
    revert_threshold: float,
) -> None:
    """Restore pre-intervention multipliers if recent TIR drops below threshold.

    "Pre-intervention" is always the Phase 3 start multipliers stored in
    ``branch.pre_int_*`` — NOT the previous decision-interval values.
    This avoids compounding rollbacks and makes the counterfactual clean.
    """
    bg_rows = branch.rows.get("bg_hourly", [])
    recent = bg_rows[-(5 * 24):]  # last 5 days
    if not recent:
        return
    pct_low_vals = [r[5] for r in recent if r[5] is not None]
    pct_high_vals = [r[6] for r in recent if r[6] is not None]
    if not pct_low_vals or not pct_high_vals:
        return
    current_tir = max(0.0, 1.0 - float(np.mean(pct_low_vals)) - float(np.mean(pct_high_vals)))
    if current_tir < phase2_baseline_tir - revert_threshold:
        branch.isf_mult = branch.pre_int_isf_mult
        branch.cr_mult = branch.pre_int_cr_mult
        branch.basal_mult = branch.pre_int_basal_mult

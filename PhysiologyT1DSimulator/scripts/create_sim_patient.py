"""Create or extend a mature synthetic InSite patient in Firebase."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

import firebase_admin
from firebase_admin import auth, credentials, firestore
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from t1d_sim.chamelia_client import ChameliaClient, ChameliaError
from t1d_sim.constants import PERSONAS
from t1d_sim.firebase_writer import FirebaseWriter
from t1d_sim.local_writer import LocalArtifactWriter, list_local_users
from t1d_sim.population import PatientConfig, sample_population
from t1d_sim.questionnaire import (
    ColdStartTargets,
    QuestionnaireAnswers,
    physical_priors_from_twins,
    questionnaire_to_patientconfig_priors,
    sample_twins_from_priors,
)
from t1d_sim.simulate import SimulationCarryState, simulate_day
from t1d_sim.therapy import SegmentDelta, StructureEdit, TherapySchedule, make_default_schedule

SIM_USER_TAG = "insite_sim_user"


class SimulationState:
    def __init__(
        self,
        cfg: PatientConfig,
        initial_schedule: TherapySchedule,
        profile_policy: str = "single",
        prior_profile_history: list[dict[str, Any]] | None = None,
        prior_recommendation_history: list[dict[str, Any]] | None = None,
    ) -> None:
        self.cfg = cfg
        self.current_schedule = initial_schedule
        self.profile_policy = profile_policy
        self.carry_state = SimulationCarryState()
        self.recommendation_history: list[dict[str, Any]] = list(prior_recommendation_history or [])
        self.profile_history: list[dict[str, Any]] = list(prior_profile_history or [{
            "timestamp": None,
            "event": "initial",
            "profile_name": "Default",
            "segment_count": len(initial_schedule.segments),
        }])
        self.allow_structural_recommendations = profile_policy in {"limited-multi", "multi"}

    def patient_response(self, recommendation: dict, day_index: int) -> tuple[str, float]:
        agency = self.cfg.agency_profile
        if agency is None:
            return "accept", 1.0

        momentum = 0.0
        recent_success = [r for r in self.recommendation_history[-5:] if r.get("accepted") and r.get("successful")]
        if recent_success:
            momentum = min(0.2, 0.04 * len(recent_success))

        magnitude = _action_magnitude(recommendation)
        accept_prob = float(max(0.05, min(0.95, agency.initial_trust + momentum - 0.35 * magnitude)))

        rng = random.Random((self.cfg.seed or 1) * 1000 + day_index)
        draw = rng.random()
        if draw < accept_prob * 0.7:
            compliance = max(0.3, min(1.0, 1.0 - agency.compliance_noise * rng.gauss(0.0, 1.0)))
            return "accept", compliance
        if draw < accept_prob:
            compliance = max(0.2, min(0.8, 0.5 - agency.compliance_noise * rng.gauss(0.0, 0.5)))
            return "partial", compliance
        return "reject", 0.0

    def apply_recommendation(self, recommendation: dict, compliance: float, timestamp: datetime) -> bool:
        action = recommendation.get("action", {})
        kind = action.get("kind")
        changed = False

        if kind == "scheduled":
            if action.get("segment_deltas"):
                deltas = [
                    SegmentDelta(
                        segment_id=str(delta["segment_id"]),
                        isf_delta=float(delta.get("isf_delta", 0.0)) * compliance,
                        cr_delta=float(delta.get("cr_delta", 0.0)) * compliance,
                        basal_delta=float(delta.get("basal_delta", 0.0)) * compliance,
                    )
                    for delta in action["segment_deltas"]
                ]
                self.current_schedule = self.current_schedule.apply_level1_action(deltas)
                changed = True
            if action.get("structural_edits") and self.allow_structural_recommendations:
                for edit in action["structural_edits"]:
                    self.current_schedule = self.current_schedule.apply_structural_proposal(
                        StructureEdit(
                            edit_type=str(edit["edit_type"]),
                            target_segment_id=str(edit["target_segment_id"]),
                            split_at_minute=None if edit.get("split_at_minute") is None else int(edit["split_at_minute"]),
                            neighbor_segment_id=None if edit.get("neighbor_segment_id") is None else str(edit["neighbor_segment_id"]),
                        )
                    )
                changed = True
        else:
            deltas = action.get("deltas", {})
            if any(abs(float(deltas.get(key, 0.0))) > 1e-9 for key in ("isf_delta", "cr_delta", "basal_delta")):
                self.current_schedule = self.current_schedule.apply_level1_action([
                    SegmentDelta(
                        segment_id=seg.segment_id,
                        isf_delta=float(deltas.get("isf_delta", 0.0)) * compliance,
                        cr_delta=float(deltas.get("cr_delta", 0.0)) * compliance,
                        basal_delta=float(deltas.get("basal_delta", 0.0)) * compliance,
                    )
                    for seg in self.current_schedule.segments
                ])
                changed = True

        if changed:
            self.profile_history.append({
                "timestamp": timestamp.isoformat(),
                "event": "recommendation_applied",
                "profile_name": "Default",
                "segment_count": len(self.current_schedule.segments),
            })
        return changed

    def to_connected_app_capabilities(self) -> dict:
        return {
            "app_id": "insite",
            "supports_scalar_schedule": True,
            "supports_piecewise_schedule": True,
            "supports_continuous_schedule": False,
            "max_segments": 8,
            "min_segment_duration_min": 120,
            "max_segments_addable": 2,
            "level_1_enabled": True,
            "level_2_enabled": self.allow_structural_recommendations,
            "level_3_enabled": False,
            "structural_change_requires_consent": True,
        }

    def to_connected_app_state(self) -> dict:
        return {
            "schedule_version": _schedule_version(self.current_schedule),
            "current_segments": [
                {
                    "segment_id": seg.segment_id,
                    "start_min": seg.start_min,
                    "end_min": seg.end_min,
                    "isf": seg.isf,
                    "cr": seg.cr,
                    "basal": seg.basal,
                }
                for seg in self.current_schedule.segments
            ],
            "allow_structural_recommendations": self.allow_structural_recommendations,
            "allow_continuous_schedule": False,
        }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.no_firebase:
        _init_firebase(args.service_account, args.bucket, args.project)

    if args.list:
        list_sim_users(args)
        return
    if args.delete:
        delete_sim_user(args)
        return

    uid, email = resolve_user(args)
    cfg = build_patient_config(args, uid)
    writer = make_writer(args, uid)
    existing_report = writer.load_latest_report() if args.append else None
    existing_entries = writer.load_sim_log_entries() if args.append else []
    writer.write_user_profile(cfg, email=email, namespace=args.namespace)

    initial_schedule = writer.load_latest_therapy_schedule() if args.append else None
    if initial_schedule is None:
        initial_schedule = cfg.therapy_schedule or cfg.baseline_therapy_schedule or make_default_schedule(cfg)
    sim_state = SimulationState(
        cfg,
        initial_schedule,
        profile_policy=args.profile_policy,
        prior_profile_history=list(existing_report.get("profile_history", [])) if existing_report else None,
        prior_recommendation_history=_recommendation_history_from_logs(existing_entries),
    )

    chamelia = ChameliaClient(args.chamelia_url, timeout=args.timeout)
    preferences = build_preferences(cfg, args)
    run_id = _run_id(uid, args, existing_entries)

    if args.append:
        try:
            chamelia.load(uid)
        except ChameliaError:
            chamelia.initialize(uid, preferences, weights_dir=args.weights_dir)
    else:
        chamelia.initialize(uid, preferences, weights_dir=args.weights_dir)
        bootstrap_now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        bootstrap_start = bootstrap_now - timedelta(days=args.days)
        writer.write_therapy_snapshot(sim_state.current_schedule, timestamp=bootstrap_start)
    chamelia.save(uid)

    now = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    latest_entry = existing_entries[-1] if existing_entries else None
    if args.append and latest_entry and latest_entry.get("date"):
        start_date = _parse_datetime(str(latest_entry["date"])).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        starting_day = int(latest_entry.get("day", 0))
    else:
        start_date = now - timedelta(days=args.days)
        starting_day = 0

    log_entries: list[dict] = []
    for day_index in range(args.days):
        cumulative_day = starting_day + day_index + 1
        current_date = start_date + timedelta(days=day_index)
        daily_result = simulate_day(
            cfg=cfg,
            schedule=sim_state.current_schedule,
            date=current_date,
            rng_seed=args.seed,
            day_index=day_index,
            carry_state=sim_state.carry_state,
        )
        sim_state.carry_state = daily_result.carry_state
        writer.write_daily_result(daily_result)

        signals = daily_result.decision_frame.to_signal_dict()
        timestamp = daily_result.decision_frame.hour_start_utc.timestamp()

        observe_response = chamelia.observe(uid, timestamp, signals)
        step_response = chamelia.step(
            uid,
            timestamp,
            signals,
            connected_app_capabilities=sim_state.to_connected_app_capabilities(),
            connected_app_state=sim_state.to_connected_app_state(),
        )
        recommendation = step_response.get("recommendation")
        response_name = None
        compliance = 0.0
        schedule_changed = False
        status = step_response.get("status") or observe_response.get("status") or {}
        realized_cost = compute_realized_cost(signals)

        rec_id = step_response.get("rec_id")
        if recommendation and rec_id is not None:
            response_name, compliance = sim_state.patient_response(recommendation, cumulative_day)
            if response_name in {"accept", "partial"}:
                schedule_changed = sim_state.apply_recommendation(recommendation, compliance, current_date)
                if schedule_changed:
                    writer.write_therapy_snapshot(sim_state.current_schedule, timestamp=current_date + timedelta(hours=23))

            chamelia.record_outcome(
                uid,
                int(rec_id),
                response_name,
                signals,
                realized_cost,
            )
            sim_state.recommendation_history.append({
                "day": cumulative_day,
                "date": current_date.date().isoformat(),
                "recommendation": recommendation,
                "response": response_name,
                "compliance": compliance,
                "accepted": response_name in {"accept", "partial"},
                "successful": _successful_day(daily_result),
                "action_kind": recommendation.get("action", {}).get("kind"),
                "action_level": recommendation.get("action_level"),
                "action_family": recommendation.get("action_family"),
                "confidence": recommendation.get("confidence"),
                "effect_size": recommendation.get("effect_size"),
                "predicted_improvement": recommendation.get("predicted_improvement"),
                "confidence_breakdown": recommendation.get("confidence_breakdown"),
                "predicted_outcomes": recommendation.get("predicted_outcomes"),
                "predicted_uncertainty": recommendation.get("predicted_uncertainty"),
                "segment_summaries": recommendation.get("segment_summaries") or [],
                "structure_summaries": recommendation.get("structure_summaries") or [],
            })
        elif rec_id is not None:
            chamelia.record_outcome(
                uid,
                int(rec_id),
                None,
                signals,
                realized_cost,
            )

        if rec_id is not None:
            latest_status = chamelia.graduation_status(uid).get("status") or {}
            if latest_status:
                status = latest_status

        chamelia.save(uid)

        log_entries.append({
            "run_id": run_id,
            "namespace": args.namespace,
            "day": cumulative_day,
            "date": current_date.isoformat(),
            "bg_avg": signals.get("bg_avg"),
            "tir_7d": signals.get("tir_7d"),
            "pct_low_7d": signals.get("pct_low_7d"),
            "pct_high_7d": signals.get("pct_high_7d"),
            "realized_cost": realized_cost,
            "mood_valence": signals.get("valence"),
            "mood_arousal": signals.get("arousal"),
            "stress_acute": signals.get("stress_acute"),
            "graduation_status": status,
            "recommendation": recommendation,
            "recommendation_returned": recommendation is not None,
            "patient_response": response_name,
            "compliance": compliance,
            "schedule_changed": schedule_changed,
            "action_kind": recommendation.get("action", {}).get("kind") if recommendation else None,
            "action_level": recommendation.get("action_level") if recommendation else None,
            "action_family": recommendation.get("action_family") if recommendation else None,
            "burnout_attribution": recommendation.get("burnout_attribution") if recommendation else None,
            "predicted_improvement": recommendation.get("predicted_improvement") if recommendation else None,
            "confidence_breakdown": recommendation.get("confidence_breakdown") if recommendation else None,
            "predicted_outcomes": recommendation.get("predicted_outcomes") if recommendation else None,
            "predicted_uncertainty": recommendation.get("predicted_uncertainty") if recommendation else None,
            "segment_summaries": recommendation.get("segment_summaries") if recommendation else None,
            "structure_summaries": recommendation.get("structure_summaries") if recommendation else None,
            "decision_withheld": bool(status.get("graduated")) and recommendation is None,
            "jepa_status": status.get("belief_mode"),
            "jepa_active": status.get("jepa_active"),
            "jepa_weights_loaded": status.get("jepa_weights_loaded"),
            "configurator_mode": status.get("configurator_mode"),
            "belief_entropy": status.get("belief_entropy"),
            "familiarity": status.get("familiarity"),
            "concordance": status.get("concordance"),
            "calibration": status.get("calibration"),
            "trust_level": status.get("trust_level"),
            "burnout_level": status.get("burnout_level"),
            "no_surface_streak": status.get("no_surface_streak"),
            "decision_block_reason": status.get("last_decision_reason"),
            "safety_diagnostics": status.get("last_safety_diagnostics"),
        })

        if args.verbose:
            print_progress(day_index + 1, args.days, current_date, log_entries[-1])

    writer.write_sim_log(log_entries)
    full_entries = writer.load_sim_log_entries()
    report = build_run_report(
        full_entries,
        sim_state,
        uid=uid,
        email=email,
        namespace=args.namespace,
        persona=cfg.persona,
        run_id=run_id,
        days_this_run=len(log_entries),
    )
    writer.write_run_report(report)
    report_path = _write_report_artifact(args, uid, report)

    print(f"Simulation complete for {uid}")
    if email and args.password and not args.no_firebase:
        print(f"Login with: {email} / {args.password}")
    print(f"Final status: {report['final_status']}")
    print(f"Accept+partial rate: {report['accept_or_partial_rate']:.3f}")
    print(f"Realized positive outcome rate: {report['realized_positive_outcome_rate']:.3f}")
    print(f"Graduated on day: {report['graduated_day']}")
    print(f"Recommendations surfaced: {report['recommendation_count']} (post-grad withheld: {report['post_graduation_no_surface_days']})")
    print(f"Report written to: {report_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="insitev2")
    parser.add_argument("--bucket", default="insitev2.appspot.com")
    parser.add_argument("--chamelia-url", default="https://chamelia-136217612465.us-central1.run.app")
    parser.add_argument("--namespace", default="dev-sim")
    parser.add_argument("--service-account", default=str(_default_service_account_path()))
    parser.add_argument("--no-firebase", action="store_true")
    parser.add_argument("--local-root", default=str(Path(__file__).resolve().parent / "local_runs"))
    parser.add_argument("--email")
    parser.add_argument("--password")
    parser.add_argument("--uid")
    parser.add_argument("--persona", default="athlete")
    parser.add_argument("--questionnaire")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--delete")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--profile-policy", choices=["single", "limited-multi", "multi"], default="single")
    parser.add_argument("--coldstart-targets", choices=["synthetic", "none"], default="synthetic")
    parser.add_argument("--weights-dir")
    parser.add_argument("--report-file")
    return parser


def _init_firebase(service_account_path: str, bucket: str, project: str) -> None:
    if firebase_admin._apps:
        return
    service_account = Path(service_account_path)
    if service_account.exists():
        cred = credentials.Certificate(service_account)
        firebase_admin.initialize_app(cred, {"storageBucket": bucket, "projectId": project})
        return
    firebase_admin.initialize_app(credentials.ApplicationDefault(), {"storageBucket": bucket, "projectId": project})


def _default_service_account_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        Path(__file__).with_name("service-account.json"),
        repo_root / "insitev2_chamelia.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_user(args) -> tuple[str, str | None]:
    if args.no_firebase:
        if args.uid:
            return args.uid, args.email
        if args.email:
            local_uid = _local_uid(args.email, args.namespace)
            return local_uid, args.email
        local_uid = _local_uid(f"{args.namespace}:{args.persona}:{args.seed}", args.namespace)
        return local_uid, None

    if args.uid:
        try:
            user = auth.get_user(args.uid)
            return user.uid, user.email
        except auth.UserNotFoundError as exc:
            raise SystemExit(f"Unknown uid: {args.uid}") from exc
    if not args.email or not args.password:
        raise SystemExit("Either --uid or both --email/--password are required")
    try:
        user = auth.get_user_by_email(args.email)
        return user.uid, user.email
    except auth.UserNotFoundError:
        user = auth.create_user(
            email=args.email,
            password=args.password,
            display_name=args.email.split("@")[0],
            email_verified=True,
        )
        firestore.client().collection("users").document(user.uid).set({
            "email": args.email,
            "displayName": args.email.split("@")[0],
            "isSim": True,
            "simTag": SIM_USER_TAG,
            "createdAt": firestore.SERVER_TIMESTAMP,
        }, merge=True)
        return user.uid, args.email


def build_patient_config(args, uid: str) -> PatientConfig:
    if args.questionnaire:
        answers = QuestionnaireAnswers.from_json(args.questionnaire)
        priors = questionnaire_to_patientconfig_priors(answers)
        twins = sample_twins_from_priors(priors, n=32, seed=args.seed)
        cfg = twins[0]
    else:
        cfg = sample_patient_for_persona(args.persona, args.seed)
    cfg.patient_id = uid
    cfg.seed = args.seed
    cfg.n_days = args.days
    return cfg


def sample_patient_for_persona(persona: str, seed: int) -> PatientConfig:
    if persona not in PERSONAS:
        raise SystemExit(f"Unknown persona `{persona}`")
    patients = sample_population(64, seed=seed)
    for patient in patients:
        if patient.persona == persona:
            return patient
    patient = patients[0]
    patient.persona = persona
    return patient


def build_preferences(cfg: PatientConfig, args) -> dict:
    prefs = {
        "aggressiveness": cfg.agency_profile.aggressiveness if cfg.agency_profile else 0.5,
        "hypoglycemia_fear": 0.9 if cfg.stress_reactivity > 0.6 else 0.6,
        "burden_sensitivity": 0.3,
        "persona": cfg.persona,
    }
    if getattr(args, "coldstart_targets", "synthetic") == "synthetic":
        targets = ColdStartTargets.from_patient_config(cfg).to_chamelia_targets()
        if targets:
            prefs["calibration_targets"] = targets
    if args.questionnaire:
        answers = QuestionnaireAnswers.from_json(args.questionnaire)
        priors = questionnaire_to_patientconfig_priors(answers)
        twins = sample_twins_from_priors(priors, n=32, seed=args.seed)
        prefs["physical_priors"] = {
            key: [float(values[0]), float(values[1])]
            for key, values in physical_priors_from_twins(twins).items()
        }
    return prefs


def compute_realized_cost(signals: dict) -> float:
    pct_low = float(signals.get("pct_low_7d", 0.0))
    pct_high = float(signals.get("pct_high_7d", 0.0))
    return pct_low * 5.0 + pct_high


def build_run_report(
    log_entries: list[dict[str, Any]],
    sim_state: SimulationState,
    *,
    uid: str,
    email: str | None,
    namespace: str,
    persona: str,
    run_id: str,
    days_this_run: int,
) -> dict[str, Any]:
    recs = [entry for entry in log_entries if entry.get("recommendation")]
    accepted = [r for r in recs if r.get("patient_response") in {"accept", "partial"}]
    accepted_exact = [r for r in recs if r.get("patient_response") == "accept"]
    partials = [r for r in recs if r.get("patient_response") == "partial"]
    rejected = [r for r in recs if r.get("patient_response") == "reject"]
    schedule_changed_entries = [r for r in recs if r.get("schedule_changed")]
    graduated_day = next((int(entry["day"]) for entry in log_entries if (entry.get("graduation_status") or {}).get("graduated")), None)
    post_grad_entries = [
        entry for entry in log_entries
        if graduated_day is not None and int(entry.get("day", 0)) >= graduated_day
    ]
    post_grad_withheld = [entry for entry in post_grad_entries if entry.get("recommendation") is None]
    post_grad_surfaced = [entry for entry in post_grad_entries if entry.get("recommendation") is not None]

    tirs = [float(entry["tir_7d"]) for entry in log_entries if entry.get("tir_7d") is not None]
    baseline_tirs = tirs[:14]
    final_tirs = tirs[-14:]
    lows = [float(entry.get("pct_low_7d", 0.0)) for entry in log_entries if entry.get("pct_low_7d") is not None]
    highs = [float(entry.get("pct_high_7d", 0.0)) for entry in log_entries if entry.get("pct_high_7d") is not None]
    jepa_entries = [entry for entry in log_entries if entry.get("jepa_active")]
    first_jepa_day = next((int(entry["day"]) for entry in log_entries if entry.get("jepa_active")), None)
    configurator_modes = Counter(
        str(entry.get("configurator_mode"))
        for entry in log_entries
        if entry.get("configurator_mode")
    )
    trend_series = _build_trend_series(log_entries)
    realized_outcomes = _recommendation_outcome_summary(log_entries)
    realized_positive_count = sum(1 for item in realized_outcomes if item.get("positive"))
    recommendations_with_followup = len(realized_outcomes)
    calibration_summary = _calibration_summary(recs, realized_outcomes)
    uncertainty_summary = _uncertainty_summary(recs)

    final_status = log_entries[-1].get("graduation_status") if log_entries else {}
    return {
        "run_id": run_id,
        "uid": uid,
        "email": email,
        "namespace": namespace,
        "persona": persona,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "days_total": len(log_entries),
        "days_this_run": days_this_run,
        "graduated_day": graduated_day,
        "recommendation_count": len(recs),
        "accepted_count": len(accepted_exact),
        "partial_count": len(partials),
        "rejected_count": len(rejected),
        "accept_or_partial_count": len(accepted),
        "accept_or_partial_rate": len(accepted) / max(len(recs), 1),
        "exact_accept_rate": len(accepted_exact) / max(len(recs), 1),
        "schedule_changed_count": len(schedule_changed_entries),
        "schedule_change_rate": len(schedule_changed_entries) / max(len(recs), 1),
        "recommendations_with_followup": recommendations_with_followup,
        "realized_positive_outcome_count": realized_positive_count,
        "realized_positive_outcome_rate": realized_positive_count / max(recommendations_with_followup, 1),
        # Backward-compatible alias; now defined in terms of explicit post-recommendation followup windows.
        "recommendation_success_rate": realized_positive_count / max(recommendations_with_followup, 1),
        "tir_mean": _safe_mean(tirs),
        "tir_baseline_14d_mean": _safe_mean(baseline_tirs),
        "tir_final_14d_mean": _safe_mean(final_tirs),
        "tir_delta_baseline_vs_final_14d": (
            _safe_mean(final_tirs) - _safe_mean(baseline_tirs)
        ),
        "pct_low_mean": _safe_mean(lows),
        "pct_high_mean": _safe_mean(highs),
        "accepted_recommendation_days": len(accepted),
        "jepa_status": final_status.get("belief_mode"),
        "jepa_active_days": len(jepa_entries),
        "first_jepa_day": first_jepa_day,
        "jepa_weights_loaded": bool(final_status.get("jepa_weights_loaded")),
        "competence_snapshot": {
            "belief_entropy": final_status.get("belief_entropy"),
            "familiarity": final_status.get("familiarity"),
            "concordance": final_status.get("concordance"),
            "calibration": final_status.get("calibration"),
            "trust_level": final_status.get("trust_level"),
            "burnout_level": final_status.get("burnout_level"),
            "no_surface_streak": final_status.get("no_surface_streak"),
        },
        "configurator_mode_counts": dict(configurator_modes),
        "post_graduation_days": len(post_grad_entries),
        "post_graduation_no_surface_days": len(post_grad_withheld),
        "post_graduation_surface_days": len(post_grad_surfaced),
        "block_reasons": _block_reason_counts(log_entries),
        "post_graduation_block_reasons": _block_reason_counts(post_grad_withheld),
        "trend_series": trend_series,
        "realized_outcome_timeline": realized_outcomes,
        "calibration_summary": calibration_summary,
        "uncertainty_summary": uncertainty_summary,
        "recommendation_timeline": [
            {
                "day": entry.get("day"),
                "date": entry.get("date"),
                "action_kind": entry.get("action_kind"),
                "action_level": entry.get("action_level"),
                "action_family": entry.get("action_family"),
                "patient_response": entry.get("patient_response"),
                "schedule_changed": entry.get("schedule_changed"),
                "realized_cost": entry.get("realized_cost"),
                "predicted_improvement": entry.get("predicted_improvement"),
                "confidence": entry.get("confidence"),
                "confidence_breakdown": entry.get("confidence_breakdown"),
                "predicted_outcomes": entry.get("predicted_outcomes"),
                "predicted_uncertainty": entry.get("predicted_uncertainty"),
                "segment_summaries": entry.get("segment_summaries") or [],
                "structure_summaries": entry.get("structure_summaries") or [],
            }
            for entry in recs
        ],
        "profile_history": sim_state.profile_history,
        "final_schedule": [
            {
                "segment_id": seg.segment_id,
                "start_min": seg.start_min,
                "end_min": seg.end_min,
                "isf": seg.isf,
                "cr": seg.cr,
                "basal": seg.basal,
            }
            for seg in sim_state.current_schedule.segments
        ],
        "final_status": final_status,
    }


def _calibration_summary(
    recommendations: list[dict[str, Any]],
    realized_outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    realized_by_day = {
        int(item["day"]): item
        for item in realized_outcomes
        if item.get("day") is not None
    }


def _uncertainty_summary(recommendations: list[dict[str, Any]]) -> dict[str, Any]:
    uncertainty_items = [
        recommendation.get("predicted_uncertainty") or {}
        for recommendation in recommendations
        if recommendation.get("predicted_uncertainty")
    ]
    confidence_items = [
        recommendation.get("confidence_breakdown") or {}
        for recommendation in recommendations
        if recommendation.get("confidence_breakdown")
    ]
    return {
        "count": len(uncertainty_items),
        "tir_std_mean": _safe_mean([float(item.get("tir_std", 0.0)) for item in uncertainty_items]),
        "pct_low_std_mean": _safe_mean([float(item.get("pct_low_std", 0.0)) for item in uncertainty_items]),
        "pct_high_std_mean": _safe_mean([float(item.get("pct_high_std", 0.0)) for item in uncertainty_items]),
        "bg_avg_std_mean": _safe_mean([float(item.get("bg_avg_std", 0.0)) for item in uncertainty_items]),
        "cost_std_mean": _safe_mean([float(item.get("cost_std", 0.0)) for item in uncertainty_items]),
        "mean_confidence": _safe_mean([float(item.get("final_confidence", 0.0)) for item in confidence_items]),
        "mean_familiarity": _safe_mean([float(item.get("familiarity", 0.0)) for item in confidence_items]),
        "mean_concordance": _safe_mean([float(item.get("concordance", 0.0)) for item in confidence_items]),
        "mean_calibration": _safe_mean([float(item.get("calibration", 0.0)) for item in confidence_items]),
        "mean_effect_support": _safe_mean([float(item.get("effect_support", 0.0)) for item in confidence_items]),
    }
    paired: list[dict[str, float | bool | int]] = []

    for recommendation in recommendations:
        day = recommendation.get("day")
        predicted = recommendation.get("predicted_outcomes") or {}
        realized = realized_by_day.get(int(day)) if day is not None else None
        if not predicted or realized is None:
            continue
        paired.append({
            "day": int(day),
            "tir_abs_error": abs(float(predicted.get("delta_tir", 0.0)) - float(realized.get("tir_delta", 0.0))),
            "pct_low_abs_error": abs(float(predicted.get("delta_pct_low", 0.0)) - float(realized.get("pct_low_delta", 0.0))),
            "pct_high_abs_error": abs(float(predicted.get("delta_pct_high", 0.0)) - float(realized.get("pct_high_delta", 0.0))),
            "bg_avg_abs_error": abs(float(predicted.get("delta_bg_avg", 0.0)) - float(realized.get("bg_avg_delta", 0.0))),
            "cost_abs_error": abs(float(predicted.get("delta_cost_mean", 0.0)) - float(realized.get("cost_delta", 0.0))),
            "tir_direction_match": _direction_match(
                float(predicted.get("delta_tir", 0.0)),
                float(realized.get("tir_delta", 0.0)),
                prefer_positive=True,
            ),
            "pct_low_direction_match": _direction_match(
                float(predicted.get("delta_pct_low", 0.0)),
                float(realized.get("pct_low_delta", 0.0)),
                prefer_positive=False,
            ),
            "pct_high_direction_match": _direction_match(
                float(predicted.get("delta_pct_high", 0.0)),
                float(realized.get("pct_high_delta", 0.0)),
                prefer_positive=False,
            ),
        })

    return {
        "paired_count": len(paired),
        "tir_mae": _safe_mean([float(item["tir_abs_error"]) for item in paired]),
        "pct_low_mae": _safe_mean([float(item["pct_low_abs_error"]) for item in paired]),
        "pct_high_mae": _safe_mean([float(item["pct_high_abs_error"]) for item in paired]),
        "bg_avg_mae": _safe_mean([float(item["bg_avg_abs_error"]) for item in paired]),
        "cost_mae": _safe_mean([float(item["cost_abs_error"]) for item in paired]),
        "tir_direction_match_rate": _safe_mean([1.0 if item["tir_direction_match"] else 0.0 for item in paired]),
        "pct_low_direction_match_rate": _safe_mean([1.0 if item["pct_low_direction_match"] else 0.0 for item in paired]),
        "pct_high_direction_match_rate": _safe_mean([1.0 if item["pct_high_direction_match"] else 0.0 for item in paired]),
    }


def _direction_match(predicted: float, realized: float, *, prefer_positive: bool) -> bool:
    if abs(predicted) < 1e-9 or abs(realized) < 1e-9:
        return abs(predicted - realized) < 1e-9
    if prefer_positive:
        return (predicted > 0 and realized > 0) or (predicted < 0 and realized < 0)
    return (predicted < 0 and realized < 0) or (predicted > 0 and realized > 0)


def print_progress(day: int, total_days: int, current_date: datetime, entry: dict) -> None:
    status = entry.get("graduation_status") or {}
    print(f"Day {day} / {total_days} — {current_date.date()}")
    print(f"  BG avg: {entry.get('bg_avg', 0) or 0:.1f} mg/dL")
    tir = entry.get("tir_7d")
    print(f"  TIR: {(tir or 0) * 100:.1f}%")
    print(
        "  Shadow: "
        f"{status.get('n_days', 0)} days | "
        f"Win rate: {status.get('win_rate', 0) * 100:.1f}% | "
        f"Graduated: {status.get('graduated', False)}"
    )
    if entry.get("recommendation"):
        print(f"  Patient response: {entry.get('patient_response')} (compliance: {entry.get('compliance', 0):.2f})")
    if entry.get("decision_block_reason"):
        print(f"  Blocked by: {entry['decision_block_reason']}")


def list_sim_users(args) -> None:
    if args.no_firebase:
        for user in list_local_users(args.local_root, args.namespace):
            print(f"{user.get('email') or '<local-only>'} -> {user['uid']} ({user.get('persona') or 'unknown'})")
        return
    users = firestore.client().collection("users").where("isSim", "==", True).get()
    for doc in users:
        data = doc.to_dict() or {}
        print(f"{data.get('email', '<unknown>')} -> {doc.id}")


def delete_sim_user(args) -> None:
    email_or_uid = args.delete
    if args.no_firebase:
        uid = email_or_uid if "@" not in email_or_uid else _local_uid(email_or_uid, args.namespace)
        writer = LocalArtifactWriter(uid, args.local_root, namespace=args.namespace)
        writer.delete_all_user_data()
        print(f"Deleted local sim user {uid}")
        return
    try:
        user = auth.get_user(email_or_uid)
    except auth.UserNotFoundError:
        user = auth.get_user_by_email(email_or_uid)
    uid = user.uid
    writer = FirebaseWriter(uid)
    _delete_document_recursive(firestore.client().collection("users").document(uid))
    writer.delete_all_user_data()
    auth.delete_user(uid)
    print(f"Deleted sim user {uid}")


def _delete_document_recursive(doc_ref) -> None:
    for collection in doc_ref.collections():
        for doc in collection.list_documents():
            _delete_document_recursive(doc)
            doc.delete()
    doc_ref.delete()


def _action_magnitude(recommendation: dict) -> float:
    action = recommendation.get("action", {})
    if action.get("kind") == "scheduled":
        segment_deltas = action.get("segment_deltas", [])
        mags = [
            abs(float(delta.get("isf_delta", 0.0)))
            + abs(float(delta.get("cr_delta", 0.0)))
            + abs(float(delta.get("basal_delta", 0.0)))
            for delta in segment_deltas
        ]
        return sum(mags) / max(len(mags), 1)
    deltas = action.get("deltas", {})
    return (
        abs(float(deltas.get("isf_delta", 0.0)))
        + abs(float(deltas.get("cr_delta", 0.0)))
        + abs(float(deltas.get("basal_delta", 0.0)))
    ) / 3.0


def _schedule_version(schedule: TherapySchedule) -> str:
    encoded = "|".join(
        f"{seg.segment_id}:{seg.start_min}:{seg.end_min}:{seg.isf:.4f}:{seg.cr:.4f}:{seg.basal:.4f}"
        for seg in schedule.segments
    )
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:10]


def _block_reason_counts(log_entries: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in log_entries:
        reason = entry.get("decision_block_reason")
        if not reason:
            continue
        counts[str(reason)] = counts.get(str(reason), 0) + 1
    return counts


def _safe_mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _rolling_mean(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        out.append(_safe_mean(values[start:idx + 1]))
    return out


def _entry_cost(entry: dict[str, Any]) -> float:
    explicit = entry.get("realized_cost")
    if explicit is not None:
        return float(explicit)
    pct_low = float(entry.get("pct_low_7d") or 0.0)
    pct_high = float(entry.get("pct_high_7d") or 0.0)
    return pct_low * 5.0 + pct_high


def _build_trend_series(log_entries: list[dict[str, Any]]) -> dict[str, list[Any]]:
    ordered = sorted(log_entries, key=lambda entry: int(entry.get("day", 0)))
    dates = [entry.get("date") for entry in ordered]
    tir = [float(entry.get("tir_7d") or 0.0) for entry in ordered]
    pct_low = [float(entry.get("pct_low_7d") or 0.0) for entry in ordered]
    pct_high = [float(entry.get("pct_high_7d") or 0.0) for entry in ordered]
    bg_avg = [float(entry.get("bg_avg") or 0.0) for entry in ordered]
    realized_cost = [_entry_cost(entry) for entry in ordered]
    mood_valence = [None if entry.get("mood_valence") is None else float(entry["mood_valence"]) for entry in ordered]
    mood_arousal = [None if entry.get("mood_arousal") is None else float(entry["mood_arousal"]) for entry in ordered]
    stress_acute = [None if entry.get("stress_acute") is None else float(entry["stress_acute"]) for entry in ordered]
    return {
        "dates": dates,
        "tir_daily": tir,
        "tir_rolling_14d": _rolling_mean(tir, 14),
        "pct_low_daily": pct_low,
        "pct_low_rolling_14d": _rolling_mean(pct_low, 14),
        "pct_high_daily": pct_high,
        "pct_high_rolling_14d": _rolling_mean(pct_high, 14),
        "bg_avg_daily": bg_avg,
        "bg_avg_rolling_14d": _rolling_mean(bg_avg, 14),
        "realized_cost_daily": realized_cost,
        "realized_cost_rolling_14d": _rolling_mean(realized_cost, 14),
        "mood_valence_daily": mood_valence,
        "mood_arousal_daily": mood_arousal,
        "stress_acute_daily": stress_acute,
    }


def _recommendation_outcome_summary(
    log_entries: list[dict[str, Any]],
    *,
    horizon_days: int = 3,
    min_tir_gain: float = 0.01,
    min_cost_drop: float = 0.01,
) -> list[dict[str, Any]]:
    ordered = sorted(log_entries, key=lambda entry: int(entry.get("day", 0)))
    by_day = {int(entry.get("day", 0)): entry for entry in ordered}
    outcomes: list[dict[str, Any]] = []
    for entry in ordered:
        if entry.get("patient_response") not in {"accept", "partial"}:
            continue
        if not entry.get("recommendation"):
            continue
        day = int(entry.get("day", 0))
        before_days = [candidate for candidate in range(max(1, day - horizon_days), day + 1) if candidate in by_day]
        after_days = [candidate for candidate in range(day + 1, day + horizon_days + 1) if candidate in by_day]
        if not after_days:
            continue

        before_entries = [by_day[candidate] for candidate in before_days]
        after_entries = [by_day[candidate] for candidate in after_days]
        before_tir = _safe_mean([float(item.get("tir_7d") or 0.0) for item in before_entries])
        after_tir = _safe_mean([float(item.get("tir_7d") or 0.0) for item in after_entries])
        before_pct_low = _safe_mean([float(item.get("pct_low_7d") or 0.0) for item in before_entries])
        after_pct_low = _safe_mean([float(item.get("pct_low_7d") or 0.0) for item in after_entries])
        before_pct_high = _safe_mean([float(item.get("pct_high_7d") or 0.0) for item in before_entries])
        after_pct_high = _safe_mean([float(item.get("pct_high_7d") or 0.0) for item in after_entries])
        before_bg_avg = _safe_mean([float(item.get("bg_avg") or 0.0) for item in before_entries])
        after_bg_avg = _safe_mean([float(item.get("bg_avg") or 0.0) for item in after_entries])
        before_cost = _safe_mean([_entry_cost(item) for item in before_entries])
        after_cost = _safe_mean([_entry_cost(item) for item in after_entries])

        tir_delta = after_tir - before_tir
        pct_low_delta = after_pct_low - before_pct_low
        pct_high_delta = after_pct_high - before_pct_high
        bg_avg_delta = after_bg_avg - before_bg_avg
        cost_delta = after_cost - before_cost
        outcomes.append({
            "day": day,
            "date": entry.get("date"),
            "patient_response": entry.get("patient_response"),
            "horizon_days": horizon_days,
            "before_tir": before_tir,
            "after_tir": after_tir,
            "tir_delta": tir_delta,
            "before_pct_low": before_pct_low,
            "after_pct_low": after_pct_low,
            "pct_low_delta": pct_low_delta,
            "before_pct_high": before_pct_high,
            "after_pct_high": after_pct_high,
            "pct_high_delta": pct_high_delta,
            "before_bg_avg": before_bg_avg,
            "after_bg_avg": after_bg_avg,
            "bg_avg_delta": bg_avg_delta,
            "before_cost": before_cost,
            "after_cost": after_cost,
            "cost_delta": cost_delta,
            "positive": (
                (cost_delta <= -min_cost_drop or tir_delta >= min_tir_gain)
                and pct_low_delta <= 0.0
            ),
        })
    return outcomes


def _successful_day(result) -> bool:
    finite = result.observed_cgm[np.isfinite(result.observed_cgm)]
    if finite.size == 0:
        return False
    tir = float(np.mean((finite >= 70) & (finite <= 180)))
    return tir >= 0.7


def _run_id(uid: str, args, existing_entries: list[dict[str, Any]]) -> str:
    ordinal = len(existing_entries) + args.seed
    payload = f"{uid}|{args.namespace}|{args.seed}|{ordinal}|{datetime.now(timezone.utc).isoformat()}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _recommendation_history_from_logs(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for entry in entries:
        if not entry.get("recommendation"):
            continue
        history.append({
            "day": entry.get("day"),
            "date": entry.get("date"),
            "recommendation": entry.get("recommendation"),
            "response": entry.get("patient_response"),
            "compliance": entry.get("compliance", 0.0),
            "accepted": entry.get("patient_response") in {"accept", "partial"},
            "successful": entry.get("schedule_changed", False),
            "action_kind": entry.get("action_kind"),
            "action_level": entry.get("action_level"),
            "action_family": entry.get("action_family"),
        })
    return history


def _write_report_artifact(args, uid: str, report: dict[str, Any]) -> Path:
    if args.report_file:
        path = Path(args.report_file).expanduser()
    else:
        reports_dir = Path(__file__).resolve().parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        path = reports_dir / f"{args.namespace}_{uid}_{report['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def make_writer(args, uid: str):
    if args.no_firebase:
        return LocalArtifactWriter(uid, args.local_root, namespace=args.namespace)
    return FirebaseWriter(uid)


def _local_uid(seed_text: str, namespace: str) -> str:
    return f"local_{hashlib.sha1(f'{namespace}|{seed_text}'.encode('utf-8')).hexdigest()[:12]}"


if __name__ == "__main__":
    main()

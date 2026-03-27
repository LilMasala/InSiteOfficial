"""Seed a synthetic InSite patient history without running the Chamelia loop."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.create_sim_patient import (
    SimulationState,
    _default_service_account_path,
    _parse_datetime,
    build_patient_config,
    build_run_report,
    make_writer,
    resolve_user,
    _init_firebase,
)
from t1d_sim.simulate import simulate_day
from t1d_sim.therapy import make_default_schedule


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.no_firebase:
        _init_firebase(args.service_account, args.bucket, args.project)

    uid, email = resolve_user(args)
    cfg = build_patient_config(args, uid)
    writer = make_writer(args, uid)
    existing_entries = writer.load_sim_log_entries() if args.append else []
    existing_report = writer.load_latest_report() if args.append else None

    writer.write_user_profile(cfg, email=email, namespace=args.namespace)
    initial_schedule = writer.load_latest_therapy_schedule() if args.append else None
    if initial_schedule is None:
        initial_schedule = cfg.therapy_schedule or cfg.baseline_therapy_schedule or make_default_schedule(cfg)

    sim_state = SimulationState(
        cfg,
        initial_schedule,
        profile_policy=args.profile_policy,
        prior_profile_history=list(existing_report.get("profile_history", [])) if existing_report else None,
        prior_recommendation_history=[],
    )

    if args.append and existing_entries and existing_entries[-1].get("date"):
        start_date = _parse_datetime(str(existing_entries[-1]["date"])).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)
        starting_day = int(existing_entries[-1].get("day", 0))
    else:
        start_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=args.days)
        starting_day = 0
        writer.write_therapy_snapshot(sim_state.current_schedule, timestamp=start_date)

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
        log_entries.append({
            "namespace": args.namespace,
            "day": cumulative_day,
            "date": current_date.isoformat(),
            "bg_avg": signals.get("bg_avg"),
            "tir_7d": signals.get("tir_7d"),
            "pct_low_7d": signals.get("pct_low_7d"),
            "pct_high_7d": signals.get("pct_high_7d"),
            "realized_cost": float(signals.get("pct_low_7d", 0.0)) * 5.0 + float(signals.get("pct_high_7d", 0.0)),
            "mood_valence": signals.get("valence"),
            "mood_arousal": signals.get("arousal"),
            "stress_acute": signals.get("stress_acute"),
            "graduation_status": {
                "graduated": False,
                "n_days": 0,
                "win_rate": 0.0,
                "safety_violations": 0,
                "consecutive_days": 0,
            },
            "recommendation": None,
            "recommendation_returned": False,
            "patient_response": None,
            "compliance": 0.0,
            "schedule_changed": False,
            "action_kind": None,
            "action_level": None,
            "action_family": None,
            "burnout_attribution": None,
            "decision_withheld": False,
            "jepa_status": None,
            "jepa_active": False,
            "jepa_weights_loaded": False,
            "configurator_mode": None,
            "decision_block_reason": None,
            "safety_diagnostics": None,
        })

        if args.verbose:
            print(
                f"Day {cumulative_day}: "
                f"bg_avg={signals.get('bg_avg', 0.0):.1f} "
                f"tir={100 * float(signals.get('tir_7d', 0.0)):.1f}%"
            )

    writer.write_sim_log(log_entries)
    full_entries = writer.load_sim_log_entries()
    report = build_run_report(
        full_entries,
        sim_state,
        uid=uid,
        email=email,
        namespace=args.namespace,
        persona=cfg.persona,
        run_id=f"seed-{args.seed}-{args.days}",
        days_this_run=len(log_entries),
    )
    writer.write_run_report(report)

    print(f"Seeded {len(log_entries)} historical days for {uid}")
    if email and args.password and not args.no_firebase:
        print(f"Login with: {email} / {args.password}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="insitev2")
    parser.add_argument("--bucket", default="insitev2.appspot.com")
    parser.add_argument("--service-account", default=str(_default_service_account_path()))
    parser.add_argument("--no-firebase", action="store_true")
    parser.add_argument("--local-root", default=str(Path(__file__).resolve().parent / "local_runs"))
    parser.add_argument("--namespace", default="seed-only")
    parser.add_argument("--email")
    parser.add_argument("--password")
    parser.add_argument("--uid")
    parser.add_argument("--persona", default="athlete")
    parser.add_argument("--questionnaire")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--profile-policy", choices=["single", "limited-multi", "multi"], default="single")
    return parser


if __name__ == "__main__":
    main()

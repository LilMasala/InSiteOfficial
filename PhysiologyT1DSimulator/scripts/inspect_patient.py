"""Inspect a simulated InSite patient and its Chamelia/Firebase state."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import firebase_admin
from firebase_admin import auth, credentials

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from t1d_sim.chamelia_client import ChameliaClient, ChameliaError
from t1d_sim.firebase_writer import FirebaseWriter
from t1d_sim.local_writer import LocalArtifactWriter, list_local_users


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.no_firebase:
        _init_firebase(args.service_account, args.bucket, args.project)
    if args.list:
        _list_users(args)
        return
    uid, email = _resolve_user(args)
    writer = _make_writer(args, uid)
    report = writer.load_latest_report()
    logs = writer.load_sim_log_entries()
    schedule = writer.load_latest_therapy_schedule()

    if args.status or _no_explicit_view(args):
        _print_status(uid, email, report, args.chamelia_url)
    if args.report or _no_explicit_view(args):
        _print_report(report)
    if args.schedule:
        _print_schedule(schedule)
    if args.recommendations:
        _print_recommendations(logs)
    if args.log:
        _print_log_tail(logs, args.days)
    if args.json and report is not None:
        print(json.dumps(report, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="insitev2")
    parser.add_argument("--bucket", default="insitev2.appspot.com")
    parser.add_argument("--chamelia-url", default="https://chamelia-136217612465.us-central1.run.app")
    parser.add_argument("--service-account", default=str(_default_service_account_path()))
    parser.add_argument("--no-firebase", action="store_true")
    parser.add_argument("--local-root", default=str(Path(__file__).resolve().parent / "local_runs"))
    parser.add_argument("--namespace", default="dev-sim")
    parser.add_argument("--uid")
    parser.add_argument("--email")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--recommendations", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    return parser


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


def _init_firebase(service_account_path: str, bucket: str, project: str) -> None:
    if firebase_admin._apps:
        return
    service_account = Path(service_account_path)
    if service_account.exists():
        cred = credentials.Certificate(service_account)
        firebase_admin.initialize_app(cred, {"storageBucket": bucket, "projectId": project})
        return
    firebase_admin.initialize_app(credentials.ApplicationDefault(), {"storageBucket": bucket, "projectId": project})


def _resolve_user(args) -> tuple[str, str | None]:
    if args.no_firebase:
        if args.uid:
            return args.uid, args.email
        if args.email:
            return _local_uid(args.email, args.namespace), args.email
        raise SystemExit("Provide either --uid or --email")
    if args.uid:
        user = auth.get_user(args.uid)
        return user.uid, user.email
    if args.email:
        user = auth.get_user_by_email(args.email)
        return user.uid, user.email
    raise SystemExit("Provide either --uid or --email")


def _no_explicit_view(args) -> bool:
    return not any([args.status, args.report, args.schedule, args.recommendations, args.log, args.json])


def _print_status(uid: str, email: str | None, report: dict | None, chamelia_url: str) -> None:
    print("Status")
    print(f"  uid: {uid}")
    if email:
        print(f"  email: {email}")
    if report:
        print(f"  persona: {report.get('persona')}")
        print(f"  namespace: {report.get('namespace')}")
        print(f"  days_total: {report.get('days_total')}")
        print(f"  graduated_day: {report.get('graduated_day')}")
        print(f"  recommendation_count: {report.get('recommendation_count')}")
        print(f"  tir_final_14d_mean: {report.get('tir_final_14d_mean', 0.0):.3f}")
        print(f"  accept_or_partial_rate: {report.get('accept_or_partial_rate', 0.0):.3f}")
        print(f"  realized_positive_outcome_rate: {report.get('realized_positive_outcome_rate', 0.0):.3f}")
    try:
        client = ChameliaClient(chamelia_url)
        status = client.graduation_status(uid).get("status", {})
        print(f"  chamelia_status: {status}")
    except ChameliaError as exc:
        print(f"  chamelia_status_error: {exc}")


def _print_report(report: dict | None) -> None:
    print("\nReport")
    if not report:
        print("  no report found")
        return
    print(f"  run_id: {report.get('run_id')}")
    print(f"  generatedAt: {report.get('generatedAt')}")
    print(f"  final_status: {report.get('final_status')}")
    print(f"  post_graduation_no_surface_days: {report.get('post_graduation_no_surface_days')}")
    print(f"  post_graduation_surface_days: {report.get('post_graduation_surface_days')}")
    print(f"  block_reasons: {report.get('block_reasons')}")
    print(f"  post_graduation_block_reasons: {report.get('post_graduation_block_reasons')}")
    print(
        "  rates: "
        f"accept_or_partial={report.get('accept_or_partial_rate', 0.0):.3f} | "
        f"schedule_change={report.get('schedule_change_rate', 0.0):.3f} | "
        f"realized_positive={report.get('realized_positive_outcome_rate', 0.0):.3f}"
    )
    print(
        "  trends: "
        f"TIR14d {report.get('tir_baseline_14d_mean', 0.0):.3f} -> {report.get('tir_final_14d_mean', 0.0):.3f} | "
        f"delta={report.get('tir_delta_baseline_vs_final_14d', 0.0):+.3f}"
    )


def _print_schedule(schedule) -> None:
    print("\nSchedule")
    if schedule is None:
        print("  no therapy schedule found")
        return
    print(f"  tz: {schedule.tz_identifier}")
    for seg in schedule.segments:
        print(
            f"  {seg.segment_id}: {seg.start_min:04d}-{seg.end_min:04d} "
            f"isf={seg.isf:.2f} cr={seg.cr:.2f} basal={seg.basal:.2f}"
        )


def _print_recommendations(logs: list[dict]) -> None:
    print("\nRecommendations")
    recs = [entry for entry in logs if entry.get("recommendation")]
    if not recs:
        print("  none")
        return
    for entry in recs[-20:]:
        print(
            f"  day {entry.get('day')}: kind={entry.get('action_kind')} "
            f"level={entry.get('action_level')} family={entry.get('action_family')} "
            f"response={entry.get('patient_response')} changed={entry.get('schedule_changed')}"
        )


def _print_log_tail(logs: list[dict], days: int) -> None:
    print("\nLog Tail")
    for entry in logs[-days:]:
        status = entry.get("graduation_status") or {}
        print(
            f"  day {entry.get('day')}: date={entry.get('date')} "
            f"bg_avg={entry.get('bg_avg')} tir={entry.get('tir_7d')} "
            f"graduated={status.get('graduated')} rec={entry.get('recommendation_returned')}"
        )


def _list_users(args) -> None:
    if args.no_firebase:
        for user in list_local_users(args.local_root, args.namespace):
            print(f"{user.get('email') or '<local-only>'} -> {user['uid']} ({user.get('persona') or 'unknown'})")
        return
    raise SystemExit("--list is only implemented for --no-firebase in inspect_patient.py")


def _make_writer(args, uid: str):
    if args.no_firebase:
        return LocalArtifactWriter(uid, args.local_root, namespace=args.namespace)
    return FirebaseWriter(uid)


def _local_uid(seed_text: str, namespace: str) -> str:
    import hashlib
    return f"local_{hashlib.sha1(f'{namespace}|{seed_text}'.encode('utf-8')).hexdigest()[:12]}"


if __name__ == "__main__":
    main()

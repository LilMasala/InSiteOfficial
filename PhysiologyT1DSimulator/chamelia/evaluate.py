"""CLI entry point for the Chamelia evaluation layer (Block 9).

Usage:
    python -m chamelia.evaluate closedloop_5.db
    python -m chamelia.evaluate closedloop_5.db --report results_dev/
    python -m chamelia.evaluate closedloop_5.db --model zoo_dev/zoo.pkl

Connects to a SQLite database produced by t1d_sim closed-loop simulation,
loads shadow records and forked timeline metadata, and runs every evaluation
method that has sufficient data.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

from chamelia.evaluation import (
    EvaluationResult,
    build_robustness_report,
    forked_timeline_analysis,
    off_policy_evaluation,
    shadow_retrospective,
    surrogate_replay,
)
from chamelia.shadow import ShadowRecord


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_shadow_records(conn: sqlite3.Connection) -> list[ShadowRecord]:
    """Reconstruct ShadowRecord objects from the shadow_records table."""
    rows = conn.execute(
        "SELECT record_id, patient_id, day_index, timestamp_utc, "
        "feature_snapshot, proposed_action, baseline_action, "
        "proposed_predictions, baseline_predictions, "
        "gate_passed, gate_composite_score, gate_layer_scores, gate_blocked_by, "
        "familiarity_score, calibration_scores, "
        "actual_outcomes, actual_user_action, actual_settings, "
        "counterfactual_estimate, per_model_accuracy, shadow_score_delta "
        "FROM shadow_records"
    ).fetchall()

    records: list[ShadowRecord] = []
    for r in rows:
        rec = ShadowRecord(
            record_id=r[0],
            patient_id=r[1],
            day_index=r[2],
            timestamp_utc=r[3] or "",
            feature_snapshot=json.loads(r[4]) if r[4] else {},
            proposed_action=json.loads(r[5]) if r[5] else [],
            baseline_action=json.loads(r[6]) if r[6] else [],
            proposed_predictions=json.loads(r[7]) if r[7] else {},
            baseline_predictions=json.loads(r[8]) if r[8] else {},
            gate_passed=bool(r[9]),
            gate_composite_score=r[10] or 0.0,
            gate_layer_scores=json.loads(r[11]) if r[11] else {},
            gate_blocked_by=r[12],
            familiarity_score=r[13] or 0.0,
            calibration_scores=json.loads(r[14]) if r[14] else {},
            actual_outcomes=json.loads(r[15]) if r[15] else None,
            actual_user_action=r[16],
            actual_settings=json.loads(r[17]) if r[17] else None,
            counterfactual_estimate=json.loads(r[18]) if r[18] else None,
            per_model_accuracy=json.loads(r[19]) if r[19] else None,
            shadow_score_delta=r[20],
        )
        records.append(rec)
    return records


def _load_forked_payloads(conn: sqlite3.Connection) -> list[dict]:
    """Build pseudo-payloads from ground_truth_daily for forked timeline analysis.

    ``bg_hourly`` does not carry a ``path_id`` column, so per-branch BG data
    cannot be recovered from a cold database.  Instead we compute branch-level
    TIR from the *daily BG statistics* stored in ``bg_hourly`` keyed by date
    ranges derived from ``ground_truth_daily`` path records.

    When that is not possible (e.g. the database only contains a single
    branch's BG rows) the function still builds stub payloads so the
    ``forked_timeline_analysis`` function can count sibling pairs even if
    it cannot compute TIR deltas.
    """
    # Identify all (user_id, path_id) branches.
    branches = conn.execute(
        "SELECT user_id, path_id, MIN(date_utc) AS start_date, "
        "MAX(date_utc) AS end_date, COUNT(*) AS n_days "
        "FROM ground_truth_daily "
        "WHERE path_id != '' "
        "GROUP BY user_id, path_id"
    ).fetchall()

    if not branches:
        return []

    # For each user, load their bg_hourly rows once.
    user_bg: dict[str, list[tuple]] = {}
    for user_id, *_ in branches:
        if user_id not in user_bg:
            user_bg[user_id] = conn.execute(
                "SELECT user_id, hour_utc, start_bg, end_bg, avg_bg, "
                "percent_low, percent_high, uroc, expected_end_bg, "
                "therapy_profile_id FROM bg_hourly WHERE user_id = ? "
                "ORDER BY hour_utc",
                (user_id,),
            ).fetchall()

    payloads: list[dict] = []
    for user_id, path_id, start_date, end_date, n_days in branches:
        # Filter bg_hourly to this branch's date range.
        bg_rows = [
            r for r in user_bg.get(user_id, [])
            if start_date <= r[1][:10] <= end_date
        ]

        payloads.append({
            "bg_hourly": bg_rows,
            "branch_meta": {
                "path_id": path_id,
                "user_id": user_id,
                "start_date": start_date,
                "end_date": end_date,
                "n_days": n_days,
            },
        })

    return payloads


def _load_scorecard_snapshots(conn: sqlite3.Connection) -> list[dict]:
    """Load scorecard snapshots for display."""
    rows = conn.execute(
        "SELECT timestamp_utc, window_size, n_records, win_rate, "
        "safety_violations, coverage_80, familiarity_rate, "
        "cross_context_spread, acceptance_rate, "
        "consecutive_pass_days, status "
        "FROM scorecard_snapshots"
    ).fetchall()
    return [
        {
            "timestamp_utc": r[0],
            "window_size": r[1],
            "n_records": r[2],
            "win_rate": r[3],
            "safety_violations": r[4],
            "coverage_80": r[5],
            "familiarity_rate": r[6],
            "cross_context_spread": r[7],
            "acceptance_rate": r[8],
            "consecutive_pass_days": r[9],
            "status": r[10],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def main(
    db_path: str,
    report_dir: str | None = None,
    model_path: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run available evaluation methods against *db_path* and return a summary.

    Args:
        db_path:    Path to the SQLite database from a closed-loop simulation.
        report_dir: If provided, save detailed JSON results to this directory.
        model_path: Optional path to a trained PredictorCard pickle for
                    surrogate replay evaluation.
        verbose:    Print summary to stdout.

    Returns:
        Summary dict (same structure as ``RobustnessReport.summary()``).
    """
    conn = sqlite3.connect(db_path)

    # --- Load data ---------------------------------------------------------
    shadow_records = _load_shadow_records(conn)
    forked_payloads = _load_forked_payloads(conn)
    scorecards = _load_scorecard_snapshots(conn)

    n_patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    conn.close()

    enriched = [r for r in shadow_records if r.actual_outcomes is not None]

    if verbose:
        print(f"[evaluate] Database: {db_path}")
        print(f"  patients: {n_patients}")
        print(f"  shadow_records: {len(shadow_records)} ({len(enriched)} enriched)")
        print(f"  forked branches: {len(forked_payloads)}")
        print(f"  scorecard snapshots: {len(scorecards)}")
        print()

    # --- Run evaluation methods --------------------------------------------
    results: list[EvaluationResult] = []

    # Method 4: Shadow retrospective (always available if records exist).
    res_shadow = shadow_retrospective(shadow_records)
    results.append(res_shadow)
    if verbose and res_shadow.n_samples > 0:
        m = res_shadow.metrics
        print(f"[shadow_retrospective]  n={res_shadow.n_samples}")
        print(f"  win_rate:           {m.get('win_rate', 0):.3f}")
        print(f"  prediction_error:   {m.get('mean_prediction_error', 0):.4f}")
        print(f"  empirical_coverage: {m.get('empirical_coverage', 0):.3f}")
        print()

    # Method 3: Off-policy evaluation.
    res_ope = off_policy_evaluation(shadow_records)
    results.append(res_ope)
    if verbose and res_ope.n_samples > 0:
        m = res_ope.metrics
        print(f"[off_policy]  n={res_ope.n_samples}")
        print(f"  naive_ate:       {m.get('naive_ate', 0):+.4f}")
        print(f"  ipw_ate:         {m.get('ipw_ate', 0):+.4f}")
        print(f"  mean_tir_accept: {m.get('mean_tir_accepted', 0):.3f}")
        print(f"  mean_tir_reject: {m.get('mean_tir_rejected', 0):.3f}")
        print()

    # Method 1: Surrogate replay (requires a model artifact).
    if model_path is not None:
        from chamelia.models.aggregate import AggregateOutcomePredictor

        surrogate = AggregateOutcomePredictor.load(model_path)
        res_surr = surrogate_replay(shadow_records, surrogate)
        results.append(res_surr)
        if verbose and res_surr.n_samples > 0:
            m = res_surr.metrics
            print(f"[surrogate_replay]  n={res_surr.n_samples}")
            print(f"  mean_effect:    {m.get('mean_effect', 0):+.4f}")
            print(f"  positive_rate:  {m.get('positive_rate', 0):.3f}")
            print()
    elif verbose:
        print("[surrogate_replay]  skipped (no --model provided)")
        print()

    # Method 5: Forked timeline causal analysis.
    res_fork = forked_timeline_analysis(forked_payloads)
    results.append(res_fork)
    if verbose:
        m = res_fork.metrics
        n_pairs = m.get("n_pairs", 0)
        if n_pairs > 0:
            print(f"[forked_timeline]  n_pairs={n_pairs}")
            print(f"  mean_causal_effect: {m.get('mean_causal_effect', 0):+.4f}")
            print(f"  positive_rate:      {m.get('positive_effect_rate', 0):.3f}")
            print(f"  median_effect:      {m.get('median_effect', 0):+.4f}")
        else:
            print(f"[forked_timeline]  n_branches={res_fork.n_samples}, "
                  f"no sibling pairs found")
        print()

    # --- Robustness report -------------------------------------------------
    report = build_robustness_report(results)
    summary = report.summary()

    if verbose:
        print("=" * 60)
        print("ROBUSTNESS REPORT")
        print("=" * 60)
        print(f"  methods with data: {summary['n_methods']}")
        print(f"  consistent:        {summary['consistent']}")
        print(f"  overall_effect:    {summary['overall_effect']:+.4f}")
        print(f"  overall_confidence: {summary['overall_confidence']:.3f}")

        if scorecards:
            print()
            print("SCORECARD SUMMARY (latest per-patient)")
            for sc in scorecards[-5:]:
                print(
                    f"  win_rate={sc['win_rate']:.2f}  "
                    f"safety={sc['safety_violations']}  "
                    f"coverage={sc['coverage_80']:.2f}  "
                    f"status={sc['status']}"
                )

    # --- Save report -------------------------------------------------------
    if report_dir is not None:
        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Full summary JSON.
        with open(out / "robustness_report.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Per-method details.
        for res in results:
            if res.n_samples == 0:
                continue
            payload = {
                "method": res.method,
                "n_samples": res.n_samples,
                "metrics": res.metrics,
                "details": res.details,
            }
            fname = f"{res.method}.json"
            with open(out / fname, "w") as f:
                json.dump(payload, f, indent=2, default=str)

        # Scorecard snapshots.
        if scorecards:
            with open(out / "scorecard_snapshots.json", "w") as f:
                json.dump(scorecards, f, indent=2, default=str)

        if verbose:
            print(f"\n[evaluate] Results saved to {out}/")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m chamelia.evaluate",
        description="Run Chamelia evaluation methods against a closed-loop simulation database.",
    )
    p.add_argument(
        "db_path",
        help="Path to the SQLite database (e.g. closedloop_5.db)",
    )
    p.add_argument(
        "--report",
        metavar="DIR",
        default=None,
        help="Directory to save detailed JSON results",
    )
    p.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help="Path to a trained PredictorCard pickle for surrogate replay",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress stdout output",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(
        db_path=args.db_path,
        report_dir=args.report,
        model_path=args.model,
        verbose=not args.quiet,
    )

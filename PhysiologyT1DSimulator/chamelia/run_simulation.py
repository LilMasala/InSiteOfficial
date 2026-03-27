"""Single-entry MVP Chamelia simulation runner.

This module exposes an honest pre-JEPA simulation path built on top of the
existing ``WorldRunner``. It keeps the aggregate model + grid-search optimizer
as the only recommendation path and emits both SQLite and JSON artifacts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chamelia.run import WorldRunner


def run_chamelia_simulation(
    n_patients: int = 200,
    days: int = 180,
    seed: int = 42,
    outdb: str | None = None,
    report: str | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the canonical MVP Chamelia simulation and optionally persist a report."""
    runner = WorldRunner(
        n_patients=n_patients,
        n_days=days,
        seed=seed,
        learning_mode="hybrid",
        outdb_path=outdb,
        verbose=not quiet,
    )
    summary = runner.run()
    if report:
        report_path = Path(report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _print_final_summary(summary, outdb=outdb, report=report)
    return summary


def _print_final_summary(summary: dict[str, Any], outdb: str | None, report: str | None) -> None:
    """Print a concise human-readable summary answering the MVP questions."""
    print("\n[chamelia-sim] Final summary")
    print(f"  patients: {summary['n_patients']}")
    print(f"  days: {summary['n_days']}")
    print(f"  seed: {summary['seed']}")
    print(
        "  TIR early→late: "
        f"{summary['mean_tir_early_window']:.3f} → {summary['mean_tir_late_window']:.3f} "
        f"(Δ {summary['overall_tir_delta']:+.3f})"
    )
    print(
        "  surfaced recommendations / acceptance: "
        f"{summary['number_of_surfaced_recommendations']} / "
        f"{summary['recommendation_acceptance_rate']:.1%}"
    )
    print(
        "  shadow / intervention: "
        f"{summary['patients_entering_shadow']} patients / "
        f"{summary['patients_graduating_to_intervention']} patients"
    )
    print(
        "  burnout: "
        f"{summary['burnout_count']} patients ({summary['burnout_rate']:.1%})"
    )
    print(f"  did mean TIR improve? {'yes' if summary['did_mean_tir_improve'] else 'no'}")
    print(
        "  did burnout/disengagement remain acceptable? "
        f"{'yes' if summary['did_burnout_remain_acceptable'] else 'no'}"
    )
    print(f"  verdict: {summary['viability_verdict']}")
    print(f"  burnout definition: {summary['burnout_definition']}")
    if outdb:
        print(f"  sqlite: {outdb}")
    if report:
        print(f"  report: {report}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MVP Chamelia simulation.")
    parser.add_argument("--n-patients", type=int, default=200)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdb", type=str, default="artifacts/chamelia_sim.db")
    parser.add_argument("--report", type=str, default="artifacts/chamelia_report.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    run_chamelia_simulation(
        n_patients=args.n_patients,
        days=args.days,
        seed=args.seed,
        outdb=args.outdb,
        report=args.report,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()

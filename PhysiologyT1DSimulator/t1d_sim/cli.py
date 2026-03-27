"""CLI entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from t1d_sim import simulate_population
from t1d_sim.patient_threephase import PhaseConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--outdb")
    p.add_argument("--firebase-project")
    p.add_argument("--outdir", default="./t1d_sim_output/")
    p.add_argument("--n_patients", type=int, default=100)
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="80/10/10")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--male_fraction", type=float, default=0.45)
    p.add_argument("--aid_fraction", type=float, default=0.35,
        help="Fraction of population using AID systems (default 0.35)")

    # Closed-loop (Chamelia) options.
    p.add_argument("--closed-loop", action="store_true",
        help="Enable three-phase closed-loop simulation with fork branching")
    p.add_argument("--obs-days", type=int, default=30,
        help="Phase 1 observation days (default 30)")
    p.add_argument("--shadow-days", type=int, default=30,
        help="Phase 2 shadow days (default 30)")
    p.add_argument("--decision-interval", type=int, default=5,
        help="Days between decision points in Phase 3 (default 5)")
    p.add_argument("--fork-probability", type=float, default=0.3,
        help="P(fork) at each decision point (default 0.3)")
    p.add_argument("--max-depth", type=int, default=8,
        help="Maximum fork depth (default 8)")
    p.add_argument("--recommender-artifact",
        help="Path to a trained PredictorCard pickle for recommendations")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if not args.outdb and not args.firebase_project:
        raise SystemExit("Either --outdb or --firebase-project is required")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Load recommender if specified.
    recommender = None
    if args.recommender_artifact:
        from chamelia.models.aggregate import AggregateOutcomePredictor
        recommender = AggregateOutcomePredictor.load(args.recommender_artifact)

    # Build phase config for closed-loop mode.
    phase_cfg = None
    if args.closed_loop:
        phase_cfg = PhaseConfig(
            obs_days=args.obs_days,
            shadow_days=args.shadow_days,
            total_days=args.days,
            decision_interval=args.decision_interval,
            fork_probability=args.fork_probability,
            max_depth=args.max_depth,
        )

    simulate_population(
        outdb=args.outdb,
        firebase_project=args.firebase_project,
        outdir=args.outdir,
        n_patients=args.n_patients,
        days=args.days,
        seed=args.seed,
        split=args.split,
        jobs=args.jobs,
        male_fraction=args.male_fraction,
        aid_fraction=args.aid_fraction,
        closed_loop=args.closed_loop,
        phase_cfg=phase_cfg,
        recommender=recommender,
    )


if __name__ == "__main__":
    main()

"""t1d synthetic simulator."""
from __future__ import annotations

from datetime import datetime, timezone
from multiprocessing import Pool

from t1d_sim.feature_frame import FeatureFrameHourly
from t1d_sim.patient import simulate_patient
from t1d_sim.patient_threephase import simulate_patient_threephase, PhaseConfig
from t1d_sim.population import sample_population, PatientConfig
from t1d_sim.questionnaire import (
    QuestionnaireAnswers,
    questionnaire_to_patientconfig_priors,
    questionnaire_to_agency_priors,
    sample_twins_from_priors,
    physical_priors_from_twins,
)
from t1d_sim.therapy import (
    TherapySegment,
    TherapySchedule,
    SegmentDelta,
    StructureEdit,
    make_default_schedule,
)
from t1d_sim.simulate import DailySimResult, SimulationCarryState, simulate_day
from t1d_sim.writers import SQLiteWriter, FirebaseWriter


def _simulate_one(args: tuple[PatientConfig, int, datetime]) -> dict:
    return simulate_patient(*args)


def _simulate_one_threephase(args: tuple) -> list[dict]:
    cfg, phase_cfg, start_utc, recommender = args
    return simulate_patient_threephase(cfg, phase_cfg, start_utc, recommender)


def simulate_population(
    outdb: str | None = None,
    firebase_project: str | None = None,
    outdir: str = "./t1d_sim_output/",
    n_patients: int = 100,
    days: int = 180,
    seed: int = 42,
    split: str = "80/10/10",
    jobs: int = 1,
    male_fraction: float = 0.45,
    aid_fraction: float = 0.35,
    closed_loop: bool = False,
    phase_cfg: PhaseConfig | None = None,
    recommender=None,
) -> None:
    """Run full population simulation.

    Args:
        outdb:            Path to SQLite output database.
        firebase_project: Firebase project ID (alternative to outdb).
        outdir:           Output directory.
        n_patients:       Number of patients.
        days:             Simulation length in days.
        seed:             RNG seed.
        split:            Train/val/test split ratios (e.g. "80/10/10").
        jobs:             Parallel workers (1 = sequential).
        male_fraction:    Fraction of male patients.
        aid_fraction:     Fraction using AID technology.
        closed_loop:      If True, use three-phase simulation with fork branching.
        phase_cfg:        PhaseConfig for closed-loop mode.
        recommender:      PredictorCard for closed-loop recommendations.
    """
    import pandas as pd  # noqa: F401  # imported lazily for environments that only need core simulation APIs
    from tqdm import tqdm

    patients = sample_population(n_patients, seed=seed, male_fraction=male_fraction, aid_fraction=aid_fraction)
    train, val, test = [int(x) for x in split.split("/")]
    cuts = [n_patients * train // 100, n_patients * (train + val) // 100]
    for i, p in enumerate(patients):
        p.n_days = days
        p.split = "train" if i < cuts[0] else "val" if i < cuts[1] else "test"
    writer = SQLiteWriter(outdb) if outdb else FirebaseWriter(firebase_project or "")
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)

    if closed_loop:
        _run_closed_loop(patients, days, start, writer, outdb, jobs, phase_cfg, recommender)
    else:
        _run_open_loop(patients, days, start, writer, outdb, jobs)

    writer.finalize()


def _run_open_loop(patients, days, start, writer, outdb, jobs):
    """Original open-loop simulation pipeline."""
    from tqdm import tqdm

    tasks = [(p, days, start) for p in patients]
    results = []
    if jobs > 1:
        with Pool(jobs) as pool:
            for payload in tqdm(pool.imap_unordered(_simulate_one, tasks), total=len(tasks), desc="patients"):
                writer.write_patient(payload)
                results.append(payload)
    else:
        for t in tqdm(tasks, total=len(tasks), desc="patients"):
            payload = _simulate_one(t)
            writer.write_patient(payload)
            results.append(payload)
    if outdb:
        _write_feature_frames(writer, results)
    _print_summary(results, len(patients), days)


def _run_closed_loop(patients, days, start, writer, outdb, jobs, phase_cfg, recommender):
    """Three-phase closed-loop simulation with fork branching."""
    from tqdm import tqdm

    if phase_cfg is None:
        phase_cfg = PhaseConfig(total_days=days)

    all_branch_payloads = []
    scorecard_written: set = set()
    # Closed-loop is harder to parallelise due to branching; run sequentially.
    for p in tqdm(patients, desc="patients (closed-loop)"):
        branch_payloads = simulate_patient_threephase(
            p, phase_cfg, start, recommender,
        )
        for payload in branch_payloads:
            writer.write_patient(payload)
            # Write shadow records if present.
            shadow_recs = payload.get("shadow_records", [])
            if shadow_recs and hasattr(writer, "write_shadow_records"):
                writer.write_shadow_records([r.to_row() for r in shadow_recs])
            # Write scorecard snapshot (once per patient).
            scorecard = payload.get("scorecard")
            pid = payload["patient"]["patient_id"]
            if scorecard is not None and pid not in scorecard_written:
                if hasattr(writer, "write_scorecard_snapshot"):
                    writer.write_scorecard_snapshot(scorecard.to_row())
                scorecard_written.add(pid)
            all_branch_payloads.append(payload)

    if outdb:
        _write_feature_frames(writer, all_branch_payloads)

    n_branches = len(all_branch_payloads)
    n_patients = len(patients)
    n_shadow = sum(len(p.get("shadow_records", [])) for p in all_branch_payloads)
    n_scorecards = len(scorecard_written)
    print(
        f"patients={n_patients} branches={n_branches} "
        f"avg_branches/patient={n_branches/max(n_patients,1):.1f} "
        f"days={days} shadow_records={n_shadow} "
        f"scorecard_snapshots={n_scorecards}"
    )


def _write_feature_frames(writer, results):
    """Compute and write feature frames from simulation results."""
    import pandas as pd
    from t1d_sim.features import build_feature_frames

    raw = writer.raw_for_features()
    rdf = pd.DataFrame(raw, columns=[
        "user_id", "hour_utc", "avg_bg", "percent_low", "percent_high",
        "uroc", "heart_rate", "active_energy", "exercise_minutes",
        "valence", "arousal", "days_since_change", "location",
    ])
    ff = build_feature_frames(rdf)
    rows = []
    for _, r in ff.iterrows():
        rows.append((
            r.user_id, r.hour_utc, r.avg_bg,
            1 - r.percent_low - r.percent_high,
            r.percent_low, r.percent_high, r.uroc,
            r.bg_delta_avg_7h, r.bg_z_avg_7h,
            r.heart_rate, r.hr_delta_7h, r.hr_z_7h, None,
            r.active_energy, r.kcal_active_last3h, r.kcal_active_last6h,
            None, None, None, None, None, None,
            r.exercise_minutes, r.ex_min_last3h, None, None,
            0, 0, 0, r.days_since_change, r.location, 1,
            r.valence, r.arousal,
            int(r.valence >= 0 and r.arousal >= 0),
            int(r.valence >= 0 and r.arousal < 0),
            int(r.valence < 0 and r.arousal >= 0),
            int(r.valence < 0 and r.arousal < 0),
            1.0,
        ))
    writer.write_features(rows)


def _print_summary(results, n_patients, days):
    """Print summary statistics."""
    all_bg = [x[4] for payload in results for x in payload["bg_hourly"] if x[4] is not None]
    mean_meals = (
        sum(len(eval(r[5])) for p in results for r in p["ground_truth"])
        / max(1, sum(len(p["ground_truth"]) for p in results))
    ) if results else 0
    mean_bg = sum(all_bg) / len(all_bg) if all_bg else 0
    print(f"patients={n_patients} days={days} mean_meals/day={mean_meals:.2f} mean_cgm={mean_bg:.2f}")

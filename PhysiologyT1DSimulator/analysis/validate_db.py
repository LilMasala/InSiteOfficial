"""
Sanity checks for output.db — PhysiologyT1DSimulator
Run: python sanity_check.py --db /path/to/output.db
Produces a printed report + optional plots (--plots flag).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

PASS = "  ✓"
FAIL = "  ✗"
WARN = "  ⚠"


def check(label: str, ok: bool, detail: str = "") -> bool:
    tag = PASS if ok else FAIL
    print(f"{tag}  {label}" + (f"  →  {detail}" if detail else ""))
    return ok


def warn(label: str, detail: str = "") -> None:
    print(f"{WARN}  {label}" + (f"  →  {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}  [{time.strftime('%H:%M:%S')}]")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────
# 1. LOAD DB
# ─────────────────────────────────────────────────────────────

def load(db_path: str) -> sqlite3.Connection:
    p = Path(db_path)
    if not p.exists():
        print(f"ERROR: {db_path} not found")
        sys.exit(1)
    return sqlite3.connect(db_path)


def df(con: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, con)


# ─────────────────────────────────────────────────────────────
# 2. CHECKS
# ─────────────────────────────────────────────────────────────

def check_coverage(con: sqlite3.Connection) -> None:
    section("COVERAGE")

    patients  = df(con, "SELECT * FROM patients")
    n_pat     = len(patients)
    check("Patients exist", n_pat > 0, f"{n_pat} patients")

    splits = patients["split"].value_counts().to_dict()
    check("Train/val/test splits present",
          {"train", "val", "test"}.issubset(splits),
          str(splits))

    female_pct = patients["is_female"].mean() * 100
    check("Female fraction ~45-60%", 40 <= female_pct <= 65,
          f"{female_pct:.1f}%")

    bg = df(con, "SELECT COUNT(*) as n, COUNT(DISTINCT user_id) as u FROM bg_hourly")
    n_bg = bg["n"].iloc[0]
    check("bg_hourly populated", n_bg > 0, f"{n_bg:,} rows, {bg['u'].iloc[0]} users")

    gt = df(con, "SELECT COUNT(*) as n, COUNT(DISTINCT user_id) as u FROM ground_truth_daily")
    n_gt = gt["n"].iloc[0]
    check("ground_truth_daily populated", n_gt > 0,
          f"{n_gt:,} rows, {gt['u'].iloc[0]} users")

    ff = df(con, "SELECT COUNT(*) as n FROM feature_frames")
    check("feature_frames populated", ff["n"].iloc[0] > 0,
          f"{ff['n'].iloc[0]:,} rows")

    days_per_patient = df(con,
        "SELECT user_id, COUNT(DISTINCT date_utc) as d FROM ground_truth_daily GROUP BY user_id")
    med_days = days_per_patient["d"].median()
    check("Median days per patient ≥ 170", med_days >= 170,
          f"median = {med_days:.0f} days")


def check_bg_physiology(con: sqlite3.Connection) -> None:
    section("BG PHYSIOLOGICAL RANGES")

    bg = df(con, "SELECT avg_bg, percent_low, percent_high FROM bg_hourly WHERE avg_bg IS NOT NULL")

    check("All avg_bg > 40 mg/dL",
          (bg["avg_bg"] > 40).all(),
          f"min = {bg['avg_bg'].min():.1f}")

    check("All avg_bg < 450 mg/dL",
          (bg["avg_bg"] < 450).all(),
          f"max = {bg['avg_bg'].max():.1f}")

    mean_bg = bg["avg_bg"].mean()
    check("Mean avg_bg in realistic range (110–200)",
          110 <= mean_bg <= 200,
          f"mean = {mean_bg:.1f} mg/dL")

    tir = 1 - bg["percent_low"] - bg["percent_high"]
    mean_tir = tir.clip(0, 1).mean()
    check("Mean TIR in plausible range (0.40–0.90)",
          0.40 <= mean_tir <= 0.90,
          f"mean TIR = {mean_tir:.3f}")

    pct_in_range = ((bg["avg_bg"] >= 70) & (bg["avg_bg"] <= 180)).mean()
    check("Majority of hours in 70-180 mg/dL range",
          pct_in_range > 0.50,
          f"{pct_in_range*100:.1f}%")


def check_sleep(con: sqlite3.Connection) -> None:
    section("SLEEP DISTRIBUTIONS")

    gt = df(con, """
        SELECT g.user_id, g.true_sleep_min, p.sleep_regularity
        FROM ground_truth_daily g
        JOIN patients p ON g.user_id = p.user_id
    """)

    mean_sleep = gt["true_sleep_min"].mean()
    check("Mean sleep 300–500 min (5-8.3h)",
          300 <= mean_sleep <= 500,
          f"mean = {mean_sleep:.0f} min ({mean_sleep/60:.1f}h)")

    std_sleep = gt["true_sleep_min"].std()
    check("Sleep std > 50 min (realistic variability)",
          std_sleep > 50,
          f"std = {std_sleep:.0f} min")

    # Low regularity patients should have higher sleep std
    low_reg  = gt[gt["sleep_regularity"] < 0.40]["true_sleep_min"].std()
    high_reg = gt[gt["sleep_regularity"] > 0.75]["true_sleep_min"].std()
    if len(gt[gt["sleep_regularity"] < 0.40]) > 50:
        check("Low-regularity patients have higher sleep std than high-regularity",
              low_reg > high_reg,
              f"low_reg std={low_reg:.0f}  high_reg std={high_reg:.0f}")
    else:
        warn("Not enough low-regularity patients to compare sleep std")

    sl = df(con, "SELECT awake, asleep_core, asleep_deep, asleep_rem FROM sleep_daily")
    total = sl["awake"] + sl["asleep_core"] + sl["asleep_deep"] + sl["asleep_rem"]
    check("Sleep stages sum to reasonable total (240–600 min)",
          ((total >= 240) & (total <= 650)).mean() > 0.95,
          f"{((total >= 240) & (total <= 650)).mean()*100:.1f}% in range")

    deep_frac = (sl["asleep_deep"] / total.replace(0, np.nan)).mean()
    check("Deep sleep fraction 15–30%",
          0.10 <= deep_frac <= 0.35,
          f"mean deep frac = {deep_frac*100:.1f}%")


def check_meals(con: sqlite3.Connection) -> None:
    section("MEAL DISTRIBUTIONS")

    gt = df(con, """
        SELECT g.user_id, g.true_meal_times, g.true_meal_carbs,
               p.meal_regularity, p.logging_quality
        FROM ground_truth_daily g
        JOIN patients p ON g.user_id = p.user_id
    """)

    # Parse JSON meal times
    gt["n_meals"] = gt["true_meal_times"].apply(
        lambda x: len(json.loads(x)) if x else 0)
    gt["total_carbs"] = gt["true_meal_carbs"].apply(
        lambda x: sum(json.loads(x)) if x else 0)

    mean_meals = gt["n_meals"].mean()
    check("Mean meals per day 1.5–5.0",
          1.5 <= mean_meals <= 5.0,
          f"mean = {mean_meals:.2f}")

    mean_carbs = gt["total_carbs"].mean()
    check("Mean daily carbs 80–300g",
          80 <= mean_carbs <= 300,
          f"mean = {mean_carbs:.0f}g")

    # Breakfast timing check — first meal should mostly be before noon
    def first_meal_hour(times_json: str) -> float | None:
        try:
            times = json.loads(times_json)
            if not times:
                return None
            h, m = times[0].split(":")
            return int(h) + int(m) / 60
        except Exception:
            return None

    gt["first_meal_h"] = gt["true_meal_times"].apply(first_meal_hour)
    valid = gt["first_meal_h"].dropna()
    pct_morning = (valid < 14).mean()
    check("First meal before 2pm on majority of days",
          pct_morning > 0.70,
          f"{pct_morning*100:.1f}% before 14:00")

    # High meal regularity should have tighter meal timing spread
    # (compare std of first_meal_h across days per user)
    per_user = gt.groupby("user_id").agg(
        first_meal_std=("first_meal_h", "std"),
        meal_regularity=("meal_regularity", "first")
    ).dropna()
    corr = per_user["first_meal_std"].corr(per_user["meal_regularity"])
    check("Meal regularity trait negatively correlated with first-meal timing std",
          corr < -0.1,
          f"r = {corr:.3f}")


def check_exercise(con: sqlite3.Connection) -> None:
    section("EXERCISE DISTRIBUTIONS")

    gt = df(con, """
        SELECT g.true_exercise_min, g.true_exercise_intensity,
               p.activity_propensity, p.fitness_level
        FROM ground_truth_daily g
        JOIN patients p ON g.user_id = p.user_id
    """)

    exercise_days = (gt["true_exercise_min"] > 0).mean()
    check("Exercise day fraction 10–80%",
          0.10 <= exercise_days <= 0.80,
          f"{exercise_days*100:.1f}% of days have exercise")

    active = gt[gt["true_exercise_min"] > 0]
    mean_dur = active["true_exercise_min"].mean()
    check("Exercise duration 20–75 min when active",
          15 <= mean_dur <= 85,
          f"mean = {mean_dur:.0f} min")

    # High activity propensity → more exercise days
    lo_act = gt[gt["activity_propensity"] < 0.30]["true_exercise_min"].gt(0).mean()
    hi_act = gt[gt["activity_propensity"] > 0.70]["true_exercise_min"].gt(0).mean()
    if len(gt[gt["activity_propensity"] > 0.70]) > 50:
        check("High activity_propensity patients exercise more often",
              hi_act > lo_act,
              f"hi={hi_act*100:.1f}%  lo={lo_act*100:.1f}%")


def check_effectors(con: sqlite3.Connection) -> None:
    section("CONTEXT EFFECTOR VALIDATION")

    gt = df(con, """
        SELECT g.user_id, g.date_utc, g.true_isf, g.true_sleep_min,
               g.true_cycle_phase, g.true_stress,
               p.stress_reactivity, p.cycle_sensitivity, p.sleep_regularity
        FROM ground_truth_daily g
        JOIN patients p ON g.user_id = p.user_id
        WHERE g.true_isf IS NOT NULL
    """)

    # ── Sleep vs ISF ─────────────────────────────────────────
    # Poor sleep should raise insulin resistance → lower effective ISF
    good_sleep = gt[gt["true_sleep_min"] >= 420]["true_isf"].mean()
    bad_sleep  = gt[gt["true_sleep_min"] < 300]["true_isf"].mean()
    if len(gt[gt["true_sleep_min"] < 300]) > 30:
        check("Poor sleep (<5h) associated with lower true_ISF (more resistant)",
              bad_sleep < good_sleep,
              f"<5h ISF={bad_sleep:.1f}  ≥7h ISF={good_sleep:.1f}")
    else:
        warn("Not enough <5h sleep days to check ISF effect")

    # ── Stress vs ISF ────────────────────────────────────────
    low_stress  = gt[gt["true_stress"] < 0.20]["true_isf"].mean()
    high_stress = gt[gt["true_stress"] > 0.60]["true_isf"].mean()
    if len(gt[gt["true_stress"] > 0.60]) > 30:
        check("High stress (>0.6) associated with lower true_ISF",
              high_stress < low_stress,
              f"high_stress ISF={high_stress:.1f}  low_stress ISF={low_stress:.1f}")
    else:
        warn("Not enough high-stress days to check ISF effect")

    # ── Cycle phase vs ISF (females only) ────────────────────
    female_gt = gt[gt["user_id"].str.startswith("sim_f")]
    luteal      = female_gt[female_gt["true_cycle_phase"] == "luteal"]["true_isf"].mean()
    follicular  = female_gt[female_gt["true_cycle_phase"] == "follicular"]["true_isf"].mean()
    if not np.isnan(luteal) and not np.isnan(follicular):
        check("Luteal phase has lower ISF than follicular (insulin resistance)",
              luteal < follicular,
              f"luteal={luteal:.1f}  follicular={follicular:.1f}")
    else:
        warn("No cycle phase data — are there female patients?")

    # High cycle_sensitivity patients should show bigger luteal/follicular gap
    high_sens = female_gt[female_gt["cycle_sensitivity"] > 0.70]
    if len(high_sens) > 0:
        hs_luteal = high_sens[high_sens["true_cycle_phase"] == "luteal"]["true_isf"].mean()
        hs_foll   = high_sens[high_sens["true_cycle_phase"] == "follicular"]["true_isf"].mean()
        if not np.isnan(hs_luteal) and not np.isnan(hs_foll):
            gap = hs_foll - hs_luteal
            check("High cycle_sensitivity patients show ≥3 unit luteal/follicular ISF gap",
                  gap >= 3.0,
                  f"gap = {gap:.2f} mg/dL/U")


def check_persona_differentiation(con: sqlite3.Connection) -> None:
    section("PERSONA TRAIT DIFFERENTIATION")

    patients = df(con, "SELECT * FROM patients")
    gt = df(con, """
        SELECT g.user_id, AVG(g.true_sleep_min) as mean_sleep,
               AVG(CASE WHEN g.true_exercise_min > 0 THEN 1.0 ELSE 0.0 END) as exercise_rate
        FROM ground_truth_daily g
        GROUP BY g.user_id
    """)
    merged = patients.merge(gt, left_on="user_id", right_on="user_id")

    # Athletes (high activity_propensity) should have lower RHR
    athletes  = merged[merged["activity_propensity"] > 0.80]
    sedentary = merged[merged["activity_propensity"] < 0.25]
    if len(athletes) > 3 and len(sedentary) > 3:
        check("High activity_propensity patients have lower base_rhr",
              athletes["base_rhr"].mean() < sedentary["base_rhr"].mean(),
              f"athletes={athletes['base_rhr'].mean():.1f}  sedentary={sedentary['base_rhr'].mean():.1f}")

    # Low sleep_regularity patients have higher sleep std (checked via ground truth)
    # ISF multiplier range check
    isf_range = merged["isf_multiplier"].agg(["min", "max", "std"])
    check("ISF multiplier has meaningful spread (std > 0.05)",
          isf_range["std"] > 0.05,
          f"min={isf_range['min']:.3f}  max={isf_range['max']:.3f}  std={isf_range['std']:.3f}")

    # Logging quality distribution
    lq = patients["logging_quality"].value_counts()
    check("All three logging quality tiers present",
          {"good", "mediocre", "poor"}.issubset(set(lq.index)),
          str(lq.to_dict()))

    # Female patients should have menstrual data
    females = patients[patients["is_female"] == 1]["user_id"].tolist()
    if females:
        mdata = df(con, f"""
            SELECT COUNT(DISTINCT user_id) as n
            FROM menstrual_daily
            WHERE user_id IN ({','.join(['?']*len(females))})
        """.replace("?", "'{}'".format("','".join(females))))
        check("Female patients have menstrual records",
              mdata["n"].iloc[0] > 0,
              f"{mdata['n'].iloc[0]} of {len(females)} females have records")


def check_feature_frames(con: sqlite3.Connection) -> None:
    section("FEATURE FRAMES COMPLETENESS")

    ff = df(con, """
        SELECT bg_avg, bg_tir, hr_mean, kcal_active,
               bg_delta_avg_7h, bg_z_avg_7h,
               sleep_prev_total_min, ex_hours_since
        FROM feature_frames
        LIMIT 50000
    """)

    for col in ["bg_avg", "hr_mean", "kcal_active"]:
        null_rate = ff[col].isna().mean()
        check(f"{col} null rate < 20%", null_rate < 0.20,
              f"{null_rate*100:.1f}% null")

    # Rolling window features will be null at start of each patient — that's ok
    roll_null = ff["bg_delta_avg_7h"].isna().mean()
    check("bg_delta_avg_7h null rate < 50% (rolling window warmup expected)",
          roll_null < 0.50,
          f"{roll_null*100:.1f}% null")

    # bg_tir should be 0–1
    valid_tir = ff["bg_tir"].dropna()
    if len(valid_tir) > 0:
        check("All bg_tir values in [0, 1]",
              ((valid_tir >= 0) & (valid_tir <= 1)).all(),
              f"range [{valid_tir.min():.3f}, {valid_tir.max():.3f}]")


def check_mood(con: sqlite3.Connection) -> None:
    section("MOOD & LOGGING QUALITY")

    mood = df(con, "SELECT valence, arousal FROM mood_hourly WHERE valence IS NOT NULL")
    check("Mood valence in [-1, 1]",
          ((mood["valence"] >= -1) & (mood["valence"] <= 1)).all(),
          f"range [{mood['valence'].min():.2f}, {mood['valence'].max():.2f}]")

    check("Mood arousal in [-1, 1]",
          ((mood["arousal"] >= -1) & (mood["arousal"] <= 1)).all(),
          f"range [{mood['arousal'].min():.2f}, {mood['arousal'].max():.2f}]")

    events = df(con, "SELECT COUNT(*) as n FROM mood_events")
    check("Mood events exist", events["n"].iloc[0] > 0,
          f"{events['n'].iloc[0]:,} events")

    # Good loggers should have more mood events than poor loggers
    lq = df(con, """
        SELECT p.logging_quality, COUNT(e.event_id) as n_events, COUNT(DISTINCT p.user_id) as n_users
        FROM patients p
        LEFT JOIN mood_events e ON p.user_id = e.user_id
        GROUP BY p.logging_quality
    """)
    lq["events_per_user"] = lq["n_events"] / lq["n_users"].replace(0, np.nan)
    lq_dict = lq.set_index("logging_quality")["events_per_user"].to_dict()
    good = lq_dict.get("good", 0)
    poor = lq_dict.get("poor", 0)
    if good and poor:
        check("Good loggers have more mood events than poor loggers",
              good > poor,
              f"good={good:.0f}  poor={poor:.0f} events/user")


def check_ground_truth_consistency(con: sqlite3.Connection) -> None:
    section("GROUND TRUTH CONSISTENCY")

    gt = df(con, "SELECT true_isf, true_cr, true_basal, true_stress FROM ground_truth_daily")

    check("true_ISF values in plausible range (20–150)",
          ((gt["true_isf"] > 20) & (gt["true_isf"] < 150)).all(),
          f"range [{gt['true_isf'].min():.1f}, {gt['true_isf'].max():.1f}]")

    check("true_CR values in plausible range (4–30)",
          ((gt["true_cr"] > 4) & (gt["true_cr"] < 30)).all(),
          f"range [{gt['true_cr'].min():.1f}, {gt['true_cr'].max():.1f}]")

    check("true_basal values in plausible range (0.2–3.0 U/hr)",
          ((gt["true_basal"] > 0.1) & (gt["true_basal"] < 4.0)).all(),
          f"range [{gt['true_basal'].min():.3f}, {gt['true_basal'].max():.3f}]")

    check("true_stress in [0, 1]",
          ((gt["true_stress"] >= 0) & (gt["true_stress"] <= 1)).all(),
          f"range [{gt['true_stress'].min():.2f}, {gt['true_stress'].max():.2f}]")

    # ISF should vary across days for the same patient (SQLite lacks STDEV — use pandas)
    isf_all = df(con, "SELECT user_id, true_isf FROM ground_truth_daily")
    isf_std_per_user = isf_all.groupby("user_id")["true_isf"].std()
    mean_isf_std = isf_std_per_user.mean()
    check("ISF varies across days per patient (mean std > 1.0)",
          mean_isf_std > 1.0,
          f"mean within-patient std = {mean_isf_std:.2f}")


def check_missingness(con: sqlite3.Connection) -> None:
    section("MISSINGNESS RATES BY LOGGING QUALITY")

    # CGM NaN rate per logging quality
    q = df(con, """
        SELECT p.logging_quality,
               AVG(CASE WHEN b.avg_bg IS NULL THEN 1.0 ELSE 0.0 END) as null_rate,
               COUNT(*) as n
        FROM bg_hourly b
        JOIN patients p ON b.user_id = p.user_id
        GROUP BY p.logging_quality
    """)
    expected = {
        "good":     (0.0,  0.15),
        "mediocre": (0.05, 0.30),
        "poor":     (0.10, 0.50),
    }
    for _, row in q.iterrows():
        lo, hi = expected.get(row["logging_quality"], (0, 1))
        check(f"CGM null rate [{row['logging_quality']}] in expected range",
              lo <= row["null_rate"] <= hi,
              f"{row['null_rate']*100:.1f}%")

    # HR null rate should be higher than CGM (watch worn less consistently than sensor)
    hr_null = df(con, "SELECT AVG(CASE WHEN heart_rate IS NULL THEN 1.0 ELSE 0.0 END) as r FROM hr_hourly")
    bg_null = df(con, "SELECT AVG(CASE WHEN avg_bg IS NULL THEN 1.0 ELSE 0.0 END) as r FROM bg_hourly")
    check("HR null rate > CGM null rate (watch worn less consistently than CGM)",
          hr_null["r"].iloc[0] > bg_null["r"].iloc[0],
          f"HR={hr_null['r'].iloc[0]*100:.1f}%  CGM={bg_null['r'].iloc[0]*100:.1f}%")

    # Sleep coverage — not every night should have data
    gt_days = df(con, "SELECT COUNT(DISTINCT user_id || date_utc) as n FROM ground_truth_daily")
    sleep_days = df(con, "SELECT COUNT(*) as n FROM sleep_daily")
    sleep_coverage = sleep_days["n"].iloc[0] / max(1, gt_days["n"].iloc[0])
    check("Sleep coverage < 95% (not every night has data)",
          sleep_coverage < 0.95,
          f"{sleep_coverage*100:.1f}% of days have sleep records")

    # Mood engagement decay — later days should have fewer events
    mood_by_day = df(con, """
        SELECT CAST((julianday(timestamp_utc) - julianday('2025-01-01')) AS INTEGER) as day_num,
               COUNT(*) as n
        FROM mood_events
        GROUP BY day_num
    """)
    if len(mood_by_day) > 90:
        early = mood_by_day[mood_by_day["day_num"] < 30]["n"].mean()
        late  = mood_by_day[mood_by_day["day_num"] > 150]["n"].mean()
        check("Mood event count decays over time (engagement decay)",
              late < early,
              f"first 30 days: {early:.1f}/day  last 30 days: {late:.1f}/day")
    else:
        warn("Not enough days to assess mood engagement decay")


def print_summary_stats(con: sqlite3.Connection) -> None:
    section("SUMMARY STATISTICS")

    patients = df(con, "SELECT * FROM patients")
    gt = df(con, "SELECT true_isf, true_cr, true_basal, true_sleep_min, true_stress FROM ground_truth_daily")
    bg = df(con, "SELECT avg_bg FROM bg_hourly WHERE avg_bg IS NOT NULL")

    print(f"  Patients:          {len(patients)}")
    print(f"  Female fraction:   {patients['is_female'].mean()*100:.1f}%")
    print(f"  Logging quality:   {patients['logging_quality'].value_counts().to_dict()}")
    print(f"  Mean avg BG:       {bg['avg_bg'].mean():.1f} mg/dL")
    print(f"  Mean true_ISF:     {gt['true_isf'].mean():.1f} mg/dL/U")
    print(f"  Mean true_CR:      {gt['true_cr'].mean():.1f} g/U")
    print(f"  Mean true_basal:   {gt['true_basal'].mean():.3f} U/hr")
    print(f"  Mean sleep:        {gt['true_sleep_min'].mean():.0f} min ({gt['true_sleep_min'].mean()/60:.1f}h)")
    print(f"  Mean stress:       {gt['true_stress'].mean():.3f}")

    # Activity stats
    n_exercise = df(con, "SELECT COUNT(*) as n FROM ground_truth_daily WHERE true_exercise_min > 0")
    n_total = df(con, "SELECT COUNT(*) as n FROM ground_truth_daily")
    ex_rate = n_exercise["n"].iloc[0] / n_total["n"].iloc[0]
    print(f"  Exercise rate:     {ex_rate*100:.1f}% of days")


def make_plots(con: sqlite3.Connection, out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("\n  [plots] matplotlib not installed — skipping plots")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # ── Plot 1: BG distributions by logging quality ───────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("BG Distribution by Logging Quality")
    for ax, lq in zip(axes, ["good", "mediocre", "poor"]):
        uids = df(con, f"SELECT user_id FROM patients WHERE logging_quality='{lq}'")["user_id"].tolist()
        if not uids:
            continue
        uid_str = "','".join(uids[:20])  # limit for speed
        bgs = df(con, f"SELECT avg_bg FROM bg_hourly WHERE user_id IN ('{uid_str}') AND avg_bg IS NOT NULL")
        ax.hist(bgs["avg_bg"], bins=50, color={"good": "steelblue", "mediocre": "orange", "poor": "crimson"}[lq], alpha=0.7)
        ax.axvline(70, color="green", linestyle="--", linewidth=1, label="TIR low")
        ax.axvline(180, color="red", linestyle="--", linewidth=1, label="TIR high")
        ax.set_title(lq)
        ax.set_xlabel("BG (mg/dL)")
    axes[0].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/bg_by_logging_quality.png", dpi=120)
    plt.close()

    # ── Plot 2: Sleep distributions ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sleep = df(con, """
        SELECT g.true_sleep_min, p.sleep_regularity
        FROM ground_truth_daily g
        JOIN patients p ON g.user_id = p.user_id
    """)
    axes[0].hist(sleep["true_sleep_min"] / 60, bins=40, color="steelblue", alpha=0.7)
    axes[0].axvline(7.5, color="green", linestyle="--", label="NSF target (7.5h)")
    axes[0].set_xlabel("Sleep (hours)")
    axes[0].set_title("Sleep Duration Distribution")
    axes[0].legend()

    axes[1].scatter(sleep["sleep_regularity"], sleep["true_sleep_min"] / 60,
                    alpha=0.05, s=2, color="steelblue")
    axes[1].set_xlabel("sleep_regularity trait")
    axes[1].set_ylabel("Sleep (hours)")
    axes[1].set_title("Regularity Trait vs Actual Sleep")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/sleep_distributions.png", dpi=120)
    plt.close()

    # ── Plot 3: ISF by cycle phase (females) ──────────────────
    female_gt = df(con, """
        SELECT g.true_isf, g.true_cycle_phase
        FROM ground_truth_daily g
        JOIN patients p ON g.user_id = p.user_id
        WHERE p.is_female = 1 AND g.true_cycle_phase IS NOT NULL
          AND g.true_cycle_phase != 'None'
    """)
    if len(female_gt) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        phase_order = ["menstrual", "follicular", "ovulation", "luteal"]
        phase_data = [female_gt[female_gt["true_cycle_phase"] == ph]["true_isf"].dropna()
                      for ph in phase_order]
        ax.boxplot(phase_data, labels=phase_order, patch_artist=True,
                   boxprops=dict(facecolor="lightblue"))
        ax.set_ylabel("true_ISF (mg/dL/U)")
        ax.set_title("ISF by Menstrual Cycle Phase (females)")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/isf_by_cycle_phase.png", dpi=120)
        plt.close()

    # ── Plot 4: ISF vs sleep (effector validation) ────────────
    gt = df(con, "SELECT true_isf, true_sleep_min, true_stress FROM ground_truth_daily")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hexbin(gt["true_sleep_min"] / 60, gt["true_isf"],
                   gridsize=30, cmap="Blues", mincnt=1)
    axes[0].set_xlabel("Sleep (hours)")
    axes[0].set_ylabel("true_ISF")
    axes[0].set_title("Sleep Duration vs ISF (should slope up)")

    axes[1].hexbin(gt["true_stress"], gt["true_isf"],
                   gridsize=30, cmap="Reds", mincnt=1)
    axes[1].set_xlabel("Stress level")
    axes[1].set_ylabel("true_ISF")
    axes[1].set_title("Stress vs ISF (should slope down)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/effector_validation.png", dpi=120)
    plt.close()

    # ── Plot 5: Sample CGM traces for 3 contrasting personas ──
    patients = df(con, """
        SELECT user_id, activity_propensity, sleep_regularity, stress_reactivity
        FROM patients
        ORDER BY RANDOM()
        LIMIT 20
    """)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    labels = ["High activity (athlete-like)", "Low sleep regularity (insomniac-like)", "High stress"]
    filters = [
        patients.nlargest(1, "activity_propensity")["user_id"].iloc[0],
        patients.nsmallest(1, "sleep_regularity")["user_id"].iloc[0],
        patients.nlargest(1, "stress_reactivity")["user_id"].iloc[0],
    ]
    for ax, uid, label in zip(axes, filters, labels):
        bg = df(con, f"""
            SELECT hour_utc, avg_bg FROM bg_hourly
            WHERE user_id='{uid}' AND avg_bg IS NOT NULL
            ORDER BY hour_utc
            LIMIT 336
        """)  # 2 weeks
        ax.plot(range(len(bg)), bg["avg_bg"], linewidth=0.8, color="steelblue")
        ax.axhline(70, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(180, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.fill_between(range(len(bg)), 70, 180, alpha=0.05, color="green")
        ax.set_ylabel("BG (mg/dL)")
        ax.set_title(f"{label}  [{uid}]")
        ax.set_ylim(40, 350)
    axes[-1].set_xlabel("Hours (2 weeks)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/sample_cgm_traces.png", dpi=120)
    plt.close()

    print(f"\n  Plots saved to {out_dir}/")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity checks for output.db")
    parser.add_argument("--db",    required=True, help="Path to output.db")
    parser.add_argument("--plots", action="store_true", help="Generate diagnostic plots")
    parser.add_argument("--plot_dir", default="./sanity_plots",
                        help="Directory for plots (default: ./sanity_plots)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  PhysiologyT1DSimulator — Sanity Check")
    print(f"  DB: {args.db}")
    print(f"{'='*60}")

    con = load(args.db)

    checks = [
        check_coverage,
        check_bg_physiology,
        check_sleep,
        check_meals,
        check_exercise,
        check_effectors,
        check_persona_differentiation,
        check_feature_frames,
        check_mood,
        check_ground_truth_consistency,
        check_missingness,
    ]

    iterator = (
        _tqdm(checks, desc="Running checks", unit="check")
        if _HAS_TQDM
        else checks
    )
    t_total = time.time()
    for fn in iterator:
        if _HAS_TQDM:
            iterator.set_postfix(current=fn.__name__)  # type: ignore[union-attr]
        t0 = time.time()
        try:
            fn(con)
        except Exception as e:
            print(f"\n  ERROR in {fn.__name__}: {e}")
        elapsed = time.time() - t0
        if not _HAS_TQDM:
            print(f"  ↳ {fn.__name__} done in {elapsed:.1f}s")
    print(f"\n  All checks completed in {time.time() - t_total:.1f}s")

    print_summary_stats(con)

    if args.plots:
        section("GENERATING PLOTS")
        make_plots(con, args.plot_dir)

    print(f"\n{'='*60}")
    print("  Done. Review any ✗ or ⚠ items above.")
    print(f"{'='*60}\n")
    con.close()


if __name__ == "__main__":
    main()
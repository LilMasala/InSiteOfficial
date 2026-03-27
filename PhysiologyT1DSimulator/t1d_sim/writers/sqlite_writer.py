"""SQLite backend writer."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from t1d_sim.writers.base_writer import BaseWriter


SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (user_id TEXT PRIMARY KEY, split TEXT, logging_quality TEXT, is_female INTEGER, activity_propensity REAL, sleep_regularity REAL, stress_reactivity REAL, cycle_sensitivity REAL, mood_stability REAL, meal_regularity REAL, base_rhr REAL, fitness_level REAL, base_patient_name TEXT, isf_multiplier REAL, cr_multiplier REAL, basal_multiplier REAL);
CREATE TABLE IF NOT EXISTS bg_hourly (user_id TEXT, hour_utc TEXT, start_bg REAL, end_bg REAL, avg_bg REAL, percent_low REAL, percent_high REAL, uroc REAL, expected_end_bg REAL, therapy_profile_id TEXT, PRIMARY KEY (user_id, hour_utc));
CREATE TABLE IF NOT EXISTS hr_hourly (user_id TEXT, hour_utc TEXT, heart_rate REAL, PRIMARY KEY (user_id, hour_utc));
CREATE TABLE IF NOT EXISTS energy_hourly (user_id TEXT, hour_utc TEXT, basal_energy REAL, active_energy REAL, total_energy REAL, PRIMARY KEY (user_id, hour_utc));
CREATE TABLE IF NOT EXISTS exercise_hourly (user_id TEXT, hour_utc TEXT, move_minutes REAL, exercise_minutes REAL, total_minutes REAL, PRIMARY KEY (user_id, hour_utc));
CREATE TABLE IF NOT EXISTS sleep_daily (user_id TEXT, date_utc TEXT, awake REAL, asleep_core REAL, asleep_deep REAL, asleep_rem REAL, asleep_unspecified REAL, PRIMARY KEY (user_id, date_utc));
CREATE TABLE IF NOT EXISTS menstrual_daily (user_id TEXT, date_utc TEXT, days_since_period_start INTEGER, PRIMARY KEY (user_id, date_utc));
CREATE TABLE IF NOT EXISTS site_changes_daily (user_id TEXT, date_utc TEXT, days_since_change INTEGER, location TEXT, PRIMARY KEY (user_id, date_utc));
CREATE TABLE IF NOT EXISTS therapy_settings_hourly (user_id TEXT, hour_utc TEXT, profile_id TEXT, profile_name TEXT, carb_ratio REAL, basal_rate REAL, insulin_sensitivity REAL, PRIMARY KEY (user_id, hour_utc));
CREATE TABLE IF NOT EXISTS mood_hourly (user_id TEXT, hour_utc TEXT, valence REAL, arousal REAL, quad_pos_pos INTEGER, quad_pos_neg INTEGER, quad_neg_pos INTEGER, quad_neg_neg INTEGER, hours_since_mood REAL, PRIMARY KEY (user_id, hour_utc));
CREATE TABLE IF NOT EXISTS mood_events (user_id TEXT, event_id TEXT, timestamp_utc TEXT, valence REAL, arousal REAL, PRIMARY KEY (user_id, event_id));
CREATE TABLE IF NOT EXISTS feature_frames (user_id TEXT, hour_utc TEXT, bg_avg REAL, bg_tir REAL, bg_percent_low REAL, bg_percent_high REAL, bg_uroc REAL, bg_delta_avg_7h REAL, bg_z_avg_7h REAL, hr_mean REAL, hr_delta_7h REAL, hr_z_7h REAL, rhr_daily REAL, kcal_active REAL, kcal_active_last3h REAL, kcal_active_last6h REAL, kcal_active_delta_7h REAL, kcal_active_z_7h REAL, sleep_prev_total_min REAL, sleep_debt_7d_min REAL, minutes_since_wake INTEGER, ex_move_min REAL, ex_exercise_min REAL, ex_min_last3h REAL, ex_hours_since REAL, days_since_period_start INTEGER, cycle_follicular INTEGER, cycle_ovulation INTEGER, cycle_luteal INTEGER, days_since_site_change INTEGER, site_loc_current TEXT, site_loc_same_as_last INTEGER, mood_valence REAL, mood_arousal REAL, mood_quad_pos_pos INTEGER, mood_quad_pos_neg INTEGER, mood_quad_neg_pos INTEGER, mood_quad_neg_neg INTEGER, mood_hours_since REAL, PRIMARY KEY (user_id, hour_utc));
CREATE TABLE IF NOT EXISTS ground_truth_daily (user_id TEXT, date_utc TEXT, true_isf REAL, true_cr REAL, true_basal REAL, true_meal_times TEXT, true_meal_carbs TEXT, true_exercise_min INTEGER, true_exercise_intensity REAL, true_sleep_min INTEGER, true_cycle_phase TEXT, true_mood_valence REAL, true_mood_arousal REAL, true_stress REAL, phase INTEGER DEFAULT 0, path_id TEXT DEFAULT '', effective_isf REAL, effective_fitness REAL, active_events TEXT, PRIMARY KEY (user_id, date_utc, path_id));

-- Chamelia Block 5: Shadow records
CREATE TABLE IF NOT EXISTS shadow_records (
    record_id TEXT PRIMARY KEY,
    patient_id TEXT,
    day_index INTEGER,
    timestamp_utc TEXT,
    feature_snapshot TEXT,
    proposed_action TEXT,
    baseline_action TEXT,
    proposed_predictions TEXT,
    baseline_predictions TEXT,
    gate_passed INTEGER,
    gate_composite_score REAL,
    gate_layer_scores TEXT,
    gate_blocked_by TEXT,
    familiarity_score REAL,
    calibration_scores TEXT,
    actual_outcomes TEXT,
    actual_user_action TEXT,
    actual_settings TEXT,
    counterfactual_estimate TEXT,
    per_model_accuracy TEXT,
    shadow_score_delta REAL
);

-- Chamelia Block 6: Model registry
CREATE TABLE IF NOT EXISTS model_registry (
    model_id TEXT PRIMARY KEY,
    version TEXT,
    architecture TEXT,
    target TEXT,
    training_date TEXT,
    data_window TEXT,
    hyperparameters TEXT,
    validation_metrics TEXT,
    trust_weight REAL,
    status TEXT DEFAULT 'active',
    drift_sensitivity REAL DEFAULT 0.5,
    regime_tags TEXT DEFAULT '[]'
);

-- Chamelia Block 5: Scorecard snapshots
CREATE TABLE IF NOT EXISTS scorecard_snapshots (
    timestamp_utc TEXT,
    window_size INTEGER,
    n_records INTEGER,
    win_rate REAL,
    safety_violations INTEGER,
    coverage_80 REAL,
    familiarity_rate REAL,
    cross_context_spread REAL,
    acceptance_rate REAL,
    consecutive_pass_days INTEGER,
    status TEXT
);

-- Chamelia Block 7: Recommendation log
CREATE TABLE IF NOT EXISTS recommendation_log (
    record_id TEXT PRIMARY KEY,
    patient_id TEXT,
    day_index INTEGER,
    timestamp_utc TEXT,
    decision TEXT,
    proposed_isf REAL,
    proposed_cr REAL,
    proposed_basal REAL,
    baseline_isf REAL,
    baseline_cr REAL,
    baseline_basal REAL,
    predicted_tir REAL,
    predicted_pct_low REAL,
    confidence REAL,
    reward REAL,
    explanation TEXT
);

-- Chamelia Block 9: Evaluation snapshots
CREATE TABLE IF NOT EXISTS evaluation_snapshots (
    timestamp_utc TEXT,
    method TEXT,
    n_samples INTEGER,
    metrics TEXT,
    details TEXT
);

-- Chamelia MVP simulation summaries
CREATE TABLE IF NOT EXISTS patient_run_summary (
    patient_id TEXT PRIMARY KEY,
    final_phase INTEGER,
    current_isf REAL,
    current_cr REAL,
    current_basal REAL,
    days_observed INTEGER,
    mean_tir REAL,
    early_tir REAL,
    late_tir REAL,
    tir_delta REAL,
    recommendations_surfaced INTEGER,
    recommendations_accepted INTEGER,
    acceptance_rate REAL,
    graduated_to_intervention INTEGER,
    entered_shadow INTEGER,
    degraduation_count INTEGER,
    burnout_flag INTEGER,
    burnout_definition TEXT
);

CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id TEXT PRIMARY KEY,
    created_at_utc TEXT,
    summary_json TEXT
);
"""


class SQLiteWriter(BaseWriter):
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)

    def write_patient(self, payload: dict) -> None:
        c = self.conn.cursor()
        p = payload["patient"]
        c.execute("INSERT OR REPLACE INTO patients VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            p["patient_id"], p["split"], p["logging_quality"], int(p["is_female"]), p["activity_propensity"], p["sleep_regularity"], p["stress_reactivity"], p["cycle_sensitivity"], p["mood_stability"], p["meal_regularity"], p["base_rhr"], p["fitness_level"], p["base_patient_name"], p["isf_multiplier"], p["cr_multiplier"], p["basal_multiplier"],
        ))
        c.executemany("INSERT OR REPLACE INTO bg_hourly VALUES (?,?,?,?,?,?,?,?,?,?)", payload["bg_hourly"])
        c.executemany("INSERT OR REPLACE INTO hr_hourly VALUES (?,?,?)", payload["hr_hourly"])
        c.executemany("INSERT OR REPLACE INTO energy_hourly VALUES (?,?,?,?,?)", payload["energy_hourly"])
        c.executemany("INSERT OR REPLACE INTO exercise_hourly VALUES (?,?,?,?,?)", payload["exercise_hourly"])
        c.executemany("INSERT OR REPLACE INTO sleep_daily VALUES (?,?,?,?,?,?,?)", payload["sleep_daily"])
        c.executemany("INSERT OR REPLACE INTO menstrual_daily VALUES (?,?,?)", payload["menstrual_daily"])
        c.executemany("INSERT OR REPLACE INTO site_changes_daily VALUES (?,?,?,?)", payload["site_daily"])
        c.executemany("INSERT OR REPLACE INTO therapy_settings_hourly VALUES (?,?,?,?,?,?,?)", payload["therapy"])
        c.executemany("INSERT OR REPLACE INTO mood_hourly VALUES (?,?,?,?,?,?,?,?,?)", payload["mood_hourly"])
        c.executemany("INSERT OR REPLACE INTO mood_events VALUES (?,?,?,?,?)", payload["mood_events"])
        # ground_truth_daily now has 19 columns (phase, path_id, effective_isf,
        # effective_fitness, active_events).
        gt = payload["ground_truth"]
        if gt:
            ncols = len(gt[0])
            if ncols == 14:
                # Legacy open-loop: append phase=0, path_id='', + 3 feedback cols.
                gt = [(*r, 0, "", None, None, "[]") for r in gt]
            elif ncols == 16:
                # Pre-feedback: append 3 feedback cols.
                gt = [(*r, None, None, "[]") for r in gt]
        c.executemany(
            "INSERT OR REPLACE INTO ground_truth_daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            gt,
        )
        self.conn.commit()

    def write_features(self, rows: list[tuple]) -> None:
        self.conn.executemany("INSERT OR REPLACE INTO feature_frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        self.conn.commit()

    def write_shadow_records(self, records: list[tuple]) -> None:
        """Write shadow records from Block 5."""
        self.conn.executemany(
            "INSERT OR REPLACE INTO shadow_records VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            records,
        )
        self.conn.commit()

    def write_model_registry(self, entries: list[tuple]) -> None:
        """Write model registry entries from Block 6."""
        self.conn.executemany(
            "INSERT OR REPLACE INTO model_registry VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            entries,
        )
        self.conn.commit()

    def write_scorecard_snapshot(self, row: tuple) -> None:
        """Write a scorecard snapshot from Block 5."""
        self.conn.execute(
            "INSERT INTO scorecard_snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            row,
        )
        self.conn.commit()

    def write_recommendation_log(self, rows: list[tuple]) -> None:
        """Write recommendation log entries from Block 7."""
        self.conn.executemany(
            "INSERT OR REPLACE INTO recommendation_log VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def write_evaluation_snapshot(self, row: tuple) -> None:
        """Write an evaluation snapshot from Block 9."""
        self.conn.execute(
            "INSERT INTO evaluation_snapshots VALUES (?,?,?,?,?)",
            row,
        )
        self.conn.commit()

    def write_patient_run_summary(self, rows: list[tuple]) -> None:
        """Write MVP patient-level simulation summaries."""
        self.conn.executemany(
            "INSERT OR REPLACE INTO patient_run_summary VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def write_simulation_run(self, row: tuple) -> None:
        """Write the top-level simulation summary payload."""
        self.conn.execute(
            "INSERT OR REPLACE INTO simulation_runs VALUES (?,?,?)",
            row,
        )
        self.conn.commit()

    def raw_for_features(self):
        return self.conn.execute("SELECT b.user_id,b.hour_utc,b.avg_bg,b.percent_low,b.percent_high,b.uroc,h.heart_rate,e.active_energy,x.exercise_minutes,m.valence,m.arousal,s.days_since_change,s.location FROM bg_hourly b LEFT JOIN hr_hourly h ON b.user_id=h.user_id AND b.hour_utc=h.hour_utc LEFT JOIN energy_hourly e ON b.user_id=e.user_id AND b.hour_utc=e.hour_utc LEFT JOIN exercise_hourly x ON b.user_id=x.user_id AND b.hour_utc=x.hour_utc LEFT JOIN mood_hourly m ON b.user_id=m.user_id AND b.hour_utc=m.hour_utc LEFT JOIN site_changes_daily s ON b.user_id=s.user_id AND substr(b.hour_utc,1,10)=s.date_utc").fetchall()

    def finalize(self) -> None:
        self.conn.commit()
        self.conn.close()

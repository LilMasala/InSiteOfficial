"""Training pipeline for AggregateOutcomePredictor (Chamelia Model 1).

Usage:
    python -m chamelia.train output.db
    python -m chamelia.train output.db --artifact chamelia/artifacts/aggregate_v1.pkl
    python -m chamelia.train output.db --artifact my_model.pkl --val-only

The script loads from the SQLite file produced by t1d_sim, aggregates hourly
feature_frames to daily, joins ground_truth_daily (for action inputs) and
bg_hourly (for outcome targets), then trains and evaluates the model.

Columns dropped from feature_frames because they are NULL in all rows of the
existing database (never populated by the current __init__.py pipeline):

    DROPPED_COLS = [
        "rhr_daily", "kcal_active_delta_7h", "kcal_active_z_7h",
        "sleep_debt_7d_min", "minutes_since_wake",
        "ex_move_min", "ex_hours_since",
    ]

Note on path IDs from fork_timeline:
    Rows contributed by forked simulation branches whose ``path_id`` contains
    a trailing "+" are excluded from training.  These represent convergence-
    pruned branches that lack a clean counterfactual sibling and can bias
    causal estimates.  The current ground_truth_daily table from the open-loop
    simulator does not include a path_id column, so this filter is a no-op
    for the existing output.db; it will activate once the closed-loop simulator
    writes branch-tagged data.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Add project root to path so this script works when run as __main__.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from chamelia.models.aggregate import AggregateOutcomePredictor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns confirmed NULL in all rows of the existing output.db (never
# computed by the current __init__.py pipeline).  Dropping them removes
# useless columns and avoids inflating NaN imputation.
DROPPED_COLS: list[str] = [
    "rhr_daily",
    "kcal_active_delta_7h",
    "kcal_active_z_7h",
    "sleep_debt_7d_min",
    "minutes_since_wake",
    "ex_move_min",
    "ex_hours_since",
]

# Therapy setting absolute baselines used to convert ground_truth_daily
# true_isf / true_cr / true_basal into dimensionless multipliers.
ISF_BASE = 45.0
CR_BASE = 12.0
BASAL_BASE = 0.85

TARGET_COLS: list[str] = ["percent_low", "percent_high", "tir", "mean_bg"]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_tables(db_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the four tables needed for training from *db_path*."""
    conn = sqlite3.connect(db_path)
    try:
        ff  = pd.read_sql("SELECT * FROM feature_frames", conn)
        gt  = pd.read_sql(
            "SELECT user_id, date_utc, true_isf, true_cr, true_basal FROM ground_truth_daily",
            conn,
        )
        bg  = pd.read_sql(
            "SELECT user_id, hour_utc, avg_bg, percent_low, percent_high FROM bg_hourly",
            conn,
        )
        pts = pd.read_sql("SELECT user_id, split FROM patients", conn)
    finally:
        conn.close()
    return ff, gt, bg, pts


def _aggregate_features_daily(ff: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly feature_frames to one row per (user_id, date_utc).

    Strategy:
    - Numeric columns: mean over the 24 hourly readings.
    - site_loc_current (TEXT): mode (most frequent value in the day).
    - Columns in DROPPED_COLS are excluded entirely.
    """
    ff = ff.copy()
    ff["date_utc"] = ff["hour_utc"].str[:10]

    numeric_cols = [
        c for c in ff.columns
        if c not in ("user_id", "hour_utc", "date_utc", "site_loc_current")
        and c not in DROPPED_COLS
        and ff[c].dtype != object
    ]

    daily_numeric = (
        ff.groupby(["user_id", "date_utc"])[numeric_cols]
        .mean()
        .reset_index()
    )

    daily_site = (
        ff.groupby(["user_id", "date_utc"])["site_loc_current"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        .reset_index()
    )

    return daily_numeric.merge(daily_site, on=["user_id", "date_utc"])


def _compute_action_inputs(gt: pd.DataFrame) -> pd.DataFrame:
    """Convert absolute therapy params to dimensionless multipliers."""
    gt = gt.copy()
    gt["isf_multiplier"]   = gt["true_isf"]   / ISF_BASE
    gt["cr_multiplier"]    = gt["true_cr"]    / CR_BASE
    gt["basal_multiplier"] = gt["true_basal"] / BASAL_BASE
    return gt[["user_id", "date_utc", "isf_multiplier", "cr_multiplier", "basal_multiplier"]]


def _compute_daily_targets(bg: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bg_hourly to daily outcome metrics.

    bg_hourly.percent_low and percent_high are within-hour fractions
    (fraction of 5-min bins in that hour below/above threshold).  Averaging
    across all 24 hours gives the daily fraction, which is a valid
    approximation of daily TIR given near-uniform bin distribution.
    """
    bg = bg.copy()
    bg["date_utc"] = bg["hour_utc"].str[:10]

    daily = (
        bg.groupby(["user_id", "date_utc"])
        .agg(
            mean_bg=("avg_bg", "mean"),
            percent_low=("percent_low", "mean"),
            percent_high=("percent_high", "mean"),
        )
        .reset_index()
    )
    daily["tir"] = (1.0 - daily["percent_low"] - daily["percent_high"]).clip(0.0, 1.0)
    return daily[["user_id", "date_utc", "mean_bg", "percent_low", "percent_high", "tir"]]


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main(
    db_path: str,
    artifact_path: str = "chamelia/artifacts/aggregate_v1.pkl",
    verbose: bool = True,
) -> AggregateOutcomePredictor:
    """Full training pipeline.

    Args:
        db_path:       Path to the SQLite database produced by t1d_sim.
        artifact_path: Where to save the trained model pickle.
        verbose:       Print progress and evaluation metrics.

    Returns:
        Trained AggregateOutcomePredictor.
    """
    if verbose:
        print(f"[train] Loading tables from {db_path} ...")

    ff, gt, bg, pts = _load_tables(db_path)

    if verbose:
        print(f"  feature_frames: {len(ff):,} rows")
        print(f"  ground_truth_daily: {len(gt):,} rows")
        print(f"  bg_hourly: {len(bg):,} rows")
        print(f"  patients: {len(pts):,} patients, splits: {dict(pts['split'].value_counts())}")

    # --- Feature aggregation ------------------------------------------------
    if verbose:
        print("[train] Aggregating feature_frames to daily ...")
    daily_ff = _aggregate_features_daily(ff)

    # --- Action inputs from ground truth ------------------------------------
    action_df = _compute_action_inputs(gt)

    # --- Outcome targets from bg_hourly -------------------------------------
    target_df = _compute_daily_targets(bg)

    # --- Assemble full frame ------------------------------------------------
    merged = (
        daily_ff
        .merge(action_df, on=["user_id", "date_utc"], how="inner")
        .merge(target_df, on=["user_id", "date_utc"], how="inner")
        .merge(pts[["user_id", "split"]], on="user_id", how="inner")
    )

    if verbose:
        print(f"  Merged rows: {len(merged):,}")

    # --- Feature engineering: encode site_loc_current -----------------------
    encoder = LabelEncoder()
    merged["site_loc_current_enc"] = encoder.fit_transform(
        merged["site_loc_current"].fillna("unknown")
    )

    # --- Define feature and action column lists -----------------------------
    feature_only_cols = [
        c for c in daily_ff.columns
        if c not in ("user_id", "date_utc", "site_loc_current")
        and c not in DROPPED_COLS
    ]
    feature_only_cols.append("site_loc_current_enc")
    action_cols = ["isf_multiplier", "cr_multiplier", "basal_multiplier"]
    all_feature_cols = feature_only_cols + action_cols

    X_all = merged[all_feature_cols].copy()
    y_all = merged[TARGET_COLS].copy()

    # --- Train / val split by patients.split column -------------------------
    train_mask = merged["split"] == "train"
    val_mask   = merged["split"] == "val"

    X_train = X_all[train_mask].copy()
    y_train = y_all[train_mask].copy()
    X_val   = X_all[val_mask].copy()
    y_val   = y_all[val_mask].copy()

    # --- Drop rows with any NULL in retained numeric feature columns ---------
    retain_mask = X_train[feature_only_cols].notna().all(axis=1)
    n_before = len(X_train)
    X_train = X_train[retain_mask]
    y_train = y_train[retain_mask]
    n_dropped = n_before - len(X_train)
    if verbose:
        pct = 100.0 * n_dropped / max(1, n_before)
        print(f"  Dropped {n_dropped} train rows due to NULL features ({pct:.1f}%)")
        print(f"  Train: {len(X_train):,} rows  |  Val: {len(X_val):,} rows")

    # --- Impute remaining NaN with training-set column means ----------------
    feature_means = X_train[feature_only_cols].mean()
    X_train = X_train.fillna(feature_means)
    X_val   = X_val.fillna(feature_means)

    # --- Train model --------------------------------------------------------
    if verbose:
        print("[train] Training AggregateOutcomePredictor (12 XGBoost models) ...")

    model = AggregateOutcomePredictor()
    model._site_encoder = encoder
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # Copy feature means into model for inference-time imputation.
    model._feature_means = feature_means

    # --- Evaluate on validation set -----------------------------------------
    if verbose:
        print("[train] Evaluating on validation set ...")
        X_val_feat = X_val[feature_only_cols].fillna(feature_means)
        X_val_action = X_val[action_cols].values
        envelope = model.predict(X_val_feat, action=X_val_action)
        # envelope.point shape: (n_val, 4)
        pts_arr = np.asarray(envelope.point)
        for i, target in enumerate(AggregateOutcomePredictor.TARGETS):
            col_vals = y_val[target].values.astype(float)
            valid_mask = np.isfinite(col_vals)
            if valid_mask.sum() == 0:
                continue
            pred_col = pts_arr[:, i] if pts_arr.ndim == 2 else pts_arr
            mae  = float(np.mean(np.abs(pred_col[valid_mask] - col_vals[valid_mask])))
            rmse = float(np.sqrt(np.mean((pred_col[valid_mask] - col_vals[valid_mask]) ** 2)))
            print(f"  val [{target:>15s}]  MAE={mae:.4f}  RMSE={rmse:.4f}")

    # --- Save artifact -------------------------------------------------------
    model.save(artifact_path)
    if verbose:
        print(f"[train] Saved model → {artifact_path}")

    return model


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the Chamelia AggregateOutcomePredictor from a t1d_sim SQLite database."
    )
    parser.add_argument("db_path", help="Path to the SQLite database (e.g. output.db)")
    parser.add_argument(
        "--artifact",
        default="chamelia/artifacts/aggregate_v1.pkl",
        help="Output path for the trained model pickle (default: chamelia/artifacts/aggregate_v1.pkl)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    main(
        db_path=args.db_path,
        artifact_path=args.artifact,
        verbose=not args.quiet,
    )

#!/usr/bin/env bash
# train_insite_jepa.sh
#
# InSite-specific wrapper around Chamelia's domain-agnostic
# train_local_jepa_weights.jl.
#
# Provides the exact column lists that match InSite's feature_frames
# schema (from sqlite_writer.py / export_firestore_to_sqlite.py).
#
# Usage:
#   # Synthetic init only
#   bash scripts/train_insite_jepa.sh
#
#   # Real data from simulator output
#   bash scripts/train_insite_jepa.sh --db-path /path/to/sim_output.db
#
#   # Real data exported from Firestore
#   bash scripts/train_insite_jepa.sh --db-path /path/to/firestore_export.db
#
#   # Extra Chamelia args are forwarded as-is
#   bash scripts/train_insite_jepa.sh --db-path data.db --epochs 100 --batch-size 32

set -euo pipefail

CHAMELIA_ROOT="$(cd "$(dirname "$0")/../../Chamelia" && pwd)"

CTX_COLS="bg_avg,bg_tir,bg_percent_low,bg_percent_high,bg_uroc,bg_delta_avg_7h,bg_z_avg_7h,hr_mean,hr_delta_7h,hr_z_7h,rhr_daily,kcal_active,kcal_active_last3h,kcal_active_last6h,kcal_active_delta_7h,kcal_active_z_7h,sleep_prev_total_min,sleep_debt_7d_min,minutes_since_wake,ex_move_min,ex_exercise_min,ex_min_last3h,ex_hours_since,site_loc_same_as_last,mood_valence,mood_arousal,mood_quad_pos_pos,mood_quad_pos_neg,mood_quad_neg_pos,mood_quad_neg_neg,mood_hours_since,stress_acute,insulin_iob,insulin_cob,insulin_recent_bolus_count,insulin_recent_carb_count,insulin_recent_temp_basal_count"

DAILY_COLS="days_since_period_start,cycle_follicular,cycle_ovulation,cycle_luteal,cycle_menstrual,days_since_site_change"

# If --db-path is provided, pass column lists to the training script.
# Otherwise, let the script use synthetic data with matching dimensions.
if echo "$@" | grep -q -- "--db-path"; then
    exec julia --project="$CHAMELIA_ROOT" "$CHAMELIA_ROOT/scripts/train_local_jepa_weights.jl" \
        --ctx-cols "$CTX_COLS" \
        --daily-cols "$DAILY_COLS" \
        "$@"
else
    exec julia --project="$CHAMELIA_ROOT" "$CHAMELIA_ROOT/scripts/train_local_jepa_weights.jl" \
        --n-ctx 37 --n-daily 6 \
        "$@"
fi

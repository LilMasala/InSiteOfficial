"""
export_firestore_to_sqlite.py

Exports InSite feature frames from Firestore to a SQLite database that can be
passed directly to Chamelia's SQLiteTrainingDataset for JEPA encoder pretraining.

Source collection:  users/{uid}/features/ml_feature_frames/items/{hourId}
  — already computed, joined, and normalized by the iOS app.
  — each document is one hourly feature frame containing all 43 signals.

Output table:  feature_frames
  — one row per (user_id, hour_utc)
  — columns match SQLiteTrainingDataset's ctx_cols + daily_cols exactly

Usage:
    python export_firestore_to_sqlite.py \\
        --project <firebase-project-id> \\
        --out training_data.db \\
        [--uid uid1 uid2 ...]   # omit to export all users
        [--days 90]             # max history per user (default: all)
        [--credentials path/to/serviceAccount.json]

After export, train the JEPA encoder:
    julia Chamelia/scripts/train_local_jepa_weights.jl \\
        --db-path training_data.db --epochs 50

Requirements:
    pip install firebase-admin
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any

# ─────────────────────────────────────────────────────────────────
# Signal column definitions — must match SQLiteTrainingDataset in
# Chamelia/src/Perception/jepa_training.jl AND the simulator's
# feature_frames table in sqlite_writer.py (canonical schema).
#
# Column names use the simulator's snake_case convention so that
# a simulator-generated SQLite can be passed directly to
# train_local_jepa_weights.jl without any conversion step.
# ─────────────────────────────────────────────────────────────────

# Hourly contextual features.
CTX_COLS: list[str] = [
    "bg_avg",
    "bg_tir",
    "bg_percent_low",
    "bg_percent_high",
    "bg_uroc",
    "bg_delta_avg_7h",
    "bg_z_avg_7h",
    "hr_mean",
    "hr_delta_7h",
    "hr_z_7h",
    "rhr_daily",
    "kcal_active",
    "kcal_active_last3h",
    "kcal_active_last6h",
    "kcal_active_delta_7h",
    "kcal_active_z_7h",
    "sleep_prev_total_min",
    "sleep_debt_7d_min",
    "minutes_since_wake",
    "ex_move_min",
    "ex_exercise_min",
    "ex_min_last3h",
    "ex_hours_since",
    "site_loc_same_as_last",
    "mood_valence",
    "mood_arousal",
    "mood_quad_pos_pos",
    "mood_quad_pos_neg",
    "mood_quad_neg_pos",
    "mood_quad_neg_neg",
    "mood_hours_since",
    "stress_acute",           # derived: valence > 0.3 and arousal > 0.3
    "insulin_iob",
    "insulin_cob",
    "insulin_recent_bolus_count",
    "insulin_recent_carb_count",
    "insulin_recent_temp_basal_count",
]

# Daily-resolution features.  SQLiteTrainingDataset accumulates these
# across the 24 hours of each day and divides by 24, so values that
# are constant within a day (like cycle phase) survive the averaging.
DAILY_COLS: list[str] = [
    "days_since_period_start",
    "cycle_follicular",
    "cycle_ovulation",
    "cycle_luteal",
    "cycle_menstrual",           # derived: days_since_period_start <= 4
    "days_since_site_change",
]

# All columns stored in the SQLite table.
ALL_SIGNAL_COLS = CTX_COLS + DAILY_COLS

# ─────────────────────────────────────────────────────────────────
# Firestore field → SQLite column mapping
#
# Firestore stores fields in camelCase (iOS Codable convention).
# SQLite uses snake_case (simulator convention).
# List every field whose Firestore name differs from its SQLite column.
# Fields not listed here are read from Firestore using the column name as-is.
# ─────────────────────────────────────────────────────────────────
_FIRESTORE_TO_SQLITE: dict[str, str] = {
    # SQLite column name       →  Firestore field name
    "bg_percent_low":           "bg_percentLow",
    "bg_percent_high":          "bg_percentHigh",
    "bg_uroc":                  "bg_uRoc",
    "bg_delta_avg_7h":          "bg_deltaAvg7h",
    "bg_z_avg_7h":              "bg_zAvg7h",
    "hr_delta_7h":              "hr_delta7h",
    "hr_z_7h":                  "hr_z7h",
    "kcal_active_delta_7h":     "kcal_active_delta7h",
    "kcal_active_z_7h":         "kcal_active_z7h",
    "mood_quad_pos_pos":        "mood_quad_posPos",
    "mood_quad_pos_neg":        "mood_quad_posNeg",
    "mood_quad_neg_pos":        "mood_quad_negPos",
    "mood_quad_neg_neg":        "mood_quad_negNeg",
    "insulin_iob":              "insulin_iob",          # same
    "insulin_cob":              "insulin_cob",          # same
    "insulin_recent_bolus_count":    "insulin_recent_bolus_count",    # same
    "insulin_recent_carb_count":     "insulin_recent_carb_count",     # same
    "insulin_recent_temp_basal_count": "insulin_recent_temp_basal_count",  # same
}

# Fields to derive rather than copy directly from Firestore.
_DERIVED_FIELDS = {"stress_acute", "cycle_menstrual"}


def _derive_fields(doc: dict[str, Any]) -> dict[str, float]:
    """Compute fields that are not stored in Firestore but must appear in the DB."""
    v = float(doc.get("mood_valence") or 0.0)
    a = float(doc.get("mood_arousal") or 0.0)
    stress = 1.0 if (v > 0.3 and a > 0.3) else 0.0

    dpstart = doc.get("days_since_period_start")
    cycle_m = 1.0 if (dpstart is not None and float(dpstart) <= 4.0) else 0.0

    return {
        "stress_acute": stress,
        "cycle_menstrual": cycle_m,
    }


def _safe_float(value: Any) -> float:
    """Convert a Firestore field value to float, returning 0.0 for missing/null."""
    if value is None:
        return 0.0
    try:
        f = float(value)
        return 0.0 if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return 0.0


def _parse_hour_utc(doc_id: str) -> str | None:
    """
    Validate and normalise the hour document ID.
    Firestore doc IDs for hourly collections are ISO8601 UTC strings like
    '2025-10-08T14:00:00Z'.  We store them as-is in SQLite for sorting.
    Returns None if the ID cannot be parsed.
    """
    try:
        dt = datetime.fromisoformat(doc_id.replace("Z", "+00:00"))
        # Round to nearest hour (tolerate off-by-one-second writes)
        return dt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00:00Z")
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────
# SQLite schema helpers
# ─────────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feature_frames (
    user_id  TEXT NOT NULL,
    hour_utc TEXT NOT NULL,
    {signal_cols},
    PRIMARY KEY (user_id, hour_utc)
);
""".format(signal_cols=",\n    ".join(f"{c} REAL NOT NULL DEFAULT 0.0" for c in ALL_SIGNAL_COLS))

INSERT_SQL = """
INSERT OR REPLACE INTO feature_frames (user_id, hour_utc, {cols})
VALUES ({placeholders})
""".format(
    cols=", ".join(ALL_SIGNAL_COLS),
    placeholders=", ".join(["?"] * (2 + len(ALL_SIGNAL_COLS))),
)


def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(CREATE_TABLE_SQL)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_hour ON feature_frames (user_id, hour_utc)")
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────────
# Firestore helpers
# ─────────────────────────────────────────────────────────────────

def init_firestore(project: str, credentials_path: str | None):
    import firebase_admin
    from firebase_admin import credentials, firestore as fs

    if not firebase_admin._apps:
        if credentials_path:
            cred = credentials.Certificate(credentials_path)
        else:
            cred = credentials.ApplicationDefault()
        firebase_admin.initialize_app(cred, {"projectId": project})

    return fs.client()


def fetch_user_ids(db, uids: list[str] | None) -> list[str]:
    """Return the list of UIDs to export."""
    if uids:
        return list(dict.fromkeys(uids))  # deduplicate, preserve order

    # Collect all UIDs that have at least one feature frame.
    print("Discovering users from Firestore (this may take a moment)…")
    users_ref = db.collection("users")
    return [doc.id for doc in users_ref.list_documents()]


def fetch_feature_frames(
    db,
    uid: str,
    days_limit: int | None,
) -> list[dict[str, Any]]:
    """
    Fetch ml_feature_frames documents for one user, newest-first so that
    days_limit trims the oldest data.
    Returns list of dicts: {'doc_id': str, 'data': dict}.
    """
    col = db.collection(f"users/{uid}/features/ml_feature_frames/items")
    query = col.order_by("hourStartUtc", direction="DESCENDING")
    if days_limit is not None:
        query = query.limit(days_limit * 24)

    return [{"doc_id": doc.id, "data": doc.to_dict() or {}} for doc in query.stream()]


# ─────────────────────────────────────────────────────────────────
# Export logic
# ─────────────────────────────────────────────────────────────────

def export_user(conn: sqlite3.Connection, uid: str, frames: list[dict]) -> int:
    """
    Write feature frames for one user to SQLite.
    Returns the number of rows written.
    """
    rows: list[tuple] = []

    for frame in frames:
        doc_id = frame["doc_id"]
        hour_utc = _parse_hour_utc(doc_id)
        if hour_utc is None:
            continue

        doc = frame["data"]
        derived = _derive_fields(doc)

        values: list[float] = []
        for col in ALL_SIGNAL_COLS:
            if col in _DERIVED_FIELDS:
                values.append(derived[col])
            else:
                firestore_key = _FIRESTORE_TO_SQLITE.get(col, col)
                values.append(_safe_float(doc.get(firestore_key)))

        rows.append((uid, hour_utc, *values))

    if rows:
        conn.executemany(INSERT_SQL, rows)
        conn.commit()

    return len(rows)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export InSite Firestore feature frames to SQLite for JEPA training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--project", required=True, help="Firebase project ID")
    p.add_argument("--out", default="training_data.db", help="Output SQLite path (default: training_data.db)")
    p.add_argument("--uid", nargs="+", metavar="UID", help="One or more user IDs to export (default: all users)")
    p.add_argument("--days", type=int, default=None, help="Max days of history per user (default: all available)")
    p.add_argument("--credentials", default=None, help="Path to Firebase service account JSON (default: ADC)")
    return p.parse_args(argv)


def main(argv: list[str] = sys.argv[1:]) -> None:
    args = parse_args(argv)

    print(f"Initialising Firestore (project={args.project})…")
    db = init_firestore(args.project, args.credentials)

    print(f"Opening SQLite: {args.out}")
    conn = init_db(args.out)

    uids = fetch_user_ids(db, args.uid)
    if not uids:
        print("No users found — nothing to export.")
        return

    print(f"Exporting {len(uids)} user(s)…")
    total_rows = 0
    for i, uid in enumerate(uids, 1):
        frames = fetch_feature_frames(db, uid, args.days)
        n = export_user(conn, uid, frames)
        total_rows += n
        print(f"  [{i}/{len(uids)}] {uid}: {n} rows")

    conn.close()
    print(f"\nDone. {total_rows} total rows written to {args.out}")
    print("\nTo train the JEPA encoder:")
    print(f"  julia Chamelia/scripts/train_local_jepa_weights.jl --db-path {args.out} --epochs 50")


if __name__ == "__main__":
    main()

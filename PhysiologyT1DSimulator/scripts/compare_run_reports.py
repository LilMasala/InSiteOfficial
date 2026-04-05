"""Summarize local simulation reports side by side."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


FIELDS = [
    "experiment_label",
    "persona",
    "python_bridge_enabled",
    "python_bridge_mode",
    "python_bridge_model_version",
    "legacy_jepa_compat_enabled",
    "legacy_jepa_mode",
    "jepa_status",
    "jepa_weights_loaded",
    "graduated_day",
    "recommendation_count",
    "accepted_count",
    "partial_count",
    "rejected_count",
    "accept_or_partial_rate",
    "schedule_change_rate",
    "realized_positive_outcome_rate",
    "recommendations_with_followup",
    "recommendation_success_rate",
    "tir_baseline_14d_mean",
    "tir_final_14d_mean",
    "tir_delta_baseline_vs_final_14d",
    "pct_low_mean",
    "pct_high_mean",
]


def load_report(path: Path) -> dict:
    data = json.loads(path.read_text())
    data["_path"] = str(path)
    return data


def format_value(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def print_table(reports: list[dict]) -> None:
    headers = ["run"] + FIELDS
    rows = []
    for report in reports:
        rows.append([Path(report["_path"]).stem] + [format_value(report.get(field)) for field in FIELDS])

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def emit(row):
        print(" | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))

    emit(headers)
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        emit(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()

    reports = [load_report(Path(path)) for path in args.reports]
    print_table(reports)


if __name__ == "__main__":
    main()

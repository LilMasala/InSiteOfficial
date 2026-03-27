"""Plot local no-Firebase simulation run metrics."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Path to local run report JSON")
    parser.add_argument("--output-dir", help="Optional output directory for generated plots")
    args = parser.parse_args()

    report_path = Path(args.report).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run_root = _infer_run_root(report_path, report)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    log_entries = _load_log_entries(run_root)
    daily_results = _load_daily_results(run_root)

    metrics = _build_daily_metrics(log_entries, daily_results)
    if not metrics["dates"]:
        raise SystemExit("No daily metrics found for plotting")

    dashboard_path = output_dir / "dashboard.png"
    response_path = output_dir / "recommendations.png"

    _plot_dashboard(metrics, report, dashboard_path)
    _plot_recommendations(metrics, report, response_path)

    print(f"Dashboard written to: {dashboard_path}")
    print(f"Recommendation plot written to: {response_path}")


def _infer_run_root(report_path: Path, report: dict[str, Any]) -> Path:
    namespace = str(report.get("namespace") or "")
    uid = str(report.get("uid") or "")
    candidate = report_path.parents[1] / namespace / uid
    if candidate.exists():
        return candidate
    # fallback for report-file output locations
    local_runs = Path(__file__).resolve().parent / "local_runs" / namespace / uid
    if local_runs.exists():
        return local_runs
    raise SystemExit(f"Unable to infer local run root for report: {report_path}")


def _load_log_entries(run_root: Path) -> list[dict[str, Any]]:
    log_dir = run_root / "sim_log"
    files = sorted(log_dir.glob("day_*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in files]


def _load_daily_results(run_root: Path) -> dict[str, dict[str, Any]]:
    daily_dir = run_root / "daily_results"
    files = sorted(daily_dir.glob("*.json"))
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in files
    }


def _build_daily_metrics(
    log_entries: list[dict[str, Any]],
    daily_results: dict[str, dict[str, Any]],
) -> dict[str, list[Any]]:
    dates: list[datetime] = []
    tir: list[float] = []
    pct_low: list[float] = []
    pct_high: list[float] = []
    bg_avg: list[float] = []
    mood_valence: list[float | None] = []
    mood_arousal: list[float | None] = []
    mood_event_count: list[int] = []
    decision_reasons: list[str | None] = []
    recommendation_returned: list[bool] = []
    schedule_changed: list[bool] = []
    response: list[str | None] = []

    for entry in log_entries:
        date = _parse_dt(str(entry["date"]))
        date_key = date.strftime("%Y-%m-%d")
        daily = daily_results.get(date_key, {})
        decision_frame = daily.get("decision_frame") or {}
        mood_events = daily.get("mood_events") or []

        dates.append(date)
        tir.append(_float_or_none(entry.get("tir_7d")) or 0.0)
        pct_low.append(_float_or_none(entry.get("pct_low_7d")) or 0.0)
        pct_high.append(_float_or_none(entry.get("pct_high_7d")) or 0.0)
        bg_avg.append(_float_or_none(entry.get("bg_avg")) or 0.0)
        mood_valence.append(_float_or_none(decision_frame.get("mood_valence")))
        mood_arousal.append(_float_or_none(decision_frame.get("mood_arousal")))
        mood_event_count.append(len(mood_events))
        decision_reasons.append(entry.get("decision_block_reason"))
        recommendation_returned.append(bool(entry.get("recommendation_returned")))
        schedule_changed.append(bool(entry.get("schedule_changed")))
        response.append(entry.get("patient_response"))

    return {
        "dates": dates,
        "tir": tir,
        "pct_low": pct_low,
        "pct_high": pct_high,
        "bg_avg": bg_avg,
        "mood_valence": mood_valence,
        "mood_arousal": mood_arousal,
        "mood_event_count": mood_event_count,
        "decision_reasons": decision_reasons,
        "recommendation_returned": recommendation_returned,
        "schedule_changed": schedule_changed,
        "response": response,
    }


def _plot_dashboard(metrics: dict[str, list[Any]], report: dict[str, Any], output_path: Path) -> None:
    dates = metrics["dates"]
    graduated_day = report.get("graduated_day")
    graduated_date = dates[graduated_day - 1] if graduated_day and 0 < graduated_day <= len(dates) else None
    tir_roll = _rolling_mean(metrics["tir"], 14)
    low_roll = _rolling_mean(metrics["pct_low"], 14)
    high_roll = _rolling_mean(metrics["pct_high"], 14)
    bg_roll = _rolling_mean(metrics["bg_avg"], 14)
    mood_valence_roll = _rolling_mean(_fill_none(metrics["mood_valence"]), 14)
    mood_arousal_roll = _rolling_mean(_fill_none(metrics["mood_arousal"]), 14)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"Local Run Dashboard: {report.get('namespace')} | {report.get('persona')} | {report.get('days_total')} days",
        fontsize=14,
    )

    ax = axes[0]
    ax.plot(dates, [100 * x for x in metrics["tir"]], label="TIR % (daily)", color="#1b9e77", linewidth=1.0, alpha=0.28)
    ax.plot(dates, [100 * x for x in tir_roll], label="TIR % (14d)", color="#1b9e77", linewidth=2.4)
    ax.plot(dates, [100 * x for x in metrics["pct_low"]], label="% Low (daily)", color="#d95f02", linewidth=0.9, alpha=0.22)
    ax.plot(dates, [100 * x for x in low_roll], label="% Low (14d)", color="#d95f02", linewidth=1.8)
    ax.plot(dates, [100 * x for x in metrics["pct_high"]], label="% High (daily)", color="#7570b3", linewidth=0.9, alpha=0.22)
    ax.plot(dates, [100 * x for x in high_roll], label="% High (14d)", color="#7570b3", linewidth=1.8)
    ax.set_ylabel("Percent")
    ax.set_title("Rolling Glycemic Metrics")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(dates, metrics["bg_avg"], color="#4c78a8", linewidth=0.9, alpha=0.25, label="BG avg (daily)")
    ax.plot(dates, bg_roll, color="#4c78a8", linewidth=2.2, label="BG avg (14d)")
    ax.axhline(70, color="#d95f02", linestyle="--", linewidth=1)
    ax.axhline(180, color="#7570b3", linestyle="--", linewidth=1)
    ax.set_ylabel("mg/dL")
    ax.set_title("Average BG")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(dates, _fill_none(metrics["mood_valence"]), label="Mood valence (daily)", color="#e7298a", linewidth=0.9, alpha=0.25)
    ax.plot(dates, mood_valence_roll, label="Mood valence (14d)", color="#e7298a", linewidth=1.8)
    ax.plot(dates, _fill_none(metrics["mood_arousal"]), label="Mood arousal (daily)", color="#66a61e", linewidth=0.9, alpha=0.25)
    ax.plot(dates, mood_arousal_roll, label="Mood arousal (14d)", color="#66a61e", linewidth=1.8)
    ax.bar(dates, metrics["mood_event_count"], width=0.8, alpha=0.15, color="#444444", label="Mood event count")
    ax.set_ylabel("Mood")
    ax.set_title("Mood Trend")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[3]
    response_y = {"reject": -1.0, "partial": 0.0, "accept": 1.0, None: None}
    for date, returned, changed, resp in zip(
        dates,
        metrics["recommendation_returned"],
        metrics["schedule_changed"],
        metrics["response"],
    ):
        if returned:
            ax.scatter(date, response_y.get(resp, 0.0), color="#1f78b4", s=35, zorder=3)
        if changed:
            ax.axvline(date, color="#ff7f00", alpha=0.18, linewidth=2)
    ax.set_yticks([-1, 0, 1], labels=["Reject", "Partial", "Accept"])
    ax.set_title("Recommendation Responses and Applied Schedule Changes")
    ax.grid(alpha=0.25)

    if graduated_date is not None:
        for ax in axes:
            ax.axvline(graduated_date, color="#111111", linestyle="--", linewidth=1.2)
            ax.text(graduated_date, ax.get_ylim()[1], " graduation", va="top", ha="left", fontsize=8)

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_recommendations(metrics: dict[str, list[Any]], report: dict[str, Any], output_path: Path) -> None:
    dates = metrics["dates"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, constrained_layout=True)
    fig.suptitle("Recommendation Cadence and Blocks", fontsize=14)

    cumulative_surfaced = []
    total = 0
    for returned in metrics["recommendation_returned"]:
        total += 1 if returned else 0
        cumulative_surfaced.append(total)

    axes[0].plot(dates, cumulative_surfaced, color="#1f78b4", linewidth=2)
    axes[0].set_ylabel("Cumulative surfaced")
    axes[0].grid(alpha=0.25)

    reason_to_y = {}
    next_y = 0
    for reason in sorted({r for r in metrics["decision_reasons"] if r}):
        reason_to_y[reason] = next_y
        next_y += 1
    plotted = set()
    for date, reason, returned in zip(dates, metrics["decision_reasons"], metrics["recommendation_returned"]):
        if returned:
            axes[1].scatter(date, -1, color="#1f78b4", s=30, label="surfaced" if "surfaced" not in plotted else None)
            plotted.add("surfaced")
        elif reason:
            axes[1].scatter(date, reason_to_y[reason], color="#888888", s=18)
    axes[1].set_yticks([-1] + list(reason_to_y.values()), labels=["surfaced"] + list(reason_to_y.keys()))
    axes[1].set_title(
        "Daily decision outcome | "
        f"surfaced={report.get('recommendation_count')} | "
        f"withheld={report.get('post_graduation_no_surface_days')}"
    )
    axes[1].grid(alpha=0.25)
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _fill_none(values: list[float | None]) -> list[float]:
    out: list[float] = []
    last = 0.0
    for value in values:
        if value is None:
            out.append(last)
        else:
            last = float(value)
            out.append(last)
    return out


def _rolling_mean(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    acc = 0.0
    for idx, value in enumerate(values):
        acc += value
        if idx >= window:
            acc -= values[idx - window]
        out.append(acc / min(idx + 1, window))
    return out


if __name__ == "__main__":
    main()

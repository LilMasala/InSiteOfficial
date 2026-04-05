"""Run comparable local Chamelia simulator runs across bridge/runtime modes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "create_sim_patient.py"
DEFAULT_OUT_DIR = REPO_ROOT / "scripts" / "reports" / "mode_comparisons"
SUMMARY_FIELDS = [
    "python_bridge_enabled",
    "python_bridge_mode",
    "legacy_jepa_compat_enabled",
    "graduated_day",
    "recommendation_count",
    "accept_or_partial_rate",
    "realized_positive_outcome_rate",
    "tir_delta_baseline_vs_final_14d",
    "pct_low_mean",
    "pct_high_mean",
    "post_graduation_no_surface_days",
]


def _slug(text: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in text.strip())
    clean = clean.strip("-_")
    return clean or "run"


def build_variant_specs(args) -> list[dict[str, Any]]:
    variants = [
        {
            "label": mode,
            "bridge_mode": mode,
            "legacy_jepa_compat": False,
        }
        for mode in args.modes
    ]
    if args.include_legacy_jepa_compat:
        variants.append({
            "label": "legacy-jepa-compat",
            "bridge_mode": "v3",
            "legacy_jepa_compat": True,
        })
    return variants


def build_run_command(args, variant: dict[str, Any], *, report_path: Path) -> list[str]:
    mode_slug = _slug(variant["label"])
    namespace = f"{args.namespace_prefix}-{_slug(args.comparison_label)}-{mode_slug}"
    uid = f"{args.uid_prefix}-{mode_slug}" if args.uid_prefix else None

    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--no-firebase",
        "--local-root",
        args.local_root,
        "--namespace",
        namespace,
        "--persona",
        args.persona,
        "--days",
        str(args.days),
        "--seed",
        str(args.seed),
        "--chamelia-url",
        args.chamelia_url,
        "--timeout",
        str(args.timeout),
        "--profile-policy",
        args.profile_policy,
        "--coldstart-targets",
        args.coldstart_targets,
        "--bridge-mode",
        variant["bridge_mode"],
        "--bridge-model-version",
        args.bridge_model_version,
        "--experiment-label",
        variant["label"],
        "--report-file",
        str(report_path),
    ]
    if uid:
        command.extend(["--uid", uid])
    if args.python_bridge_url:
        command.extend(["--python-bridge-url", args.python_bridge_url])
    if args.weights_dir:
        command.extend(["--weights-dir", args.weights_dir])
    if variant["legacy_jepa_compat"]:
        command.append("--legacy-jepa-compat")
    if args.verbose:
        command.append("--verbose")
    return command


def build_summary(
    args,
    variants: list[dict[str, Any]],
    reports: list[dict[str, Any]],
    report_paths: list[Path],
) -> dict[str, Any]:
    baseline_report = reports[0] if reports else {}
    baseline_label = variants[0]["label"] if variants else None
    comparisons = []

    for variant, report, report_path in zip(variants, reports, report_paths, strict=True):
        entry = {
            "label": variant["label"],
            "bridge_mode": variant["bridge_mode"],
            "legacy_jepa_compat": variant["legacy_jepa_compat"],
            "report_path": str(report_path),
        }
        for field in SUMMARY_FIELDS:
            entry[field] = report.get(field)

        if baseline_report and variant["label"] != baseline_label:
            entry["delta_vs_baseline"] = {
                "tir_delta_baseline_vs_final_14d": _numeric_delta(report, baseline_report, "tir_delta_baseline_vs_final_14d"),
                "realized_positive_outcome_rate": _numeric_delta(report, baseline_report, "realized_positive_outcome_rate"),
                "accept_or_partial_rate": _numeric_delta(report, baseline_report, "accept_or_partial_rate"),
                "pct_low_mean": _numeric_delta(report, baseline_report, "pct_low_mean"),
                "pct_high_mean": _numeric_delta(report, baseline_report, "pct_high_mean"),
                "graduated_day": _numeric_delta(report, baseline_report, "graduated_day"),
            }
        comparisons.append(entry)

    return {
        "comparison_label": args.comparison_label,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_label": baseline_label,
        "modes": comparisons,
    }


def _numeric_delta(report: dict[str, Any], baseline_report: dict[str, Any], field: str) -> float | None:
    current = report.get(field)
    baseline = baseline_report.get(field)
    if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
        return None
    return float(current) - float(baseline)


def _print_summary(summary: dict[str, Any]) -> None:
    print(f"Comparison: {summary['comparison_label']}")
    print(f"Baseline: {summary.get('baseline_label')}")
    for mode in summary["modes"]:
        print(
            f"- {mode['label']}: bridge={mode.get('python_bridge_mode')} "
            f"legacy={mode.get('legacy_jepa_compat_enabled')} "
            f"tir_delta={_format_value(mode.get('tir_delta_baseline_vs_final_14d'))} "
            f"positive_rate={_format_value(mode.get('realized_positive_outcome_rate'))} "
            f"accept_rate={_format_value(mode.get('accept_or_partial_rate'))}"
        )


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=["v1.1", "v1.5", "v3"])
    parser.add_argument("--include-legacy-jepa-compat", action="store_true")
    parser.add_argument("--comparison-label", default=datetime.now(timezone.utc).strftime("compare-%Y%m%d-%H%M%S"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--namespace-prefix", default="mode-compare")
    parser.add_argument("--uid-prefix", default="mode-compare")
    parser.add_argument("--persona", default="athlete")
    parser.add_argument("--days", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chamelia-url", default="http://127.0.0.1:8080")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--profile-policy", choices=["single", "limited-multi", "multi"], default="single")
    parser.add_argument("--coldstart-targets", choices=["synthetic", "none"], default="synthetic")
    parser.add_argument("--local-root", default=str(REPO_ROOT / "scripts" / "local_runs"))
    parser.add_argument("--python-bridge-url")
    parser.add_argument("--bridge-model-version", default="unknown")
    parser.add_argument("--weights-dir")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    variants = build_variant_specs(args)

    if args.include_legacy_jepa_compat and not args.weights_dir:
        raise SystemExit("--include-legacy-jepa-compat requires --weights-dir so the legacy JEPA path can load weights.")

    out_dir = Path(args.out_dir).expanduser().resolve() / _slug(args.comparison_label)
    out_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    report_paths: list[Path] = []
    for variant in variants:
        mode_dir = out_dir / _slug(variant["label"])
        mode_dir.mkdir(parents=True, exist_ok=True)
        report_path = mode_dir / "report.json"
        command = build_run_command(args, variant, report_path=report_path)
        subprocess.run(command, check=True, cwd=REPO_ROOT)
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        report_paths.append(report_path)

    summary = build_summary(args, variants, reports, report_paths)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _print_summary(summary)
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()

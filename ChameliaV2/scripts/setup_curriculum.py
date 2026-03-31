#!/usr/bin/env python3
"""Plan and partially execute curriculum asset setup from the YAML manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.curriculum.data.setup_manifest import (
    asset_specs_from_manifest,
    blocking_failures,
    load_setup_manifest,
    parse_csv_arg,
    prepare_asset,
    select_asset_specs,
    summarize_results,
    unresolved_domains_from_manifest,
    write_followup_script,
    write_setup_report,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for curriculum setup."""
    parser = argparse.ArgumentParser(description="Set up Chamelia V2 curriculum assets")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "configs" / "curriculum_setup_manifest.yaml"),
        help="Path to the curriculum setup manifest YAML",
    )
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "artifacts" / "setup" / "curriculum_setup_report.json"),
        help="Where to write the setup report JSON",
    )
    parser.add_argument(
        "--followup-script",
        default=str(PROJECT_ROOT / "artifacts" / "setup" / "curriculum_followups.sh"),
        help="Where to write shell follow-up hints",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Stage selector; may be repeated or comma-separated",
    )
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help="Domain selector; may be repeated or comma-separated",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional assets in the selected set",
    )
    parser.add_argument(
        "--include-gated",
        action="store_true",
        help="Include gated assets in the selected set",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan setup without cloning or writing destination content",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-fetch for git-backed assets and ignore present destination state",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero code if required selected assets remain unavailable",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    report_path = Path(args.report).resolve()
    followup_path = Path(args.followup_script).resolve()

    raw_stages = parse_csv_arg(args.stage)
    stages = [int(stage) for stage in raw_stages]
    domains = parse_csv_arg(args.domain)

    manifest = load_setup_manifest(manifest_path)
    asset_specs = asset_specs_from_manifest(manifest)
    unresolved_domains = unresolved_domains_from_manifest(manifest)
    selected_assets = select_asset_specs(asset_specs, stages=stages, domains=domains)

    results = [
        prepare_asset(
            asset,
            PROJECT_ROOT,
            include_optional=args.include_optional,
            include_gated=args.include_gated,
            dry_run=args.dry_run,
            force=args.force,
        )
        for asset in selected_assets
    ]

    write_setup_report(
        report_path,
        manifest_path=manifest_path,
        selected_assets=selected_assets,
        unresolved_domains=unresolved_domains,
        results=results,
        strict=args.strict,
        dry_run=args.dry_run,
    )
    write_followup_script(followup_path, results)

    summary = summarize_results(results)
    print("Curriculum setup summary:")
    for status, count in sorted(summary.items()):
        print(f"  {status}: {count}")
    print(f"Report: {report_path}")
    print(f"Follow-ups: {followup_path}")

    if args.strict:
        blocking = blocking_failures(selected_assets, results)
        if blocking:
            print("Blocking assets remain:")
            for result in blocking:
                print(f"  {result.key}: {result.status} -> {result.followup_hint}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

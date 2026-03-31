"""Data utilities for the curriculum scaffold."""

from training.curriculum.data.downloaders import download_all
from training.curriculum.data.setup_manifest import (
    SetupAssetResult,
    SetupAssetSpec,
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
from training.curriculum.data.sources import DataSourceSpec, stage_source_specs

__all__ = [
    "DataSourceSpec",
    "SetupAssetResult",
    "SetupAssetSpec",
    "asset_specs_from_manifest",
    "blocking_failures",
    "download_all",
    "load_setup_manifest",
    "parse_csv_arg",
    "prepare_asset",
    "select_asset_specs",
    "stage_source_specs",
    "summarize_results",
    "unresolved_domains_from_manifest",
    "write_followup_script",
    "write_setup_report",
]

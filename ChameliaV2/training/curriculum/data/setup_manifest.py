"""Setup-manifest planning and execution utilities for the curriculum pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
from urllib.parse import urlparse

import yaml


@dataclass(frozen=True)
class SetupAssetSpec:
    """One asset entry from the curriculum setup manifest."""

    key: str
    stage: int
    domains: list[str]
    kind: str
    install_mode: str
    setup_policy: str
    fetch_method: str
    source_url: str
    destination: str
    estimated_size_gb: float
    consumers: list[str]
    notes: str


@dataclass(frozen=True)
class SetupAssetResult:
    """Execution or planning result for one selected asset."""

    key: str
    stage: int
    status: str
    install_mode: str
    fetch_method: str
    destination: str
    source_url: str
    domains: list[str]
    notes: str
    followup_hint: str


def load_setup_manifest(path: str | Path) -> dict[str, Any]:
    """Load the curriculum setup manifest from YAML."""
    return yaml.safe_load(Path(path).read_text())


def asset_specs_from_manifest(manifest: dict[str, Any]) -> list[SetupAssetSpec]:
    """Convert raw YAML manifest data into typed asset specs."""
    assets = manifest.get("assets", [])
    return [SetupAssetSpec(**asset) for asset in assets]


def unresolved_domains_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return unresolved-domain entries from the manifest."""
    return list(manifest.get("unresolved_domains", []))


def parse_csv_arg(values: list[str] | None) -> list[str]:
    """Split repeated or comma-separated CLI values into a flat list."""
    if not values:
        return []
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def select_asset_specs(
    asset_specs: list[SetupAssetSpec],
    stages: list[int] | None = None,
    domains: list[str] | None = None,
) -> list[SetupAssetSpec]:
    """Filter asset specs by stage and/or domain selection."""
    stage_filter = set(stages or [])
    domain_filter = set(domains or [])
    selected: list[SetupAssetSpec] = []
    for asset in asset_specs:
        if stage_filter and asset.stage not in stage_filter:
            continue
        if domain_filter and domain_filter.isdisjoint(asset.domains):
            continue
        selected.append(asset)
    return selected


def resolve_destination(project_root: Path, destination: str) -> Path:
    """Resolve a manifest destination against the project root."""
    path = Path(destination)
    if path.is_absolute():
        return path
    return project_root / path


def _looks_like_file(path: Path) -> bool:
    """Best-effort check for whether a destination should be treated as a file."""
    return path.suffix != ""


def _path_has_payload(path: Path) -> bool:
    """Return whether an asset destination appears populated."""
    if not path.exists():
        return False
    if path.is_file():
        return path.stat().st_size > 0
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    return True


def _ensure_destination(path: Path, dry_run: bool) -> None:
    """Create the destination parent or directory, unless running dry."""
    target = path.parent if _looks_like_file(path) else path
    if dry_run:
        return
    target.mkdir(parents=True, exist_ok=True)


def _repo_id_from_hf_url(url: str) -> str:
    """Extract a Hugging Face repository identifier from a dataset or model URL."""
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if not parts:
        return ""
    if parts[0] == "datasets" and len(parts) >= 3:
        return "/".join(parts[1:3])
    return "/".join(parts[:2])


def followup_hint_for_asset(asset: SetupAssetSpec, project_root: Path) -> str:
    """Generate a shell-oriented next-step hint for an asset."""
    destination = resolve_destination(project_root, asset.destination)
    if asset.fetch_method == "local_generator":
        return f"Generated locally into {destination}"
    if asset.fetch_method == "manual_drop":
        return f"Place the file at {destination}"
    if asset.fetch_method == "git_repo":
        return f"git clone --depth 1 {asset.source_url} {destination}"
    if asset.fetch_method == "huggingface_dataset":
        repo_id = _repo_id_from_hf_url(asset.source_url)
        return (
            "huggingface-cli download "
            f"--repo-type dataset {repo_id} --local-dir {destination}"
        )
    if asset.fetch_method == "huggingface_model":
        repo_id = _repo_id_from_hf_url(asset.source_url)
        return f"huggingface-cli download {repo_id} --local-dir {destination}"
    return f"Review source and install manually from {asset.source_url} into {destination}"


def _run_git_repo_fetch(destination: Path, source_url: str, force: bool, dry_run: bool) -> str:
    """Clone or update a git-backed asset."""
    if destination.exists() and (destination / ".git").exists():
        if dry_run:
            return "git_update_planned"
        subprocess.run(
            ["git", "-C", str(destination), "pull", "--ff-only"],
            check=True,
        )
        return "git_updated"

    if destination.exists() and _path_has_payload(destination) and not force:
        return "present"

    if dry_run:
        return "git_clone_planned"

    if destination.exists() and force:
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", source_url, str(destination)],
        check=True,
    )
    return "git_cloned"


def prepare_asset(
    asset: SetupAssetSpec,
    project_root: Path,
    *,
    include_optional: bool,
    include_gated: bool,
    dry_run: bool,
    force: bool,
) -> SetupAssetResult:
    """Plan or perform setup for one asset."""
    destination = resolve_destination(project_root, asset.destination)
    _ensure_destination(destination, dry_run=dry_run)
    followup_hint = followup_hint_for_asset(asset, project_root)

    if _path_has_payload(destination) and not force:
        status = "present"
    elif asset.install_mode == "optional" and not include_optional:
        status = "optional_skipped"
    elif asset.install_mode == "gated" and not include_gated:
        status = "gated_skipped"
    elif asset.install_mode == "manual":
        status = "manual_required"
    elif asset.install_mode == "generated":
        status = "generated_ready"
    elif asset.fetch_method == "git_repo":
        try:
            status = _run_git_repo_fetch(destination, asset.source_url, force=force, dry_run=dry_run)
        except subprocess.CalledProcessError:
            status = "fetch_failed"
    else:
        status = "planned_fetch"

    return SetupAssetResult(
        key=asset.key,
        stage=asset.stage,
        status=status,
        install_mode=asset.install_mode,
        fetch_method=asset.fetch_method,
        destination=str(destination),
        source_url=asset.source_url,
        domains=list(asset.domains),
        notes=asset.notes,
        followup_hint=followup_hint,
    )


def blocking_failures(
    asset_specs: list[SetupAssetSpec],
    results: list[SetupAssetResult],
) -> list[SetupAssetResult]:
    """Return assets that are still blocking under strict setup semantics."""
    spec_by_key = {asset.key: asset for asset in asset_specs}
    blocking: list[SetupAssetResult] = []
    ok_statuses = {"present", "generated_ready", "git_cloned", "git_updated"}
    for result in results:
        spec = spec_by_key[result.key]
        if spec.setup_policy != "hard_fail_if_stage_enabled":
            continue
        if result.status not in ok_statuses:
            blocking.append(result)
    return blocking


def write_setup_report(
    report_path: Path,
    *,
    manifest_path: Path,
    selected_assets: list[SetupAssetSpec],
    unresolved_domains: list[dict[str, Any]],
    results: list[SetupAssetResult],
    strict: bool,
    dry_run: bool,
) -> None:
    """Write a machine-readable setup report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "manifest_path": str(manifest_path),
        "strict": strict,
        "dry_run": dry_run,
        "selected_assets": [asdict(asset) for asset in selected_assets],
        "results": [asdict(result) for result in results],
        "unresolved_domains": unresolved_domains,
    }
    report_path.write_text(json.dumps(report, indent=2))


def write_followup_script(path: Path, results: list[SetupAssetResult]) -> None:
    """Write a shell script of follow-up commands and manual instructions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated follow-up hints for curriculum setup.",
        "",
    ]
    for result in results:
        lines.append(f"# {result.key} [{result.status}]")
        lines.append(f"# {result.followup_hint}")
        lines.append("")
    path.write_text("\n".join(lines))


def summarize_results(results: list[SetupAssetResult]) -> dict[str, int]:
    """Count results by status."""
    summary: dict[str, int] = {}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    return summary


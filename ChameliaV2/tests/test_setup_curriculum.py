"""Tests for the curriculum setup manifest planner."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from training.curriculum.data.setup_manifest import (
    asset_specs_from_manifest,
    blocking_failures,
    load_setup_manifest,
    prepare_asset,
    select_asset_specs,
    summarize_results,
    unresolved_domains_from_manifest,
    write_followup_script,
    write_setup_report,
)


def _write_manifest(path: Path) -> Path:
    manifest = {
        "version": 1,
        "assets": [
            {
                "key": "generated_asset",
                "stage": 1,
                "domains": ["basic_arithmetic"],
                "kind": "generated_dataset",
                "install_mode": "generated",
                "setup_policy": "hard_fail_if_stage_enabled",
                "fetch_method": "local_generator",
                "source_url": "local://generator.py",
                "destination": "data/generated_asset",
                "estimated_size_gb": 0.1,
                "consumers": ["training/curriculum/domains/stage1_reasoning.py"],
                "notes": "Generated locally.",
            },
            {
                "key": "optional_asset",
                "stage": 1,
                "domains": ["lsat"],
                "kind": "dataset",
                "install_mode": "optional",
                "setup_policy": "soft_skip",
                "fetch_method": "huggingface_dataset",
                "source_url": "https://huggingface.co/datasets/example/reasoning",
                "destination": "data/optional_asset",
                "estimated_size_gb": 0.1,
                "consumers": ["training/curriculum/domains/stage1_reasoning.py"],
                "notes": "Optional reasoning corpus.",
            },
            {
                "key": "manual_asset",
                "stage": 5,
                "domains": ["medical_knowledge"],
                "kind": "manual_corpus",
                "install_mode": "manual",
                "setup_policy": "soft_skip",
                "fetch_method": "manual_drop",
                "source_url": "https://example.com/manual.pdf",
                "destination": "data/manual/manual.pdf",
                "estimated_size_gb": 0.2,
                "consumers": ["training/curriculum/domains/stage5_health.py"],
                "notes": "Manual PDF.",
            },
            {
                "key": "git_asset",
                "stage": 3,
                "domains": ["poker"],
                "kind": "framework",
                "install_mode": "default",
                "setup_policy": "hard_fail_if_stage_enabled",
                "fetch_method": "git_repo",
                "source_url": "https://github.com/example/repo.git",
                "destination": "external/repo",
                "estimated_size_gb": 0.1,
                "consumers": ["training/curriculum/generators/poker_env.py"],
                "notes": "Git-backed framework.",
            },
        ],
        "unresolved_domains": [{"domain": "gre", "fallback": "Use LSAT-style corpora"}],
    }
    path.write_text(yaml.safe_dump(manifest))
    return path


def test_setup_manifest_selection_and_statuses(tmp_path: Path) -> None:
    """Select assets by stage/domain and compute stable statuses."""
    manifest_path = _write_manifest(tmp_path / "manifest.yaml")
    manifest = load_setup_manifest(manifest_path)
    specs = asset_specs_from_manifest(manifest)

    selected = select_asset_specs(specs, stages=[1], domains=["basic_arithmetic", "lsat"])
    assert {asset.key for asset in selected} == {"generated_asset", "optional_asset"}

    generated = next(asset for asset in selected if asset.key == "generated_asset")
    optional = next(asset for asset in selected if asset.key == "optional_asset")

    generated_result = prepare_asset(
        generated,
        tmp_path,
        include_optional=False,
        include_gated=False,
        dry_run=False,
        force=False,
    )
    optional_result = prepare_asset(
        optional,
        tmp_path,
        include_optional=False,
        include_gated=False,
        dry_run=False,
        force=False,
    )

    assert generated_result.status == "generated_ready"
    assert optional_result.status == "optional_skipped"
    assert (tmp_path / "data" / "generated_asset").exists()


def test_setup_manifest_report_and_followups(tmp_path: Path) -> None:
    """Write setup report and follow-up shell hints."""
    manifest_path = _write_manifest(tmp_path / "manifest.yaml")
    manifest = load_setup_manifest(manifest_path)
    specs = asset_specs_from_manifest(manifest)
    unresolved = unresolved_domains_from_manifest(manifest)

    manual_asset = next(asset for asset in specs if asset.key == "manual_asset")
    result = prepare_asset(
        manual_asset,
        tmp_path,
        include_optional=False,
        include_gated=False,
        dry_run=False,
        force=False,
    )
    assert result.status == "manual_required"
    assert (tmp_path / "data" / "manual").exists()

    report_path = tmp_path / "report.json"
    followup_path = tmp_path / "followups.sh"
    write_setup_report(
        report_path,
        manifest_path=manifest_path,
        selected_assets=[manual_asset],
        unresolved_domains=unresolved,
        results=[result],
        strict=False,
        dry_run=False,
    )
    write_followup_script(followup_path, [result])

    report = json.loads(report_path.read_text())
    assert report["results"][0]["status"] == "manual_required"
    assert report["unresolved_domains"][0]["domain"] == "gre"
    followup_text = followup_path.read_text()
    assert "Place the file at" in followup_text


def test_setup_manifest_blocking_summary(tmp_path: Path) -> None:
    """Strict blocking uses setup policy, not just install mode."""
    manifest_path = _write_manifest(tmp_path / "manifest.yaml")
    manifest = load_setup_manifest(manifest_path)
    specs = asset_specs_from_manifest(manifest)
    git_asset = next(asset for asset in specs if asset.key == "git_asset")

    result = prepare_asset(
        git_asset,
        tmp_path,
        include_optional=False,
        include_gated=False,
        dry_run=True,
        force=False,
    )
    assert result.status == "git_clone_planned"

    blocking = blocking_failures([git_asset], [result])
    assert len(blocking) == 1
    summary = summarize_results([result])
    assert summary["git_clone_planned"] == 1

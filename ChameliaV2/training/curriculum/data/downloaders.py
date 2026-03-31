"""Scaffold download planners for curriculum data sources."""

from __future__ import annotations

import json
from pathlib import Path

from training.curriculum.data.sources import DataSourceSpec, stage_source_specs


def _write_manifest_entry(output_dir: str, source: DataSourceSpec) -> None:
    """Append one source plan to the download manifest."""
    manifest_path = Path(output_dir) / "download_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, object]] = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
    existing.append(
        {
            "name": source.name,
            "description": source.description,
            "estimated_size_gb": source.estimated_size_gb,
            "stage": source.stage,
            "status": "planned",
        }
    )
    manifest_path.write_text(json.dumps(existing, indent=2))


def _plan_download(stage: int, output_dir: str) -> None:
    """Write placeholder download plans for one stage."""
    for source in stage_source_specs().get(stage, []):
        _write_manifest_entry(output_dir, source)


def download_wikipedia(languages: list[str], output_dir: str) -> None:
    """Plan Wikipedia dump downloads for specified languages."""
    _ = languages
    _plan_download(0, output_dir)


def download_oeis_sequences(output_dir: str) -> None:
    """Plan OEIS sequence download."""
    _plan_download(2, output_dir)


def download_lsat_materials(output_dir: str) -> None:
    """Plan LSAT material download."""
    _plan_download(1, output_dir)


def download_chess_data(output_dir: str) -> None:
    """Plan chess data and engine download."""
    _plan_download(3, output_dir)


def download_pubmed_abstracts(output_dir: str) -> None:
    """Plan medical abstract download."""
    _plan_download(5, output_dir)


def download_arc_dataset(output_dir: str) -> None:
    """Plan ARC dataset download."""
    _plan_download(2, output_dir)


def download_medical_textbooks(output_dir: str) -> None:
    """Plan medical textbook download."""
    _plan_download(5, output_dir)


def download_all(output_dir: str, stages: list[int] | None = None) -> None:
    """Write a manifest for all requested curriculum downloads."""
    stage_ids = stages if stages is not None else [0, 1, 2, 3, 4, 5]
    for stage in stage_ids:
        _plan_download(stage, output_dir)

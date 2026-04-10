"""Path helpers for the protein DTI domain."""

from __future__ import annotations

import os
from pathlib import Path

_SWALLOWTAIL_ROOT = Path("/zfshomes/aparikh02/InSite/InSiteOfficial/ChameliaV2")
_SWALLOWTAIL_SCRATCH = Path("/sanscratch/aparikh02/protein_dti_tmp")


def repo_root() -> Path:
    """Return the local repository root."""
    return Path(__file__).resolve().parents[4]


def default_data_dir() -> Path:
    """Return the default local protein DTI data directory."""
    override = os.getenv("CHAMELIA_PROTEIN_DTI_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "data" / "protein_dti"


def default_scratch_dir() -> Path:
    """Return the default local scratch directory for acquisition jobs."""
    override = os.getenv("CHAMELIA_PROTEIN_DTI_SCRATCH_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "artifacts" / "protein_dti_tmp"


def swallowtail_root() -> Path:
    """Return the canonical swallowtail repo path."""
    return Path(os.getenv("CHAMELIA_SWALLOWTAIL_ROOT", str(_SWALLOWTAIL_ROOT)))


def swallowtail_data_dir() -> Path:
    """Return the canonical swallowtail data directory."""
    return swallowtail_root() / "data" / "protein_dti"


def swallowtail_scratch_dir() -> Path:
    """Return the canonical swallowtail scratch directory."""
    return Path(os.getenv("CHAMELIA_SWALLOWTAIL_SCRATCH_DIR", str(_SWALLOWTAIL_SCRATCH)))


def default_db_path() -> Path:
    """Return the default protein DTI metadata database path."""
    return default_data_dir() / "db" / "protein_dti.sqlite3"


def default_log_db_path() -> Path:
    """Return the default acquisition-log database path."""
    return default_data_dir() / "db" / "acquisition_log.sqlite3"


"""Shared helpers for protein DTI acquisition scripts."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def repo_root() -> Path:
    """Return the repository root."""
    return _REPO_ROOT


def default_data_dir() -> Path:
    """Return the repo-relative default data directory."""
    override = os.getenv("CHAMELIA_PROTEIN_DTI_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "data" / "protein_dti"


def default_scratch_dir() -> Path:
    """Return the repo-relative default scratch directory."""
    override = os.getenv("CHAMELIA_PROTEIN_DTI_SCRATCH_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "artifacts" / "protein_dti_tmp"


def default_db_path() -> Path:
    """Return the metadata DB path."""
    return default_data_dir() / "db" / "protein_dti.sqlite3"


def default_log_db_path() -> Path:
    """Return the acquisition-log DB path."""
    return default_data_dir() / "db" / "acquisition_log.sqlite3"


def default_log_dir() -> Path:
    """Return the log directory used by acquisition scripts."""
    return repo_root() / "logs" / "protein_data"


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).isoformat()


def stable_hash(*parts: Any) -> str:
    """Return a deterministic short hash key."""
    hasher = hashlib.blake2b(digest_size=16)
    for part in parts:
        hasher.update(str(part).encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def configure_logging(name: str, *, log_path: str | Path | None = None) -> logging.Logger:
    """Configure a simple console-plus-file logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def path_argument(parser: argparse.ArgumentParser, flag: str, default: Path, help_text: str) -> None:
    """Add a path-like CLI argument with a resolved default."""
    parser.add_argument(flag, type=str, default=str(default), help=help_text)


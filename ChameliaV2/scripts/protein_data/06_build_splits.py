"""Build train/val/test splits for the protein DTI dataset."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from common import configure_logging, default_data_dir, default_db_path, default_log_dir
from src.chamelia.domains.protein_dti.dataset import (
    build_protein_family_splits,
    build_random_splits,
)
from src.chamelia.domains.protein_dti.features import write_feature_hdf5

LOG = configure_logging(
    "protein_data.build_splits",
    log_path=default_log_dir() / "06_build_splits.log",
)


def run(
    db_path: Path,
    data_dir: Path,
    *,
    affinity_type: str,
    strategy: str,
    seed: int,
    write_hdf5: bool,
) -> None:
    conn = sqlite3.connect(db_path)
    try:
        if strategy in {"random", "all"}:
            count = build_random_splits(conn, affinity_type=affinity_type, seed=seed)
            LOG.info("Built random splits for %s pair assignments", count)
        if strategy in {"protein_family", "all"}:
            count = build_protein_family_splits(conn, affinity_type=affinity_type, seed=seed)
            LOG.info("Built protein-family splits for %s pair assignments", count)
        if strategy == "temporal":
            raise NotImplementedError("Temporal split generation is still a TODO.")
    finally:
        conn.close()

    if write_hdf5:
        report = write_feature_hdf5(
            db_path,
            data_dir / "hdf5" / "features.h5",
        )
        LOG.info("Wrote feature HDF5: %s", report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build protein DTI split tables.")
    parser.add_argument("--db", type=str, default=str(default_db_path()))
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir()))
    parser.add_argument("--affinity-type", type=str, default="Kd")
    parser.add_argument(
        "--strategy",
        type=str,
        default="all",
        choices=["random", "protein_family", "all", "temporal"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-hdf5", action="store_true")
    args = parser.parse_args()
    run(
        Path(args.db),
        Path(args.data_dir),
        affinity_type=args.affinity_type,
        strategy=args.strategy,
        seed=args.seed,
        write_hdf5=bool(args.write_hdf5),
    )


if __name__ == "__main__":
    main()

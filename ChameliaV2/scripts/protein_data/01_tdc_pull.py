"""Pull DTI datasets from TDC into the protein DTI metadata database."""

from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

from common import (
    configure_logging,
    default_db_path,
    default_log_dir,
    default_scratch_dir,
    stable_hash,
    utc_now,
)

LOG = configure_logging(
    "protein_data.tdc_pull",
    log_path=default_log_dir() / "01_tdc_pull.log",
)

DATASETS = [
    ("BindingDB_Kd", "Kd"),
    ("BindingDB_Ki", "Ki"),
    ("DAVIS", "Ki"),
    ("KIBA", "KIBA"),
]


def to_affinity_scale(value: float, affinity_type: str) -> float | None:
    """Convert raw values to the shared training scale."""
    if affinity_type == "KIBA":
        return float(value)
    if value <= 0.0:
        return None
    return -math.log10(float(value) * 1.0e-9)


def _string_field(row: object, *keys: str) -> str:
    for key in keys:
        value = getattr(row, "get", lambda _key, _default=None: None)(key, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def run(db_path: Path, scratch_dir: Path) -> None:
    from tdc.multi_pred import DTI  # type: ignore[import-untyped]

    scratch_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")

        for dataset_name, affinity_type in DATASETS:
            LOG.info("Pulling %s", dataset_name)
            dataset = DTI(name=dataset_name, path=str(scratch_dir))
            frame = dataset.get_data()
            inserted = 0
            skipped = 0

            for _, row in frame.iterrows():
                uniprot_id = _string_field(row, "Target_ID", "UniProt_ID")
                smiles = _string_field(row, "Drug")
                sequence = _string_field(row, "Target")
                chembl_id = _string_field(row, "Drug_ID", "ChEMBL_ID")
                if not chembl_id:
                    chembl_id = stable_hash("drug", smiles)[:20]
                raw_value = row.get("Y")
                assay_id = _string_field(row, "Assay_ID", "AssayID")
                if not uniprot_id or not smiles or not sequence or raw_value is None:
                    skipped += 1
                    continue
                affinity_value = to_affinity_scale(float(raw_value), affinity_type)
                if affinity_value is None:
                    skipped += 1
                    continue
                measurement_key = stable_hash(
                    dataset_name,
                    uniprot_id,
                    chembl_id,
                    affinity_type,
                    raw_value,
                    assay_id,
                )
                created_at = utc_now()
                conn.execute(
                    """
                    INSERT OR IGNORE INTO proteins
                        (uniprot_id, sequence, sequence_length, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (uniprot_id, sequence, len(sequence), created_at, created_at),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO drugs
                        (chembl_id, smiles, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (chembl_id, smiles, created_at, created_at),
                )
                conn.execute(
                    """
                    INSERT OR IGNORE INTO binding_affinities
                        (measurement_key, uniprot_id, chembl_id, affinity_value, affinity_type,
                         source_dataset, original_value, original_units, assay_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        measurement_key,
                        uniprot_id,
                        chembl_id,
                        affinity_value,
                        affinity_type,
                        dataset_name,
                        float(raw_value),
                        None if affinity_type == "KIBA" else "nM",
                        assay_id or None,
                        created_at,
                    ),
                )
                inserted += 1
            conn.commit()
            LOG.info("%s complete: inserted=%s skipped=%s", dataset_name, inserted, skipped)

        total_proteins = conn.execute("SELECT COUNT(*) FROM proteins").fetchone()[0]
        total_drugs = conn.execute("SELECT COUNT(*) FROM drugs").fetchone()[0]
        total_pairs = conn.execute("SELECT COUNT(*) FROM binding_affinities").fetchone()[0]
        LOG.info(
            "Totals: proteins=%s drugs=%s measurements=%s",
            total_proteins,
            total_drugs,
            total_pairs,
        )
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull TDC DTI datasets into SQLite.")
    parser.add_argument("--db", type=str, default=str(default_db_path()))
    parser.add_argument("--scratch", type=str, default=str(default_scratch_dir()))
    args = parser.parse_args()
    run(Path(args.db), Path(args.scratch))


if __name__ == "__main__":
    main()


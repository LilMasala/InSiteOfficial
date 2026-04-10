"""Initialize the protein DTI SQLite schemas."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from common import configure_logging, default_db_path, default_log_db_path

LOG = configure_logging("protein_data.init_schema")

DATA_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS proteins (
    uniprot_id TEXT PRIMARY KEY,
    sequence TEXT NOT NULL,
    sequence_length INTEGER NOT NULL,
    organism TEXT,
    protein_name TEXT,
    gene_name TEXT,
    best_pdb_id TEXT,
    best_pdb_resolution REAL,
    structure_source TEXT,
    structure_path TEXT,
    graph_path TEXT,
    uniprot_fetched_at TEXT,
    structure_fetched_at TEXT,
    go_fetched_at TEXT,
    cath_fetched_at TEXT,
    graph_built_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS protein_go_terms (
    uniprot_id TEXT NOT NULL,
    go_id TEXT NOT NULL,
    go_aspect TEXT NOT NULL,
    go_term TEXT NOT NULL,
    evidence_code TEXT,
    PRIMARY KEY (uniprot_id, go_id),
    FOREIGN KEY (uniprot_id) REFERENCES proteins(uniprot_id)
);

CREATE TABLE IF NOT EXISTS protein_cath (
    uniprot_id TEXT NOT NULL,
    cath_id TEXT NOT NULL,
    cath_class TEXT,
    cath_arch TEXT,
    cath_topology TEXT,
    cath_homology TEXT,
    PRIMARY KEY (uniprot_id, cath_id),
    FOREIGN KEY (uniprot_id) REFERENCES proteins(uniprot_id)
);

CREATE TABLE IF NOT EXISTS protein_pdb_structures (
    uniprot_id TEXT NOT NULL,
    pdb_id TEXT NOT NULL,
    resolution REAL,
    has_ligand INTEGER NOT NULL DEFAULT 0,
    chain_id TEXT,
    deposition_date TEXT,
    rank INTEGER,
    PRIMARY KEY (uniprot_id, pdb_id),
    FOREIGN KEY (uniprot_id) REFERENCES proteins(uniprot_id)
);

CREATE TABLE IF NOT EXISTS drugs (
    chembl_id TEXT PRIMARY KEY,
    smiles TEXT NOT NULL,
    inchi_key TEXT,
    mol_weight REAL,
    logp REAL,
    hbd INTEGER,
    hba INTEGER,
    rotatable_bonds INTEGER,
    graph_path TEXT,
    pubchem_cid TEXT,
    drug_name TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS binding_affinities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    measurement_key TEXT NOT NULL UNIQUE,
    uniprot_id TEXT NOT NULL,
    chembl_id TEXT NOT NULL,
    affinity_value REAL NOT NULL,
    affinity_type TEXT NOT NULL,
    source_dataset TEXT NOT NULL,
    original_value REAL,
    original_units TEXT,
    assay_id TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (uniprot_id) REFERENCES proteins(uniprot_id),
    FOREIGN KEY (chembl_id) REFERENCES drugs(chembl_id)
);

CREATE VIEW IF NOT EXISTS binding_affinities_deduped AS
SELECT
    uniprot_id,
    chembl_id,
    AVG(affinity_value) AS affinity_value,
    affinity_type,
    COUNT(*) AS num_measurements,
    MIN(source_dataset) AS primary_source,
    GROUP_CONCAT(DISTINCT source_dataset) AS all_sources
FROM binding_affinities
GROUP BY uniprot_id, chembl_id, affinity_type;

CREATE TABLE IF NOT EXISTS dataset_splits (
    uniprot_id TEXT NOT NULL,
    chembl_id TEXT NOT NULL,
    affinity_type TEXT NOT NULL,
    split TEXT NOT NULL,
    split_strategy TEXT NOT NULL,
    PRIMARY KEY (uniprot_id, chembl_id, affinity_type, split_strategy)
);

CREATE INDEX IF NOT EXISTS idx_affinities_protein ON binding_affinities(uniprot_id);
CREATE INDEX IF NOT EXISTS idx_affinities_drug ON binding_affinities(chembl_id);
CREATE INDEX IF NOT EXISTS idx_affinities_type ON binding_affinities(affinity_type);
CREATE INDEX IF NOT EXISTS idx_go_protein ON protein_go_terms(uniprot_id);
CREATE INDEX IF NOT EXISTS idx_cath_protein ON protein_cath(uniprot_id);
CREATE INDEX IF NOT EXISTS idx_pdb_protein ON protein_pdb_structures(uniprot_id);
CREATE INDEX IF NOT EXISTS idx_splits_split ON dataset_splits(split);
"""

LOG_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS api_health_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checked_at TEXT NOT NULL,
    api_name TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms REAL,
    http_status INTEGER,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS acquisition_jobs (
    job_id TEXT PRIMARY KEY,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    records_attempted INTEGER DEFAULT 0,
    records_succeeded INTEGER DEFAULT 0,
    records_failed INTEGER DEFAULT 0,
    records_skipped INTEGER DEFAULT 0,
    error_log TEXT
);

CREATE TABLE IF NOT EXISTS acquisition_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uniprot_id TEXT,
    chembl_id TEXT,
    job_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    attempt_count INTEGER DEFAULT 1,
    last_attempted TEXT NOT NULL,
    resolved INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_failures_unresolved
    ON acquisition_failures(resolved, job_type);
CREATE INDEX IF NOT EXISTS idx_health_api
    ON api_health_checks(api_name, checked_at);
"""


def _apply_schema(db_path: Path, schema: str) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema)
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize protein DTI SQLite schemas.")
    parser.add_argument("--db", type=str, default=str(default_db_path()))
    parser.add_argument("--log-db", type=str, default=str(default_log_db_path()))
    args = parser.parse_args()

    data_db = Path(args.db)
    log_db = Path(args.log_db)
    _apply_schema(data_db, DATA_SCHEMA)
    _apply_schema(log_db, LOG_SCHEMA)
    LOG.info("Initialized %s", data_db)
    LOG.info("Initialized %s", log_db)


if __name__ == "__main__":
    main()


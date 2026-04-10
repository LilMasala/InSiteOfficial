"""Enrich protein DTI drug stubs with RDKit and PubChem metadata."""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import requests

from common import configure_logging, default_db_path, default_log_db_path, default_log_dir, utc_now

LOG = configure_logging(
    "protein_data.enrich_drugs",
    log_path=default_log_dir() / "03_enrich_drugs.log",
)
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"


def compute_molecule_properties(smiles: str) -> dict[str, object] | None:
    from rdkit import Chem  # type: ignore[import-untyped]
    from rdkit.Chem import Descriptors, rdMolDescriptors  # type: ignore[import-untyped]

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return {
        "inchi_key": Chem.MolToInchiKey(molecule),
        "mol_weight": Descriptors.MolWt(molecule),
        "logp": Descriptors.MolLogP(molecule),
        "hbd": rdMolDescriptors.CalcNumHBD(molecule),
        "hba": rdMolDescriptors.CalcNumHBA(molecule),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(molecule),
    }


def fetch_pubchem_cid(smiles: str) -> str | None:
    try:
        response = requests.get(
            f"{PUBCHEM_BASE}/smiles/{requests.utils.quote(smiles)}/cids/JSON",
            timeout=15,
        )
    except requests.RequestException:
        return None
    if not response.ok:
        return None
    cids = response.json().get("IdentifierList", {}).get("CID", [])
    return str(cids[0]) if cids else None


def run(db_path: Path, log_db_path: Path, requests_per_second: float) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    sleep_time = 1.0 / max(requests_per_second, 0.1)
    pending = conn.execute(
        """
        SELECT chembl_id, smiles
        FROM drugs
        WHERE mol_weight IS NULL
        ORDER BY chembl_id
        """
    ).fetchall()
    LOG.info("Pending drugs: %s", len(pending))
    succeeded = 0
    failed = 0
    for index, row in enumerate(pending, start=1):
        chembl_id = str(row["chembl_id"])
        smiles = str(row["smiles"])
        try:
            props = compute_molecule_properties(smiles)
            if props is None:
                raise ValueError("Invalid SMILES")
            cid = fetch_pubchem_cid(smiles)
            conn.execute(
                """
                UPDATE drugs
                SET inchi_key = ?, mol_weight = ?, logp = ?, hbd = ?, hba = ?,
                    rotatable_bonds = ?, pubchem_cid = ?, updated_at = ?
                WHERE chembl_id = ?
                """,
                (
                    props["inchi_key"],
                    props["mol_weight"],
                    props["logp"],
                    props["hbd"],
                    props["hba"],
                    props["rotatable_bonds"],
                    cid,
                    utc_now(),
                    chembl_id,
                ),
            )
            conn.commit()
            succeeded += 1
        except Exception as exc:
            failed += 1
            log_conn = sqlite3.connect(log_db_path)
            try:
                log_conn.execute(
                    """
                    INSERT INTO acquisition_failures
                        (chembl_id, job_type, error_message, attempt_count, last_attempted)
                    VALUES (?, 'enrich_drugs', ?, 1, ?)
                    """,
                    (chembl_id, str(exc), utc_now()),
                )
                log_conn.commit()
            finally:
                log_conn.close()
            LOG.warning("Failed %s: %s", chembl_id, exc)
        time.sleep(sleep_time)
        if index % 500 == 0:
            LOG.info("Progress %s/%s succeeded=%s failed=%s", index, len(pending), succeeded, failed)
    conn.close()
    LOG.info("Completed enrich_drugs succeeded=%s failed=%s", succeeded, failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich drug records.")
    parser.add_argument("--db", type=str, default=str(default_db_path()))
    parser.add_argument("--log-db", type=str, default=str(default_log_db_path()))
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    args = parser.parse_args()
    run(Path(args.db), Path(args.log_db), args.requests_per_second)


if __name__ == "__main__":
    main()


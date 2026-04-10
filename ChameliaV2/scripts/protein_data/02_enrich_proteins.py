"""Enrich protein records with UniProt, PDB, and AlphaFold metadata."""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path
from typing import Any

import requests

from common import configure_logging, default_data_dir, default_db_path, default_log_db_path, default_log_dir, utc_now

LOG = configure_logging(
    "protein_data.enrich_proteins",
    log_path=default_log_dir() / "02_enrich_proteins.log",
)

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"
RCSB_BASE = "https://data.rcsb.org/rest/v1/core"
RCSB_FILE_BASE = "https://files.rcsb.org/download"
ALPHAFOLD_BASE = "https://alphafold.ebi.ac.uk/api/prediction"
ALLOWED_NONPOLYMERS = {
    "HOH",
    "DOD",
    "GOL",
    "EDO",
    "PEG",
    "ACT",
    "FMT",
    "TRS",
    "MES",
    "SO4",
    "PO4",
    "CL",
    "BR",
    "IOD",
    "ZN",
    "MG",
    "CA",
    "NA",
    "K",
    "MN",
    "FE",
    "CU",
    "CO",
    "NI",
    "HEM",
    "HEC",
    "FAD",
    "FMN",
    "NAD",
    "NAP",
    "SAM",
    "SAH",
    "PLP",
    "TDP",
    "TPP",
    "COA",
}


def get_with_retry(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 20.0,
    max_retries: int = 4,
    backoff: float = 2.0,
) -> requests.Response | None:
    """Make a GET request with small retry/backoff handling."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 429:
                wait = backoff * (2 ** attempt)
                LOG.warning("Rate limited at %s; sleeping %.1fs", url, wait)
                time.sleep(wait)
                continue
            return response
        except requests.RequestException as exc:
            if attempt + 1 >= max_retries:
                raise exc
            time.sleep(backoff * (2 ** attempt))
    return None


def _cross_references(data: dict[str, Any]) -> list[dict[str, Any]]:
    refs = data.get("uniProtKBCrossReferences")
    if isinstance(refs, list):
        return refs
    refs = data.get("dbReferences")
    if isinstance(refs, list):
        return refs
    return []


def _ref_database(ref: dict[str, Any]) -> str:
    return str(ref.get("database") or ref.get("type") or "")


def _ref_id(ref: dict[str, Any]) -> str:
    return str(ref.get("id") or "")


def _ref_properties(ref: dict[str, Any]) -> dict[str, str]:
    props = ref.get("properties")
    if isinstance(props, list):
        return {str(item.get("key")): str(item.get("value")) for item in props if isinstance(item, dict)}
    if isinstance(props, dict):
        return {str(key): str(value) for key, value in props.items()}
    return {}


def parse_go_terms(data: dict[str, Any]) -> list[dict[str, str | None]]:
    terms: list[dict[str, str | None]] = []
    for ref in _cross_references(data):
        if _ref_database(ref) != "GO":
            continue
        props = _ref_properties(ref)
        aspect_raw = props.get("GoTerm", "")
        if ":" in aspect_raw:
            aspect_letter, term = aspect_raw.split(":", 1)
        else:
            aspect_letter, term = "?", aspect_raw
        terms.append(
            {
                "go_id": _ref_id(ref),
                "go_aspect": aspect_letter.strip(),
                "go_term": term.strip(),
                "evidence_code": props.get("GoEvidenceType"),
            }
        )
    return terms


def parse_cath_ids(data: dict[str, Any]) -> list[str]:
    return [_ref_id(ref) for ref in _cross_references(data) if _ref_database(ref) == "CATH"]


def parse_pdb_ids(data: dict[str, Any]) -> list[str]:
    return [_ref_id(ref) for ref in _cross_references(data) if _ref_database(ref) == "PDB"]


def _nested(payload: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = payload
        ok = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                ok = False
                break
            current = current[key]
        if ok:
            return current
    return None


def _extract_formula_weight(entity: dict[str, Any]) -> float | None:
    value = _nested(
        entity,
        ("chem_comp", "formula_weight"),
        ("rcsb_chem_comp_info", "formula_weight"),
        ("rcsb_nonpolymer_entity", "formula_weight"),
    )
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_heavy_atom_count(entity: dict[str, Any]) -> int | None:
    value = _nested(
        entity,
        ("chem_comp", "heavy_atom_count"),
        ("rcsb_chem_comp_info", "heavy_atom_count"),
        ("rcsb_nonpolymer_entity", "heavy_atom_count"),
    )
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_comp_id(entity: dict[str, Any]) -> str:
    value = _nested(
        entity,
        ("pdbx_entity_nonpoly", "comp_id"),
        ("chem_comp", "id"),
        ("chem_comp", "chem_comp_id"),
    )
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def _entity_ids(entry: dict[str, Any]) -> list[str]:
    raw = _nested(
        entry,
        ("rcsb_entry_container_identifiers", "non_polymer_entity_ids"),
        ("entry", "nonpolymer_entity_ids"),
    )
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def has_drug_like_ligand(pdb_id: str, entry: dict[str, Any]) -> bool:
    """Return ``True`` when a structure appears to contain a drug-like ligand."""
    entity_ids = _entity_ids(entry)
    if not entity_ids:
        count = _nested(entry, ("rcsb_entry_info", "nonpolymer_entity_count"))
        try:
            return int(count or 0) > 0
        except (TypeError, ValueError):
            return False
    for entity_id in entity_ids:
        response = get_with_retry(f"{RCSB_BASE}/nonpolymer_entity/{pdb_id}/{entity_id}")
        if response is None or not response.ok:
            continue
        entity = response.json()
        comp_id = _extract_comp_id(entity).upper()
        if comp_id in ALLOWED_NONPOLYMERS:
            continue
        formula_weight = _extract_formula_weight(entity)
        heavy_atoms = _extract_heavy_atom_count(entity)
        if (formula_weight is not None and formula_weight > 150.0) or (
            heavy_atoms is not None and heavy_atoms > 5
        ):
            return True
    return False


def rank_pdb_structures(pdb_ids: list[str]) -> list[dict[str, Any]]:
    structures: list[dict[str, Any]] = []
    for pdb_id in pdb_ids:
        response = get_with_retry(f"{RCSB_BASE}/entry/{pdb_id}")
        if response is None or not response.ok:
            continue
        entry = response.json()
        resolution_raw = _nested(entry, ("rcsb_entry_info", "resolution_combined"))
        if isinstance(resolution_raw, list):
            resolution = resolution_raw[0] if resolution_raw else None
        else:
            resolution = resolution_raw
        deposition_date = _nested(entry, ("rcsb_accession_info", "initial_release_date"))
        has_ligand = has_drug_like_ligand(pdb_id, entry)
        structures.append(
            {
                "pdb_id": pdb_id,
                "resolution": float(resolution) if resolution is not None else None,
                "has_ligand": int(has_ligand),
                "deposition_date": deposition_date,
                "chain_id": None,
            }
        )
    structures.sort(
        key=lambda item: (
            item["has_ligand"],
            item["resolution"] if item["resolution"] is not None else 99.0,
            str(item.get("deposition_date") or ""),
        )
    )
    for rank, item in enumerate(structures, start=1):
        item["rank"] = rank
    return structures


def fetch_uniprot(uniprot_id: str) -> dict[str, Any] | None:
    response = get_with_retry(f"{UNIPROT_BASE}/{uniprot_id}", params={"format": "json"})
    if response is None or not response.ok:
        return None
    return response.json()


def fetch_pdb_structure(pdb_id: str, destination: Path) -> bool:
    if destination.exists():
        return True
    response = get_with_retry(f"{RCSB_FILE_BASE}/{pdb_id}.cif", timeout=60.0)
    if response is None or not response.ok:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    return True


def fetch_alphafold_structure(uniprot_id: str, destination: Path) -> bool:
    if destination.exists():
        return True
    response = get_with_retry(f"{ALPHAFOLD_BASE}/{uniprot_id}")
    if response is None or not response.ok:
        return False
    entries = response.json()
    if not isinstance(entries, list) or not entries:
        return False
    cif_url = entries[0].get("cifUrl")
    if not cif_url:
        return False
    cif_response = get_with_retry(str(cif_url), timeout=60.0)
    if cif_response is None or not cif_response.ok:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(cif_response.content)
    return True


def _protein_name(data: dict[str, Any]) -> str | None:
    description = data.get("proteinDescription", {})
    recommended = description.get("recommendedName", {})
    full_name = recommended.get("fullName", {})
    if isinstance(full_name, dict):
        value = full_name.get("value")
        if value:
            return str(value)
    submission = description.get("submissionNames")
    if isinstance(submission, list) and submission:
        full = submission[0].get("fullName", {})
        if isinstance(full, dict) and full.get("value"):
            return str(full["value"])
    return None


def _gene_name(data: dict[str, Any]) -> str | None:
    genes = data.get("genes")
    if not isinstance(genes, list) or not genes:
        return None
    gene_name = genes[0].get("geneName", {})
    if isinstance(gene_name, dict) and gene_name.get("value"):
        return str(gene_name["value"])
    return None


def enrich_one(
    conn: sqlite3.Connection,
    log_db_path: Path,
    *,
    uniprot_id: str,
    data_dir: Path,
    requests_per_second: float,
) -> bool:
    sleep_time = 1.0 / max(requests_per_second, 0.1)
    data = fetch_uniprot(uniprot_id)
    time.sleep(sleep_time)
    if data is None:
        return False

    organism = _nested(data, ("organism", "scientificName"))
    protein_name = _protein_name(data)
    gene_name = _gene_name(data)
    go_terms = parse_go_terms(data)
    cath_ids = parse_cath_ids(data)
    pdb_ids = parse_pdb_ids(data)
    now = utc_now()

    conn.execute(
        """
        UPDATE proteins
        SET organism = ?, protein_name = ?, gene_name = ?,
            uniprot_fetched_at = ?, updated_at = ?
        WHERE uniprot_id = ?
        """,
        (organism, protein_name, gene_name, now, now, uniprot_id),
    )
    for go_term in go_terms:
        conn.execute(
            """
            INSERT OR REPLACE INTO protein_go_terms
                (uniprot_id, go_id, go_aspect, go_term, evidence_code)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                uniprot_id,
                go_term["go_id"],
                go_term["go_aspect"],
                go_term["go_term"],
                go_term["evidence_code"],
            ),
        )
    for cath_id in cath_ids:
        parts = cath_id.split(".")
        conn.execute(
            """
            INSERT OR REPLACE INTO protein_cath
                (uniprot_id, cath_id, cath_class, cath_arch, cath_topology, cath_homology)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                uniprot_id,
                cath_id,
                parts[0] if len(parts) > 0 else None,
                parts[1] if len(parts) > 1 else None,
                parts[2] if len(parts) > 2 else None,
                parts[3] if len(parts) > 3 else None,
            ),
        )
    conn.execute(
        """
        UPDATE proteins
        SET go_fetched_at = ?, cath_fetched_at = ?, updated_at = ?
        WHERE uniprot_id = ?
        """,
        (utc_now(), utc_now(), utc_now(), uniprot_id),
    )

    best_pdb_id: str | None = None
    best_resolution: float | None = None
    structure_source: str | None = None
    structure_path: str | None = None
    if pdb_ids:
        structures = rank_pdb_structures(pdb_ids)
        time.sleep(sleep_time)
        for structure in structures:
            conn.execute(
                """
                INSERT OR REPLACE INTO protein_pdb_structures
                    (uniprot_id, pdb_id, resolution, has_ligand, chain_id, deposition_date, rank)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uniprot_id,
                    structure["pdb_id"],
                    structure["resolution"],
                    structure["has_ligand"],
                    structure["chain_id"],
                    structure["deposition_date"],
                    structure["rank"],
                ),
            )
        apo_structures = [item for item in structures if not item["has_ligand"]]
        if apo_structures:
            best = apo_structures[0]
            destination = data_dir / "structures" / "pdb" / f"{best['pdb_id']}.cif"
            if fetch_pdb_structure(str(best["pdb_id"]), destination):
                best_pdb_id = str(best["pdb_id"])
                best_resolution = best["resolution"]
                structure_source = "experimental"
                structure_path = f"structures/pdb/{best_pdb_id}.cif"
            time.sleep(sleep_time)

    if structure_source is None:
        destination = data_dir / "structures" / "alphafold" / f"{uniprot_id}.cif"
        if fetch_alphafold_structure(uniprot_id, destination):
            structure_source = "alphafold"
            structure_path = f"structures/alphafold/{uniprot_id}.cif"
        time.sleep(sleep_time)

    conn.execute(
        """
        UPDATE proteins
        SET best_pdb_id = ?, best_pdb_resolution = ?, structure_source = ?, structure_path = ?,
            structure_fetched_at = ?, updated_at = ?
        WHERE uniprot_id = ?
        """,
        (best_pdb_id, best_resolution, structure_source, structure_path, utc_now(), utc_now(), uniprot_id),
    )
    conn.commit()
    return True


def run(db_path: Path, data_dir: Path, log_db_path: Path, requests_per_second: float) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    pending = conn.execute(
        """
        SELECT uniprot_id
        FROM proteins
        WHERE uniprot_fetched_at IS NULL
        ORDER BY uniprot_id
        """
    ).fetchall()
    LOG.info("Pending proteins: %s", len(pending))
    succeeded = 0
    failed = 0
    for index, row in enumerate(pending, start=1):
        uniprot_id = str(row["uniprot_id"])
        try:
            ok = enrich_one(
                conn,
                log_db_path,
                uniprot_id=uniprot_id,
                data_dir=data_dir,
                requests_per_second=requests_per_second,
            )
            if ok:
                succeeded += 1
            else:
                raise RuntimeError("UniProt enrichment failed")
        except Exception as exc:
            failed += 1
            log_conn = sqlite3.connect(log_db_path)
            try:
                log_conn.execute(
                    """
                    INSERT INTO acquisition_failures
                        (uniprot_id, job_type, error_message, attempt_count, last_attempted)
                    VALUES (?, 'enrich_proteins', ?, 1, ?)
                    """,
                    (uniprot_id, str(exc), utc_now()),
                )
                log_conn.commit()
            finally:
                log_conn.close()
            LOG.warning("Failed %s: %s", uniprot_id, exc)
        if index % 100 == 0:
            LOG.info("Progress %s/%s succeeded=%s failed=%s", index, len(pending), succeeded, failed)
    conn.close()
    LOG.info("Completed protein enrichment succeeded=%s failed=%s", succeeded, failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich protein records from UniProt and structure APIs.")
    parser.add_argument("--db", type=str, default=str(default_db_path()))
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir()))
    parser.add_argument("--log-db", type=str, default=str(default_log_db_path()))
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    args = parser.parse_args()
    run(
        Path(args.db),
        Path(args.data_dir),
        Path(args.log_db),
        args.requests_per_second,
    )


if __name__ == "__main__":
    main()

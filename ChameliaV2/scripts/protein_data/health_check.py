"""Health checks for protein DTI acquisition jobs."""

from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import requests

from common import configure_logging, default_db_path, default_log_db_path, default_log_dir, utc_now

LOG = configure_logging(
    "protein_data.health_check",
    log_path=default_log_dir() / "health.log",
)

INTERVAL_SECONDS = 4 * 3600
ENDPOINTS = {
    "uniprot": ("https://rest.uniprot.org/uniprotkb/P12345", {"format": "json"}),
    "rcsb": ("https://data.rcsb.org/rest/v1/core/entry/1CRN", None),
    "alphafold": ("https://alphafold.ebi.ac.uk/api/prediction/P12345", None),
    "pubchem": ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/JSON", None),
}
INTEGRITY_QUERIES = {
    "total_proteins": "SELECT COUNT(*) FROM proteins",
    "proteins_with_structure": "SELECT COUNT(*) FROM proteins WHERE structure_path IS NOT NULL",
    "proteins_with_graph": "SELECT COUNT(*) FROM proteins WHERE graph_path IS NOT NULL",
    "total_drugs": "SELECT COUNT(*) FROM drugs",
    "drugs_with_graph": "SELECT COUNT(*) FROM drugs WHERE graph_path IS NOT NULL",
    "total_measurements": "SELECT COUNT(*) FROM binding_affinities",
}


def check_api(log_conn: sqlite3.Connection, name: str, url: str, params: dict[str, str] | None) -> None:
    start = time.perf_counter()
    try:
        response = requests.get(url, params=params, timeout=15)
        latency_ms = (time.perf_counter() - start) * 1000.0
        status = "ok" if response.ok else "error"
        if latency_ms > 5000.0 and response.ok:
            status = "slow"
        http_status = response.status_code
        error_message = None if response.ok else response.text[:200]
    except requests.Timeout:
        latency_ms = None
        status = "timeout"
        http_status = None
        error_message = "Request timed out"
    except requests.RequestException as exc:
        latency_ms = None
        status = "error"
        http_status = None
        error_message = str(exc)[:200]

    log_conn.execute(
        """
        INSERT INTO api_health_checks
            (checked_at, api_name, endpoint, status, latency_ms, http_status, error_message)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (utc_now(), name, url, status, latency_ms, http_status, error_message),
    )
    LOG.info("%s status=%s latency_ms=%s", name, status, latency_ms)


def check_integrity(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        for label, query in INTEGRITY_QUERIES.items():
            count = conn.execute(query).fetchone()[0]
            LOG.info("%s=%s", label, count)
    finally:
        conn.close()


def run_once(db_path: Path, log_db_path: Path) -> None:
    LOG.info("=== health check %s ===", utc_now())
    log_conn = sqlite3.connect(log_db_path)
    try:
        for name, (url, params) in ENDPOINTS.items():
            check_api(log_conn, name, url, params)
        log_conn.commit()
    finally:
        log_conn.close()
    check_integrity(db_path)


def run_forever(db_path: Path, log_db_path: Path, interval_seconds: int) -> None:
    while True:
        run_once(db_path, log_db_path)
        time.sleep(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run protein DTI API and integrity health checks.")
    parser.add_argument("--db", type=str, default=str(default_db_path()))
    parser.add_argument("--log-db", type=str, default=str(default_log_db_path()))
    parser.add_argument("--interval-seconds", type=int, default=INTERVAL_SECONDS)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.once:
        run_once(Path(args.db), Path(args.log_db))
    else:
        run_forever(Path(args.db), Path(args.log_db), int(args.interval_seconds))


if __name__ == "__main__":
    main()

"""Build drug graph payloads from SMILES strings."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from common import configure_logging, default_data_dir, default_db_path, default_log_dir, utc_now
from src.chamelia.domains.protein_dti.graph_builder import build_drug_graph, save_graph_record

LOG = configure_logging(
    "protein_data.build_drug_graphs",
    log_path=default_log_dir() / "05_build_drug_graphs.log",
)


def run(db_path: Path, data_dir: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    pending = conn.execute(
        """
        SELECT chembl_id, smiles
        FROM drugs
        WHERE smiles IS NOT NULL
          AND graph_path IS NULL
        ORDER BY chembl_id
        """
    ).fetchall()
    LOG.info("Pending drug graphs: %s", len(pending))
    succeeded = 0
    failed = 0
    for index, row in enumerate(pending, start=1):
        chembl_id = str(row["chembl_id"])
        smiles = str(row["smiles"])
        output_path = data_dir / "graphs" / "drugs" / f"{chembl_id}.pt"
        try:
            graph = build_drug_graph(smiles, chembl_id)
            if graph is None:
                raise ValueError("Could not build drug graph")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_graph_record(graph, output_path)
            conn.execute(
                """
                UPDATE drugs
                SET graph_path = ?, updated_at = ?
                WHERE chembl_id = ?
                """,
                (f"graphs/drugs/{chembl_id}.pt", utc_now(), chembl_id),
            )
            conn.commit()
            succeeded += 1
        except Exception as exc:
            failed += 1
            LOG.warning("Failed %s: %s", chembl_id, exc)
        if index % 1000 == 0:
            LOG.info("Progress %s/%s succeeded=%s failed=%s", index, len(pending), succeeded, failed)
    conn.close()
    LOG.info("Completed drug graph build succeeded=%s failed=%s", succeeded, failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build drug graph payloads.")
    parser.add_argument("--db", type=str, default=str(default_db_path()))
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir()))
    args = parser.parse_args()
    run(Path(args.db), Path(args.data_dir))


if __name__ == "__main__":
    main()


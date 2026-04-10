"""Build protein graph payloads from mmCIF structures."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from common import configure_logging, default_data_dir, default_db_path, default_log_dir, utc_now
from src.chamelia.domains.protein_dti.graph_builder import build_protein_graph, save_graph_record

LOG = configure_logging(
    "protein_data.build_protein_graphs",
    log_path=default_log_dir() / "04_build_protein_graphs.log",
)


def run(db_path: Path, data_dir: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    pending = conn.execute(
        """
        SELECT uniprot_id, structure_path
        FROM proteins
        WHERE structure_path IS NOT NULL
          AND graph_built_at IS NULL
        ORDER BY uniprot_id
        """
    ).fetchall()
    LOG.info("Pending protein graphs: %s", len(pending))
    succeeded = 0
    failed = 0
    for index, row in enumerate(pending, start=1):
        uniprot_id = str(row["uniprot_id"])
        structure_path = data_dir / str(row["structure_path"])
        output_path = data_dir / "graphs" / "proteins" / f"{uniprot_id}.pt"
        try:
            graph = build_protein_graph(structure_path, uniprot_id)
            if graph is None:
                raise ValueError("Could not build protein graph")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_graph_record(graph, output_path)
            conn.execute(
                """
                UPDATE proteins
                SET graph_path = ?, graph_built_at = ?, updated_at = ?
                WHERE uniprot_id = ?
                """,
                (f"graphs/proteins/{uniprot_id}.pt", utc_now(), utc_now(), uniprot_id),
            )
            conn.commit()
            succeeded += 1
        except Exception as exc:
            failed += 1
            LOG.warning("Failed %s: %s", uniprot_id, exc)
        if index % 200 == 0:
            LOG.info("Progress %s/%s succeeded=%s failed=%s", index, len(pending), succeeded, failed)
    conn.close()
    LOG.info("Completed protein graph build succeeded=%s failed=%s", succeeded, failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build protein graph payloads.")
    parser.add_argument("--db", type=str, default=str(default_db_path()))
    parser.add_argument("--data-dir", type=str, default=str(default_data_dir()))
    args = parser.parse_args()
    run(Path(args.db), Path(args.data_dir))


if __name__ == "__main__":
    main()


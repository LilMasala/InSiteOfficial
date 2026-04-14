"""Summarize protein DTI dataset coverage, measurements, and split integrity."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from common import default_db_path

CORE_QUERIES = {
    "total_proteins": "SELECT COUNT(*) FROM proteins",
    "proteins_with_structure": "SELECT COUNT(*) FROM proteins WHERE structure_path IS NOT NULL",
    "proteins_with_graph": "SELECT COUNT(*) FROM proteins WHERE graph_path IS NOT NULL",
    "proteins_with_go_terms": "SELECT COUNT(DISTINCT uniprot_id) FROM protein_go_terms",
    "proteins_with_cath": "SELECT COUNT(DISTINCT uniprot_id) FROM protein_cath",
    "total_drugs": "SELECT COUNT(*) FROM drugs",
    "drugs_with_graph": "SELECT COUNT(*) FROM drugs WHERE graph_path IS NOT NULL",
    "total_measurements": "SELECT COUNT(*) FROM binding_affinities",
    "deduped_pairs": "SELECT COUNT(*) FROM binding_affinities_deduped",
}


def _group_rows(conn: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
    rows = conn.execute(query).fetchall()
    return [dict(row) for row in rows]


def run(db_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        core = {
            label: int(conn.execute(query).fetchone()[0])
            for label, query in CORE_QUERIES.items()
        }
        by_affinity_type = _group_rows(
            conn,
            """
            SELECT affinity_type, COUNT(*) AS pair_count
            FROM binding_affinities_deduped
            GROUP BY affinity_type
            ORDER BY affinity_type
            """,
        )
        by_source_dataset = _group_rows(
            conn,
            """
            SELECT source_dataset, affinity_type, COUNT(*) AS measurement_count
            FROM binding_affinities
            GROUP BY source_dataset, affinity_type
            ORDER BY source_dataset, affinity_type
            """,
        )
        split_counts = _group_rows(
            conn,
            """
            SELECT
                split_strategy,
                split,
                affinity_type,
                COUNT(*) AS pair_count,
                COUNT(DISTINCT uniprot_id) AS protein_count
            FROM dataset_splits
            GROUP BY split_strategy, split, affinity_type
            ORDER BY split_strategy, affinity_type, split
            """,
        )
        return {
            "db_path": str(db_path),
            "core": core,
            "by_affinity_type": by_affinity_type,
            "by_source_dataset": by_source_dataset,
            "split_counts": split_counts,
        }
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Report protein DTI dataset coverage and split sizes.")
    parser.add_argument("--db", type=str, default=str(default_db_path()))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = run(Path(args.db))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"db={report['db_path']}")
    for label, value in report["core"].items():
        print(f"{label}={value}")
    print("affinity_types:")
    for row in report["by_affinity_type"]:
        print(f"  {row['affinity_type']}: {row['pair_count']}")
    print("split_counts:")
    for row in report["split_counts"]:
        print(
            "  "
            f"{row['split_strategy']} | {row['affinity_type']} | {row['split']} "
            f"| pairs={row['pair_count']} proteins={row['protein_count']}"
        )


if __name__ == "__main__":
    main()

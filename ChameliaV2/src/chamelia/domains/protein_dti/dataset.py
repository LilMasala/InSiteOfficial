"""Dataset utilities for the protein DTI domain."""

from __future__ import annotations

import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from .paths import default_data_dir, default_db_path
from .tokenizer import ProteinDTIObservation


def build_random_splits(
    conn: sqlite3.Connection,
    *,
    affinity_type: str,
    seed: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> int:
    """Populate random pair-level train/val/test splits."""
    rows = conn.execute(
        """
        SELECT uniprot_id, chembl_id
        FROM binding_affinities_deduped
        WHERE affinity_type = ?
        ORDER BY uniprot_id, chembl_id
        """,
        (affinity_type,),
    ).fetchall()
    pairs = [(str(row[0]), str(row[1])) for row in rows]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    train_cutoff = int(len(pairs) * train_fraction)
    val_cutoff = int(len(pairs) * (train_fraction + val_fraction))

    written = 0
    for index, (uniprot_id, chembl_id) in enumerate(pairs):
        if index < train_cutoff:
            split = "train"
        elif index < val_cutoff:
            split = "val"
        else:
            split = "test"
        conn.execute(
            """
            INSERT OR REPLACE INTO dataset_splits
                (uniprot_id, chembl_id, affinity_type, split, split_strategy)
            VALUES (?, ?, ?, ?, 'random')
            """,
            (uniprot_id, chembl_id, affinity_type, split),
        )
        written += 1
    conn.commit()
    return written


def _protein_family_keys(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT p.uniprot_id, pc.cath_id, pc.cath_topology, pc.cath_homology
        FROM proteins p
        LEFT JOIN protein_cath pc ON pc.uniprot_id = p.uniprot_id
        ORDER BY p.uniprot_id, pc.cath_id
        """
    ).fetchall()
    families: dict[str, Counter[str]] = defaultdict(Counter)
    for uniprot_id, cath_id, cath_topology, cath_homology in rows:
        protein_id = str(uniprot_id)
        family_key = str(cath_homology or cath_topology or cath_id or f"unknown:{protein_id}")
        families[protein_id][family_key] += 1
    resolved: dict[str, str] = {}
    for protein_id, counts in families.items():
        if counts:
            resolved[protein_id] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        else:
            resolved[protein_id] = f"unknown:{protein_id}"
    return resolved


def build_protein_family_splits(
    conn: sqlite3.Connection,
    *,
    affinity_type: str,
    seed: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> int:
    """Populate held-out family splits using CATH-derived protein families."""
    family_by_protein = _protein_family_keys(conn)
    family_keys = sorted(set(family_by_protein.values()))
    rng = random.Random(seed)
    rng.shuffle(family_keys)
    train_cutoff = int(len(family_keys) * train_fraction)
    val_cutoff = int(len(family_keys) * (train_fraction + val_fraction))
    split_by_family: dict[str, str] = {}
    for index, family_key in enumerate(family_keys):
        if index < train_cutoff:
            split_by_family[family_key] = "train"
        elif index < val_cutoff:
            split_by_family[family_key] = "val"
        else:
            split_by_family[family_key] = "test"

    rows = conn.execute(
        """
        SELECT uniprot_id, chembl_id
        FROM binding_affinities_deduped
        WHERE affinity_type = ?
        ORDER BY uniprot_id, chembl_id
        """,
        (affinity_type,),
    ).fetchall()
    written = 0
    for uniprot_id, chembl_id in rows:
        protein_id = str(uniprot_id)
        family_key = family_by_protein.get(protein_id, f"unknown:{protein_id}")
        split = split_by_family[family_key]
        conn.execute(
            """
            INSERT OR REPLACE INTO dataset_splits
                (uniprot_id, chembl_id, affinity_type, split, split_strategy)
            VALUES (?, ?, ?, ?, 'protein_family')
            """,
            (protein_id, str(chembl_id), affinity_type, split),
        )
        written += 1
    conn.commit()
    return written


def ensure_dataset_splits(
    conn: sqlite3.Connection,
    *,
    affinity_type: str,
    split_strategy: str,
    seed: int = 42,
) -> int:
    """Ensure the requested split strategy exists."""
    existing = conn.execute(
        """
        SELECT COUNT(*)
        FROM dataset_splits
        WHERE affinity_type = ? AND split_strategy = ?
        """,
        (affinity_type, split_strategy),
    ).fetchone()
    existing_count = int(existing[0]) if existing is not None else 0
    if existing_count > 0:
        return existing_count
    if split_strategy == "random":
        return build_random_splits(conn, affinity_type=affinity_type, seed=seed)
    if split_strategy == "protein_family":
        return build_protein_family_splits(conn, affinity_type=affinity_type, seed=seed)
    if split_strategy == "temporal":
        raise NotImplementedError("Temporal protein DTI splits are not implemented yet.")
    raise ValueError(f"Unsupported split strategy '{split_strategy}'.")


class ProteinDTIDataset:
    """Read protein DTI episodes from SQLite plus precomputed graph payloads."""

    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        data_base_dir: str | Path | None = None,
        split: str = "train",
        split_strategy: str = "protein_family",
        affinity_type: str = "Kd",
        max_candidate_drugs: int = 20,
        min_ranked_candidates: int = 2,
        seed: int = 42,
        auto_build_splits: bool = True,
    ) -> None:
        self.db_path = Path(db_path or default_db_path())
        self.base_dir = Path(data_base_dir or default_data_dir())
        self.split = str(split)
        self.split_strategy = str(split_strategy)
        self.affinity_type = str(affinity_type)
        self.max_candidate_drugs = int(max_candidate_drugs)
        self.min_ranked_candidates = int(min_ranked_candidates)
        self.rng = random.Random(seed)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        if auto_build_splits:
            ensure_dataset_splits(
                self.conn,
                affinity_type=self.affinity_type,
                split_strategy=self.split_strategy,
                seed=seed,
            )
        self._protein_ids = self._load_protein_ids()

    def close(self) -> None:
        self.conn.close()

    @property
    def protein_ids(self) -> list[str]:
        """Return the proteins available for the configured split."""
        return list(self._protein_ids)

    def _load_protein_ids(self) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT ba.uniprot_id
            FROM binding_affinities_deduped ba
            JOIN proteins p
              ON p.uniprot_id = ba.uniprot_id
             AND p.graph_path IS NOT NULL
            JOIN drugs d
              ON d.chembl_id = ba.chembl_id
             AND d.graph_path IS NOT NULL
            JOIN dataset_splits ds
              ON ds.uniprot_id = ba.uniprot_id
             AND ds.chembl_id = ba.chembl_id
             AND ds.affinity_type = ba.affinity_type
             AND ds.split = ?
             AND ds.split_strategy = ?
            WHERE ba.affinity_type = ?
            GROUP BY ba.uniprot_id
            HAVING COUNT(DISTINCT ba.chembl_id) >= ?
            ORDER BY ba.uniprot_id
            """,
            (
                self.split,
                self.split_strategy,
                self.affinity_type,
                self.min_ranked_candidates,
            ),
        ).fetchall()
        return [str(row["uniprot_id"]) for row in rows]

    def _load_graph_payload(self, relative_path: str | None) -> Any | None:
        if not relative_path:
            return None
        path = self.base_dir / relative_path
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu")

    def _load_protein_graph(self, uniprot_id: str) -> Any | None:
        row = self.conn.execute(
            "SELECT graph_path FROM proteins WHERE uniprot_id = ?",
            (uniprot_id,),
        ).fetchone()
        if row is None:
            return None
        return self._load_graph_payload(row["graph_path"])

    def _load_drug_graph(self, chembl_id: str) -> Any | None:
        row = self.conn.execute(
            "SELECT graph_path FROM drugs WHERE chembl_id = ?",
            (chembl_id,),
        ).fetchone()
        if row is None:
            return None
        return self._load_graph_payload(row["graph_path"])

    def _load_annotations(self, uniprot_id: str) -> tuple[list[str], list[str]]:
        go_ids = [
            str(row["go_id"])
            for row in self.conn.execute(
                "SELECT go_id FROM protein_go_terms WHERE uniprot_id = ? ORDER BY go_id",
                (uniprot_id,),
            )
        ]
        cath_ids = [
            str(row["cath_id"])
            for row in self.conn.execute(
                "SELECT cath_id FROM protein_cath WHERE uniprot_id = ? ORDER BY cath_id",
                (uniprot_id,),
            )
        ]
        return go_ids, cath_ids

    def _candidate_rows(self, uniprot_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT ba.chembl_id, ba.affinity_value
            FROM binding_affinities_deduped ba
            JOIN dataset_splits ds
              ON ds.uniprot_id = ba.uniprot_id
             AND ds.chembl_id = ba.chembl_id
             AND ds.affinity_type = ba.affinity_type
             AND ds.split = ?
             AND ds.split_strategy = ?
            JOIN drugs d
              ON d.chembl_id = ba.chembl_id
             AND d.graph_path IS NOT NULL
            WHERE ba.uniprot_id = ?
              AND ba.affinity_type = ?
            ORDER BY ba.affinity_value DESC, ba.chembl_id ASC
            """,
            (self.split, self.split_strategy, uniprot_id, self.affinity_type),
        ).fetchall()

    def _candidate_rows_for_ids(self, uniprot_id: str, candidate_ids: list[str]) -> list[sqlite3.Row]:
        if not candidate_ids:
            return []
        placeholders = ", ".join("?" for _ in candidate_ids)
        rows = self.conn.execute(
            f"""
            SELECT ba.chembl_id, ba.affinity_value
            FROM binding_affinities_deduped ba
            JOIN dataset_splits ds
              ON ds.uniprot_id = ba.uniprot_id
             AND ds.chembl_id = ba.chembl_id
             AND ds.affinity_type = ba.affinity_type
             AND ds.split = ?
             AND ds.split_strategy = ?
            JOIN drugs d
              ON d.chembl_id = ba.chembl_id
             AND d.graph_path IS NOT NULL
            WHERE ba.uniprot_id = ?
              AND ba.affinity_type = ?
              AND ba.chembl_id IN ({placeholders})
            ORDER BY ba.affinity_value DESC, ba.chembl_id ASC
            """,
            (
                self.split,
                self.split_strategy,
                uniprot_id,
                self.affinity_type,
                *candidate_ids,
            ),
        ).fetchall()
        row_by_id = {str(row["chembl_id"]): row for row in rows}
        return [row_by_id[chembl_id] for chembl_id in candidate_ids if chembl_id in row_by_id]

    def _select_candidates(self, rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
        if len(rows) <= self.max_candidate_drugs:
            return list(rows)
        selected: list[sqlite3.Row] = [rows[0], rows[-1]]
        middle = rows[1:-1]
        needed = max(0, self.max_candidate_drugs - len(selected))
        if needed > 0:
            if len(middle) <= needed:
                selected.extend(middle)
            else:
                sampled = self.rng.sample(middle, needed)
                selected.extend(sampled)
        selected = selected[: self.max_candidate_drugs]
        self.rng.shuffle(selected)
        return selected

    def load_observation(
        self,
        uniprot_id: str,
        *,
        candidate_ids: list[str] | None = None,
        deterministic: bool = False,
    ) -> ProteinDTIObservation | None:
        protein_graph = self._load_protein_graph(uniprot_id)
        if protein_graph is None:
            return None
        if candidate_ids is not None:
            candidate_rows = self._candidate_rows_for_ids(uniprot_id, list(candidate_ids))
        else:
            rows = self._candidate_rows(uniprot_id)
            candidate_rows = (
                list(rows[: self.max_candidate_drugs])
                if deterministic
                else self._select_candidates(rows)
            )

        resolved_candidate_ids: list[str] = []
        candidate_graphs: list[Any] = []
        affinity_values: list[float] = []
        for row in candidate_rows:
            chembl_id = str(row["chembl_id"])
            graph = self._load_drug_graph(chembl_id)
            if graph is None:
                continue
            resolved_candidate_ids.append(chembl_id)
            candidate_graphs.append(graph)
            affinity_values.append(float(row["affinity_value"]))
        if len(resolved_candidate_ids) < self.min_ranked_candidates:
            return None
        go_terms, cath_ids = self._load_annotations(uniprot_id)
        return ProteinDTIObservation(
            uniprot_id=uniprot_id,
            protein_graph=protein_graph,
            candidate_drugs=candidate_graphs,
            candidate_ids=resolved_candidate_ids,
            affinity_values=affinity_values,
            go_terms=go_terms,
            cath_ids=cath_ids,
            affinity_type=self.affinity_type,
        )

    def sample_episode(self) -> ProteinDTIObservation | None:
        if not self._protein_ids:
            return None
        for _ in range(max(1, len(self._protein_ids) * 2)):
            uniprot_id = self.rng.choice(self._protein_ids)
            observation = self.load_observation(uniprot_id)
            if observation is not None:
                return observation
        return None

    def __len__(self) -> int:
        return len(self._protein_ids)

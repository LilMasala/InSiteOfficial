"""Tests for the protein DTI tokenizer, dataset, and domain plugin."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import torch

from src.chamelia.domains.protein_dti.dataset import ProteinDTIDataset
from src.chamelia.domains.protein_dti.tokenizer import ProteinDTIObservation, ProteinDrugTokenizer
from src.chamelia.plugins.protein_dti import ProteinDTIDomain

TEST_SCHEMA = """
CREATE TABLE proteins (
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

CREATE TABLE protein_go_terms (
    uniprot_id TEXT NOT NULL,
    go_id TEXT NOT NULL,
    go_aspect TEXT NOT NULL,
    go_term TEXT NOT NULL,
    evidence_code TEXT,
    PRIMARY KEY (uniprot_id, go_id)
);

CREATE TABLE protein_cath (
    uniprot_id TEXT NOT NULL,
    cath_id TEXT NOT NULL,
    cath_class TEXT,
    cath_arch TEXT,
    cath_topology TEXT,
    cath_homology TEXT,
    PRIMARY KEY (uniprot_id, cath_id)
);

CREATE TABLE drugs (
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

CREATE TABLE binding_affinities (
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
    created_at TEXT NOT NULL
);

CREATE VIEW binding_affinities_deduped AS
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

CREATE TABLE dataset_splits (
    uniprot_id TEXT NOT NULL,
    chembl_id TEXT NOT NULL,
    affinity_type TEXT NOT NULL,
    split TEXT NOT NULL,
    split_strategy TEXT NOT NULL,
    PRIMARY KEY (uniprot_id, chembl_id, affinity_type, split_strategy)
);
"""


def _graph_payload(identifier: str, node_dim: int, edge_dim: int, *, num_nodes: int = 4) -> dict[str, torch.Tensor | str | dict[str, str]]:
    edge_src = []
    edge_dst = []
    for index in range(num_nodes - 1):
        edge_src.extend([index, index + 1])
        edge_dst.extend([index + 1, index])
    edge_count = len(edge_src)
    return {
        "identifier": identifier,
        "x": torch.randn(num_nodes, node_dim, dtype=torch.float32),
        "edge_index": torch.tensor([edge_src, edge_dst], dtype=torch.long),
        "edge_attr": torch.randn(edge_count, edge_dim, dtype=torch.float32),
        "metadata": {},
    }


def _write_graph(path: Path, payload: dict[str, torch.Tensor | str | dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _seed_test_db(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data" / "protein_dti"
    db_path = data_dir / "db" / "protein_dti.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(TEST_SCHEMA)

    now = "2026-04-09T00:00:00+00:00"
    proteins = [
        ("P11111", "M" * 12, "graphs/proteins/P11111.pt"),
        ("P22222", "A" * 14, "graphs/proteins/P22222.pt"),
    ]
    for uniprot_id, sequence, graph_path in proteins:
        conn.execute(
            """
            INSERT INTO proteins
                (uniprot_id, sequence, sequence_length, graph_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (uniprot_id, sequence, len(sequence), graph_path, now, now),
        )
    conn.executemany(
        """
        INSERT INTO protein_go_terms
            (uniprot_id, go_id, go_aspect, go_term, evidence_code)
        VALUES (?, ?, 'F', 'term', 'EXP')
        """,
        [("P11111", "GO:0001"), ("P22222", "GO:0002")],
    )
    conn.executemany(
        """
        INSERT INTO protein_cath
            (uniprot_id, cath_id, cath_class, cath_arch, cath_topology, cath_homology)
        VALUES (?, ?, '1', '10', ?, ?)
        """,
        [
            ("P11111", "1.10.20.30", "20", "30"),
            ("P22222", "2.40.50.60", "50", "60"),
        ],
    )

    drugs = [
        ("CHEMBL1", "CCO", "graphs/drugs/CHEMBL1.pt"),
        ("CHEMBL2", "CCC", "graphs/drugs/CHEMBL2.pt"),
        ("CHEMBL3", "CCN", "graphs/drugs/CHEMBL3.pt"),
        ("CHEMBL4", "CCCl", "graphs/drugs/CHEMBL4.pt"),
        ("CHEMBL5", "CCBr", "graphs/drugs/CHEMBL5.pt"),
    ]
    for chembl_id, smiles, graph_path in drugs:
        conn.execute(
            """
            INSERT INTO drugs
                (chembl_id, smiles, graph_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chembl_id, smiles, graph_path, now, now),
        )

    affinities = [
        ("m1", "P11111", "CHEMBL1", 9.0),
        ("m2", "P11111", "CHEMBL2", 7.0),
        ("m3", "P11111", "CHEMBL3", 5.0),
        ("m4", "P22222", "CHEMBL4", 8.5),
        ("m5", "P22222", "CHEMBL5", 6.0),
    ]
    for measurement_key, uniprot_id, chembl_id, affinity_value in affinities:
        conn.execute(
            """
            INSERT INTO binding_affinities
                (measurement_key, uniprot_id, chembl_id, affinity_value, affinity_type,
                 source_dataset, original_value, original_units, assay_id, created_at)
            VALUES (?, ?, ?, ?, 'Kd', 'synthetic', ?, 'nM', NULL, ?)
            """,
            (measurement_key, uniprot_id, chembl_id, affinity_value, affinity_value, now),
        )

    conn.commit()
    conn.close()

    _write_graph(data_dir / "graphs" / "proteins" / "P11111.pt", _graph_payload("P11111", 21, 19, num_nodes=6))
    _write_graph(data_dir / "graphs" / "proteins" / "P22222.pt", _graph_payload("P22222", 21, 19, num_nodes=5))
    for chembl_id in ("CHEMBL1", "CHEMBL2", "CHEMBL3", "CHEMBL4", "CHEMBL5"):
        _write_graph(data_dir / "graphs" / "drugs" / f"{chembl_id}.pt", _graph_payload(chembl_id, 22, 6, num_nodes=4))
    return db_path, data_dir


def test_protein_drug_tokenizer_emits_padded_standard_tokens() -> None:
    tokenizer = ProteinDrugTokenizer(embed_dim=32, max_candidate_drugs=4, protein_summary_tokens=3)
    first = ProteinDTIObservation(
        uniprot_id="P11111",
        protein_graph=_graph_payload("P11111", 21, 19, num_nodes=6),
        candidate_drugs=[
            _graph_payload("CHEMBL1", 22, 6, num_nodes=4),
            _graph_payload("CHEMBL2", 22, 6, num_nodes=4),
        ],
        candidate_ids=["CHEMBL1", "CHEMBL2"],
        affinity_values=[9.0, 7.0],
        go_terms=["GO:0001"],
        cath_ids=["1.10.20.30"],
    )
    second = ProteinDTIObservation(
        uniprot_id="P22222",
        protein_graph=_graph_payload("P22222", 21, 19, num_nodes=5),
        candidate_drugs=[_graph_payload("CHEMBL3", 22, 6, num_nodes=4)],
        candidate_ids=["CHEMBL3"],
        affinity_values=[8.0],
        go_terms=["GO:0002"],
        cath_ids=["2.40.50.60"],
    )

    batch = tokenizer.collate([first, second])
    output = tokenizer(batch)

    assert output.tokens.shape == (2, 7, 32)
    assert output.position_ids.shape == (2, 7)
    assert output.padding_mask is not None
    assert output.padding_mask.shape == (2, 7)
    assert bool(output.padding_mask[1, -1].item()) is True
    assert bool(output.padding_mask[0].any().item()) is False


def test_protein_dti_domain_ranking_cost_prefers_correct_order(tmp_path: Path) -> None:
    db_path, data_dir = _seed_test_db(tmp_path)
    domain = ProteinDTIDomain(
        db_path=str(db_path),
        data_base_dir=str(data_dir),
        embed_dim=32,
        max_candidate_drugs=4,
        action_dim=4,
        split="train",
        split_strategy="random",
        seed=7,
    )
    observation = ProteinDTIObservation(
        uniprot_id="P11111",
        protein_graph=_graph_payload("P11111", 21, 19, num_nodes=6),
        candidate_drugs=[
            _graph_payload("CHEMBL1", 22, 6, num_nodes=4),
            _graph_payload("CHEMBL2", 22, 6, num_nodes=4),
            _graph_payload("CHEMBL3", 22, 6, num_nodes=4),
        ],
        candidate_ids=["CHEMBL1", "CHEMBL2", "CHEMBL3"],
        affinity_values=[9.0, 7.0, 5.0],
        go_terms=["GO:0001"],
        cath_ids=["1.10.20.30"],
    )
    domain_state = domain.get_domain_state(observation)
    ranking_cost = domain.get_intrinsic_cost_fns()[0][0]
    good = torch.tensor([[3.0, 1.0, -1.0, 0.0]], dtype=torch.float32)
    bad = torch.tensor([[-1.0, 1.0, 3.0, 0.0]], dtype=torch.float32)

    good_cost = ranking_cost(torch.zeros(1, 1), good, domain_state)
    bad_cost = ranking_cost(torch.zeros(1, 1), bad, domain_state)

    assert good_cost.shape == (1,)
    assert float(good_cost.item()) < float(bad_cost.item())
    decoded = domain.decode_action(good)
    assert decoded["ranked_candidate_ids"][0] == "CHEMBL1"


def test_protein_dti_dataset_builds_family_splits_and_samples_measured_candidates(tmp_path: Path) -> None:
    db_path, data_dir = _seed_test_db(tmp_path)
    dataset = ProteinDTIDataset(
        db_path=str(db_path),
        data_base_dir=str(data_dir),
        split="train",
        split_strategy="protein_family",
        affinity_type="Kd",
        max_candidate_drugs=3,
        seed=5,
        auto_build_splits=True,
    )

    assert len(dataset) >= 1
    sample = dataset.sample_episode()
    assert sample is not None
    measured_candidates = {
        "P11111": {"CHEMBL1", "CHEMBL2", "CHEMBL3"},
        "P22222": {"CHEMBL4", "CHEMBL5"},
    }
    assert set(sample.candidate_ids).issubset(measured_candidates[sample.uniprot_id])
    assert len(sample.candidate_ids) == len(sample.affinity_values)
    assert len(sample.candidate_ids) >= 2

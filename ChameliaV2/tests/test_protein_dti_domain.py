"""Tests for the protein DTI tokenizer, dataset, and domain plugin."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.chamelia.domains.protein_dti.dataset import ProteinDTIDataset
from src.chamelia.domains.protein_dti.tokenizer import ProteinDTIObservation, ProteinDrugTokenizer
from src.chamelia.plugins.protein_dti import ProteinDTIDomain
from src.serving.bridge_runtime import BridgeRuntime
from tests.protein_dti_test_utils import graph_payload, seed_test_db


def test_protein_drug_tokenizer_emits_padded_standard_tokens() -> None:
    tokenizer = ProteinDrugTokenizer(embed_dim=32, max_candidate_drugs=4, protein_summary_tokens=3)
    first = ProteinDTIObservation(
        uniprot_id="P11111",
        protein_graph=graph_payload("P11111", 21, 19, num_nodes=6),
        candidate_drugs=[
            graph_payload("CHEMBL1", 22, 6, num_nodes=4),
            graph_payload("CHEMBL2", 22, 6, num_nodes=4),
        ],
        candidate_ids=["CHEMBL1", "CHEMBL2"],
        affinity_values=[9.0, 7.0],
        go_terms=["GO:0001"],
        cath_ids=["1.10.20.30"],
    )
    second = ProteinDTIObservation(
        uniprot_id="P22222",
        protein_graph=graph_payload("P22222", 21, 19, num_nodes=5),
        candidate_drugs=[graph_payload("CHEMBL3", 22, 6, num_nodes=4)],
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
    db_path, data_dir = seed_test_db(tmp_path)
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
        protein_graph=graph_payload("P11111", 21, 19, num_nodes=6),
        candidate_drugs=[
            graph_payload("CHEMBL1", 22, 6, num_nodes=4),
            graph_payload("CHEMBL2", 22, 6, num_nodes=4),
            graph_payload("CHEMBL3", 22, 6, num_nodes=4),
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
    db_path, data_dir = seed_test_db(tmp_path)
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


def test_protein_dti_bridge_payload_round_trips_through_chamelia(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path, data_dir = seed_test_db(tmp_path)
    monkeypatch.setenv("CHAMELIA_PROTEIN_DTI_DB_PATH", str(db_path))
    monkeypatch.setenv("CHAMELIA_PROTEIN_DTI_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CHAMELIA_PROTEIN_DTI_SPLIT", "train")
    monkeypatch.setenv("CHAMELIA_PROTEIN_DTI_SPLIT_STRATEGY", "protein_family")
    monkeypatch.setenv("CHAMELIA_PROTEIN_DTI_MAX_CANDIDATES", "4")
    monkeypatch.setenv("CHAMELIA_PROTEIN_DTI_ACTION_DIM", "4")

    runtime = BridgeRuntime(
        backbone_mode="stub",
        device="cpu",
        model_version="protein-dti-test-model",
    )
    session = runtime.get_session("protein-1", "protein_dti")
    uniprot_id = session.domain.dataset.protein_ids[0]  # type: ignore[attr-defined]
    observation = session.domain.dataset.load_observation(  # type: ignore[attr-defined]
        uniprot_id,
        deterministic=True,
    )
    assert observation is not None
    payload = {
        "uniprot_id": uniprot_id,
        "candidate_ids": list(observation.candidate_ids),
    }

    batch = session.domain.prepare_bridge_observation(payload)
    tokenized = session.domain.get_tokenizer()(batch)
    mask = tokenized.padding_mask.to(dtype=torch.float32) if tokenized.padding_mask is not None else torch.zeros(
        tokenized.tokens.shape[0],
        tokenized.tokens.shape[1],
        dtype=torch.float32,
    )
    outputs = session.model(
        tokenized.tokens,
        mask,
        session.domain.get_domain_state(payload),
        input_kind="embedded_tokens",
    )

    assert outputs["action_vec"].shape == (1, session.domain.get_action_dim())
    assert outputs["cost"]["ic"].shape == (1,)
    delayed = session.domain.simulate_delayed_outcome(outputs["action_vec"], session.domain.get_domain_state(payload))
    assert delayed is not None
    record_id = session.model._pending_record_indices[0]
    session.model.fill_outcome(
        ic_realized=delayed["realized_intrinsic_cost"],
        outcome_observation=delayed["outcome_observation"],
    )
    stored = session.model.memory.get_record_by_id(record_id)
    assert stored is not None
    assert stored.ic_realized is not None
    assert stored.outcome_key is not None
    critic_loss = session.model.train_critic_from_memory()
    assert critic_loss is not None

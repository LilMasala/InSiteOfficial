"""Runtime tests for the Julia ↔ Python bridge surface."""

from __future__ import annotations

from pathlib import Path

import torch

from src.serving.bridge_runtime import (
    BridgeRuntime,
    configure_session,
    critic_session,
    encode_session,
    ingest_replay_examples,
    propose_session,
    retrieve_session,
    rollout_session,
)
from tests.protein_dti_test_utils import seed_test_db


def _bridge_observation() -> dict[str, object]:
    return {
        "timestamp": 1.0,
        "signals": {
            "bg_avg": 112.0,
            "tir_7d": 0.72,
            "pct_low_7d": 0.01,
            "pct_high_7d": 0.18,
            "bg_var": 0.14,
        },
    }


def test_bridge_runtime_scopes_sessions_and_model_version() -> None:
    """Bridge runtime should keep sessions separate and expose the active model version."""
    runtime = BridgeRuntime(
        backbone_mode="stub",
        device="cpu",
        model_version="bridge-test-model-v1",
    )
    assert runtime.model_version == "bridge-test-model-v1"
    assert len(runtime.sessions) == 0

    first = runtime.get_session("patient-1", "insite_t1d")
    second = runtime.get_session("patient-2", "insite_t1d")
    first_again = runtime.get_session("patient-1", "insite_t1d")

    assert first.session_id == "patient-1"
    assert second.session_id == "patient-2"
    assert first.model_version == "bridge-test-model-v1"
    assert first is first_again
    assert len(runtime.sessions) == 2


def test_bridge_runtime_round_trip_returns_consistent_transport_objects() -> None:
    """The bridge runtime helpers should round-trip a real repaired-model session."""
    runtime = BridgeRuntime(
        backbone_mode="stub",
        device="cpu",
        model_version="bridge-test-model-v1",
    )
    session = runtime.get_session("patient-1", "insite_t1d")
    observation = _bridge_observation()

    encoded = encode_session(
        session,
        input_kind="plugin_observation",
        observation=observation,
    )
    assert encoded["model_version"] == "bridge-test-model-v1"
    assert len(encoded["z_t"]) > 0

    retrieved = retrieve_session(session, z_t=encoded["z_t"])
    configured = configure_session(
        session,
        encoded_state=encoded,
        retrieved_memory=retrieved,
    )
    proposed = propose_session(
        session,
        mode="v3",
        encoded_state=encoded,
        configurator_output=configured,
        retrieved_memory=retrieved,
    )
    rolled = rollout_session(
        session,
        encoded_state=encoded,
        configurator_output=configured,
        proposal_bundle=proposed,
        rollout_horizon=2,
    )
    scored = critic_session(
        session,
        encoded_state=encoded,
        configurator_output=configured,
        proposal_bundle=proposed,
        rollout_bundle=rolled,
        domain_state=observation,
    )

    num_candidates = len(proposed["candidate_paths"])
    assert num_candidates >= 1
    assert proposed["model_version"] == "bridge-test-model-v1"
    assert configured["model_version"] == "bridge-test-model-v1"
    assert retrieved["model_version"] == "bridge-test-model-v1"
    assert rolled["model_version"] == "bridge-test-model-v1"
    assert scored["model_version"] == "bridge-test-model-v1"
    assert len(proposed["candidate_actions"]) == num_candidates
    assert len(rolled["trajectory"]) == num_candidates
    assert len(rolled["terminal_latents"]) == num_candidates
    assert len(scored["candidate_total"]) == num_candidates


def test_bridge_runtime_mode_profiles_share_planner_substrate() -> None:
    """All bridge modes should use the same repaired planner substrate with explicit budgets."""
    runtime = BridgeRuntime(
        backbone_mode="stub",
        device="cpu",
        model_version="bridge-test-model-v1",
    )
    session = runtime.get_session("patient-1", "insite_t1d")
    observation = _bridge_observation()

    encoded = encode_session(
        session,
        input_kind="plugin_observation",
        observation=observation,
    )
    retrieved = retrieve_session(session, z_t=encoded["z_t"])
    configured = configure_session(
        session,
        encoded_state=encoded,
        retrieved_memory=retrieved,
    )

    proposed_v11 = propose_session(
        session,
        mode="v1.1",
        encoded_state=encoded,
        configurator_output=configured,
        retrieved_memory=retrieved,
    )
    proposed_v15 = propose_session(
        session,
        mode="v1.5",
        encoded_state=encoded,
        configurator_output=configured,
        retrieved_memory=retrieved,
    )
    proposed_v3 = propose_session(
        session,
        mode="v3",
        encoded_state=encoded,
        configurator_output=configured,
        retrieved_memory=retrieved,
    )

    assert proposed_v11["proposal_diagnostics"]["shared_planner_substrate"] is True
    assert proposed_v15["proposal_diagnostics"]["shared_planner_substrate"] is True
    assert proposed_v3["proposal_diagnostics"]["shared_planner_substrate"] is True

    assert proposed_v11["proposal_diagnostics"]["mode"] == "v1.1"
    assert proposed_v15["proposal_diagnostics"]["mode"] == "v1.5"
    assert proposed_v3["proposal_diagnostics"]["mode"] == "v3"

    assert proposed_v11["proposal_diagnostics"]["planner_profile"] == "conservative_shared_planner"
    assert proposed_v15["proposal_diagnostics"]["planner_profile"] == "lightweight_shared_planner"
    assert proposed_v3["proposal_diagnostics"]["planner_profile"] == "default_shared_planner"

    assert proposed_v11["proposal_diagnostics"]["full_candidate_budget"] == session.model.actor.num_candidates
    assert proposed_v3["proposal_diagnostics"]["full_candidate_budget"] == session.model.actor.num_candidates

    assert len(proposed_v11["candidate_paths"]) == 2
    assert len(proposed_v15["candidate_paths"]) == 4
    assert len(proposed_v3["candidate_paths"]) == session.model.actor.num_candidates

    assert proposed_v11["proposal_diagnostics"]["uses_retrieved_postures"] is False
    assert proposed_v15["proposal_diagnostics"]["uses_retrieved_postures"] is True
    assert proposed_v3["proposal_diagnostics"]["uses_retrieved_postures"] is True

    assert all(abs(value) < 1.0e-6 for step in proposed_v11["candidate_paths"][0] for value in step)
    assert all(abs(value) < 1.0e-6 for step in proposed_v15["candidate_paths"][0] for value in step)
    assert all(abs(value) < 1.0e-6 for step in proposed_v3["candidate_paths"][0] for value in step)

    rollout_v11 = rollout_session(
        session,
        encoded_state=encoded,
        configurator_output=configured,
        proposal_bundle=proposed_v11,
        rollout_horizon=2,
    )
    scores_v11 = critic_session(
        session,
        encoded_state=encoded,
        configurator_output=configured,
        proposal_bundle=proposed_v11,
        rollout_bundle=rollout_v11,
        domain_state=observation,
    )
    assert len(rollout_v11["trajectory"]) == 2
    assert len(scores_v11["candidate_total"]) == 2


def test_bridge_runtime_uses_checkpoint_embedded_model_config(tmp_path: Path) -> None:
    """Bridge runtime should rebuild from checkpoint config when one is embedded in the artifact."""
    checkpoint_path = tmp_path / "bridge_embedded_config.pth"
    torch.save(
        {
            "model_state_dict": {},
            "config": {
                "embed_dim": 64,
                "configurator": {
                    "num_ctx_tokens": 7,
                    "num_heads": 4,
                    "num_layers": 2,
                    "mlp_ratio": 2.0,
                    "dropout": 0.0,
                    "memory_read_k": 4,
                },
                "actor": {"num_heads": 4, "num_layers": 2, "mlp_ratio": 2.0, "dropout": 0.0},
                "cost": {
                    "critic_num_heads": 4,
                    "critic_num_layers": 2,
                    "critic_mlp_ratio": 2.0,
                    "critic_dropout": 0.0,
                    "critic_horizon": 30,
                },
                "memory": {"max_episodes": 512, "retrieval_k": 4, "device": "cpu"},
            },
            "model_version": "bridge-embedded-config-v1",
        },
        checkpoint_path,
    )

    runtime = BridgeRuntime(
        backbone_mode="stub",
        device="cpu",
        checkpoint_path=str(checkpoint_path),
    )
    session = runtime.get_session("patient-2", "insite_t1d")

    assert runtime.model_version == "bridge-embedded-config-v1"
    assert session.model.num_ctx_tokens == 7


def test_bridge_runtime_supports_protein_dti_bridge_payloads(
    tmp_path: Path,
    monkeypatch,
) -> None:
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
        model_version="bridge-protein-dti-v1",
    )
    session = runtime.get_session("protein-bridge-1", "protein_dti")
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

    encoded = encode_session(
        session,
        input_kind="plugin_observation",
        observation=payload,
    )
    retrieved = retrieve_session(session, z_t=encoded["z_t"])
    configured = configure_session(
        session,
        encoded_state=encoded,
        retrieved_memory=retrieved,
    )
    proposed = propose_session(
        session,
        mode="v3",
        encoded_state=encoded,
        configurator_output=configured,
        retrieved_memory=retrieved,
    )
    rolled = rollout_session(
        session,
        encoded_state=encoded,
        configurator_output=configured,
        proposal_bundle=proposed,
        rollout_horizon=2,
    )
    scored = critic_session(
        session,
        encoded_state=encoded,
        configurator_output=configured,
        proposal_bundle=proposed,
        rollout_bundle=rolled,
        domain_state=payload,
    )

    assert encoded["domain_name"] == "protein_dti"
    assert encoded["model_version"] == "bridge-protein-dti-v1"
    assert len(proposed["candidate_actions"]) == len(proposed["candidate_paths"])
    assert len(rolled["trajectory"]) == len(proposed["candidate_paths"])
    assert len(scored["candidate_total"]) == len(proposed["candidate_paths"])


def test_bridge_runtime_ingests_julia_replay_examples_with_dedup_and_version_checks() -> None:
    """Replay ingest should rebuild durable Julia exports into Python session memory."""
    runtime = BridgeRuntime(
        backbone_mode="stub",
        device="cpu",
        model_version="bridge-test-model-v1",
    )
    session = runtime.get_session("patient-1", "insite_t1d")
    observation = _bridge_observation()

    encoded = encode_session(
        session,
        input_kind="plugin_observation",
        observation=observation,
    )
    retrieved = retrieve_session(session, z_t=encoded["z_t"])
    configured = configure_session(
        session,
        encoded_state=encoded,
        retrieved_memory=retrieved,
    )
    proposed = propose_session(
        session,
        mode="v3",
        encoded_state=encoded,
        configurator_output=configured,
        retrieved_memory=retrieved,
    )
    rolled = rollout_session(
        session,
        encoded_state=encoded,
        configurator_output=configured,
        proposal_bundle=proposed,
        rollout_horizon=2,
    )
    scored = critic_session(
        session,
        encoded_state=encoded,
        configurator_output=configured,
        proposal_bundle=proposed,
        rollout_bundle=rolled,
        domain_state=observation,
    )

    selected_slot = 2
    replay_example = {
        "bridge_version": "v1",
        "domain_name": "insite_t1d",
        "model_version": "bridge-test-model-v1",
        "source_patient_id": "patient-1",
        "record_id": 7,
        "day": 3,
        "z_t": encoded["z_t"],
        "ctx_tokens": configured["ctx_tokens"],
        "selected_candidate_idx": 1,
        "selected_candidate_slot": selected_slot,
        "selected_action_vec": proposed["candidate_actions"][selected_slot - 1],
        "selected_path": proposed["candidate_paths"][selected_slot - 1],
        "selected_posture": proposed["candidate_postures"][selected_slot - 1],
        "selected_reasoning_state": proposed["candidate_reasoning_states"][selected_slot - 1],
        "candidate_actions": proposed["candidate_actions"],
        "candidate_paths": proposed["candidate_paths"],
        "candidate_postures": proposed["candidate_postures"],
        "candidate_reasoning_states": proposed["candidate_reasoning_states"],
        "candidate_ic": scored["candidate_ic"],
        "candidate_tc": scored["candidate_tc"],
        "candidate_total": scored["candidate_total"],
        "selected_candidate_ic": scored["candidate_ic"][selected_slot - 1],
        "selected_candidate_tc": scored["candidate_tc"][selected_slot - 1],
        "selected_candidate_total": scored["candidate_total"][selected_slot - 1],
        "realized_ic": 0.12,
        "outcome_z_tH": rolled["terminal_latents"][selected_slot - 1],
        "retrieval_trace": [
            {
                "query_key": encoded["z_t"],
                "memory_keys": [encoded["z_t"]],
                "memory_summaries": [configured["ctx_tokens"][0]],
                "base_quality_scores": [0.1],
                "query_posture": proposed["candidate_postures"][selected_slot - 1],
                "memory_postures": [proposed["candidate_postures"][selected_slot - 1]],
                "base_scores": [0.1],
                "relevance_scores": [0.2],
                "relevance_weights": [1.0],
            }
        ],
        "julia_selection": {"selected_bridge_candidate_idx": 1},
        "selected_candidate": {"decode_metadata": {"decoder": "insite_scalar_delta"}},
    }
    wrong_version = dict(replay_example)
    wrong_version["record_id"] = 8
    wrong_version["model_version"] = "bridge-test-model-v2"

    result = ingest_replay_examples(session, examples=[replay_example, wrong_version])
    assert result["ingested"] == 1
    assert result["skipped"] == 1
    assert result["duplicates"] == 0
    assert result["memory_size"] == 1

    stored = session.model.memory.records[0]
    assert stored.ic_realized == 0.12
    assert stored.model_version == "bridge-test-model-v1"
    assert stored.metadata is not None
    assert stored.metadata["bridge_replay_record_id"] == 7
    assert stored.retrieval_trace is not None
    assert len(stored.retrieval_trace) == 1

    duplicate = ingest_replay_examples(session, examples=[replay_example])
    assert duplicate["ingested"] == 0
    assert duplicate["duplicates"] == 1
    assert duplicate["memory_size"] == 1

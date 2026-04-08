"""Tests for the unified Chamelia training orchestrator."""

from __future__ import annotations

from pathlib import Path

import torch

from src.chamelia.cognitive.mamba_world_model import MambaActionConditionedWorldModel
from src.chamelia.memory import EpisodeRecord, LatentMemory
from src.chamelia.plugins import CartPoleDomain, Connect4Domain
from src.chamelia.tokenizers import StateVectorTokenizer
from training.orchestrator import (
    DomainPhaseConfig,
    DomainRunConfig,
    OrchestratorConfig,
    ReplayRecord,
    TransitionReplayBuffer,
    UnifiedTrainingOrchestrator,
)


def test_state_vector_tokenizer_shapes() -> None:
    """State-vector tokenization should emit one token per feature."""
    tokenizer = StateVectorTokenizer(num_features=4, embed_dim=16)
    batch = tokenizer.collate(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6, 0.7],
        ]
    )
    tokenized = tokenizer(batch)
    assert tokenized.tokens.shape == (2, 4, 16)
    assert tokenized.position_ids.shape == (2, 4)
    assert tokenized.padding_mask is not None
    assert tokenized.padding_mask.shape == (2, 4)


def test_interactive_domain_single_observation_tokenization() -> None:
    """Interactive adapters should batch singleton observations internally."""
    cartpole = CartPoleDomain(embed_dim=16)
    observation, info = cartpole.reset(seed=123)
    tokenized = cartpole.tokenize_observation(observation)
    assert tokenized.tokens.shape == (1, 4, 16)
    legal = cartpole.legal_action_mask(observation, info)
    assert legal is not None
    assert legal.shape == (2,)

    connect4 = Connect4Domain(embed_dim=16)
    board_obs, board_info = connect4.reset(seed=123)
    tokenized_board = connect4.tokenize_observation(board_obs)
    assert tokenized_board.tokens.shape == (1, 43, 16)
    greedy_action = connect4.baseline_action("greedy", board_obs, board_info)
    assert int(greedy_action.item()) == 3


def test_replay_and_latent_memory_roundtrip() -> None:
    """Replay records and episodic-memory snapshots should round-trip cleanly."""
    replay = TransitionReplayBuffer(capacity=8)
    record = ReplayRecord(
        domain_id="cartpole",
        modality_family="state_vector_hjepa",
        episode_id=1,
        step_idx=0,
        obs_raw={"state": [0.0, 0.1, 0.2, 0.3]},
        tokenizer_input=torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float32),
        tokens=torch.randn(4, 16),
        mask=torch.zeros(4),
        domain_state={"state_vector": torch.randn(1, 4)},
        action=torch.tensor([1]),
        action_logits_or_vec=torch.tensor([0.2, 0.8], dtype=torch.float32),
        legal_actions_mask=torch.tensor([True, True]),
        reward=1.0,
        cost=0.25,
        done=False,
        next_obs_raw={"state": [0.1, 0.0, 0.25, 0.4]},
        next_tokens=torch.randn(4, 16),
        next_mask=torch.zeros(4),
        next_domain_state={"state_vector": torch.randn(1, 4)},
        latent_z=torch.randn(16),
        next_latent_z=torch.randn(16),
        ctx_tokens=torch.randn(4, 16),
        search_policy=torch.tensor([0.1, 0.9], dtype=torch.float32),
        search_value=0.5,
        selected_path=torch.randn(2, 2),
        selected_posture=torch.randn(16),
        memory_hits=2,
        procedural_skill_ids=(3, 4),
        reasoning_trace_id="cartpole:1:0",
    )
    replay.add(record)
    restored = TransitionReplayBuffer(capacity=1)
    restored.load_state_dict(replay.state_dict())
    assert len(restored) == 1
    assert restored.records[0].domain_id == "cartpole"
    assert restored.records[0].procedural_skill_ids == (3, 4)

    memory = LatentMemory(embed_dim=16, max_episodes=8, retrieval_k=2, device="cpu")
    record_id = memory.store(
        EpisodeRecord(
            key=torch.randn(16),
            action=torch.randn(2),
            ctx_tokens=torch.randn(4, 16),
            ic_at_decision=0.2,
            ic_realized=None,
            tc_predicted=0.1,
            outcome_key=None,
            step=0,
            domain_name="cartpole",
            selected_posture=torch.randn(8),
            selected_path=torch.randn(2, 2),
        )
    )
    memory.fill_outcome(record_id, ic_realized=0.15, outcome_key=torch.randn(16))
    restored_memory = LatentMemory(embed_dim=16, max_episodes=8, retrieval_k=2, device="cpu")
    restored_memory.load_state_dict(memory.state_dict())
    assert restored_memory.size == 1
    restored_record = restored_memory.get_record_by_id(record_id)
    assert restored_record is not None
    assert restored_record.ic_realized == 0.15


def _tiny_orchestrator_config(run_dir: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        seed=7,
        device="cpu",
        run_dir=str(run_dir),
        num_ctx_tokens=4,
        rollout_horizon=2,
        reasoning_steps=1,
        replay_capacity=128,
        family_backbones={
            "state_vector_hjepa": {
                "encoder_type": "vit_tiny_patch16_224",
                "embed_dim": 192,
                "predictor_depth": 1,
                "predictor_num_heads": 3,
                "predictor_mlp_ratio": 2.0,
                "num_hierarchies": 2,
                "sequence_mode": True,
                "use_vq": False,
            }
        },
        memory={
            "max_episodes": 128,
            "retrieval_k": 4,
            "device": "cpu",
            "use_iob": False,
            "retrieval_train_interval": 2,
        },
        procedural={
            "use_faiss": False,
            "use_lancedb": True,
            "use_csr": False,
            "use_isotropic_storage": False,
            "lora_rank": 8,
        },
        sleep={"interval_steps": 2},
        logging={
            "representation_loss": "hjepa",
            "num_candidates": 3,
            "path_length": 2,
        },
        domains=[
            DomainRunConfig(
                name="cartpole",
                family="state_vector_hjepa",
                bootstrap_random_episodes=1,
                bootstrap_pretrain_steps=1,
                bootstrap_batch_size=1,
                mask_ratio=0.25,
                max_episode_steps=4,
                optimizer_interval=1,
                retrieval_refresh_episodes=1,
                sleep_interval_episodes=1,
                checkpoint_interval_episodes=1,
                evaluation_episodes=1,
                world_model_backend="mamba",
                mcts_simulations=1,
                mcts_depth=1,
                mcts_rollout_horizon=1,
                baselines=("random",),
                phases={
                    "core_control": DomainPhaseConfig(episodes=1),
                    "episodic_memory": DomainPhaseConfig(episodes=1, use_memory=True),
                    "sleep": DomainPhaseConfig(episodes=1, use_memory=True, use_sleep=True),
                },
            )
        ],
    )


def test_orchestrator_checkpoint_load_and_smoke_run(tmp_path: Path) -> None:
    """The orchestrator should run a tiny CartPole pass and reload its checkpoint."""
    config = _tiny_orchestrator_config(tmp_path / "run")
    orchestrator = UnifiedTrainingOrchestrator(config)
    results = orchestrator.run()
    assert "cartpole" in results
    checkpoint_dir = Path(config.run_dir) / "cartpole" / "checkpoints"
    latest_checkpoint = checkpoint_dir / "sleep-latest.pt"
    assert latest_checkpoint.exists()

    restored_model, payload = orchestrator.load_checkpoint(
        latest_checkpoint,
        domain_cfg=config.domains[0],
    )
    assert payload["domain_name"] == "cartpole"
    assert len(orchestrator.replay) > 0
    assert restored_model.memory.size >= 0
    assert restored_model.world_model_backend == "mamba"
    assert isinstance(restored_model.world_model, MambaActionConditionedWorldModel)
    assert restored_model.domain_index is not None
    assert restored_model.domain_index.adapter_bank is not None

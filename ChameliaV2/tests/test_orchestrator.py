"""Tests for the unified Chamelia training orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
import random

import pytest
import torch
import chess

from src.chamelia.cognitive.mamba_world_model import MambaActionConditionedWorldModel
from src.chamelia.memory import EpisodeRecord, LatentMemory
from src.chamelia.plugins import CartPoleDomain, ChessDomain, Connect4Domain
from src.chamelia.plugins.chess import _action_from_move, _action_to_move
from src.chamelia.tokenizers import ChessTokenizer, StateVectorTokenizer
from training.orchestrator.core import ReplayWindow, _move_nested_to_device
from training.orchestrator import (
    DomainPhaseConfig,
    DomainRunConfig,
    OrchestratorConfig,
    ReplayRecord,
    TransitionReplayBuffer,
    UnifiedTrainingOrchestrator,
    load_orchestrator_config,
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

    chess_domain = ChessDomain(embed_dim=16, opponent_level=0)
    chess_obs, chess_info = chess_domain.reset(seed=123)
    tokenized_chess = chess_domain.tokenize_observation(chess_obs)
    assert tokenized_chess.tokens.shape == (1, 64, 16)
    chess_legal = chess_domain.legal_action_mask(chess_obs, chess_info)
    assert chess_legal is not None
    assert int(chess_legal.sum().item()) == 20


def test_domain_baselines_and_imagined_states_are_not_constant() -> None:
    cartpole = CartPoleDomain(embed_dim=16)
    observation, info = cartpole.reset(seed=7)
    domain_state = cartpole.build_domain_state(observation, info)
    baseline = cartpole.build_simple_baseline_path(domain_state, path_length=3, action_dim=2)
    assert baseline is not None
    assert baseline.shape == (1, 3, 2)
    assert torch.all(baseline.argmax(dim=-1).reshape(-1) == baseline.argmax(dim=-1).reshape(-1)[0])

    connect4 = Connect4Domain(embed_dim=16)
    board_obs, board_info = connect4.reset(seed=7)
    connect_state = connect4.build_domain_state(board_obs, board_info)
    assert connect_state["board"].shape == (1, 6, 7)
    assert connect_state["current_player"].shape == (1,)
    assert connect_state["legal_actions_mask"].shape == (1, 7)
    imagined = connect4.build_imagined_domain_state(connect_state, None, torch.randn(3, 16), step_idx=0)
    assert imagined["legal_actions_mask"].shape == (3, 7)
    assert imagined["winning_actions"].shape == (3, 7)


def test_chess_tokenizer_projects_alpha_zero_planes() -> None:
    tokenizer = ChessTokenizer(embed_dim=16)
    batch = tokenizer.collate([torch.zeros(8, 8, 111), torch.ones(8, 8, 111)])
    tokenized = tokenizer(batch)
    assert tokenized.tokens.shape == (2, 64, 16)
    assert tokenized.position_ids.shape == (2, 64)
    assert tokenized.padding_mask is not None
    assert tokenized.padding_mask.shape == (2, 64)


def test_chess_domain_move_roundtrip_and_legal_mask() -> None:
    domain = ChessDomain(embed_dim=16, opponent_level=0)
    observation, info = domain.reset(seed=0)
    state = domain.build_domain_state(observation, info)
    legal_mask = state["legal_actions_mask"][0]
    move = chess.Move.from_uci("e2e4")
    action = _action_from_move(move)
    assert bool(legal_mask[action].item()) is True
    assert _action_to_move(chess.Board(), action).uci() == "e2e4"


def test_chess_domain_invalid_move_is_terminal_loss() -> None:
    domain = ChessDomain(embed_dim=16, opponent_level=0)
    observation, info = domain.reset(seed=0)
    invalid_action = torch.tensor([0], dtype=torch.long)
    next_observation, reward, terminated, truncated, next_info = domain.step(invalid_action)
    assert next_observation["observation"].shape == (8, 8, 111)
    assert reward == -1.0
    assert terminated is True
    assert truncated is False
    assert next_info["invalid_action"] is True
    assert next_info["winner"] == 2


def test_chess_action_decoder_returns_null_for_offboard_action() -> None:
    assert _action_to_move(chess.Board(), 0) == chess.Move.null()


def test_chess_domain_handles_nested_fen_batches_for_replay_paths() -> None:
    domain = ChessDomain(embed_dim=16, opponent_level=0)
    observation, info = domain.reset(seed=0)
    state = domain.build_domain_state(observation, info)
    replay_like_state = {
        **state,
        "fen": [[state["fen"][0]]],
        "history_block": state["history_block"][:1],
    }

    baseline = domain.build_simple_baseline_path(
        replay_like_state,
        path_length=2,
        action_dim=domain.get_action_dim(),
    )
    imagined = domain.build_imagined_domain_state(
        replay_like_state,
        torch.zeros(1, domain.get_action_dim()),
        torch.zeros(1, 16),
        0,
    )

    assert baseline is not None
    assert baseline.shape == (1, 2, domain.get_action_dim())
    assert imagined["board_observation"].shape == (1, 8, 8, 111)


def test_chess_domain_exact_step_and_imagined_rollout_update_board() -> None:
    domain = ChessDomain(embed_dim=16, opponent_level=0)
    observation, info = domain.reset(seed=0)
    action_id = _action_from_move(chess.Move.from_uci("e2e4"))
    action_logits = torch.full((1, domain.get_action_dim()), -6.0)
    action_logits[0, action_id] = 6.0

    state = domain.build_domain_state(observation, info)
    imagined = domain.build_imagined_domain_state(
        state,
        action_logits,
        torch.zeros(1, 16),
        0,
    )
    outcome = domain.simulate_path_outcome(action_logits.unsqueeze(1), state)

    assert imagined["board_observation"].shape == (1, 8, 8, 111)
    assert imagined["legal_actions_mask"].shape == (1, domain.get_action_dim())
    assert imagined["move_count"][0].item() >= 1.0
    assert outcome is not None
    assert outcome["outcome_observation"].shape == (1, 8, 8, 111)
    assert outcome["realized_intrinsic_cost"].shape == (1,)


def test_chess_imagined_rollout_does_not_search_opponent(monkeypatch: pytest.MonkeyPatch) -> None:
    domain = ChessDomain(embed_dim=16, opponent_level=0)
    domain.set_eval_opponent_depth(4)
    observation, info = domain.reset(seed=0)
    action_id = _action_from_move(chess.Move.from_uci("e2e4"))
    action_logits = torch.full((1, domain.get_action_dim()), -6.0)
    action_logits[0, action_id] = 6.0

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("imagined rollout must not use opponent search")

    monkeypatch.setattr(domain, "_choose_opponent_move", fail_if_called)

    imagined = domain.build_imagined_domain_state(
        domain.build_domain_state(observation, info),
        action_logits,
        torch.zeros(1, 16),
        0,
    )

    assert imagined["move_count"][0].item() == 1.0
    assert imagined["fen"][0].split()[1] == "b"


def test_chess_domain_terminal_flags_from_fen_positions() -> None:
    domain = ChessDomain(embed_dim=16, opponent_level=0)
    checkmate_board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")
    checkmate_obs = {
        "observation": torch.zeros(8, 8, 111),
        "fen": checkmate_board.fen(),
        "history_block": torch.zeros(8, 8, 104),
    }
    checkmate_state = domain.build_domain_state(checkmate_obs, None)
    assert float(checkmate_state["is_checkmate"][0].item()) == 1.0

    draw_board = chess.Board("8/8/8/8/8/8/6k1/7K w - - 0 1")
    draw_obs = {
        "observation": torch.zeros(8, 8, 111),
        "fen": draw_board.fen(),
        "history_block": torch.zeros(8, 8, 104),
    }
    draw_state = domain.build_domain_state(draw_obs, None)
    assert float(draw_state["is_draw"][0].item()) == 1.0


def test_chess_domain_config_fields_and_self_play_gate() -> None:
    domain = ChessDomain(
        embed_dim=16,
        opponent_level=0,
        self_play_unlock_threshold=0.9,
        chess_data_root="/tmp/chess-data",
        stockfish_path="/definitely/not/used/by/runtime",
        move_vocabulary_source="alphazero_4672",
    )
    assert domain.chess_data_root == "/tmp/chess-data"
    assert domain.stockfish_path == "/definitely/not/used/by/runtime"
    assert domain.move_vocabulary_source == "alphazero_4672"
    assert domain.maybe_enable_self_play(
        {
            "puzzle_solve_rate": 0.95,
            "invalid_rate": 0.0,
            "blunder_rate": 0.1,
            "win_rate": 0.6,
            "draw_rate": 0.1,
        }
    )
    blocked = ChessDomain(embed_dim=16, opponent_level=0, self_play_unlock_threshold=0.9)
    assert blocked.maybe_enable_self_play(
        {
            "puzzle_solve_rate": 0.95,
            "invalid_rate": 0.0,
            "blunder_rate": 0.35,
            "win_rate": 0.8,
            "draw_rate": 0.0,
        }
    ) is False


def test_chess_domain_runtime_does_not_depend_on_stockfish_path() -> None:
    domain = ChessDomain(embed_dim=16, opponent_level=0, stockfish_path="/missing/stockfish")
    observation, info = domain.reset(seed=0)
    action = domain.baseline_action("simple", observation, info)
    next_observation, reward, terminated, truncated, next_info = domain.step(action)
    assert next_observation["observation"].shape == (8, 8, 111)
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
    assert "winner" in next_info


def test_orchestrator_can_build_chess_adapter_and_model(tmp_path: Path) -> None:
    config = _tiny_orchestrator_config(tmp_path / "chess_build")
    config.domains = [
        DomainRunConfig(
            name="chess",
            family="state_vector_hjepa",
            adapter_kwargs={"opponent_level": 0},
            bootstrap_random_episodes=1,
            bootstrap_simple_episodes=1,
            bootstrap_pretrain_steps=1,
            bootstrap_replay_warmup_steps=1,
            bootstrap_batch_size=1,
            max_episode_steps=4,
            optimizer_interval=1,
            sleep_interval_episodes=1,
            checkpoint_interval_episodes=1,
            evaluation_episodes=1,
            world_model_backend="mamba",
            mcts_simulations=1,
            mcts_depth=1,
            mcts_rollout_horizon=1,
            baselines=("random", "simple"),
            phases={"core_control": DomainPhaseConfig(episodes=1)},
        )
    ]
    orchestrator = UnifiedTrainingOrchestrator(config)
    adapter = orchestrator._build_adapter(config.domains[0])
    model = orchestrator._build_model(config.domains[0], adapter)
    observation, info = adapter.reset(seed=0)
    tokenized = adapter.tokenize_observation(observation)
    outputs = model(
        tokens=tokenized.tokens.to(orchestrator.device),
        mask=torch.zeros(tokenized.tokens.shape[:2], dtype=torch.float32, device=orchestrator.device),
        domain_state=_move_nested_to_device(adapter.build_domain_state(observation, info), orchestrator.device),
        actor_mode="mode2",
        store_to_memory=False,
    )

    assert isinstance(adapter, ChessDomain)
    assert outputs["action"].shape == (1,)


def test_orchestrator_seeds_chess_curriculum_bootstrap(tmp_path: Path) -> None:
    chess_root = tmp_path / "stage3" / "chess" / "lichess"
    chess_root.mkdir(parents=True)
    row = {
        "fen": chess.STARTING_FEN,
        "best_move": "e2e4",
        "candidate_moves": ["e2e4", "d2d4"],
        "candidate_scores_cp": [30.0, 20.0],
        "candidate_blunder_losses_cp": [0.0, 10.0],
        "phase": "opening",
        "source": "opening_book",
    }
    (chess_root / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (chess_root / "validation.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = _tiny_orchestrator_config(tmp_path / "chess_curriculum")
    config.domains = [
        DomainRunConfig(
            name="chess",
            family="state_vector_hjepa",
            adapter_kwargs={"opponent_level": 0, "chess_data_root": str(chess_root)},
            bootstrap_random_episodes=0,
            bootstrap_simple_episodes=0,
            bootstrap_curriculum_samples=1,
            bootstrap_curriculum_buckets=("openings",),
            bootstrap_pretrain_steps=1,
            bootstrap_replay_warmup_steps=1,
            bootstrap_batch_size=1,
            max_episode_steps=4,
            evaluation_episodes=1,
            world_model_backend="mamba",
            mcts_simulations=1,
            mcts_depth=1,
            mcts_rollout_horizon=1,
            phases={"core_control": DomainPhaseConfig(episodes=1)},
        )
    ]
    orchestrator = UnifiedTrainingOrchestrator(config)
    adapter = orchestrator._build_adapter(config.domains[0])
    transitions = orchestrator._collect_bootstrap_trajectories(adapter, config.domains[0])
    assert len(transitions) == 1
    assert transitions[0].source == "bootstrap_curriculum_openings"
    assert int(transitions[0].action.item()) == _action_from_move(chess.Move.from_uci("e2e4"))
    assert transitions[0].next_observation["fen"] != chess.STARTING_FEN


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


def test_replay_buffer_samples_contiguous_windows() -> None:
    replay = TransitionReplayBuffer(capacity=16)
    for step_idx in range(4):
        replay.add(
            ReplayRecord(
                domain_id="cartpole",
                modality_family="state_vector_hjepa",
                episode_id=3,
                step_idx=step_idx,
                obs_raw=step_idx,
                tokenizer_input=torch.tensor([step_idx], dtype=torch.float32),
                tokens=torch.randn(4, 8),
                mask=torch.zeros(4),
                domain_state={"state_vector": torch.randn(1, 4)},
                action=torch.tensor([1]),
                action_logits_or_vec=torch.randn(2),
                legal_actions_mask=torch.tensor([True, True]),
                reward=1.0,
                cost=0.1,
                done=False,
                next_obs_raw=step_idx + 1,
                next_tokens=torch.randn(4, 8),
                next_mask=torch.zeros(4),
                next_domain_state={"state_vector": torch.randn(1, 4)},
                latent_z=torch.randn(8),
                next_latent_z=torch.randn(8),
                ctx_tokens=torch.randn(4, 8),
                search_policy=None,
                search_value=None,
                selected_path=None,
                selected_posture=None,
            )
        )
    windows = replay.sample_windows(batch_size=8, horizon=3, domain_id="cartpole")
    assert windows
    for window in windows:
        assert len(window.records) == 3
        assert [record.step_idx for record in window.records] in ([0, 1, 2], [1, 2, 3])


def test_orchestrator_critic_targets_discounted_future_return(tmp_path: Path) -> None:
    config = _tiny_orchestrator_config(tmp_path / "critic_targets")
    orchestrator = UnifiedTrainingOrchestrator(config)
    adapter = orchestrator._build_adapter(config.domains[0])
    model = orchestrator._build_model(config.domains[0], adapter)
    model.cost.gamma = 0.5
    embed_dim = int(model.actor.embed_dim)

    class CaptureCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.captured_target: torch.Tensor | None = None
            self.captured_keys: torch.Tensor | None = None
            self.captured_ctx_tokens: torch.Tensor | None = None

        def forward(self, z: torch.Tensor, ctx_tokens: torch.Tensor) -> torch.Tensor:
            self.captured_keys = z.detach().cpu()
            self.captured_ctx_tokens = ctx_tokens.detach().cpu()
            return torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)

        def compute_critic_loss(
            self,
            predicted_value: torch.Tensor,
            realized_ic: torch.Tensor,
        ) -> torch.Tensor:
            self.captured_target = realized_ic.detach().cpu()
            return predicted_value.sum() * 0.0 + realized_ic.mean()

    capture_critic = CaptureCritic()
    model.cost.trainable_critic = capture_critic

    def make_record(step_idx: int, cost: float, *, done: bool) -> ReplayRecord:
        return ReplayRecord(
            domain_id="cartpole",
            modality_family="state_vector_hjepa",
            episode_id=11,
            step_idx=step_idx,
            obs_raw={"step": step_idx},
            tokenizer_input=torch.zeros(4),
            tokens=torch.zeros(4, embed_dim),
            mask=torch.zeros(4),
            domain_state={"state_vector": torch.zeros(1, 4)},
            action=torch.tensor([1]),
            action_logits_or_vec=torch.tensor([0.0, 1.0], dtype=torch.float32),
            legal_actions_mask=torch.tensor([True, True]),
            reward=1.0,
            cost=cost,
            done=done,
            next_obs_raw={"step": step_idx + 1},
            next_tokens=torch.zeros(4, embed_dim),
            next_mask=torch.zeros(4),
            next_domain_state={"state_vector": torch.zeros(1, 4)},
            latent_z=torch.full((embed_dim,), float(step_idx), dtype=torch.float32),
            next_latent_z=torch.full((embed_dim,), float(step_idx + 1), dtype=torch.float32),
            ctx_tokens=torch.full((4, embed_dim), float(step_idx), dtype=torch.float32),
            search_policy=None,
            search_value=None,
            selected_path=None,
            selected_posture=None,
        )

    window = ReplayWindow(
        records=(
            make_record(0, 1.0, done=False),
            make_record(1, 2.0, done=False),
            make_record(2, 3.0, done=True),
        )
    )
    loss = orchestrator._compute_critic_loss(model, [window])

    assert loss is not None
    assert capture_critic.captured_target is not None
    assert capture_critic.captured_keys is not None
    assert capture_critic.captured_ctx_tokens is not None
    assert torch.isclose(capture_critic.captured_target[0], torch.tensor(3.5), atol=1.0e-6)
    assert torch.isclose(capture_critic.captured_keys[0, 0], torch.tensor(1.0), atol=1.0e-6)
    assert torch.isclose(capture_critic.captured_ctx_tokens[0, 0, 0], torch.tensor(1.0), atol=1.0e-6)


def test_orchestrator_critic_fallback_uses_current_state_cost_target(tmp_path: Path) -> None:
    config = _tiny_orchestrator_config(tmp_path / "critic_fallback_targets")
    orchestrator = UnifiedTrainingOrchestrator(config)
    adapter = orchestrator._build_adapter(config.domains[0])
    model = orchestrator._build_model(config.domains[0], adapter)
    embed_dim = int(model.actor.embed_dim)

    class CaptureCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.captured_target: torch.Tensor | None = None
            self.captured_keys: torch.Tensor | None = None
            self.captured_ctx_tokens: torch.Tensor | None = None

        def forward(self, z: torch.Tensor, ctx_tokens: torch.Tensor) -> torch.Tensor:
            self.captured_keys = z.detach().cpu()
            self.captured_ctx_tokens = ctx_tokens.detach().cpu()
            return torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)

        def compute_critic_loss(
            self,
            predicted_value: torch.Tensor,
            realized_ic: torch.Tensor,
        ) -> torch.Tensor:
            self.captured_target = realized_ic.detach().cpu()
            return predicted_value.sum() * 0.0 + realized_ic.mean()

    capture_critic = CaptureCritic()
    model.cost.trainable_critic = capture_critic

    record = ReplayRecord(
        domain_id="cartpole",
        modality_family="state_vector_hjepa",
        episode_id=3,
        step_idx=0,
        obs_raw={"step": 0},
        tokenizer_input=torch.zeros(4),
        tokens=torch.zeros(4, embed_dim),
        mask=torch.zeros(4),
        domain_state={"state_vector": torch.zeros(1, 4)},
        action=torch.tensor([1]),
        action_logits_or_vec=torch.tensor([0.0, 1.0], dtype=torch.float32),
        legal_actions_mask=torch.tensor([True, True]),
        reward=1.0,
        cost=4.0,
        done=True,
        next_obs_raw={"step": 1},
        next_tokens=torch.zeros(4, embed_dim),
        next_mask=torch.zeros(4),
        next_domain_state={"state_vector": torch.zeros(1, 4)},
        latent_z=torch.full((embed_dim,), 7.0, dtype=torch.float32),
        next_latent_z=torch.full((embed_dim,), 9.0, dtype=torch.float32),
        ctx_tokens=torch.full((4, embed_dim), 2.0, dtype=torch.float32),
        search_policy=None,
        search_value=None,
        selected_path=None,
        selected_posture=None,
    )

    loss = orchestrator._compute_critic_loss(model, [], fallback_records=[record])

    assert loss is not None
    assert capture_critic.captured_target is not None
    assert capture_critic.captured_keys is not None
    assert capture_critic.captured_ctx_tokens is not None
    assert torch.isclose(capture_critic.captured_target[0], torch.tensor(4.0), atol=1.0e-6)
    assert torch.isclose(capture_critic.captured_keys[0, 0], torch.tensor(7.0), atol=1.0e-6)
    assert torch.isclose(capture_critic.captured_ctx_tokens[0, 0, 0], torch.tensor(2.0), atol=1.0e-6)


def test_orchestrator_world_model_uses_imagined_state_calibration_loss(tmp_path: Path) -> None:
    config = _tiny_orchestrator_config(tmp_path / "world_model_calibration")
    orchestrator = UnifiedTrainingOrchestrator(config)
    adapter = orchestrator._build_adapter(config.domains[0])
    model = orchestrator._build_model(config.domains[0], adapter)
    model.cost.gamma = 0.5
    embed_dim = int(model.actor.embed_dim)

    class StubWorldModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.max_horizon = 4

        def compute_trajectory_loss(
            self,
            z_t: torch.Tensor,
            actions: torch.Tensor,
            target_trajectory: torch.Tensor,
            ctx_tokens: torch.Tensor,
            *,
            candidate_postures: torch.Tensor | None = None,
            reasoning_states: torch.Tensor | None = None,
            step_weights: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            _ = z_t
            _ = actions
            _ = ctx_tokens
            _ = candidate_postures
            _ = reasoning_states
            _ = step_weights
            return (
                torch.full((), 1.25, dtype=target_trajectory.dtype, device=target_trajectory.device),
                (target_trajectory + 3.0).unsqueeze(1).detach().clone(),
                torch.full(
                    (target_trajectory.shape[0], 1, target_trajectory.shape[1]),
                    0.4,
                    dtype=target_trajectory.dtype,
                    device=target_trajectory.device,
                ),
            )

    model.world_model = StubWorldModel()
    model.domain.compute_latent_state_decoder_loss = lambda *args, **kwargs: torch.zeros((), device=orchestrator.device)
    model.domain.compute_imagined_state_calibration_loss = (
        lambda predicted_future_z, action, target_domain_state, step_idx: torch.tensor(
            float(step_idx + 1),
            dtype=predicted_future_z.dtype,
            device=predicted_future_z.device,
        )
    )

    def make_record(step_idx: int) -> ReplayRecord:
        state = torch.full((1, 4), float(step_idx), dtype=torch.float32)
        return ReplayRecord(
            domain_id="cartpole",
            modality_family="state_vector_hjepa",
            episode_id=21,
            step_idx=step_idx,
            obs_raw={"step": step_idx},
            tokenizer_input=torch.zeros(4),
            tokens=torch.zeros(4, embed_dim),
            mask=torch.zeros(4),
            domain_state={"state_vector": state},
            action=torch.tensor([1]),
            action_logits_or_vec=torch.tensor([0.0, 1.0], dtype=torch.float32),
            legal_actions_mask=torch.tensor([True, True]),
            reward=1.0,
            cost=0.0,
            done=False,
            next_obs_raw={"step": step_idx + 1},
            next_tokens=torch.zeros(4, embed_dim),
            next_mask=torch.zeros(4),
            next_domain_state={"state_vector": state + 1.0},
            latent_z=torch.full((embed_dim,), float(step_idx), dtype=torch.float32),
            next_latent_z=torch.full((embed_dim,), float(step_idx + 1), dtype=torch.float32),
            ctx_tokens=torch.zeros(4, embed_dim),
            search_policy=None,
            search_value=None,
            selected_path=None,
            selected_posture=None,
        )

    window = ReplayWindow(records=(make_record(0), make_record(1)))
    loss = orchestrator._compute_world_model_loss(model, [window])
    expected = 1.25 + (0.5 * ((2.0 / 3.0) * 1.0 + (1.0 / 3.0) * 2.0))

    assert loss is not None
    assert torch.isclose(
        loss,
        torch.tensor(expected, dtype=loss.dtype, device=loss.device),
        atol=1.0e-6,
    )


def test_cartpole_intrinsic_cost_subtracts_step_reward_for_stable_states() -> None:
    cartpole = CartPoleDomain(embed_dim=16)
    action = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    stable_state = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    unstable_state = torch.tensor([[0.0, 0.0, 0.35, 0.0]], dtype=torch.float32)
    stability_cost, _ = cartpole.get_intrinsic_cost_fns()[0]

    stable_cost = stability_cost(stable_state, action, cartpole.build_domain_state(stable_state, None))
    unstable_cost = stability_cost(unstable_state, action, cartpole.build_domain_state(unstable_state, None))

    assert float(stable_cost.item()) < -0.9
    assert float(stable_cost.item()) > -1.1
    assert float(unstable_cost.item()) > 0.0


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
                bootstrap_simple_episodes=1,
                bootstrap_pretrain_steps=1,
                bootstrap_replay_warmup_steps=1,
                bootstrap_batch_size=1,
                mask_ratio=0.25,
                max_episode_steps=4,
                optimizer_interval=1,
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
    assert "full_train_mode" in payload["evaluation"]
    assert "episode_reward_mean" in payload["evaluation"]["full_train_mode"]
    assert len(orchestrator.replay) > 0
    assert restored_model.memory.size >= 0
    assert restored_model.world_model_backend == "mamba"
    assert isinstance(restored_model.world_model, MambaActionConditionedWorldModel)
    assert restored_model.domain_index is not None
    assert restored_model.domain_index.adapter_bank is not None
    diagnostic_path = Path(config.run_dir) / "cartpole" / "diagnostics" / "sleep.jsonl"
    progress_events = [
        json.loads(line)
        for line in diagnostic_path.read_text().splitlines()
        if json.loads(line).get("kind") == "episode_progress"
    ]
    assert progress_events
    assert "execution_source" in progress_events[-1]
    assert "effective_epsilon" in progress_events[-1]
    assert "candidate_total_std" in progress_events[-1]
    assert "world_model_error" in progress_events[-1]
    assert "baseline_guard_fraction" in progress_events[-1]
    assert "learned_branch_fraction" in progress_events[-1]
    assert "behavior_diagnostics" in payload["evaluation"]
    assert "exploration_fraction" in payload["evaluation"]["behavior_diagnostics"]


def test_bootstrap_trajectories_seed_replay_with_random_and_simple_sources(tmp_path: Path) -> None:
    config = _tiny_orchestrator_config(tmp_path / "bootstrap_seed")
    config.domains[0].bootstrap_teacher_episodes = 1
    orchestrator = UnifiedTrainingOrchestrator(config)
    domain_cfg = config.domains[0]
    adapter = orchestrator._build_adapter(domain_cfg)
    transitions = orchestrator._collect_bootstrap_trajectories(adapter, domain_cfg)

    assert {transition.source for transition in transitions} == {
        "bootstrap_teacher",
        "bootstrap_random",
        "bootstrap_simple",
    }

    model = orchestrator._build_model(domain_cfg, adapter)
    orchestrator._seed_replay_from_bootstrap(model, adapter, domain_cfg, transitions)

    replay_sources = {record.execution_source for record in orchestrator.replay.records}
    assert replay_sources == {"bootstrap_teacher", "bootstrap_random", "bootstrap_simple"}
    assert all(record.selected_path is not None for record in orchestrator.replay.records)
    assert model._step_counter == 0


def test_bootstrap_summary_reports_teacher_quality_stats(tmp_path: Path) -> None:
    config = _tiny_orchestrator_config(tmp_path / "bootstrap_summary")
    config.domains[0].bootstrap_teacher_episodes = 1
    orchestrator = UnifiedTrainingOrchestrator(config)
    adapter = orchestrator._build_adapter(config.domains[0])
    transitions = orchestrator._collect_bootstrap_trajectories(adapter, config.domains[0])

    summary = orchestrator._summarize_bootstrap_transitions(adapter, transitions)

    assert "overall" in summary
    assert "sources" in summary
    assert "bootstrap_teacher" in summary["sources"]
    assert summary["overall"]["episodes"] >= 3.0
    assert "reward_mean" in summary["sources"]["bootstrap_teacher"]
    assert "length_ge_20_pct" in summary["sources"]["bootstrap_teacher"]


def test_bootstrap_seeded_replay_supports_replay_warmup_batches(tmp_path: Path) -> None:
    config = _tiny_orchestrator_config(tmp_path / "bootstrap_warmup")
    orchestrator = UnifiedTrainingOrchestrator(config)
    domain_cfg = config.domains[0]
    adapter = orchestrator._build_adapter(domain_cfg)
    transitions = orchestrator._collect_bootstrap_trajectories(adapter, domain_cfg)
    model = orchestrator._build_model(domain_cfg, adapter)
    orchestrator._seed_replay_from_bootstrap(model, adapter, domain_cfg, transitions)

    actor_batch = orchestrator.replay.sample(32, domain_id=domain_cfg.name, require_next_latent=False)
    world_model_windows = orchestrator.replay.sample_windows(
        32,
        horizon=min(config.rollout_horizon, model.world_model.max_horizon),
        domain_id=domain_cfg.name,
    )
    critic_windows = orchestrator.replay.sample_windows(32, horizon=2, domain_id=domain_cfg.name)

    assert orchestrator._count_actor_valid(actor_batch) > 0
    assert orchestrator._count_world_model_valid(world_model_windows) > 0
    assert orchestrator._count_critic_valid(critic_windows, fallback_records=actor_batch) > 0

    orchestrator._bootstrap_replay_warmup(
        model,
        domain_cfg=domain_cfg,
        steps=1,
    )


def test_bootstrap_pretrain_logs_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = _tiny_orchestrator_config(tmp_path / "bootstrap_pretrain_logs")
    orchestrator = UnifiedTrainingOrchestrator(config)
    domain_cfg = config.domains[0]
    adapter = orchestrator._build_adapter(domain_cfg)
    transitions = orchestrator._collect_bootstrap_trajectories(adapter, domain_cfg)
    observations = orchestrator._bootstrap_observations(transitions)

    orchestrator._pretrain_hjepa(
        domain_cfg.family,
        adapter,
        observations,
        steps=3,
        batch_size=1,
        mask_ratio=domain_cfg.mask_ratio,
        lr=1.0e-4,
    )

    captured = capsys.readouterr().out
    assert "bootstrap_pretrain start" in captured
    assert "bootstrap_pretrain step=1/3" in captured
    assert "bootstrap_pretrain step=3/3" in captured


def test_evaluate_model_never_uses_training_exploration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _tiny_orchestrator_config(tmp_path / "eval_no_explore")
    orchestrator = UnifiedTrainingOrchestrator(config)
    domain_cfg = config.domains[0]
    adapter = orchestrator._build_adapter(domain_cfg)
    model = orchestrator._build_model(domain_cfg, adapter)
    random_calls = 0
    original_baseline_action = adapter.baseline_action

    def tracked_baseline_action(kind: str, observation: object, info: dict[str, object] | None = None) -> object:
        nonlocal random_calls
        if kind == "random":
            random_calls += 1
        return original_baseline_action(kind, observation, info)

    monkeypatch.setattr(adapter, "baseline_action", tracked_baseline_action)
    metrics = orchestrator._evaluate_model(model, adapter, domain_cfg, episodes=1, ablation="full")

    assert metrics["episodes"] == 1.0
    assert random_calls == 0


def test_training_exploration_override_records_executed_random_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _tiny_orchestrator_config(tmp_path / "epsilon_override")
    orchestrator = UnifiedTrainingOrchestrator(config)
    domain_cfg = config.domains[0]
    phase_cfg = DomainPhaseConfig(
        episodes=1,
        use_memory=True,
        exploration_epsilon_start=1.0,
        exploration_epsilon_end=1.0,
        exploration_uncertainty_bonus=0.0,
        exploration_uncertainty_std_threshold=0.0,
    )
    adapter = orchestrator._build_adapter(domain_cfg)
    model = orchestrator._build_model(domain_cfg, adapter)
    representation_loss_fn = orchestrator._build_representation_loss(domain_cfg.family)
    original_baseline_action = adapter.baseline_action
    original_forward = model.forward

    def fixed_baseline_action(kind: str, observation: object, info: dict[str, object] | None = None) -> object:
        if kind == "random":
            return torch.tensor([1], dtype=torch.long)
        return original_baseline_action(kind, observation, info)

    def forced_forward(*args: object, **kwargs: object) -> dict[str, object]:
        outputs = original_forward(*args, **kwargs)
        device = outputs["action_vec"].device
        dtype = outputs["action_vec"].dtype
        outputs["action"] = torch.tensor([0], dtype=torch.long, device=device)
        outputs["action_vec"] = torch.tensor([[1.0, 0.0]], dtype=dtype, device=device)
        outputs["selected_path"] = torch.tensor([[[1.0, 0.0]]], dtype=dtype, device=device)
        return outputs

    monkeypatch.setattr(adapter, "baseline_action", fixed_baseline_action)
    monkeypatch.setattr(model, "forward", forced_forward)

    phase_metrics = orchestrator._run_phase(
        phase_name="core_control",
        phase_cfg=phase_cfg,
        domain_cfg=domain_cfg,
        adapter=adapter,
        model=model,
        representation_loss_fn=representation_loss_fn,
    )

    assert phase_metrics["phase_passed"] in {True, False}
    replay_record = orchestrator.replay.records[-1]
    assert replay_record.execution_source == "epsilon_random"
    assert replay_record.exploration_taken
    assert int(torch.as_tensor(replay_record.action).reshape(-1)[0].item()) == 1
    assert int(replay_record.action_logits_or_vec.argmax().item()) == 1
    assert replay_record.selected_path is not None
    assert replay_record.selected_path.shape[0] == 1
    assert int(replay_record.selected_path[0].argmax().item()) == 1

    memory_record = model.memory.records[0]
    assert memory_record.metadata is not None
    assert memory_record.metadata["execution_source"] == "epsilon_random"
    assert memory_record.metadata["exploration_taken"] is True
    assert "world_model_error" in memory_record.metadata
    assert "surprise_priority" in memory_record.metadata
    assert int(memory_record.action.argmax().item()) == 1
    assert memory_record.selected_path is not None
    assert memory_record.selected_path.shape[0] == 1
    assert int(memory_record.selected_path[0].argmax().item()) == 1
    assert memory_record.ic_realized is not None


def test_mode2_actor_loss_uses_deliberate_path_not_mode1(tmp_path: Path) -> None:
    config = _tiny_orchestrator_config(tmp_path / "actor_loss")
    orchestrator = UnifiedTrainingOrchestrator(config)
    adapter = orchestrator._build_adapter(config.domains[0])
    model = orchestrator._build_model(config.domains[0], adapter)
    observation, info = adapter.reset(seed=123)
    domain_state = adapter.build_domain_state(observation, info)
    records: list[ReplayRecord] = []
    embed_dim = model.actor.embed_dim
    obs_tensor = (
        observation.detach().clone().float()
        if torch.is_tensor(observation)
        else torch.tensor(observation, dtype=torch.float32)
    )
    for step_idx in range(4):
        records.append(
            ReplayRecord(
                domain_id="cartpole",
                modality_family="state_vector_hjepa",
                episode_id=1,
                step_idx=step_idx,
                obs_raw=obs_tensor,
                tokenizer_input=obs_tensor,
                tokens=torch.randn(4, embed_dim),
                mask=torch.zeros(4),
                domain_state=domain_state,
                action=torch.tensor([1]),
                action_logits_or_vec=torch.randn(model.actor.action_dim),
                legal_actions_mask=torch.tensor([True, True]),
                reward=1.0,
                cost=0.1,
                done=False,
                next_obs_raw=obs_tensor,
                next_tokens=torch.randn(4, embed_dim),
                next_mask=torch.zeros(4),
                next_domain_state=domain_state,
                latent_z=torch.randn(embed_dim),
                next_latent_z=torch.randn(embed_dim),
                ctx_tokens=torch.randn(config.num_ctx_tokens, embed_dim),
                search_policy=None,
                search_value=None,
                selected_path=None,
                selected_posture=None,
            )
        )
    model.zero_grad(set_to_none=True)
    loss = orchestrator._compute_actor_loss(model, records, action_space_type="discrete")
    assert loss is not None
    loss.backward()
    assert model.actor.mode_1_policy.weight.grad is None
    assert model.actor.candidate_selection_head[0].weight.grad is not None


def test_mode2_actor_loss_trains_candidate_paths_from_replay_targets(tmp_path: Path) -> None:
    config = _tiny_orchestrator_config(tmp_path / "actor_path_loss")
    orchestrator = UnifiedTrainingOrchestrator(config)
    adapter = orchestrator._build_adapter(config.domains[0])
    model = orchestrator._build_model(config.domains[0], adapter)
    model.actor.compute_posture_diversity_loss = (
        lambda *args, **kwargs: torch.zeros((), device=next(model.parameters()).device)
    )
    observation, info = adapter.reset(seed=321)
    domain_state = adapter.build_domain_state(observation, info)
    records: list[ReplayRecord] = []
    embed_dim = model.actor.embed_dim
    obs_tensor = (
        observation.detach().clone().float()
        if torch.is_tensor(observation)
        else torch.tensor(observation, dtype=torch.float32)
    )
    for step_idx in range(4):
        records.append(
            ReplayRecord(
                domain_id="cartpole",
                modality_family="state_vector_hjepa",
                episode_id=1,
                step_idx=step_idx,
                obs_raw=obs_tensor,
                tokenizer_input=obs_tensor,
                tokens=torch.randn(4, embed_dim),
                mask=torch.zeros(4),
                domain_state=domain_state,
                action=torch.tensor([1]),
                action_logits_or_vec=torch.randn(model.actor.action_dim),
                legal_actions_mask=torch.tensor([True, True]),
                reward=1.0,
                cost=0.1,
                done=False,
                next_obs_raw=obs_tensor,
                next_tokens=torch.randn(4, embed_dim),
                next_mask=torch.zeros(4),
                next_domain_state=domain_state,
                latent_z=torch.randn(embed_dim),
                next_latent_z=torch.randn(embed_dim),
                ctx_tokens=torch.randn(config.num_ctx_tokens, embed_dim),
                search_policy=torch.tensor([0.2, 0.8], dtype=torch.float32),
                search_value=None,
                selected_path=torch.randn(model.actor.path_length, model.actor.action_dim),
                selected_posture=None,
            )
        )
    model.zero_grad(set_to_none=True)
    loss = orchestrator._compute_actor_loss(model, records, action_space_type="discrete")
    assert loss is not None
    loss.backward()
    action_head_grad = model.actor.action_head[0].weight.grad
    assert action_head_grad is not None
    assert float(action_head_grad.abs().sum().item()) > 0.0


def test_behavior_cloning_loss_prefers_executed_replay_action(tmp_path: Path) -> None:
    config = _tiny_orchestrator_config(tmp_path / "behavior_cloning")
    orchestrator = UnifiedTrainingOrchestrator(config)
    adapter = orchestrator._build_adapter(config.domains[0])
    model = orchestrator._build_model(config.domains[0], adapter)
    embed_dim = model.actor.embed_dim
    posture_dim = model.actor.posture_dim

    def make_record(action_index: int) -> ReplayRecord:
        return ReplayRecord(
            domain_id="cartpole",
            modality_family="state_vector_hjepa",
            episode_id=1,
            step_idx=0,
            obs_raw={"state": [0.0, 0.0, 0.0, 0.0]},
            tokenizer_input=torch.zeros(4),
            tokens=torch.zeros(4, embed_dim),
            mask=torch.zeros(4),
            domain_state={"state_vector": torch.zeros(1, 4)},
            action=torch.tensor([action_index]),
            action_logits_or_vec=torch.tensor(
                [1.0, 0.0] if action_index == 0 else [0.0, 1.0],
                dtype=torch.float32,
            ),
            legal_actions_mask=torch.tensor([True, True]),
            reward=1.0,
            cost=0.0,
            done=False,
            next_obs_raw={"state": [0.0, 0.0, 0.0, 0.0]},
            next_tokens=torch.zeros(4, embed_dim),
            next_mask=torch.zeros(4),
            next_domain_state={"state_vector": torch.zeros(1, 4)},
            latent_z=torch.zeros(embed_dim),
            next_latent_z=torch.zeros(embed_dim),
            ctx_tokens=torch.zeros(config.num_ctx_tokens, embed_dim),
            search_policy=None,
            search_value=None,
            selected_path=None,
            selected_posture=None,
        )

    def fake_propose(
        states: torch.Tensor,
        ctx_tokens: torch.Tensor,
        simple_baseline_path: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _ = states
        _ = ctx_tokens
        _ = simple_baseline_path
        candidate_paths = torch.tensor(
            [[[[ -6.0, 6.0], [0.0, 0.0]], [[-4.0, 4.0], [0.0, 0.0]], [[-3.0, 3.0], [0.0, 0.0]]]],
            dtype=torch.float32,
            device=orchestrator.device,
        ).requires_grad_()
        return {
            "candidate_paths": candidate_paths,
            "candidate_selection_logits": torch.zeros(1, 3, device=orchestrator.device, requires_grad=True),
            "candidate_postures": torch.zeros(1, 3, posture_dim, device=orchestrator.device),
            "reasoning_states": torch.zeros(1, 3, embed_dim, device=orchestrator.device),
        }

    class FakeWorldModel(torch.nn.Module):
        max_horizon = 2

        def forward(
            self,
            z: torch.Tensor,
            actions: torch.Tensor,
            ctx_tokens: torch.Tensor,
            candidate_postures: torch.Tensor | None = None,
            reasoning_states: torch.Tensor | None = None,
            horizon: int = 1,
        ) -> dict[str, torch.Tensor]:
            _ = z
            _ = actions
            _ = ctx_tokens
            _ = candidate_postures
            _ = reasoning_states
            trajectory = torch.zeros(1, 3, horizon, embed_dim, device=orchestrator.device)
            return {
                "trajectory": trajectory,
                "terminal_latents": torch.zeros(1, 3, embed_dim, device=orchestrator.device),
            }

    class FakeCost(torch.nn.Module):
        def score_candidates(
            self,
            *,
            z: torch.Tensor,
            actions: torch.Tensor,
            ctx_tokens: torch.Tensor,
            domain_state: dict[str, torch.Tensor],
            future_z: torch.Tensor,
            future_trajectory: torch.Tensor,
            imagined_domain_state_builder: object = None,
        ) -> dict[str, torch.Tensor]:
            _ = z
            _ = actions
            _ = ctx_tokens
            _ = domain_state
            _ = future_z
            _ = future_trajectory
            _ = imagined_domain_state_builder
            zeros = torch.zeros(1, 3, device=orchestrator.device)
            return {"ic": zeros, "tc": zeros, "total": zeros}

    model.actor.propose = fake_propose  # type: ignore[assignment]
    model.world_model = FakeWorldModel()  # type: ignore[assignment]
    model.cost = FakeCost()  # type: ignore[assignment]

    matching_loss = orchestrator._compute_actor_loss(
        model,
        [make_record(1)],
        action_space_type="discrete",
        behavior_cloning_weight=1.0,
    )
    mismatched_loss = orchestrator._compute_actor_loss(
        model,
        [make_record(0)],
        action_space_type="discrete",
        behavior_cloning_weight=1.0,
    )

    assert matching_loss is not None
    assert mismatched_loss is not None
    assert float(matching_loss.item()) < float(mismatched_loss.item())


def test_replay_sampling_prefers_surprising_records() -> None:
    replay = TransitionReplayBuffer(capacity=32)
    for step_idx in range(5):
        replay.add(
            ReplayRecord(
                domain_id="cartpole",
                modality_family="state_vector_hjepa",
                episode_id=1,
                step_idx=step_idx,
                obs_raw=step_idx,
                tokenizer_input=torch.zeros(1),
                tokens=torch.zeros(4, 8),
                mask=torch.zeros(4),
                domain_state={"state_vector": torch.zeros(1, 4)},
                action=torch.tensor([1]),
                action_logits_or_vec=torch.tensor([0.0, 1.0], dtype=torch.float32),
                legal_actions_mask=torch.tensor([True, True]),
                reward=1.0,
                cost=0.0,
                done=False,
                next_obs_raw=step_idx + 1,
                next_tokens=torch.zeros(4, 8),
                next_mask=torch.zeros(4),
                next_domain_state={"state_vector": torch.zeros(1, 4)},
                latent_z=torch.zeros(8),
                next_latent_z=torch.zeros(8),
                ctx_tokens=torch.zeros(4, 8),
                search_policy=None,
                search_value=None,
                selected_path=None,
                selected_posture=None,
                surprise_priority=10.0 if step_idx == 0 else 1.0,
            )
        )
    random.seed(0)
    weighted_hits = sum(
        1
        for _ in range(200)
        if replay.sample(1, domain_id="cartpole", priority_alpha=1.0)[0].step_idx == 0
    )
    random.seed(0)
    uniform_hits = sum(
        1
        for _ in range(200)
        if replay.sample(1, domain_id="cartpole", priority_alpha=0.0)[0].step_idx == 0
    )
    assert weighted_hits > uniform_hits


def test_replay_eviction_prefers_dropping_low_surprise_low_utility_record() -> None:
    replay = TransitionReplayBuffer(capacity=2)

    def make_record(step_idx: int, *, reward: float, cost: float, surprise_priority: float) -> ReplayRecord:
        return ReplayRecord(
            domain_id="cartpole",
            modality_family="state_vector_hjepa",
            episode_id=1,
            step_idx=step_idx,
            obs_raw=step_idx,
            tokenizer_input=torch.zeros(1),
            tokens=torch.zeros(4, 8),
            mask=torch.zeros(4),
            domain_state={"state_vector": torch.zeros(1, 4)},
            action=torch.tensor([1]),
            action_logits_or_vec=torch.tensor([0.0, 1.0], dtype=torch.float32),
            legal_actions_mask=torch.tensor([True, True]),
            reward=reward,
            cost=cost,
            done=False,
            next_obs_raw=step_idx + 1,
            next_tokens=torch.zeros(4, 8),
            next_mask=torch.zeros(4),
            next_domain_state={"state_vector": torch.zeros(1, 4)},
            latent_z=torch.zeros(8),
            next_latent_z=torch.zeros(8),
            ctx_tokens=torch.zeros(4, 8),
            search_policy=None,
            search_value=None,
            selected_path=None,
            selected_posture=None,
            surprise_priority=surprise_priority,
        )

    replay.add(make_record(0, reward=0.0, cost=1.0, surprise_priority=1.0))
    replay.add(make_record(1, reward=2.0, cost=0.0, surprise_priority=1.0))
    replay.add(make_record(2, reward=1.0, cost=0.0, surprise_priority=5.0))

    kept_steps = sorted(record.step_idx for record in replay.records)
    assert kept_steps == [1, 2]


def test_cartpole_simple_baseline_matches_fallback_dynamics() -> None:
    cartpole = CartPoleDomain(embed_dim=16)
    domain_state = {
        "state_vector": torch.tensor([[0.0, 0.0, 0.15, 0.0]], dtype=torch.float32),
    }
    baseline = cartpole.build_simple_baseline_path(domain_state, path_length=3, action_dim=2)
    assert baseline is not None
    assert int(baseline[0, 0].argmax().item()) == 0


def test_cartpole_simple_baseline_action_matches_path_heuristic() -> None:
    cartpole = CartPoleDomain(embed_dim=16)
    observation = torch.tensor([0.0, 0.0, 0.15, 0.0], dtype=torch.float32)

    simple_action = cartpole.baseline_action("simple", observation, None)

    assert int(simple_action.reshape(-1)[0].item()) == 0


def test_cartpole_teacher_action_is_available() -> None:
    cartpole = CartPoleDomain(embed_dim=16)
    observation = torch.tensor([0.0, 0.1, 0.02, 0.3], dtype=torch.float32)

    teacher_action = cartpole.baseline_action("teacher", observation, None)

    assert teacher_action.shape == (1,)
    assert int(teacher_action.reshape(-1)[0].item()) in {0, 1}


def test_imagined_mcts_replay_injection_adds_synthetic_actor_records(tmp_path: Path) -> None:
    config = _tiny_orchestrator_config(tmp_path / "imagined_replay")
    orchestrator = UnifiedTrainingOrchestrator(config)
    adapter = orchestrator._build_adapter(config.domains[0])
    observation, info = adapter.reset(seed=5)
    tokenized = adapter.tokenize_observation(observation)
    domain_state = adapter.build_domain_state(observation, info)
    embed_dim = 16
    action_dim = adapter.get_action_dim()
    outputs = {
        "selected_candidate_idx": torch.tensor([1], dtype=torch.long),
        "selected_path": torch.tensor(
            [[[0.1, 0.9], [0.8, 0.2], [0.2, 0.8]]],
            dtype=torch.float32,
        ),
        "selected_posture": torch.zeros(1, 8),
        "candidate_costs": {
            "total": torch.tensor([[0.5, -0.5]], dtype=torch.float32),
        },
        "rollout": {
            "trajectory": torch.randn(1, 2, 3, embed_dim),
        },
        "ctx_tokens": torch.randn(1, 4, embed_dim),
        "z": torch.randn(1, embed_dim),
        "mcts_traces": [
            {
                "selection_debug": {
                    "reason": "lowest_mean_cost",
                }
            }
        ],
    }

    injected = orchestrator._inject_imagined_replay_episode(
        domain_cfg=config.domains[0],
        adapter=adapter,
        episode_id=3,
        step_idx=2,
        observation=observation,
        info=info,
        tokenized=tokenized,
        domain_state=domain_state,
        outputs=outputs,
    )

    imagined_records = [
        record
        for record in orchestrator.replay.records
        if record.execution_source == "imagined_mcts"
    ]
    assert injected == len(imagined_records)
    assert injected == 3
    assert all(record.next_latent_z is None for record in imagined_records)
    assert all(record.action_logits_or_vec.shape[-1] == action_dim for record in imagined_records)
    assert all(record.promotion_source == "imagined_mcts" for record in imagined_records)


def test_connect4_simple_baseline_aliases_greedy() -> None:
    connect4 = Connect4Domain(embed_dim=16)
    observation, info = connect4.reset(seed=13)

    simple_action = connect4.baseline_action("simple", observation, info)
    greedy_action = connect4.baseline_action("greedy", observation, info)

    assert int(simple_action.reshape(-1)[0].item()) == int(greedy_action.reshape(-1)[0].item())


def test_cartpole_planner_diagnostics_flag_selected_branch_that_loses_to_baseline() -> None:
    cartpole = CartPoleDomain(embed_dim=16)
    domain_state = {
        "state_vector": torch.tensor([[0.0, 0.0, 0.15, 0.0]], dtype=torch.float32),
    }
    candidate_paths = torch.tensor(
        [
            [[6.0, -6.0], [6.0, -6.0], [6.0, -6.0]],
            [[-6.0, 6.0], [-6.0, 6.0], [-6.0, 6.0]],
        ],
        dtype=torch.float32,
    )
    diagnostics = cartpole.analyze_planner_candidates(
        candidate_paths=candidate_paths,
        candidate_ic=torch.tensor([0.1, 0.05], dtype=torch.float32),
        candidate_tc=torch.tensor([0.0, 0.0], dtype=torch.float32),
        candidate_total=torch.tensor([0.1, 0.05], dtype=torch.float32),
        candidate_terminal_latents=None,
        selected_candidate_idx=1,
        domain_state=domain_state,
        gamma=0.99,
        planner_trace={"selection_debug": {"reason": "lowest_mean_cost"}},
    )
    assert diagnostics is not None
    assert diagnostics["selected_candidate_idx"] == 1
    assert diagnostics["baseline_candidate_idx"] == 0
    assert diagnostics["best_actual_idx"] == 0
    assert diagnostics["selected_minus_baseline_actual_cost"] > 0.0
    assert diagnostics["selected_minus_baseline_predicted_ic"] < 0.0
    assert diagnostics["selected_minus_baseline_predicted_discounted_tc"] == 0.0
    assert diagnostics["harmful_pick_source"] == "ic_path_only_flip"


def test_cartpole_planner_diagnostics_identify_tc_tail_driven_misranking() -> None:
    cartpole = CartPoleDomain(embed_dim=16)
    domain_state = {
        "state_vector": torch.tensor([[0.0, 0.0, 0.15, 0.0]], dtype=torch.float32),
    }
    candidate_paths = torch.tensor(
        [
            [[6.0, -6.0], [6.0, -6.0], [6.0, -6.0]],
            [[-6.0, 6.0], [-6.0, 6.0], [-6.0, 6.0]],
        ],
        dtype=torch.float32,
    )
    gamma = 0.99
    tail_discount = gamma ** candidate_paths.shape[1]
    baseline_ic = 0.05
    selected_ic = 0.10
    baseline_tc = 0.50
    selected_tc = 0.0
    diagnostics = cartpole.analyze_planner_candidates(
        candidate_paths=candidate_paths,
        candidate_ic=torch.tensor([baseline_ic, selected_ic], dtype=torch.float32),
        candidate_tc=torch.tensor([baseline_tc, selected_tc], dtype=torch.float32),
        candidate_total=torch.tensor(
            [
                baseline_ic + (tail_discount * baseline_tc),
                selected_ic + (tail_discount * selected_tc),
            ],
            dtype=torch.float32,
        ),
        candidate_terminal_latents=None,
        selected_candidate_idx=1,
        domain_state=domain_state,
        gamma=gamma,
        planner_trace={"selection_debug": {"reason": "lowest_mean_cost"}},
    )
    assert diagnostics is not None
    assert diagnostics["selected_minus_baseline_actual_cost"] > 0.0
    assert diagnostics["selected_minus_baseline_predicted_ic"] > 0.0
    assert diagnostics["selected_minus_baseline_predicted_discounted_tc"] < 0.0
    assert diagnostics["predicted_advantage_source"] == "tc_tail_only"
    assert diagnostics["harmful_pick_source"] == "tc_tail_only_flip"


def test_cartpole_forward_exposes_planner_diagnostics(tmp_path: Path) -> None:
    config = _tiny_orchestrator_config(tmp_path / "planner_debug")
    orchestrator = UnifiedTrainingOrchestrator(config)
    adapter = orchestrator._build_adapter(config.domains[0])
    model = orchestrator._build_model(config.domains[0], adapter)
    observation, info = adapter.reset(seed=17)
    tokenized = adapter.tokenize_observation(observation)
    outputs = model(
        tokens=tokenized.tokens.to(orchestrator.device),
        mask=torch.zeros(
            tokenized.tokens.shape[0],
            tokenized.tokens.shape[1],
            dtype=torch.float32,
            device=orchestrator.device,
        ),
        domain_state=_move_nested_to_device(
            adapter.build_domain_state(observation, info),
            orchestrator.device,
        ),
        actor_mode="mode2",
        store_to_memory=False,
        input_kind="embedded_tokens",
    )
    planner_diagnostics = outputs["planner_diagnostics"]
    assert planner_diagnostics[0] is not None
    assert "selected_minus_baseline_actual_cost" in planner_diagnostics[0]


def test_cartpole_benchmark_config_loads_with_learning_first_defaults() -> None:
    config = load_orchestrator_config(
        "/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/configs/orchestrator_cartpole_benchmark.yaml"
    )
    assert config.router_top_k == 2
    assert config.surprise_replay_alpha > 0.0
    assert len(config.domains) == 1
    assert config.domains[0].name == "cartpole"
    assert config.domains[0].bootstrap_teacher_episodes == 49
    assert config.domains[0].evaluation_episodes == 100

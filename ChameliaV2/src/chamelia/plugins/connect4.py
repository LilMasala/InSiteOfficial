"""Interactive Connect4 domain adapter for unified Chamelia training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.chamelia.plugins.base import InteractiveDomainAdapter
from src.chamelia.tokenizers import BoardTokenizer

BOARD_ROWS = 6
BOARD_COLS = 7
CONNECT4_ACTIONS = BOARD_COLS


class _SimpleConnectFourEnv:
    """Self-contained 1-agent Connect4 env against a built-in opponent."""

    def __init__(self) -> None:
        self.board = torch.zeros(BOARD_ROWS, BOARD_COLS, dtype=torch.long)
        self.current_player = 1
        self.done = False

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = seed
        self.board.zero_()
        self.current_player = 1
        self.done = False
        return self.observation(), {}

    def observation(self) -> dict[str, Any]:
        return {
            "board": self.board.clone(),
            "current_player": self.current_player,
        }

    def legal_action_mask(self) -> torch.Tensor:
        return self.board[0] == 0

    def _drop_piece(self, column: int, player: int) -> bool:
        if column < 0 or column >= BOARD_COLS or self.board[0, column] != 0:
            return False
        for row in range(BOARD_ROWS - 1, -1, -1):
            if self.board[row, column] == 0:
                self.board[row, column] = player
                return True
        return False

    def _check_winner(self, player: int) -> bool:
        board = self.board
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS - 3):
                if torch.all(board[row, col : col + 4] == player):
                    return True
        for row in range(BOARD_ROWS - 3):
            for col in range(BOARD_COLS):
                if torch.all(board[row : row + 4, col] == player):
                    return True
        for row in range(BOARD_ROWS - 3):
            for col in range(BOARD_COLS - 3):
                if all(board[row + i, col + i].item() == player for i in range(4)):
                    return True
        for row in range(3, BOARD_ROWS):
            for col in range(BOARD_COLS - 3):
                if all(board[row - i, col + i].item() == player for i in range(4)):
                    return True
        return False

    def _choose_opponent_action(self) -> int:
        legal = self.legal_action_mask()
        for column in range(BOARD_COLS):
            if not legal[column]:
                continue
            clone = self.board.clone()
            self._drop_piece(column, 2)
            if self._check_winner(2):
                self.board = clone
                return column
            self.board = clone
        for column in range(BOARD_COLS):
            if legal[column]:
                return int(column)
        return 0

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.done:
            return self.observation(), 0.0, True, False, {"winner": 0}
        valid = self._drop_piece(int(action), 1)
        if not valid:
            self.done = True
            return self.observation(), -1.0, True, False, {"winner": 2, "invalid_action": True}
        if self._check_winner(1):
            self.done = True
            return self.observation(), 1.0, True, False, {"winner": 1}
        if not self.legal_action_mask().any():
            self.done = True
            return self.observation(), 0.0, True, False, {"winner": 0}

        opponent_action = self._choose_opponent_action()
        self._drop_piece(opponent_action, 2)
        if self._check_winner(2):
            self.done = True
            return self.observation(), -1.0, True, False, {"winner": 2, "opponent_action": opponent_action}
        if not self.legal_action_mask().any():
            self.done = True
            return self.observation(), 0.0, True, False, {"winner": 0, "opponent_action": opponent_action}
        return self.observation(), 0.0, False, False, {"opponent_action": opponent_action}


class Connect4Domain(InteractiveDomainAdapter):
    """Board-token Connect4 adapter with a built-in opponent."""

    modality_family = "board_token_hjepa"
    action_space_type = "discrete"

    def __init__(self, *, embed_dim: int = 128) -> None:
        self._tokenizer = BoardTokenizer(
            vocab_size=4,
            embed_dim=embed_dim,
            max_seq_len=BOARD_ROWS * BOARD_COLS + 1,
            domain_name="connect4",
            pad_token_id=0,
        )
        self.action_feature_decoder = nn.Sequential(
            nn.LazyLinear(128),
            nn.GELU(),
            nn.Linear(128, CONNECT4_ACTIONS * 3),
        )
        self._env = _SimpleConnectFourEnv()
        self._eval_opponent_depth = 0

    def get_tokenizer(self) -> BoardTokenizer:
        return self._tokenizer

    def get_action_dim(self) -> int:
        return CONNECT4_ACTIONS

    def get_trainable_modules(self) -> dict[str, nn.Module]:
        return {"action_feature_decoder": self.action_feature_decoder}

    def decode_action(self, action_vec: torch.Tensor) -> Any:
        if action_vec.dim() == 1:
            action_vec = action_vec.unsqueeze(0)
        return action_vec.argmax(dim=-1)

    def get_intrinsic_cost_fns(self) -> list[tuple[Any, float]]:
        def _feature_tensor(
            domain_state: dict[str, Any],
            key: str,
            action: torch.Tensor,
        ) -> torch.Tensor:
            value = domain_state.get(key)
            if value is None:
                return torch.zeros(action.shape[0], CONNECT4_ACTIONS, dtype=torch.float32, device=action.device)
            tensor = value.to(action.device).float()
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            return tensor

        def connect4_cost(_z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            legal_mask = domain_state.get("legal_actions_mask")
            if legal_mask is None:
                legal_mask = torch.ones(action.shape[0], CONNECT4_ACTIONS, dtype=torch.float32, device=action.device)
            else:
                legal_mask = legal_mask.to(action.device).float()
            chosen = action.argmax(dim=-1)
            legal_score = legal_mask.gather(1, chosen.unsqueeze(1)).squeeze(1).clamp(0.0, 1.0)
            center_penalty = (chosen.float() - 3.0).abs() / 3.0
            winning = _feature_tensor(domain_state, "winning_actions", action)
            blocking = _feature_tensor(domain_state, "blocking_actions", action)
            winning_bonus = winning.gather(1, chosen.unsqueeze(1)).squeeze(1).clamp(0.0, 1.0)
            blocking_bonus = blocking.gather(1, chosen.unsqueeze(1)).squeeze(1).clamp(0.0, 1.0)
            return center_penalty + ((1.0 - legal_score) * 5.0) - winning_bonus - (0.5 * blocking_bonus)

        return [(connect4_cost, 1.0)]

    def _flatten_board(self, board: torch.Tensor, current_player: int) -> torch.Tensor:
        flattened = board.reshape(-1).long() + 1
        player_token = torch.tensor([1 if current_player == 1 else 2], dtype=torch.long)
        return torch.cat([flattened, player_token], dim=0)

    def _simulate_drop(self, board: torch.Tensor, column: int, player: int) -> torch.Tensor | None:
        clone = board.clone()
        if clone[0, column] != 0:
            return None
        for row in range(BOARD_ROWS - 1, -1, -1):
            if clone[row, column] == 0:
                clone[row, column] = player
                return clone
        return None

    def _check_winner(self, board: torch.Tensor, player: int) -> bool:
        env = _SimpleConnectFourEnv()
        env.board = board.clone()
        return env._check_winner(player)

    def _action_features(self, board: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        legal = board[0] == 0
        winning = torch.zeros(CONNECT4_ACTIONS, dtype=torch.bool)
        blocking = torch.zeros(CONNECT4_ACTIONS, dtype=torch.bool)
        for column in range(CONNECT4_ACTIONS):
            if not legal[column]:
                continue
            own_board = self._simulate_drop(board, column, 1)
            opp_board = self._simulate_drop(board, column, 2)
            if own_board is not None and self._check_winner(own_board, 1):
                winning[column] = True
            if opp_board is not None and self._check_winner(opp_board, 2):
                blocking[column] = True
        return legal, winning, blocking

    def _center_score(self, column: int) -> float:
        return 3.0 - abs(float(column) - 3.0)

    def _board_heuristic(self, board: torch.Tensor) -> float:
        legal, winning, blocking = self._action_features(board)
        return (
            float(winning.float().sum().item()) * 3.0
            + float(blocking.float().sum().item()) * 1.5
            + sum(self._center_score(column) for column in range(CONNECT4_ACTIONS) if bool(legal[column].item())) * 0.05
        )

    def _minimax(
        self,
        board: torch.Tensor,
        depth: int,
        maximizing_player: bool,
        alpha: float,
        beta: float,
    ) -> tuple[float, int]:
        legal, _, _ = self._action_features(board)
        legal_columns = [column for column in range(CONNECT4_ACTIONS) if bool(legal[column].item())]
        if self._check_winner(board, 2):
            return 1000.0 + depth, 3
        if self._check_winner(board, 1):
            return -1000.0 - depth, 3
        if depth <= 0 or not legal_columns:
            return self._board_heuristic(board), 3

        ordered_columns = sorted(legal_columns, key=lambda column: abs(column - 3))
        if maximizing_player:
            best_score, best_column = float("-inf"), ordered_columns[0]
            for column in ordered_columns:
                child = self._simulate_drop(board, column, 2)
                if child is None:
                    continue
                score, _ = self._minimax(child, depth - 1, False, alpha, beta)
                score += 0.01 * self._center_score(column)
                if score > best_score:
                    best_score, best_column = score, column
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score, best_column
        best_score, best_column = float("inf"), ordered_columns[0]
        for column in ordered_columns:
            child = self._simulate_drop(board, column, 1)
            if child is None:
                continue
            score, _ = self._minimax(child, depth - 1, True, alpha, beta)
            score -= 0.01 * self._center_score(column)
            if score < best_score:
                best_score, best_column = score, column
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score, best_column

    def set_eval_opponent_depth(self, depth: int | None) -> None:
        self._eval_opponent_depth = max(0, int(depth or 0))

    def _choose_eval_opponent_action(self) -> int:
        _score, column = self._minimax(
            self._env.board.clone(),
            depth=max(1, self._eval_opponent_depth),
            maximizing_player=True,
            alpha=float("-inf"),
            beta=float("inf"),
        )
        return int(column)

    def _batched_action_features(
        self,
        board_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if board_batch.dim() == 2:
            board_batch = board_batch.unsqueeze(0)
        legal_list: list[torch.Tensor] = []
        winning_list: list[torch.Tensor] = []
        blocking_list: list[torch.Tensor] = []
        for board in board_batch:
            legal, winning, blocking = self._action_features(board.long())
            legal_list.append(legal.float())
            winning_list.append(winning.float())
            blocking_list.append(blocking.float())
        return (
            torch.stack(legal_list, dim=0),
            torch.stack(winning_list, dim=0),
            torch.stack(blocking_list, dim=0),
        )

    def _decode_action_features(self, future_z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if future_z.dim() == 1:
            future_z = future_z.unsqueeze(0)
        logits = self.action_feature_decoder(future_z.float()).view(-1, 3, CONNECT4_ACTIONS)
        return logits[:, 0, :], logits[:, 1, :], logits[:, 2, :]

    def build_domain_state(self, observation: Any, info: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(observation, dict):
            board = observation["board"].long()
            current_player = int(observation.get("current_player", 1))
        else:
            raise ValueError("Connect4 observation must be a dict containing 'board'.")
        if board.dim() == 2:
            board = board.unsqueeze(0)
        legal, winning, blocking = self._batched_action_features(board)
        state = {
            "board": board,
            "current_player": torch.full((board.shape[0],), current_player, dtype=torch.long),
            "legal_actions_mask": legal,
            "winning_actions": winning,
            "blocking_actions": blocking,
        }
        if info:
            state["info"] = dict(info)
        return state

    def get_domain_state(self, observation: Any) -> dict:
        return self.build_domain_state(observation, None)

    def prepare_bridge_observation(self, observation: Any) -> torch.Tensor:
        if not isinstance(observation, dict):
            raise ValueError("Connect4 observation must be a dict.")
        board = observation["board"].long()
        current_player = int(observation.get("current_player", 1))
        return self._flatten_board(board, current_player)

    def compute_regime_embedding(self, domain_state: dict) -> torch.Tensor | None:
        board = domain_state["board"].float()
        if board.dim() == 3:
            return board.reshape(board.shape[0], -1)
        return board.reshape(-1)

    def build_simple_baseline_path(
        self,
        domain_state: dict[str, Any],
        path_length: int,
        action_dim: int,
    ) -> torch.Tensor | None:
        legal = domain_state.get("legal_actions_mask")
        winning = domain_state.get("winning_actions")
        blocking = domain_state.get("blocking_actions")
        if legal is None:
            return None
        legal = legal.float()
        if legal.dim() == 1:
            legal = legal.unsqueeze(0)
        if winning is None:
            winning = torch.zeros_like(legal)
        else:
            winning = winning.float()
        if blocking is None:
            blocking = torch.zeros_like(legal)
        else:
            blocking = blocking.float()
        center_order = [3, 2, 4, 1, 5, 0, 6]
        chosen: list[int] = []
        for batch_idx in range(legal.shape[0]):
            selected = None
            for tensor in (winning, blocking):
                indices = torch.nonzero(tensor[batch_idx] > 0.5, as_tuple=False).squeeze(-1)
                if indices.numel() > 0:
                    selected = int(indices[0].item())
                    break
            if selected is None:
                for column in center_order:
                    if column < legal.shape[1] and float(legal[batch_idx, column].item()) > 0.5:
                        selected = column
                        break
            chosen.append(0 if selected is None else int(selected))
        logits = torch.full(
            (legal.shape[0], path_length, action_dim),
            fill_value=-6.0,
            dtype=legal.dtype,
            device=legal.device,
        )
        logits.scatter_(
            dim=-1,
            index=torch.tensor(chosen, device=legal.device).view(-1, 1, 1).expand(-1, path_length, 1),
            src=torch.full(
                (legal.shape[0], path_length, 1),
                fill_value=6.0,
                dtype=legal.dtype,
                device=legal.device,
            ),
        )
        return logits

    def build_imagined_domain_state(
        self,
        current_domain_state: dict[str, Any],
        future_z: torch.Tensor,
        step_idx: int,
    ) -> dict[str, Any]:
        _ = step_idx
        legal_logits, winning_logits, blocking_logits = self._decode_action_features(future_z)
        imagined = dict(current_domain_state)
        imagined["legal_actions_mask"] = torch.sigmoid(legal_logits)
        imagined["winning_actions"] = torch.sigmoid(winning_logits)
        imagined["blocking_actions"] = torch.sigmoid(blocking_logits)
        board = current_domain_state.get("board")
        if torch.is_tensor(board):
            if board.dim() == 2:
                board = board.unsqueeze(0)
            if board.shape[0] == future_z.shape[0]:
                imagined["board"] = board
            else:
                imagined["board"] = board[:1].expand(future_z.shape[0], -1, -1)
        current_player = current_domain_state.get("current_player")
        if torch.is_tensor(current_player):
            if current_player.dim() == 0:
                current_player = current_player.unsqueeze(0)
            if current_player.shape[0] == future_z.shape[0]:
                imagined["current_player"] = current_player
            else:
                imagined["current_player"] = current_player[:1].expand(future_z.shape[0])
        return imagined

    def compute_latent_state_decoder_loss(
        self,
        predicted_future_z: torch.Tensor,
        target_domain_state: dict[str, Any],
    ) -> torch.Tensor | None:
        legal_logits, winning_logits, blocking_logits = self._decode_action_features(predicted_future_z)
        legal = target_domain_state.get("legal_actions_mask")
        winning = target_domain_state.get("winning_actions")
        blocking = target_domain_state.get("blocking_actions")
        if legal is None or winning is None or blocking is None:
            return None
        legal_target = legal.float().to(predicted_future_z.device)
        winning_target = winning.float().to(predicted_future_z.device)
        blocking_target = blocking.float().to(predicted_future_z.device)
        if legal_target.dim() == 1:
            legal_target = legal_target.unsqueeze(0)
        if winning_target.dim() == 1:
            winning_target = winning_target.unsqueeze(0)
        if blocking_target.dim() == 1:
            blocking_target = blocking_target.unsqueeze(0)
        return (
            F.binary_cross_entropy_with_logits(legal_logits, legal_target.detach())
            + F.binary_cross_entropy_with_logits(winning_logits, winning_target.detach())
            + F.binary_cross_entropy_with_logits(blocking_logits, blocking_target.detach())
        ) / 3.0

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        return self._env.reset(seed=seed)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        chosen = int(action.item()) if torch.is_tensor(action) else int(action)
        if self._eval_opponent_depth > 0:
            if self._env.done:
                return self._env.observation(), 0.0, True, False, {"winner": 0}
            valid = self._env._drop_piece(chosen, 1)
            if not valid:
                self._env.done = True
                return self._env.observation(), -1.0, True, False, {"winner": 2, "invalid_action": True}
            if self._env._check_winner(1):
                self._env.done = True
                return self._env.observation(), 1.0, True, False, {"winner": 1}
            if not self._env.legal_action_mask().any():
                self._env.done = True
                return self._env.observation(), 0.0, True, False, {"winner": 0}
            opponent_action = self._choose_eval_opponent_action()
            self._env._drop_piece(opponent_action, 2)
            if self._env._check_winner(2):
                self._env.done = True
                return self._env.observation(), -1.0, True, False, {"winner": 2, "opponent_action": opponent_action}
            if not self._env.legal_action_mask().any():
                self._env.done = True
                return self._env.observation(), 0.0, True, False, {"winner": 0, "opponent_action": opponent_action}
            return self._env.observation(), 0.0, False, False, {"opponent_action": opponent_action}
        return self._env.step(chosen)

    def legal_action_mask(self, observation: Any, info: dict[str, Any] | None = None) -> torch.Tensor | None:
        _ = info
        legal = self.build_domain_state(observation).get("legal_actions_mask", None)
        if legal is None:
            return None
        return legal.reshape(-1).bool()

    def compute_realized_cost(
        self,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> float:
        _ = truncated
        winner = 0 if info is None else int(info.get("winner", 0))
        if terminated and winner == 1:
            return 0.0
        if terminated and winner == 2:
            return 2.0
        if terminated:
            return 1.0
        return max(0.0, 0.25 - float(reward))

    def compute_metrics(self, episode_records: list[dict[str, Any]]) -> dict[str, float]:
        if not episode_records:
            return {"episode_reward_mean": 0.0, "win_rate": 0.0}
        rewards = [float(record.get("episode_reward", 0.0)) for record in episode_records]
        wins = [1.0 if int(record.get("winner", 0)) == 1 else 0.0 for record in episode_records]
        return {
            "episode_reward_mean": sum(rewards) / len(rewards),
            "win_rate": sum(wins) / len(wins),
        }

    def baseline_action(
        self,
        kind: str,
        observation: Any,
        info: dict[str, Any] | None = None,
    ) -> Any:
        if kind in {"greedy", "simple"}:
            domain_state = self.build_domain_state(observation, info)
            for tensor_key in ("winning_actions", "blocking_actions"):
                tensor = domain_state[tensor_key].reshape(-1)
                indices = torch.nonzero(tensor > 0.5, as_tuple=False).squeeze(-1)
                if indices.numel() > 0:
                    return indices[0].unsqueeze(0)
            legal = domain_state["legal_actions_mask"].reshape(-1)
            center_order = [3, 2, 4, 1, 5, 0, 6]
            for column in center_order:
                if float(legal[column].item()) > 0.5:
                    return torch.tensor([column], dtype=torch.long)
        return super().baseline_action(kind, observation, info)

    def compute_goal_latent(
        self,
        domain_state: dict[str, Any],
        z: torch.Tensor,
    ) -> torch.Tensor | None:
        _ = domain_state
        _ = z
        return None

    @property
    def domain_name(self) -> str:
        return "connect4"

    @property
    def vocab_size(self) -> int:
        return 4

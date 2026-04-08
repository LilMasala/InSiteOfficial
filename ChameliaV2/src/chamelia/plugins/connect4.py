"""Interactive Connect4 domain adapter for unified Chamelia training."""

from __future__ import annotations

from typing import Any

import torch

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
        self._env = _SimpleConnectFourEnv()

    def get_tokenizer(self) -> BoardTokenizer:
        return self._tokenizer

    def get_action_dim(self) -> int:
        return CONNECT4_ACTIONS

    def decode_action(self, action_vec: torch.Tensor) -> Any:
        if action_vec.dim() == 1:
            action_vec = action_vec.unsqueeze(0)
        return action_vec.argmax(dim=-1)

    def get_intrinsic_cost_fns(self) -> list[tuple[Any, float]]:
        def connect4_cost(_z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            legal_mask = domain_state.get("legal_actions_mask")
            if legal_mask is None:
                legal_mask = torch.ones(action.shape[0], CONNECT4_ACTIONS, dtype=torch.bool, device=action.device)
            else:
                legal_mask = legal_mask.to(action.device)
            chosen = action.argmax(dim=-1)
            invalid = ~legal_mask.gather(1, chosen.unsqueeze(1)).squeeze(1)
            center_penalty = (chosen.float() - 3.0).abs() / 3.0
            winning = domain_state.get("winning_actions")
            if winning is None:
                winning = torch.zeros(action.shape[0], CONNECT4_ACTIONS, dtype=torch.bool, device=action.device)
            else:
                winning = winning.to(action.device)
            blocking = domain_state.get("blocking_actions")
            if blocking is None:
                blocking = torch.zeros(action.shape[0], CONNECT4_ACTIONS, dtype=torch.bool, device=action.device)
            else:
                blocking = blocking.to(action.device)
            bonus = torch.where(
                winning.gather(1, chosen.unsqueeze(1)).squeeze(1),
                torch.full_like(center_penalty, -1.0),
                torch.zeros_like(center_penalty),
            )
            bonus = bonus + torch.where(
                blocking.gather(1, chosen.unsqueeze(1)).squeeze(1),
                torch.full_like(center_penalty, -0.5),
                torch.zeros_like(center_penalty),
            )
            return center_penalty + invalid.float() * 5.0 + bonus

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

    def build_domain_state(self, observation: Any, info: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(observation, dict):
            board = observation["board"].long()
            current_player = int(observation.get("current_player", 1))
        else:
            raise ValueError("Connect4 observation must be a dict containing 'board'.")
        legal, winning, blocking = self._action_features(board)
        state = {
            "board": board,
            "current_player": current_player,
            "legal_actions_mask": legal.unsqueeze(0),
            "winning_actions": winning.unsqueeze(0),
            "blocking_actions": blocking.unsqueeze(0),
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
        return board.reshape(-1)

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        return self._env.reset(seed=seed)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        chosen = int(action.item()) if torch.is_tensor(action) else int(action)
        return self._env.step(chosen)

    def legal_action_mask(self, observation: Any, info: dict[str, Any] | None = None) -> torch.Tensor | None:
        _ = info
        return self.build_domain_state(observation).get("legal_actions_mask", None)

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
        if kind == "greedy":
            domain_state = self.build_domain_state(observation, info)
            for tensor_key in ("winning_actions", "blocking_actions"):
                tensor = domain_state[tensor_key].reshape(-1)
                indices = torch.nonzero(tensor, as_tuple=False).squeeze(-1)
                if indices.numel() > 0:
                    return indices[0].unsqueeze(0)
            legal = domain_state["legal_actions_mask"].reshape(-1)
            center_order = [3, 2, 4, 1, 5, 0, 6]
            for column in center_order:
                if bool(legal[column].item()):
                    return torch.tensor([column], dtype=torch.long)
        return super().baseline_action(kind, observation, info)

    @property
    def domain_name(self) -> str:
        return "connect4"

    @property
    def vocab_size(self) -> int:
        return 4

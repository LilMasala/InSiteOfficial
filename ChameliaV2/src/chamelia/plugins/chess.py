"""Interactive chess domain adapter for unified Chamelia training."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import chess
import torch
import torch.nn as nn

from src.chamelia.plugins.base import InteractiveDomainAdapter
from src.chamelia.tokenizers import ChessTokenizer

CHESS_ACTION_DIM = 8 * 8 * 73
CHESS_OBSERVATION_PLANES = 111
_PLANE_STRIDE = 73
_QUEEN_MOVE_PLANES = 56
_KNIGHT_MOVE_OFFSET = 56
_UNDERPROMOTION_OFFSET = 64
_HISTORY_PLANES = 104
_CURRENT_BOARD_PLANES = 13
_AUX_PLANES = 7
_PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.0,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
}
_QUEEN_DIRECTIONS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
_KNIGHT_DIRECTIONS = [
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
]
_UNDERPROMOTION_PIECES = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def _square_to_coord(square: int) -> tuple[int, int]:
    return chess.square_file(square), chess.square_rank(square)


def _coord_to_square(col: int, row: int) -> int:
    return chess.square(col, row)


def _coord_to_square_safe(col: int, row: int) -> int | None:
    if 0 <= col < 8 and 0 <= row < 8:
        return chess.square(col, row)
    return None


def _mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def _delta(move: chess.Move) -> tuple[int, int]:
    src_col, src_row = _square_to_coord(move.from_square)
    dst_col, dst_row = _square_to_coord(move.to_square)
    return dst_col - src_col, dst_row - src_row


def _queen_plane_for_delta(dx: int, dy: int) -> int:
    magnitude = max(abs(dx), abs(dy))
    if magnitude < 1 or magnitude > 7:
        raise ValueError(f"Unsupported queen-like delta {(dx, dy)}.")
    direction = (0 if dx == 0 else int(dx / abs(dx)), 0 if dy == 0 else int(dy / abs(dy)))
    return (magnitude - 1) * len(_QUEEN_DIRECTIONS) + _QUEEN_DIRECTIONS.index(direction)


def _knight_plane_for_delta(dx: int, dy: int) -> int:
    return _KNIGHT_MOVE_OFFSET + _KNIGHT_DIRECTIONS.index((dx, dy))


def _underpromotion_plane(move: chess.Move) -> int:
    dx, dy = _delta(move)
    if dy != 1 or dx < -1 or dx > 1:
        raise ValueError(f"Unsupported underpromotion move {move.uci()}.")
    direction_index = dx + 1
    piece_index = _UNDERPROMOTION_PIECES.index(int(move.promotion))
    return _UNDERPROMOTION_OFFSET + (direction_index * len(_UNDERPROMOTION_PIECES)) + piece_index


def _move_plane(move: chess.Move) -> int:
    dx, dy = _delta(move)
    if (abs(dx), abs(dy)) in {(1, 2), (2, 1)}:
        return _knight_plane_for_delta(dx, dy)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        return _underpromotion_plane(move)
    return _queen_plane_for_delta(dx, dy)


def _action_from_move(move: chess.Move) -> int:
    col, row = _square_to_coord(move.from_square)
    return (col * 8 + row) * _PLANE_STRIDE + _move_plane(move)


def _action_to_move(board: chess.Board, action: int) -> chess.Move:
    if int(action) < 0 or int(action) >= CHESS_ACTION_DIM:
        return chess.Move.null()
    base_board = board if board.turn == chess.WHITE else board.mirror()
    col = int(action) // (8 * _PLANE_STRIDE)
    row = (int(action) // _PLANE_STRIDE) % 8
    plane = int(action) % _PLANE_STRIDE
    from_square = _coord_to_square_safe(col, row)
    if from_square is None:
        return chess.Move.null()

    if plane < _QUEEN_MOVE_PLANES:
        magnitude = plane // len(_QUEEN_DIRECTIONS) + 1
        direction = _QUEEN_DIRECTIONS[plane % len(_QUEEN_DIRECTIONS)]
        to_square = _coord_to_square_safe(col + (direction[0] * magnitude), row + (direction[1] * magnitude))
        if to_square is None:
            return chess.Move.null()
        move = chess.Move(from_square, to_square)
        piece = base_board.piece_at(from_square)
        if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(to_square) == 7:
            move.promotion = chess.QUEEN
    elif plane < _UNDERPROMOTION_OFFSET:
        knight_dir = _KNIGHT_DIRECTIONS[plane - _KNIGHT_MOVE_OFFSET]
        to_square = _coord_to_square_safe(col + knight_dir[0], row + knight_dir[1])
        if to_square is None:
            return chess.Move.null()
        move = chess.Move(from_square, to_square)
    else:
        under_plane = plane - _UNDERPROMOTION_OFFSET
        direction_index = under_plane // len(_UNDERPROMOTION_PIECES)
        piece_index = under_plane % len(_UNDERPROMOTION_PIECES)
        dx = direction_index - 1
        to_square = _coord_to_square_safe(col + dx, row + 1)
        if to_square is None:
            return chess.Move.null()
        move = chess.Move(from_square, to_square, promotion=_UNDERPROMOTION_PIECES[piece_index])

    return move if board.turn == chess.WHITE else _mirror_move(move)


def _legal_actions(board: chess.Board) -> list[int]:
    working_board = board if board.turn == chess.WHITE else board.mirror()
    actions: list[int] = []
    for move in working_board.legal_moves:
        actions.append(_action_from_move(move))
    return actions


def _material_value(board: chess.Board, color: chess.Color) -> float:
    total = 0.0
    for piece_type, value in _PIECE_VALUES.items():
        total += value * len(board.pieces(piece_type, color))
    return total


def _phase_index(board: chess.Board) -> int:
    major_minor = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        major_minor += len(board.pieces(piece_type, chess.WHITE))
        major_minor += len(board.pieces(piece_type, chess.BLACK))
    if len(board.move_stack) < 16:
        return 0
    if major_minor <= 6:
        return 2
    return 1


def _phase_one_hot(index: int) -> torch.Tensor:
    result = torch.zeros(3, dtype=torch.float32)
    result[int(index)] = 1.0
    return result


def _board_planes(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros(8, 8, _CURRENT_BOARD_PLANES, dtype=torch.float32)
    channel_map = {
        (chess.WHITE, chess.PAWN): 0,
        (chess.WHITE, chess.KNIGHT): 1,
        (chess.WHITE, chess.BISHOP): 2,
        (chess.WHITE, chess.ROOK): 3,
        (chess.WHITE, chess.QUEEN): 4,
        (chess.WHITE, chess.KING): 5,
        (chess.BLACK, chess.PAWN): 6,
        (chess.BLACK, chess.KNIGHT): 7,
        (chess.BLACK, chess.BISHOP): 8,
        (chess.BLACK, chess.ROOK): 9,
        (chess.BLACK, chess.QUEEN): 10,
        (chess.BLACK, chess.KING): 11,
    }
    for square, piece in board.piece_map().items():
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        planes[row, col, channel_map[(piece.color, piece.piece_type)]] = 1.0
    if board.is_repetition(2):
        planes[:, :, 12] = 1.0

    if board.ep_square is not None:
        ep_square = int(board.ep_square)
        file_idx = chess.square_file(ep_square)
        if ep_square < 32:
            origin_square = ep_square + 8
            planes[7 - chess.square_rank(origin_square), file_idx, 0] = 0.0
            planes[7, file_idx, 0] = 1.0
        else:
            origin_square = ep_square - 8
            planes[7 - chess.square_rank(origin_square), file_idx, 6] = 0.0
            planes[0, file_idx, 6] = 1.0
    return planes


def _aux_planes(board: chess.Board) -> torch.Tensor:
    aux = torch.zeros(8, 8, _AUX_PLANES, dtype=torch.float32)
    if board.has_queenside_castling_rights(chess.WHITE):
        aux[:, :, 0] = 1.0
    if board.has_kingside_castling_rights(chess.WHITE):
        aux[:, :, 1] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        aux[:, :, 2] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        aux[:, :, 3] = 1.0
    if board.turn == chess.BLACK:
        aux[:, :, 4] = 1.0
    move_clock_index = min(63, int(board.halfmove_clock // 2))
    aux[move_clock_index // 8, move_clock_index % 8, 5] = 1.0
    aux[:, :, 6] = 1.0
    return aux


def _push_history(current_planes: torch.Tensor, history_block: torch.Tensor | None) -> torch.Tensor:
    flat_current = current_planes.reshape(8, 8, _CURRENT_BOARD_PLANES)
    if history_block is None:
        history_block = torch.zeros(8, 8, _HISTORY_PLANES, dtype=torch.float32)
    return torch.cat([flat_current, history_block[:, :, : _HISTORY_PLANES - _CURRENT_BOARD_PLANES]], dim=-1)


def _observation_from_board(board: chess.Board, history_block: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    current = _board_planes(board)
    updated_history = _push_history(current, history_block)
    return torch.cat([_aux_planes(board), updated_history], dim=-1), updated_history


def _decode_phase_label(index: int) -> str:
    return ("opening", "middlegame", "endgame")[int(index)]


def _piece_value_from_move(board: chess.Board, move: chess.Move) -> float:
    if board.is_en_passant(move):
        return _PIECE_VALUES[chess.PAWN]
    captured = board.piece_at(move.to_square)
    if captured is None:
        return 0.0
    return _PIECE_VALUES.get(captured.piece_type, 0.0)


def _hanging_value(board: chess.Board, color: chess.Color) -> float:
    total = 0.0
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        attacked = board.is_attacked_by(not color, square)
        defended = board.is_attacked_by(color, square)
        if attacked and not defended:
            total += _PIECE_VALUES.get(piece.piece_type, 0.0)
    return total


def _normalize_fen_batch(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        normalized: list[str] = []
        for item in value:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, Iterable) and not isinstance(item, (bytes, bytearray)):
                item_list = list(item)
                if len(item_list) != 1:
                    raise ValueError("Nested chess FEN batches must contain exactly one FEN per entry.")
                normalized.append(str(item_list[0]))
            else:
                normalized.append(str(item))
        return normalized
    raise ValueError("Chess FEN payload must be a string or iterable of strings.")


class ChessDomain(InteractiveDomainAdapter):
    """Single-agent chess adapter against a built-in opponent."""

    modality_family = "chess_plane_hjepa"
    action_space_type = "discrete"

    def __init__(
        self,
        *,
        embed_dim: int = 128,
        opponent_level: int = 1,
        opponent_max_branching: int = 12,
        self_play_unlock_threshold: float = 0.90,
        chess_data_root: str | None = None,
        stockfish_path: str | None = None,
        move_vocabulary_source: str = "alphazero_4672",
    ) -> None:
        self._tokenizer = ChessTokenizer(embed_dim=embed_dim, num_planes=CHESS_OBSERVATION_PLANES, domain_name="chess")
        self._eval_opponent_depth: int | None = None
        self._opponent_level = max(0, int(opponent_level))
        self._opponent_max_branching = max(4, int(opponent_max_branching))
        self.self_play_unlock_threshold = float(self_play_unlock_threshold)
        self.chess_data_root = chess_data_root
        self.stockfish_path = stockfish_path
        self.move_vocabulary_source = move_vocabulary_source
        self.self_play_enabled = False
        self._board = chess.Board()
        self._history_block = torch.zeros(8, 8, _HISTORY_PLANES, dtype=torch.float32)
        self._done = False
        self._last_terminal_info: dict[str, Any] = {}

    def get_tokenizer(self) -> ChessTokenizer:
        return self._tokenizer

    def get_action_dim(self) -> int:
        return CHESS_ACTION_DIM

    def get_intrinsic_cost_fns(self) -> list[tuple[Any, float]]:
        def chess_cost(_z: torch.Tensor, _action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            invalid = domain_state.get("realized_invalid")
            material_loss = domain_state.get("realized_material_loss")
            hanging = domain_state.get("realized_hanging_penalty")
            terminal_loss = domain_state.get("realized_terminal_loss")
            early_loss = domain_state.get("realized_early_loss")
            draw_penalty = domain_state.get("realized_draw_penalty")
            check_bonus = domain_state.get("realized_check_bonus")
            capture_bonus = domain_state.get("realized_capture_gain")
            mate_win_bonus = domain_state.get("realized_mate_win_bonus")
            tensors = [invalid, material_loss, hanging, terminal_loss, early_loss, draw_penalty]
            reference = next((tensor for tensor in tensors if torch.is_tensor(tensor)), None)
            if reference is None:
                raise ValueError("Chess domain-state is missing realized rollout features.")
            total = (
                invalid.float()
                + material_loss.float()
                + hanging.float()
                + terminal_loss.float()
                + early_loss.float()
                + draw_penalty.float()
                - check_bonus.float()
                - capture_bonus.float()
                - mate_win_bonus.float()
            )
            return total.to(reference.device)

        return [(chess_cost, 1.0)]

    def get_trainable_modules(self) -> dict[str, nn.Module]:
        return {}

    def decode_action(self, action_vec: torch.Tensor) -> Any:
        if action_vec.dim() == 1:
            action_vec = action_vec.unsqueeze(0)
        return action_vec.argmax(dim=-1)

    def get_domain_state(self, observation: Any) -> dict:
        return self.build_domain_state(observation, None)

    def prepare_bridge_observation(self, observation: Any) -> Any:
        if not isinstance(observation, dict) or "observation" not in observation:
            raise ValueError("Chess observation must be a dict containing 'observation'.")
        return observation["observation"]

    def compute_regime_embedding(self, domain_state: dict) -> torch.Tensor | None:
        material = domain_state.get("material_summary")
        if material is None:
            return None
        return material.float()

    def _ordered_moves(self, board: chess.Board, *, maximizing_white: bool) -> list[chess.Move]:
        scored: list[tuple[float, chess.Move]] = []
        for move in board.legal_moves:
            score = _piece_value_from_move(board, move)
            if board.gives_check(move):
                score += 0.5
            if move.promotion is not None:
                score += 0.75
            scored.append((score, move))
        scored.sort(key=lambda item: item[0], reverse=True)
        moves = [move for _, move in scored]
        if len(moves) > self._opponent_max_branching:
            moves = moves[: self._opponent_max_branching]
        if not maximizing_white:
            return moves
        return moves

    def _evaluate_board(self, board: chess.Board) -> float:
        if board.is_checkmate():
            return -10000.0 if board.turn == chess.WHITE else 10000.0
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0.0
        white_material = _material_value(board, chess.WHITE)
        black_material = _material_value(board, chess.BLACK)
        white_mobility = len(list(board.legal_moves)) if board.turn == chess.WHITE else 0
        board_turn = board.turn
        board.turn = not board.turn
        black_mobility = len(list(board.legal_moves))
        board.turn = board_turn
        return (
            white_material
            - black_material
            + (0.03 * (white_mobility - black_mobility))
            - (0.15 * _hanging_value(board, chess.WHITE))
            + (0.15 * _hanging_value(board, chess.BLACK))
        )

    def _search(self, board: chess.Board, depth: int, alpha: float, beta: float, maximizing_white: bool) -> float:
        if depth <= 0 or board.is_game_over(claim_draw=True):
            return self._evaluate_board(board)
        ordered = self._ordered_moves(board, maximizing_white=maximizing_white)
        if maximizing_white:
            value = float("-inf")
            for move in ordered:
                board.push(move)
                value = max(value, self._search(board, depth - 1, alpha, beta, False))
                board.pop()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        value = float("inf")
        for move in ordered:
            board.push(move)
            value = min(value, self._search(board, depth - 1, alpha, beta, True))
            board.pop()
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _best_move(self, board: chess.Board, *, maximizing_white: bool, depth: int) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        if depth <= 0:
            return legal_moves[0]
        best_value = float("-inf") if maximizing_white else float("inf")
        best_move = legal_moves[0]
        for move in self._ordered_moves(board, maximizing_white=maximizing_white):
            board.push(move)
            value = self._search(board, depth - 1, float("-inf"), float("inf"), not maximizing_white)
            board.pop()
            if maximizing_white and value > best_value:
                best_value = value
                best_move = move
            if not maximizing_white and value < best_value:
                best_value = value
                best_move = move
        return best_move

    def _choose_opponent_move(self, board: chess.Board) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        depth = self._eval_opponent_depth if self._eval_opponent_depth is not None else self._opponent_level
        if depth <= 0:
            return legal_moves[0]
        return self._best_move(board, maximizing_white=False, depth=depth)

    def _choose_simple_move(self, board: chess.Board) -> chess.Move | None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        for move in legal_moves:
            if board.is_capture(move):
                board.push(move)
                if board.is_checkmate():
                    board.pop()
                    return move
                board.pop()
        return self._best_move(board, maximizing_white=True, depth=max(1, self._opponent_level))

    def _empty_rollout_features(self, batch_size: int) -> dict[str, torch.Tensor]:
        zeros = torch.zeros(batch_size, dtype=torch.float32)
        return {
            "realized_invalid": zeros.clone(),
            "realized_material_loss": zeros.clone(),
            "realized_hanging_penalty": zeros.clone(),
            "realized_terminal_loss": zeros.clone(),
            "realized_early_loss": zeros.clone(),
            "realized_draw_penalty": zeros.clone(),
            "realized_check_bonus": zeros.clone(),
            "realized_capture_gain": zeros.clone(),
            "realized_mate_win_bonus": zeros.clone(),
        }

    def _build_state_from_boards(
        self,
        boards: list[chess.Board],
        history_blocks: list[torch.Tensor],
        *,
        rollout_features: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        observations: list[torch.Tensor] = []
        updated_history: list[torch.Tensor] = []
        legal_masks: list[torch.Tensor] = []
        material_summary: list[torch.Tensor] = []
        side_to_move: list[float] = []
        in_check: list[float] = []
        is_checkmate: list[float] = []
        is_stalemate: list[float] = []
        is_draw: list[float] = []
        move_count: list[float] = []
        halfmove_clock: list[float] = []
        phase: list[torch.Tensor] = []
        fens: list[str] = []
        for board, history in zip(boards, history_blocks, strict=False):
            observation, next_history = _observation_from_board(board, history)
            mask = torch.zeros(CHESS_ACTION_DIM, dtype=torch.bool)
            for action in _legal_actions(board):
                mask[action] = True
            observations.append(observation)
            updated_history.append(next_history)
            legal_masks.append(mask)
            material_summary.append(
                torch.tensor(
                    [
                        _material_value(board, chess.WHITE),
                        _material_value(board, chess.BLACK),
                        _material_value(board, chess.WHITE) - _material_value(board, chess.BLACK),
                    ],
                    dtype=torch.float32,
                )
            )
            side_to_move.append(1.0 if board.turn == chess.WHITE else 0.0)
            in_check.append(float(board.is_check()))
            is_checkmate.append(float(board.is_checkmate()))
            is_stalemate.append(float(board.is_stalemate()))
            is_draw.append(float(board.is_insufficient_material() or board.can_claim_draw()))
            move_count.append(float(len(board.move_stack)))
            halfmove_clock.append(float(board.halfmove_clock))
            phase_idx = _phase_index(board)
            phase.append(_phase_one_hot(phase_idx))
            fens.append(board.fen())
        state = {
            "board_observation": torch.stack(observations, dim=0),
            "history_block": torch.stack(updated_history, dim=0),
            "legal_actions_mask": torch.stack(legal_masks, dim=0),
            "material_summary": torch.stack(material_summary, dim=0),
            "side_to_move": torch.tensor(side_to_move, dtype=torch.float32),
            "in_check": torch.tensor(in_check, dtype=torch.float32),
            "is_checkmate": torch.tensor(is_checkmate, dtype=torch.float32),
            "is_stalemate": torch.tensor(is_stalemate, dtype=torch.float32),
            "is_draw": torch.tensor(is_draw, dtype=torch.float32),
            "move_count": torch.tensor(move_count, dtype=torch.float32),
            "halfmove_clock": torch.tensor(halfmove_clock, dtype=torch.float32),
            "phase": torch.stack(phase, dim=0),
            "fen": fens,
            "phase_label": [_decode_phase_label(int(item.argmax().item())) for item in phase],
        }
        features = rollout_features or self._empty_rollout_features(len(boards))
        state.update(features)
        return state

    def build_domain_state(self, observation: Any, info: dict[str, Any] | None = None) -> dict[str, Any]:
        if isinstance(observation, dict) and "fen" in observation:
            fen_value = observation["fen"]
            history_value = observation.get("history_block")
            if isinstance(fen_value, str):
                boards = [chess.Board(fen_value)]
            elif isinstance(fen_value, Iterable):
                boards = [chess.Board(str(item)) for item in fen_value]
            else:
                raise ValueError("Chess observation 'fen' must be a string or iterable of strings.")
            if torch.is_tensor(history_value):
                if history_value.dim() == 3:
                    history_blocks = [history_value.clone()]
                else:
                    history_blocks = [history_value[idx].clone() for idx in range(history_value.shape[0])]
            else:
                history_blocks = [torch.zeros(8, 8, _HISTORY_PLANES, dtype=torch.float32) for _ in boards]
            state = self._build_state_from_boards(boards, history_blocks)
            if info:
                state["info"] = dict(info)
            return state
        raise ValueError("Chess observation must provide at least a FEN and history block.")

    def build_imagined_domain_state(
        self,
        current_domain_state: dict[str, Any],
        action: torch.Tensor | None,
        future_z: torch.Tensor,
        step_idx: int,
    ) -> dict[str, Any]:
        _ = future_z
        fens = _normalize_fen_batch(current_domain_state["fen"])
        batch_size = len(fens)
        boards = [chess.Board(fen) for fen in fens]
        history_tensor = current_domain_state["history_block"]
        history_blocks = [history_tensor[idx].detach().cpu().clone() for idx in range(batch_size)]
        rollout = self._empty_rollout_features(batch_size)
        if action is None:
            return self._build_state_from_boards(boards, history_blocks, rollout_features=rollout)

        chosen = action.argmax(dim=-1)
        for batch_idx, board in enumerate(boards):
            action_id = int(chosen[batch_idx].item())
            material_before = _material_value(board, chess.WHITE)
            move = _action_to_move(board, action_id)
            if move not in board.legal_moves:
                rollout["realized_invalid"][batch_idx] = 10.0
                continue
            capture_gain = _piece_value_from_move(board, move)
            board.push(move)
            if board.is_checkmate():
                rollout["realized_mate_win_bonus"][batch_idx] = 5.0
                rollout["realized_check_bonus"][batch_idx] = 1.0
            elif board.is_check():
                rollout["realized_check_bonus"][batch_idx] = 0.25
            if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                rollout["realized_draw_penalty"][batch_idx] = 3.0
            else:
                opponent_reply = self._choose_opponent_move(board)
                if opponent_reply is not None:
                    board.push(opponent_reply)
            material_after = _material_value(board, chess.WHITE)
            rollout["realized_capture_gain"][batch_idx] = min(2.0, capture_gain * 0.1)
            rollout["realized_material_loss"][batch_idx] = max(0.0, material_before - material_after)
            rollout["realized_hanging_penalty"][batch_idx] = min(5.0, _hanging_value(board, chess.WHITE))
            if board.is_checkmate():
                rollout["realized_terminal_loss"][batch_idx] = 25.0
                rollout["realized_early_loss"][batch_idx] = max(0.0, float(8 - step_idx))
            elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
                rollout["realized_draw_penalty"][batch_idx] = max(
                    float(rollout["realized_draw_penalty"][batch_idx].item()),
                    3.0,
                )
        return self._build_state_from_boards(boards, history_blocks, rollout_features=rollout)

    def build_simple_baseline_path(
        self,
        domain_state: dict[str, Any],
        path_length: int,
        action_dim: int,
    ) -> torch.Tensor | None:
        if action_dim != CHESS_ACTION_DIM:
            return None
        fens = _normalize_fen_batch(domain_state["fen"])
        batch_size = len(fens)
        logits = torch.full((batch_size, path_length, action_dim), -8.0, dtype=torch.float32)
        for batch_idx, fen in enumerate(fens):
            board = chess.Board(fen)
            history_block = domain_state["history_block"][batch_idx].detach().cpu().clone()
            for step_idx in range(path_length):
                move = self._choose_simple_move(board)
                if move is None:
                    logits[batch_idx, step_idx, 0] = 8.0
                    break
                action_id = _action_from_move(move if board.turn == chess.WHITE else _mirror_move(move))
                logits[batch_idx, step_idx, action_id] = 8.0
                board.push(move)
                if board.is_game_over(claim_draw=True):
                    break
                opponent_reply = self._choose_opponent_move(board)
                if opponent_reply is not None:
                    board.push(opponent_reply)
                _, history_block = _observation_from_board(board, history_block)
                if board.is_game_over(claim_draw=True):
                    break
        return logits

    def compute_goal_latent(self, domain_state: dict[str, Any], z: torch.Tensor) -> torch.Tensor | None:
        _ = domain_state
        _ = z
        return None

    def legal_action_mask(self, observation: Any, info: dict[str, Any] | None = None) -> torch.Tensor | None:
        mask = self.build_domain_state(observation, info).get("legal_actions_mask")
        if mask is None:
            return None
        return mask.reshape(-1).bool()

    def _result_info(self, board: chess.Board, *, invalid: bool = False, opponent_move: chess.Move | None = None) -> dict[str, Any]:
        result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
        winner = 0
        if invalid or result == "0-1":
            winner = 2
        elif result == "1-0":
            winner = 1
        info = {
            "winner": winner,
            "result": result,
            "invalid_action": invalid,
            "draw": result == "1/2-1/2",
            "opponent_action": None if opponent_move is None else opponent_move.uci(),
            "fen": board.fen(),
        }
        return info

    def _observe(self) -> dict[str, Any]:
        observation, self._history_block = _observation_from_board(self._board, self._history_block)
        return {
            "observation": observation,
            "fen": self._board.fen(),
            "history_block": self._history_block.clone(),
            "moves": tuple(move.uci() for move in self._board.move_stack),
            "turn": "white" if self._board.turn == chess.WHITE else "black",
        }

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        _ = seed
        self._board = chess.Board()
        self._done = False
        self._last_terminal_info = {}
        current = _board_planes(self._board)
        self._history_block = torch.cat(
            [current, torch.zeros(8, 8, _HISTORY_PLANES - _CURRENT_BOARD_PLANES, dtype=torch.float32)],
            dim=-1,
        )
        observation = {
            "observation": torch.cat([_aux_planes(self._board), self._history_block], dim=-1),
            "fen": self._board.fen(),
            "history_block": self._history_block.clone(),
            "moves": (),
            "turn": "white",
        }
        info = {
            "winner": 0,
            "result": "*",
            "invalid_action": False,
            "draw": False,
            "phase": _decode_phase_label(_phase_index(self._board)),
        }
        return observation, info

    def set_eval_opponent_depth(self, depth: int | None) -> None:
        self._eval_opponent_depth = None if depth is None else max(0, int(depth))

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._done:
            return self._observe(), 0.0, True, False, self._last_terminal_info or self._result_info(self._board)
        action_id = int(action.item()) if torch.is_tensor(action) else int(action)
        material_before = _material_value(self._board, chess.WHITE)
        chosen_move = _action_to_move(self._board, action_id)
        if chosen_move not in self._board.legal_moves:
            self._done = True
            self._last_terminal_info = self._result_info(self._board, invalid=True)
            observation = self._observe()
            return observation, -1.0, True, False, self._last_terminal_info

        self._board.push(chosen_move)
        opponent_move = None
        if not self._board.is_game_over(claim_draw=True):
            opponent_move = self._choose_opponent_move(self._board)
            if opponent_move is not None:
                self._board.push(opponent_move)
        observation = self._observe()
        self._done = self._board.is_game_over(claim_draw=True)
        info = self._result_info(self._board, opponent_move=opponent_move)
        material_after = _material_value(self._board, chess.WHITE)
        info["material_delta"] = material_after - material_before
        info["hanging_value"] = _hanging_value(self._board, chess.WHITE)
        info["phase"] = _decode_phase_label(_phase_index(self._board))
        self._last_terminal_info = info if self._done else {}
        if info["winner"] == 1:
            reward = 1.0
        elif info["winner"] == 2:
            reward = -1.0
        else:
            reward = 0.0
        return observation, reward, self._done, False, info

    def compute_realized_cost(
        self,
        observation: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> float:
        _ = observation
        _ = truncated
        info = info or {}
        if bool(info.get("invalid_action")):
            return 10.0
        material_loss = max(0.0, -float(info.get("material_delta", 0.0)))
        hanging_penalty = float(info.get("hanging_value", 0.0))
        if terminated and int(info.get("winner", 0)) == 1:
            return 0.0
        if terminated and int(info.get("winner", 0)) == 2:
            return 25.0 + material_loss + hanging_penalty
        if terminated and bool(info.get("draw")):
            return 3.0 + material_loss
        return max(0.0, -float(reward)) + material_loss + (0.1 * hanging_penalty)

    def compute_metrics(self, episode_records: list[dict[str, Any]]) -> dict[str, float]:
        if not episode_records:
            return {
                "episode_reward_mean": 0.0,
                "win_rate": 0.0,
                "draw_rate": 0.0,
                "invalid_rate": 0.0,
                "blunder_rate": 0.0,
            }
        rewards = [float(record.get("episode_reward", 0.0)) for record in episode_records]
        wins = [1.0 if int(record.get("winner", 0)) == 1 else 0.0 for record in episode_records]
        draws = [1.0 if bool(record.get("draw", False)) else 0.0 for record in episode_records]
        invalids = [1.0 if bool(record.get("invalid_action", False)) else 0.0 for record in episode_records]
        blunders = [float(record.get("blunder_rate", 0.0)) for record in episode_records]
        return {
            "episode_reward_mean": sum(rewards) / len(rewards),
            "win_rate": sum(wins) / len(wins),
            "draw_rate": sum(draws) / len(draws),
            "invalid_rate": sum(invalids) / len(invalids),
            "blunder_rate": sum(blunders) / len(blunders),
        }

    def baseline_action(
        self,
        kind: str,
        observation: Any,
        info: dict[str, Any] | None = None,
    ) -> Any:
        if kind in {"simple", "greedy"}:
            state = self.build_domain_state(observation, info)
            board = chess.Board(state["fen"][0])
            move = self._choose_simple_move(board)
            if move is None:
                return torch.zeros(1, dtype=torch.long)
            return torch.tensor([_action_from_move(move if board.turn == chess.WHITE else _mirror_move(move))], dtype=torch.long)
        return super().baseline_action(kind, observation, info)

    def simulate_delayed_outcome(
        self,
        action_vec: torch.Tensor,
        domain_state: dict,
    ) -> dict[str, torch.Tensor] | None:
        imagined = self.build_imagined_domain_state(domain_state, action_vec, torch.zeros(action_vec.shape[0], 1), 0)
        realized_cost = (
            imagined["realized_invalid"]
            + imagined["realized_material_loss"]
            + imagined["realized_hanging_penalty"]
            + imagined["realized_terminal_loss"]
            + imagined["realized_early_loss"]
            + imagined["realized_draw_penalty"]
            - imagined["realized_check_bonus"]
            - imagined["realized_capture_gain"]
            - imagined["realized_mate_win_bonus"]
        )
        return {
            "outcome_observation": imagined["board_observation"],
            "realized_intrinsic_cost": realized_cost.float(),
        }

    def simulate_path_outcome(
        self,
        action_path: torch.Tensor,
        domain_state: dict,
    ) -> dict[str, torch.Tensor] | None:
        if action_path.dim() == 2:
            action_path = action_path.unsqueeze(0)
        if action_path.dim() != 3:
            raise ValueError(f"Expected action path [B, P, A], got {tuple(action_path.shape)}.")
        working_state = domain_state
        cumulative = torch.zeros(action_path.shape[0], dtype=torch.float32, device=action_path.device)
        for step_idx in range(action_path.shape[1]):
            working_state = self.build_imagined_domain_state(
                working_state,
                action_path[:, step_idx, :],
                torch.zeros(action_path.shape[0], 1, device=action_path.device),
                step_idx,
            )
            cumulative = cumulative + (
                working_state["realized_invalid"]
                + working_state["realized_material_loss"]
                + working_state["realized_hanging_penalty"]
                + working_state["realized_terminal_loss"]
                + working_state["realized_early_loss"]
                + working_state["realized_draw_penalty"]
                - working_state["realized_check_bonus"]
                - working_state["realized_capture_gain"]
                - working_state["realized_mate_win_bonus"]
            ).to(cumulative)
        return {
            "outcome_observation": working_state["board_observation"],
            "realized_intrinsic_cost": cumulative,
        }

    def maybe_enable_self_play(self, metrics: dict[str, float]) -> bool:
        puzzle_solve_rate = float(metrics.get("puzzle_solve_rate", 0.0))
        invalid_rate = float(metrics.get("invalid_rate", 1.0))
        blunder_rate = float(metrics.get("blunder_rate", 1.0))
        win_rate = float(metrics.get("win_rate", 0.0))
        draw_rate = float(metrics.get("draw_rate", 0.0))
        match_strength = win_rate + (0.5 * draw_rate)
        if (
            puzzle_solve_rate >= self.self_play_unlock_threshold
            and invalid_rate <= 0.05
            and blunder_rate <= 0.20
            and match_strength >= 0.55
        ):
            self.self_play_enabled = True
        return self.self_play_enabled

    @property
    def domain_name(self) -> str:
        return "chess"

    @property
    def vocab_size(self) -> int:
        return 2

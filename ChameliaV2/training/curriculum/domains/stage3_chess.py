"""Stage 3 chess curriculum with structured buckets and opponent ladder."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import chess
import chess.engine
import torch
import torch.nn.functional as F

from src.chamelia.plugins.chess import CHESS_ACTION_DIM, ChessDomain
from src.chamelia.plugins.chess import _HISTORY_PLANES, _action_to_move, _observation_from_board
from training.curriculum.batch import CurriculumBatch
from training.curriculum.cost_schedule import CostLevel
from training.curriculum.data.chess_curriculum import load_chess_curriculum_samples
from training.curriculum.data.public_games import DEFAULT_CURRICULUM_ROOT
from training.curriculum.domains.base import BaseCurriculumDomain, DomainSpec, MaskingStrategy, make_level


_LEVEL_BUCKETS = (
    "openings",
    "puzzles",
    "endgames",
    "middlegames",
    "short_games",
    "full_games",
)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_repo_relative_path(path_value: str) -> str:
    if not path_value:
        return ""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return str(path)
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return str(cwd_path)
    return str(_PROJECT_ROOT / path)


class ChessObservationMaskingStrategy(MaskingStrategy):
    """No-op masking for raw chess board observations."""

    def apply(self, tokens: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        _ = level
        batch_size = int(tokens.shape[0])
        return tokens, torch.zeros(batch_size, 64, dtype=torch.float32, device=tokens.device)


class _BuiltInOpponent:
    """Use ChessDomain's built-in heuristic/search opponent as an evaluation rung."""

    def __init__(self, runtime_domain: ChessDomain, *, depth: int) -> None:
        self.runtime_domain = runtime_domain
        self.depth = max(0, int(depth))

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        previous = getattr(self.runtime_domain, "_eval_opponent_depth", None)
        self.runtime_domain.set_eval_opponent_depth(self.depth)
        try:
            return self.runtime_domain._choose_opponent_move(board)  # noqa: SLF001
        finally:
            self.runtime_domain.set_eval_opponent_depth(previous)

    @property
    def name(self) -> str:
        return f"builtin_d{self.depth}"


class _StockfishOpponent:
    """Thin UCI wrapper around Stockfish for curriculum evaluation rungs."""

    def __init__(
        self,
        *,
        stockfish_path: str,
        uci_elo: int | None = None,
        skill_level: int | None = None,
        depth: int | None = None,
    ) -> None:
        if not stockfish_path:
            raise ValueError("stockfish_path is required for Stockfish curriculum opponents.")
        self.stockfish_path = stockfish_path
        self.uci_elo = uci_elo
        self.skill_level = skill_level
        self.depth = depth
        self._engine: chess.engine.SimpleEngine | None = None

    @classmethod
    def available(cls, path: str | None) -> bool:
        if not path:
            return False
        return shutil.which(path) is not None or Path(path).exists()

    def open(self) -> "_StockfishOpponent":
        self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        if self.uci_elo is not None:
            self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(self.uci_elo)})
        elif self.skill_level is not None:
            self._engine.configure({"Skill Level": int(self.skill_level)})
        self._engine.configure({"Threads": 1, "Hash": 64})
        return self

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def choose_move(self, board: chess.Board) -> chess.Move | None:
        if self._engine is None:
            raise RuntimeError("Stockfish opponent must be opened before use.")
        limit = chess.engine.Limit(depth=self.depth) if self.depth is not None else chess.engine.Limit(nodes=2000)
        return self._engine.play(board, limit).move

    @property
    def name(self) -> str:
        if self.uci_elo is not None:
            return f"stockfish_elo_{self.uci_elo}"
        if self.depth is not None:
            return f"stockfish_d{self.depth}"
        if self.skill_level is not None:
            return f"stockfish_skill_{self.skill_level}"
        return "stockfish"


def _bucket_metric_name(bucket: str) -> str:
    return {
        "openings": "opening_accuracy",
        "puzzles": "puzzle_solve_rate",
        "endgames": "endgame_solve_rate",
        "middlegames": "middlegame_accuracy",
        "short_games": "short_game_accuracy",
        "full_games": "full_game_accuracy",
    }[bucket]


def _zero_probe(bucket: str) -> dict[str, float]:
    return {
        _bucket_metric_name(bucket): 0.0,
        "valid_move_rate": 0.0,
        "safe_move_rate": 0.0,
        "result_score": 0.0,
        "win_rate": 0.0,
        "draw_rate": 0.0,
        "invalid_rate": 1.0,
        "self_play_ready": 0.0,
    }


class ChessCurriculumDomain(BaseCurriculumDomain):
    """Stage-3 chess domain with structured training buckets and runged evaluation."""

    def __init__(
        self,
        *,
        batch_size: int = 8,
        seq_len: int = 128,
        data_root: str | Path | None = None,
        curriculum_config: dict[str, Any] | None = None,
    ) -> None:
        self.data_root = Path(data_root) if data_root is not None else DEFAULT_CURRICULUM_ROOT
        self.curriculum_root = self.data_root / "stage3" / "chess" / "lichess"
        self.curriculum_config = dict(curriculum_config or {})
        self.stockfish_path = _resolve_repo_relative_path(str(self.curriculum_config.get("stockfish_path", "") or ""))
        self.depth_schedule = tuple(int(value) for value in self.curriculum_config.get("depth_schedule", [4, 6, 8, 10, 12, 14]))
        self.elo_target_per_level = tuple(self.curriculum_config.get("elo_target_per_level", [None] * len(_LEVEL_BUCKETS)))
        self.self_play_start_threshold = float(self.curriculum_config.get("self_play_start_threshold", 0.90))
        self._sample_cache: dict[tuple[int, str, int, int], list[dict[str, Any]]] = {}
        self._preprocess_runtime = ChessDomain(
            embed_dim=32,
            opponent_level=1,
            chess_data_root=str(self.curriculum_root),
            stockfish_path=self.stockfish_path or None,
        )

        def move_supervision(_z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            answers = domain_state["answer_token"].long().clamp(min=0, max=action.shape[1] - 1)
            fallback_loss = F.cross_entropy(action, answers, reduction="none")
            candidate_tokens = domain_state.get("candidate_move_tokens")
            candidate_mask = domain_state.get("candidate_move_mask")
            if candidate_tokens is None or candidate_mask is None or not candidate_mask.any():
                return fallback_loss
            log_probs = F.log_softmax(action, dim=-1)
            clipped_tokens = candidate_tokens.long().clamp(min=0, max=action.shape[1] - 1)
            candidate_log_probs = torch.gather(log_probs, 1, clipped_tokens)
            weights = domain_state.get("candidate_move_weights")
            if weights is None:
                weights = candidate_mask.float() / candidate_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
            else:
                weights = weights.float() * candidate_mask.float()
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            guided_loss = -(weights * candidate_log_probs * candidate_mask.float()).sum(dim=-1)
            return torch.where(candidate_mask.any(dim=-1), guided_loss, fallback_loss)

        def engine_blunder_cost(_z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            answers = domain_state["answer_token"].long().clamp(min=0, max=action.shape[1] - 1)
            fallback_cost = F.cross_entropy(action, answers, reduction="none")
            blunder_losses = domain_state.get("candidate_move_blunder_cp")
            candidate_tokens = domain_state.get("candidate_move_tokens")
            candidate_mask = domain_state.get("candidate_move_mask")
            if (
                blunder_losses is None
                or candidate_tokens is None
                or candidate_mask is None
                or not candidate_mask.any()
            ):
                return fallback_cost
            probs = torch.softmax(action, dim=-1)
            clipped_tokens = candidate_tokens.long().clamp(min=0, max=action.shape[1] - 1)
            candidate_probs = torch.gather(probs, 1, clipped_tokens) * candidate_mask.float()
            candidate_mass = candidate_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
            normalized_candidate_probs = candidate_probs / candidate_mass
            scaled_losses = (blunder_losses.float() / 100.0) * candidate_mask.float()
            off_candidate_penalty = (1.0 - candidate_probs.sum(dim=-1)).clamp_min(0.0)
            max_loss = scaled_losses.max(dim=-1).values.clamp_min(1.0)
            guided_cost = (normalized_candidate_probs * scaled_losses).sum(dim=-1) + off_candidate_penalty * max_loss
            return torch.where(candidate_mask.any(dim=-1), guided_cost, fallback_cost)

        schedule: list[CostLevel] = []
        for level, bucket in enumerate(_LEVEL_BUCKETS):
            thresholds = {
                _bucket_metric_name(bucket): 0.70 + (0.03 * min(level, 3)),
                "valid_move_rate": 0.98,
                "safe_move_rate": 0.55 + (0.05 * min(level, 4)),
            }
            if bucket in {"endgames", "middlegames", "short_games", "full_games"}:
                thresholds["result_score"] = 0.35 + (0.05 * min(level, 4))
            if bucket == "full_games":
                thresholds["self_play_ready"] = self.self_play_start_threshold
            schedule.append(
                make_level(
                    level,
                    description=f"chess_{bucket}",
                    cost_fns=[(move_supervision, 0.7), (engine_blunder_cost, 0.3)],
                    thresholds=thresholds,
                    min_episodes=64,
                )
            )

        def sample_builder(level: int, split: str, spec: DomainSpec) -> list[dict[str, Any]]:
            key = (level, 0 if split == "train" else 1, spec.seq_len, spec.dataset_size)
            cached = self._sample_cache.get(key)
            if cached is not None:
                return cached
            desired_bucket = _LEVEL_BUCKETS[min(level, len(_LEVEL_BUCKETS) - 1)]
            fallback_order = (
                desired_bucket,
                "middlegames",
                "openings",
                "endgames",
                "puzzles",
                "short_games",
                "full_games",
            )
            max_samples = spec.dataset_size if split == "train" else max(1, spec.dataset_size // 4)
            samples: list[dict[str, Any]] = []
            for bucket in fallback_order:
                samples = load_chess_curriculum_samples(
                    self.curriculum_root,
                    bucket=bucket,
                    split=split,
                    seq_len=spec.seq_len,
                    max_samples=max_samples,
                )
                if samples:
                    break
            self._sample_cache[key] = samples
            return samples

        super().__init__(
            spec=DomainSpec(
                name="chess",
                stage_idx=3,
                action_dim=CHESS_ACTION_DIM,
                vocab_size=CHESS_ACTION_DIM,
                batch_size=batch_size,
                seq_len=seq_len,
                dataset_size=2048,
            ),
            masking_strategy=ChessObservationMaskingStrategy(),
            cost_schedule=schedule,
            probe_fn=lambda model, level: _zero_probe(_LEVEL_BUCKETS[min(level, len(_LEVEL_BUCKETS) - 1)]),
            sample_builder=sample_builder,
        )

    def _opponent_for_level(self, level: int, runtime_domain: ChessDomain) -> Any:
        if level <= 1:
            depth = self.depth_schedule[min(level, len(self.depth_schedule) - 1)] if self.depth_schedule else 1
            return _BuiltInOpponent(runtime_domain, depth=max(1, depth // 2))
        target = self.elo_target_per_level[min(level, len(self.elo_target_per_level) - 1)] if self.elo_target_per_level else None
        if isinstance(target, str) and target.startswith("stockfish_d") and _StockfishOpponent.available(self.stockfish_path):
            return _StockfishOpponent(stockfish_path=self.stockfish_path, depth=int(target.split("d", 1)[1]))
        if isinstance(target, (int, float)) and _StockfishOpponent.available(self.stockfish_path):
            return _StockfishOpponent(stockfish_path=self.stockfish_path, uci_elo=int(target))
        depth = self.depth_schedule[min(level, len(self.depth_schedule) - 1)] if self.depth_schedule else 2
        return _BuiltInOpponent(runtime_domain, depth=depth)

    def build_curriculum_batch(self, raw_batch: dict[str, Any], split: str) -> CurriculumBatch:
        observations = raw_batch["observation"].float()
        history_block = raw_batch["history_block"].float()
        input_mask = torch.zeros(observations.shape[0], 64, dtype=torch.float32)
        base_state = self._preprocess_runtime.build_domain_state(
            {
                "fen": raw_batch["fen"],
                "history_block": history_block,
            },
            None,
        )
        answers = raw_batch["answer"].long()
        domain_state = dict(base_state)
        domain_state.update(
            {
                "answer": answers,
                "answer_token": answers,
                "target": raw_batch["target"],
                "candidate_move_tokens": raw_batch["candidate_move_tokens"].long(),
                "candidate_move_weights": raw_batch["candidate_move_weights"].float(),
                "candidate_move_mask": raw_batch["candidate_move_mask"].bool(),
                "candidate_move_blunder_cp": raw_batch["candidate_move_blunder_cp"].float(),
                "principal_variation_tokens": raw_batch["principal_variation_tokens"].long(),
                "centipawn_eval": raw_batch["centipawn_eval"].float(),
                "fen": list(raw_batch["fen"]),
                "phase": list(raw_batch["phase"]),
                "source": list(raw_batch["source"]),
                "bucket": list(raw_batch["bucket"]),
                "moves": list(raw_batch["moves"]),
                "best_move": list(raw_batch["best_move"]),
                "opening_tags": list(raw_batch["opening_tags"]),
                "motif_tags": list(raw_batch["motif_tags"]),
                "result": list(raw_batch["result"]),
                "history_block": history_block,
            }
        )
        targets = {
            "answer": answers,
            "answer_token": answers,
            "candidate_move_tokens": raw_batch["candidate_move_tokens"].long(),
            "candidate_move_weights": raw_batch["candidate_move_weights"].float(),
            "candidate_move_mask": raw_batch["candidate_move_mask"].bool(),
            "candidate_move_blunder_cp": raw_batch["candidate_move_blunder_cp"].float(),
            "principal_variation_tokens": raw_batch["principal_variation_tokens"].long(),
            "centipawn_eval": raw_batch["centipawn_eval"].float(),
        }
        return CurriculumBatch(
            domain_name=self.domain_name(),
            raw_inputs=observations,
            tokens=observations,
            embedded_tokens=None,
            input_mask=input_mask,
            targets=targets,
            domain_state=domain_state,
            metadata={"split": split, "stage": self.stage(), "level": self.cost.current_level},
        )

    def build_runtime_domain(self, embed_dim: int):
        return ChessDomain(
            embed_dim=embed_dim,
            opponent_level=1,
            chess_data_root=str(self.curriculum_root),
            stockfish_path=self.stockfish_path or None,
        )

    def _validation_metrics(
        self,
        model: Any,
        level: int,
        device: torch.device,
    ) -> dict[str, float]:
        loader = self.get_data_loader(level, split="val")
        runtime_domain = self.build_runtime_domain(model.embed_dim)
        tokenizer = runtime_domain.get_tokenizer()
        if isinstance(tokenizer, torch.nn.Module):
            tokenizer.to(device)
        model.set_domain(runtime_domain)
        correct = 0
        total = 0
        valid_predictions = 0
        safe_score_total = 0.0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to_device(device)
                tokenized = tokenizer(batch.tokens.float())
                outputs = model(
                    tokens=tokenized.tokens,
                    mask=batch.input_mask,
                    domain_state=batch.domain_state,
                    actor_mode="mode2",
                    store_to_memory=False,
                    input_kind="embedded_tokens",
                )
                pred = outputs["action_vec"].argmax(dim=-1)
                target = batch.domain_state["answer_token"].long()
                legal_mask = batch.domain_state["legal_actions_mask"].bool()
                correct += int((pred == target).sum().item())
                total += int(target.numel())
                valid_predictions += int(legal_mask[torch.arange(pred.shape[0], device=pred.device), pred].sum().item())
                candidate_ids = batch.domain_state["candidate_move_tokens"].long()
                candidate_mask = batch.domain_state["candidate_move_mask"].bool()
                candidate_blunders = batch.domain_state["candidate_move_blunder_cp"].float()
                for row_idx in range(pred.shape[0]):
                    matches = (candidate_ids[row_idx] == pred[row_idx]) & candidate_mask[row_idx]
                    if bool(matches.any().item()):
                        blunder = float(candidate_blunders[row_idx][matches][0].item())
                        safe_score_total += max(0.0, 1.0 - min(blunder / 200.0, 1.0))
        total = max(1, total)
        return {
            _bucket_metric_name(_LEVEL_BUCKETS[min(level, len(_LEVEL_BUCKETS) - 1)]): float(correct) / float(total),
            "valid_move_rate": float(valid_predictions) / float(total),
            "safe_move_rate": float(safe_score_total) / float(total),
        }

    def _play_eval_games(
        self,
        model: Any,
        level: int,
        device: torch.device,
        *,
        games: int = 2,
        max_plies: int = 80,
    ) -> dict[str, float]:
        runtime_domain = self.build_runtime_domain(model.embed_dim)
        tokenizer = runtime_domain.get_tokenizer()
        if isinstance(tokenizer, torch.nn.Module):
            tokenizer.to(device)
        model.set_domain(runtime_domain)
        opponent = self._opponent_for_level(level, runtime_domain)
        wins = 0.0
        draws = 0.0
        invalids = 0.0
        opened = False
        if isinstance(opponent, _StockfishOpponent):
            opponent.open()
            opened = True
        try:
            with torch.no_grad():
                for _ in range(games):
                    board = chess.Board()
                    history_block = torch.zeros(8, 8, _HISTORY_PLANES, dtype=torch.float32)
                    terminal_invalid = False
                    for _ply in range(max_plies):
                        observation, history_block = _observation_from_board(board, history_block)
                        domain_state = runtime_domain.build_domain_state(
                            {"fen": board.fen(), "history_block": history_block},
                            None,
                        )
                        tokenized = tokenizer(observation.unsqueeze(0).to(device))
                        outputs = model(
                            tokens=tokenized.tokens,
                            mask=torch.zeros(1, 64, dtype=torch.float32, device=device),
                            domain_state={key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in domain_state.items()},
                            actor_mode="mode2",
                            store_to_memory=False,
                            input_kind="embedded_tokens",
                        )
                        action_id = int(outputs["action_vec"].argmax(dim=-1)[0].item())
                        move = _action_to_move(board, action_id)
                        if move not in board.legal_moves:
                            invalids += 1.0
                            terminal_invalid = True
                            break
                        board.push(move)
                        if board.is_game_over(claim_draw=True):
                            break
                        reply = opponent.choose_move(board)
                        if reply is not None:
                            board.push(reply)
                        if board.is_game_over(claim_draw=True):
                            break
                    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
                    if terminal_invalid or result == "0-1":
                        pass
                    elif result == "1-0":
                        wins += 1.0
                    elif result == "1/2-1/2":
                        draws += 1.0
        finally:
            if opened:
                opponent.close()
        total_games = float(max(1, games))
        invalid_rate = invalids / total_games
        win_rate = wins / total_games
        draw_rate = draws / total_games
        return {
            "result_score": win_rate + (0.5 * draw_rate),
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "invalid_rate": invalid_rate,
        }

    def run_advancement_probe(self, model: Any, level: int) -> dict[str, float]:
        bucket = _LEVEL_BUCKETS[min(level, len(_LEVEL_BUCKETS) - 1)]
        if model is None:
            return _zero_probe(bucket)
        device = next(model.parameters()).device
        metrics = self._validation_metrics(model, level, device)
        if bucket in {"endgames", "middlegames", "short_games", "full_games"}:
            metrics.update(self._play_eval_games(model, level, device))
        else:
            metrics.update({"result_score": 0.0, "win_rate": 0.0, "draw_rate": 0.0, "invalid_rate": 1.0})
        metrics["self_play_ready"] = float(
            metrics.get("valid_move_rate", 0.0) >= 0.98
            and metrics.get("safe_move_rate", 0.0) >= 0.80
            and metrics.get("result_score", 0.0) >= 0.55
            and metrics.get(_bucket_metric_name(bucket), 0.0) >= self.self_play_start_threshold
        )
        return metrics

#!/usr/bin/env python3
"""Populate Stage 3 and Stage 4 curriculum assets into local files.

This script materializes the local file layout expected by the curriculum loaders:

- Stage 3 chess: engine-annotated JSONL or PGN files under ``data/curriculum/stage3/chess/lichess``
- Stage 3 poker: JSONL under ``data/curriculum/stage3/poker``
- Stage 3 gridworld: JSONL under ``data/curriculum/stage3/gridworld``
- Stage 3 go: SGF-like text under ``data/curriculum/stage3/go``
- Stage 4 collaboration: JSONL under ``data/curriculum/stage4/generated``

Public sources are used where they are tractable via Hugging Face datasets.
Remaining gaps are filled with local generated traces so Stage 3/4 do not fall
back to entirely in-memory synthetic samples.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path
import random
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - exercised on cluster
    raise SystemExit("datasets is required to populate Stage 3/4 curriculum assets") from exc

from training.curriculum.generators.gridworld_gen import HiddenRegimeGridworldGenerator


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_lines(path: Path, lines: Iterable[str]) -> int:
    _ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            count += 1
    return count


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    _ensure_parent(path)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def _coord(index: int) -> str:
    letters = "abcdefghijklmnopqrst"
    return letters[index % 19] + letters[(index // 19) % 19]


def _synthetic_sgf_games(count: int, moves_per_game: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    games: list[str] = []
    for _ in range(count):
        seen: set[str] = set()
        moves: list[str] = ["(;GM[1]FF[4]SZ[19]"]
        for move_idx in range(moves_per_game):
            color = "B" if move_idx % 2 == 0 else "W"
            for _attempt in range(32):
                coord = _coord(rng.randrange(19 * 19))
                if coord not in seen:
                    seen.add(coord)
                    moves.append(f";{color}[{coord}]")
                    break
        moves.append(")\n")
        games.append("".join(moves))
    return games


def _gridworld_rows(count: int, seed: int) -> list[dict[str, Any]]:
    generator = HiddenRegimeGridworldGenerator(side=5)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        regime = rng.randrange(4)
        sample = generator.sample(regime=regime)
        rows.append(
            {
                "trajectory": [str(int(token)) for token in sample.tokens.tolist()],
                "result": f"regime_{regime}",
                "source": "local_gridworld_generator",
                "sample_id": idx,
            }
        )
    return rows


def _configure_stockfish_skill(engine: Any, skill: int) -> None:
    """Best-effort Stockfish skill configuration across builds."""
    options = getattr(engine, "options", {})
    if "Skill Level" in options:
        engine.configure({"Skill Level": int(max(0, min(skill, 20)))})
        return
    if "UCI_LimitStrength" in options and "UCI_Elo" in options:
        bounded = int(max(100, min(skill, 3200)))
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": bounded})


def _stockfish_chess_rows(
    *,
    stockfish_path: str,
    count: int,
    seed: int,
    top_k: int,
    play_depth: int,
    annotate_depth: int,
    max_opening_plies: int,
    max_game_plies: int,
    positions_per_game: int,
    skill_levels: list[int],
) -> list[dict[str, Any]]:
    """Generate engine-annotated chess samples from locally played games."""
    try:
        import chess
        import chess.engine
    except ImportError as exc:  # pragma: no cover - cluster/runtime specific
        raise SystemExit("python-chess is required for Stockfish chess generation") from exc

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        while len(rows) < count:
            board = chess.Board()
            history: list[str] = []
            opening_plies = rng.randint(0, max_opening_plies)
            for _ in range(opening_plies):
                if board.is_game_over():
                    break
                move = rng.choice(list(board.legal_moves))
                history.append(move.uci())
                board.push(move)

            white_skill = rng.choice(skill_levels)
            black_skill = rng.choice(skill_levels)
            game_rows: list[dict[str, Any]] = []

            while not board.is_game_over() and len(history) < max_game_plies and len(rows) + len(game_rows) < count:
                if len(game_rows) < positions_per_game:
                    analysis = engine.analyse(
                        board,
                        chess.engine.Limit(depth=annotate_depth),
                        multipv=top_k,
                    )
                    infos = analysis if isinstance(analysis, list) else [analysis]
                    candidate_moves: list[str] = []
                    candidate_scores_cp: list[float] = []
                    principal_variation: list[str] = []
                    for info in infos:
                        pv = info.get("pv") or []
                        if not pv:
                            continue
                        candidate_moves.append(pv[0].uci())
                        score = info.get("score")
                        score_cp = 0
                        if score is not None:
                            score_cp = score.pov(board.turn).score(mate_score=100000) or 0
                        candidate_scores_cp.append(float(score_cp))
                        if not principal_variation:
                            principal_variation = [move.uci() for move in pv]
                    if candidate_moves:
                        best_score = candidate_scores_cp[0] if candidate_scores_cp else 0.0
                        game_rows.append(
                            {
                                "fen": board.fen(),
                                "moves": history + [candidate_moves[0]],
                                "best_move": candidate_moves[0],
                                "candidate_moves": candidate_moves,
                                "candidate_scores_cp": candidate_scores_cp,
                                "candidate_blunder_losses_cp": [float(best_score - score) for score in candidate_scores_cp],
                                "principal_variation": principal_variation,
                                "centipawn_eval": float(best_score),
                                "turn": "white" if board.turn else "black",
                                "engine_white": white_skill,
                                "engine_black": black_skill,
                                "source": "stockfish_local_generator",
                            }
                        )

                active_skill = white_skill if board.turn else black_skill
                _configure_stockfish_skill(engine, active_skill)
                result = engine.play(board, chess.engine.Limit(depth=play_depth))
                history.append(result.move.uci())
                board.push(result.move)

            result_text = board.result(claim_draw=True) if board.is_game_over() else "*"
            for row in game_rows:
                row["result"] = result_text
            rows.extend(game_rows[: max(0, count - len(rows))])
    finally:
        engine.quit()
    return rows[:count]


def _poker_row_from_hf(row: dict[str, Any]) -> dict[str, Any] | None:
    actions = row.get("actions")
    players = row.get("players")
    if not isinstance(actions, list) or not actions:
        return None
    serialized_actions = [json.dumps(action, ensure_ascii=True) if not isinstance(action, str) else action for action in actions]
    return {
        "actions": serialized_actions,
        "players": players if isinstance(players, list) else [],
        "variant": row.get("variant"),
        "venue": row.get("venue"),
        "source_file": row.get("source_file"),
    }


def _collab_row_from_hf(row: dict[str, Any]) -> dict[str, Any] | None:
    characters = row.get("characters")
    conversation = row.get("conversation")
    setting = row.get("setting")
    if not isinstance(characters, dict) or not isinstance(conversation, list) or not conversation:
        return None
    char_items = list(characters.items())
    midpoint = max(1, len(char_items) // 2)
    agent_a = [f"{name}: {desc}" for name, desc in char_items[:midpoint]]
    agent_b = [f"{name}: {desc}" for name, desc in char_items[midpoint:]] or agent_a[:1]
    messages: list[str] = []
    for turn in conversation:
        if not isinstance(turn, dict):
            continue
        speaker = turn.get("from", "agent")
        message = turn.get("message", "")
        if isinstance(message, str) and message.strip():
            messages.append(f"{speaker}: {message.strip()}")
    if not messages:
        return None
    solution = messages[-1]
    return {
        "task": setting if isinstance(setting, str) else "coordinate the scenario",
        "agent_a": agent_a,
        "agent_b": agent_b,
        "messages": messages,
        "solution": solution,
        "setting_after_interaction": row.get("setting after interaction"),
    }


def _split_stream_rows(
    rows: Iterable[dict[str, Any] | None],
    *,
    train_count: int,
    val_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for row in rows:
        if row is None:
            continue
        if len(train_rows) < train_count:
            train_rows.append(row)
            continue
        if len(val_rows) < val_count:
            val_rows.append(row)
            continue
        break
    return train_rows, val_rows


def populate_chess(
    root: Path,
    *,
    train_count: int,
    val_count: int,
    stockfish_path: str | None,
    top_k: int,
    play_depth: int,
    annotate_depth: int,
    max_opening_plies: int,
    max_game_plies: int,
    positions_per_game: int,
    skill_levels: list[int],
    seed: int,
) -> dict[str, int]:
    chess_root = root / "stage3" / "chess" / "lichess"
    if stockfish_path:
        train_rows = _stockfish_chess_rows(
            stockfish_path=stockfish_path,
            count=train_count,
            seed=seed,
            top_k=top_k,
            play_depth=play_depth,
            annotate_depth=annotate_depth,
            max_opening_plies=max_opening_plies,
            max_game_plies=max_game_plies,
            positions_per_game=positions_per_game,
            skill_levels=skill_levels,
        )
        val_rows = _stockfish_chess_rows(
            stockfish_path=stockfish_path,
            count=val_count,
            seed=seed + 1,
            top_k=top_k,
            play_depth=play_depth,
            annotate_depth=annotate_depth,
            max_opening_plies=max_opening_plies,
            max_game_plies=max_game_plies,
            positions_per_game=positions_per_game,
            skill_levels=skill_levels,
        )
        return {
            "train": _write_jsonl(chess_root / "train.jsonl", train_rows),
            "validation": _write_jsonl(chess_root / "validation.jsonl", val_rows),
        }

    dataset = load_dataset("Lichess/tournament-chess-games", split="train", streaming=True)
    train_games: list[str] = []
    val_games: list[str] = []
    for row in dataset:
        movetext = row.get("movetext")
        result = row.get("Result")
        if not isinstance(movetext, str) or len(movetext.strip()) < 8:
            continue
        pgn = f"{movetext.strip()} {result or '*'}\n"
        if len(train_games) < train_count:
            train_games.append(pgn)
            continue
        if len(val_games) < val_count:
            val_games.append(pgn)
            continue
        break
    return {
        "train": _write_lines(chess_root / "train.pgn", train_games),
        "validation": _write_lines(chess_root / "validation.pgn", val_games),
    }


def populate_poker(root: Path, *, train_count: int, val_count: int) -> dict[str, int]:
    dataset = load_dataset("takara-ai/poker_hands", split="train", streaming=True)
    train_rows, val_rows = _split_stream_rows(
        (_poker_row_from_hf(row) for row in dataset),
        train_count=train_count,
        val_count=val_count,
    )
    poker_root = root / "stage3" / "poker"
    return {
        "train": _write_jsonl(poker_root / "train.jsonl", train_rows),
        "validation": _write_jsonl(poker_root / "validation.jsonl", val_rows),
    }


def populate_collaboration(root: Path, *, train_count: int, val_count: int) -> dict[str, int]:
    dataset = load_dataset("agentlans/multi-character-dialogue", split="train", streaming=True)
    train_rows, val_rows = _split_stream_rows(
        (_collab_row_from_hf(row) for row in dataset),
        train_count=train_count,
        val_count=val_count,
    )
    collab_root = root / "stage4" / "generated"
    return {
        "train": _write_jsonl(collab_root / "train.jsonl", train_rows),
        "validation": _write_jsonl(collab_root / "validation.jsonl", val_rows),
    }


def populate_gridworld(root: Path, *, train_count: int, val_count: int, seed: int) -> dict[str, int]:
    grid_root = root / "stage3" / "gridworld"
    return {
        "train": _write_jsonl(grid_root / "train.jsonl", _gridworld_rows(train_count, seed)),
        "validation": _write_jsonl(grid_root / "validation.jsonl", _gridworld_rows(val_count, seed + 1)),
    }


def populate_go(root: Path, *, train_count: int, val_count: int, moves_per_game: int, seed: int) -> dict[str, int]:
    go_root = root / "stage3" / "go"
    return {
        "train": _write_lines(go_root / "train.sgf", _synthetic_sgf_games(train_count, moves_per_game, seed)),
        "validation": _write_lines(
            go_root / "validation.sgf",
            _synthetic_sgf_games(val_count, moves_per_game, seed + 1),
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Populate Stage 3 and 4 curriculum assets")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "curriculum"))
    parser.add_argument("--stockfish-path", default=None)
    parser.add_argument("--chess-train", type=int, default=20000)
    parser.add_argument("--chess-val", type=int, default=2000)
    parser.add_argument("--chess-top-k", type=int, default=4)
    parser.add_argument("--chess-play-depth", type=int, default=10)
    parser.add_argument("--chess-annotate-depth", type=int, default=14)
    parser.add_argument("--chess-max-opening-plies", type=int, default=10)
    parser.add_argument("--chess-max-game-plies", type=int, default=120)
    parser.add_argument("--chess-positions-per-game", type=int, default=6)
    parser.add_argument("--chess-skill-levels", default="0,4,8,12,16,20")
    parser.add_argument("--poker-train", type=int, default=20000)
    parser.add_argument("--poker-val", type=int, default=2000)
    parser.add_argument("--collab-train", type=int, default=20000)
    parser.add_argument("--collab-val", type=int, default=2000)
    parser.add_argument("--gridworld-train", type=int, default=20000)
    parser.add_argument("--gridworld-val", type=int, default=2000)
    parser.add_argument("--go-train", type=int, default=10000)
    parser.add_argument("--go-val", type=int, default=1000)
    parser.add_argument("--go-moves-per-game", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    skill_levels = [int(part.strip()) for part in str(args.chess_skill_levels).split(",") if part.strip()]
    summary = {
        "chess": populate_chess(
            root,
            train_count=args.chess_train,
            val_count=args.chess_val,
            stockfish_path=args.stockfish_path,
            top_k=args.chess_top_k,
            play_depth=args.chess_play_depth,
            annotate_depth=args.chess_annotate_depth,
            max_opening_plies=args.chess_max_opening_plies,
            max_game_plies=args.chess_max_game_plies,
            positions_per_game=args.chess_positions_per_game,
            skill_levels=skill_levels,
            seed=args.seed,
        ),
        "poker": populate_poker(root, train_count=args.poker_train, val_count=args.poker_val),
        "collaboration": populate_collaboration(root, train_count=args.collab_train, val_count=args.collab_val),
        "gridworld": populate_gridworld(root, train_count=args.gridworld_train, val_count=args.gridworld_val, seed=args.seed),
        "go": populate_go(
            root,
            train_count=args.go_train,
            val_count=args.go_val,
            moves_per_game=args.go_moves_per_game,
            seed=args.seed,
        ),
    }

    print(json.dumps({"data_root": str(root), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

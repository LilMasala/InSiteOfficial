"""Normalized chess curriculum data loaders for Stage 3."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import heapq
import hashlib
from io import TextIOWrapper
from pathlib import Path
import shutil
import subprocess
from typing import Any

import chess
import chess.pgn
import torch

from src.chamelia.plugins.chess import (
    CHESS_ACTION_DIM,
    _HISTORY_PLANES,
    _action_from_move,
    _mirror_move,
    _observation_from_board,
    _phase_index,
)


CHESS_BUCKETS = (
    "openings",
    "puzzles",
    "endgames",
    "middlegames",
    "short_games",
    "full_games",
)
MAX_CANDIDATE_MOVES = 8


@dataclass(frozen=True)
class ChessCurriculumRecord:
    """One normalized chess training/example row."""

    fen: str
    best_move: str
    context_moves: tuple[str, ...]
    candidate_moves: tuple[str, ...] = ()
    candidate_scores_cp: tuple[float, ...] = ()
    candidate_blunder_losses_cp: tuple[float, ...] = ()
    principal_variation: tuple[str, ...] = ()
    phase: str = "middlegame"
    source: str = "annotated"
    opening_tags: tuple[str, ...] = ()
    motif_tags: tuple[str, ...] = ()
    result: str | None = None
    bucket: str = "middlegames"


def _record_hash(*parts: str) -> int:
    material = "::".join(parts)
    return int(hashlib.sha256(material.encode("utf-8")).hexdigest()[:16], 16)


def _deterministic_select(records: list[ChessCurriculumRecord], max_samples: int) -> list[ChessCurriculumRecord]:
    if len(records) <= max_samples:
        return records
    ranked = sorted(
        records,
        key=lambda record: _record_hash(record.bucket, record.source, record.fen, record.best_move),
    )
    return ranked[:max_samples]


def _record_sort_key(record: ChessCurriculumRecord) -> int:
    return _record_hash(record.bucket, record.source, record.fen, record.best_move)


def _push_limited_record(
    heap: list[tuple[int, int, ChessCurriculumRecord]],
    record: ChessCurriculumRecord,
    *,
    max_samples: int,
    sequence: int,
) -> None:
    """Keep a deterministic bounded sample without retaining full source files."""
    key = _record_sort_key(record)
    item = (-key, sequence, record)
    if len(heap) < max_samples:
        heapq.heappush(heap, item)
        return
    if item > heap[0]:
        heapq.heapreplace(heap, item)


def _phase_label_for_board(board: chess.Board) -> str:
    return ("opening", "middlegame", "endgame")[_phase_index(board)]


def _normalize_phase(value: str | None, board: chess.Board) -> str:
    if value:
        lowered = value.strip().lower()
        if lowered in {"opening", "middlegame", "endgame"}:
            return lowered
    return _phase_label_for_board(board)


def _bucket_for_phase(phase: str) -> str:
    if phase == "opening":
        return "openings"
    if phase == "endgame":
        return "endgames"
    return "middlegames"


def _uci_action_id(board: chess.Board, move_uci: str) -> int:
    move = chess.Move.from_uci(move_uci)
    if board.turn == chess.WHITE:
        return _action_from_move(move)
    return _action_from_move(_mirror_move(move))


@contextmanager
def _open_text(path: Path) -> Iterator[Any]:
    if path.suffix != ".zst":
        with path.open("r", encoding="utf-8", newline="") as handle:
            yield handle
        return

    try:
        import zstandard as zstd  # type: ignore

        with path.open("rb") as raw_handle:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(raw_handle) as reader:
                with TextIOWrapper(reader, encoding="utf-8") as text_handle:
                    yield text_handle
                    return
    except ImportError:
        pass

    if shutil.which("zstd") is None:
        raise RuntimeError(
            f"Cannot read compressed file '{path}'. Install zstandard or ensure the 'zstd' CLI is available."
        )

    process = subprocess.Popen(
        ["zstd", "-dc", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdout is not None
    try:
        yield process.stdout
    finally:
        process.stdout.close()
        stderr = process.stderr.read() if process.stderr is not None else ""
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"zstd failed to read '{path}': {stderr.strip()}")


def _split_from_identifier(identifier: str) -> str:
    return "val" if _record_hash(identifier) % 5 == 0 else "train"


def _discover_first(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            yield __import__("json").loads(line)


def _iter_puzzle_rows(path: Path) -> Iterator[dict[str, str]]:
    with _open_text(path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {str(key): str(value) for key, value in row.items()}


def _iter_pgn_games(path: Path) -> Iterator[chess.pgn.Game]:
    with _open_text(path) as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            yield game


def _normalize_annotated_record(record: dict[str, Any]) -> ChessCurriculumRecord | None:
    fen = record.get("fen")
    best_move = record.get("best_move")
    if not isinstance(fen, str) or not fen.strip() or not isinstance(best_move, str) or not best_move.strip():
        return None
    board = chess.Board(fen.strip())
    phase = _normalize_phase(record.get("phase"), board)
    candidate_moves = tuple(str(move).strip() for move in record.get("candidate_moves", []) if str(move).strip())
    candidate_scores = tuple(float(score) for score in record.get("candidate_scores_cp", [])[: len(candidate_moves)])
    candidate_blunders = tuple(
        float(loss) for loss in record.get("candidate_blunder_losses_cp", [])[: len(candidate_moves)]
    )
    pv = tuple(str(move).strip() for move in record.get("principal_variation", []) if str(move).strip())
    context_moves = tuple(str(move).strip() for move in record.get("moves", []) if str(move).strip())
    opening_tags = tuple(str(tag).strip() for tag in record.get("opening_tags", []) if str(tag).strip())
    source = str(record.get("source", "annotated"))
    return ChessCurriculumRecord(
        fen=fen.strip(),
        best_move=best_move.strip(),
        context_moves=context_moves,
        candidate_moves=candidate_moves,
        candidate_scores_cp=candidate_scores,
        candidate_blunder_losses_cp=candidate_blunders,
        principal_variation=pv,
        phase=phase,
        source=source,
        opening_tags=opening_tags,
        motif_tags=(),
        result=(str(record["result"]) if record.get("result") is not None else None),
        bucket=_bucket_for_phase(phase),
    )


def _normalize_puzzle_row(row: dict[str, str]) -> ChessCurriculumRecord | None:
    fen = row.get("FEN", "").strip()
    moves = [move for move in row.get("Moves", "").split() if move]
    if not fen or len(moves) < 2:
        return None
    board = chess.Board(fen)
    try:
        board.push_uci(moves[0])
    except ValueError:
        return None
    themes = tuple(theme.strip() for theme in row.get("Themes", "").split() if theme.strip())
    opening_tags = tuple(tag.strip() for tag in row.get("OpeningTags", "").split() if tag.strip())
    if "endgame" in themes:
        phase = "endgame"
    elif "opening" in themes:
        phase = "opening"
    elif "middlegame" in themes:
        phase = "middlegame"
    else:
        phase = _phase_label_for_board(board)
    bucket = "puzzles"
    if phase == "endgame":
        bucket = "endgames"
    elif phase == "opening":
        bucket = "openings"
    return ChessCurriculumRecord(
        fen=board.fen(),
        best_move=moves[1],
        context_moves=(moves[0],),
        candidate_moves=(moves[1],),
        candidate_scores_cp=(100.0,),
        candidate_blunder_losses_cp=(0.0,),
        principal_variation=tuple(moves[1:]),
        phase=phase,
        source="puzzle",
        opening_tags=opening_tags,
        motif_tags=themes,
        result=None,
        bucket=bucket,
    )


def _normalize_eval_record(record: dict[str, Any]) -> ChessCurriculumRecord | None:
    fen = record.get("fen")
    evals = record.get("evals")
    if not isinstance(fen, str) or not isinstance(evals, list) or not evals:
        return None
    best_eval = max(
        (entry for entry in evals if isinstance(entry, dict) and isinstance(entry.get("pvs"), list)),
        key=lambda entry: int(entry.get("depth", 0)),
        default=None,
    )
    if best_eval is None:
        return None
    pvs = best_eval.get("pvs") or []
    candidate_moves: list[str] = []
    candidate_scores: list[float] = []
    principal_variation: list[str] = []
    for pv_entry in pvs[:MAX_CANDIDATE_MOVES]:
        if not isinstance(pv_entry, dict):
            continue
        line = str(pv_entry.get("line", "")).split()
        if not line:
            continue
        candidate_moves.append(line[0])
        candidate_scores.append(float(pv_entry.get("cp", 0.0)))
        if not principal_variation:
            principal_variation = line
    if not candidate_moves:
        return None
    board = chess.Board(fen)
    phase = _phase_label_for_board(board)
    return ChessCurriculumRecord(
        fen=fen.strip(),
        best_move=candidate_moves[0],
        context_moves=(),
        candidate_moves=tuple(candidate_moves),
        candidate_scores_cp=tuple(candidate_scores),
        candidate_blunder_losses_cp=tuple(max(0.0, candidate_scores[0] - score) for score in candidate_scores),
        principal_variation=tuple(principal_variation),
        phase=phase,
        source="eval",
        opening_tags=(),
        motif_tags=(),
        result=None,
        bucket=_bucket_for_phase(phase),
    )


def _normalize_game_records(path: Path, *, bucket: str) -> dict[str, list[ChessCurriculumRecord]]:
    split_records: dict[str, list[ChessCurriculumRecord]] = defaultdict(list)
    for game_index, game in enumerate(_iter_pgn_games(path)):
        board = game.board()
        moves = list(game.mainline_moves())
        if not moves:
            continue
        headers = game.headers
        result = headers.get("Result")
        max_ply = min(len(moves), 24 if bucket == "short_games" else len(moves))
        for ply_idx, move in enumerate(moves[:max_ply]):
            fen = board.fen()
            phase = _phase_label_for_board(board)
            move_uci = move.uci()
            split = _split_from_identifier(f"{path.name}:{game_index}:{ply_idx}:{fen}")
            split_records[split].append(
                ChessCurriculumRecord(
                    fen=fen,
                    best_move=move_uci,
                    context_moves=tuple(item.uci() for item in moves[:ply_idx]),
                    candidate_moves=(move_uci,),
                    candidate_scores_cp=(25.0,),
                    candidate_blunder_losses_cp=(0.0,),
                    principal_variation=(move_uci,),
                    phase=phase,
                    source="short_game" if bucket == "short_games" else "full_game",
                    opening_tags=(),
                    motif_tags=(),
                    result=result,
                    bucket=bucket,
                )
            )
            board.push(move)
    return split_records


def load_normalized_chess_records(
    root: str | Path,
    *,
    max_samples_per_bucket: int = 4096,
) -> dict[str, dict[str, list[ChessCurriculumRecord]]]:
    """Load and bucket chess curriculum records from local data files."""
    root_path = Path(root)
    heaps: dict[str, dict[str, list[tuple[int, int, ChessCurriculumRecord]]]] = {
        bucket: {"train": [], "val": []} for bucket in CHESS_BUCKETS
    }
    sequence = 0

    def add_record(bucket: str, split_name: str, record: ChessCurriculumRecord) -> None:
        nonlocal sequence
        sequence += 1
        _push_limited_record(
            heaps[bucket][split_name],
            record,
            max_samples=max_samples_per_bucket,
            sequence=sequence,
        )

    for split_name, candidate_names in (
        ("train", ("train.jsonl",)),
        ("val", ("validation.jsonl", "val.jsonl")),
    ):
        annotated_path = _discover_first(root_path, candidate_names)
        if annotated_path is not None:
            for raw_record in _iter_jsonl(annotated_path):
                normalized = _normalize_annotated_record(raw_record)
                if normalized is None:
                    continue
                add_record(_bucket_for_phase(normalized.phase), split_name, normalized)

    puzzle_paths = {
        "train": _discover_first(root_path, ("puzzles_train.csv", "train_puzzles.csv")),
        "val": _discover_first(root_path, ("puzzles_validation.csv", "validation_puzzles.csv", "val_puzzles.csv")),
    }
    shared_puzzle_path = _discover_first(root_path, ("lichess_db_puzzle.csv", "puzzles.csv", "lichess_db_puzzle.csv.zst"))
    if shared_puzzle_path is not None and puzzle_paths["train"] is None and puzzle_paths["val"] is None:
        for row in _iter_puzzle_rows(shared_puzzle_path):
            normalized = _normalize_puzzle_row(row)
            if normalized is None:
                continue
            split_name = _split_from_identifier(str(row.get("PuzzleId", normalized.fen)))
            add_record(normalized.bucket, split_name, normalized)
            if normalized.bucket == "middlegames":
                add_record("puzzles", split_name, ChessCurriculumRecord(**{**normalized.__dict__, "bucket": "puzzles"}))
    else:
        for split_name, puzzle_path in puzzle_paths.items():
            if puzzle_path is None:
                continue
            for row in _iter_puzzle_rows(puzzle_path):
                normalized = _normalize_puzzle_row(row)
                if normalized is None:
                    continue
                add_record(normalized.bucket, split_name, normalized)
                if normalized.bucket == "middlegames":
                    add_record("puzzles", split_name, ChessCurriculumRecord(**{**normalized.__dict__, "bucket": "puzzles"}))

    eval_path = _discover_first(root_path, ("lichess_db_eval.jsonl", "eval.jsonl"))
    if eval_path is not None:
        for raw_record in _iter_jsonl(eval_path):
            normalized = _normalize_eval_record(raw_record)
            if normalized is None:
                continue
            split_name = _split_from_identifier(normalized.fen)
            add_record(_bucket_for_phase(normalized.phase), split_name, normalized)

    for bucket, names in (
        (
            "short_games",
            (
                "train_short.pgn",
                "short_games.pgn",
                "train.pgn",
                "train_short.pgn.zst",
                "short_games.pgn.zst",
                "lichess_db_standard_rated_2013-01.pgn.zst",
            ),
        ),
        (
            "full_games",
            (
                "train_full.pgn",
                "full_games.pgn",
                "train.pgn",
                "train_full.pgn.zst",
                "full_games.pgn.zst",
                "lichess_db_standard_rated_2013-01.pgn.zst",
            ),
        ),
    ):
        path = _discover_first(root_path, names)
        if path is None:
            continue
        split_records = _normalize_game_records(path, bucket=bucket)
        for split_name, records in split_records.items():
            for record in records:
                add_record(bucket, split_name, record)

    buckets: dict[str, dict[str, list[ChessCurriculumRecord]]] = {
        bucket: {"train": [], "val": []} for bucket in CHESS_BUCKETS
    }
    for bucket in CHESS_BUCKETS:
        for split_name in ("train", "val"):
            selected = [item[2] for item in heaps[bucket][split_name]]
            buckets[bucket][split_name] = _deterministic_select(selected, max_samples_per_bucket)
    return buckets


def curriculum_samples_from_records(
    records: list[ChessCurriculumRecord],
    *,
    seq_len: int,
) -> list[dict[str, Any]]:
    """Convert normalized chess records into curriculum sample dictionaries."""
    samples: list[dict[str, Any]] = []
    for record in records:
        board = chess.Board(record.fen)
        history_block = torch.zeros(8, 8, _HISTORY_PLANES, dtype=torch.float32)
        observation, history_block = _observation_from_board(board, history_block)
        try:
            answer_token = _uci_action_id(board, record.best_move)
        except ValueError:
            continue
        candidate_ids = torch.zeros(MAX_CANDIDATE_MOVES, dtype=torch.long)
        candidate_weights = torch.zeros(MAX_CANDIDATE_MOVES, dtype=torch.float32)
        candidate_mask = torch.zeros(MAX_CANDIDATE_MOVES, dtype=torch.bool)
        candidate_blunder = torch.zeros(MAX_CANDIDATE_MOVES, dtype=torch.float32)
        valid_candidates = []
        for move in record.candidate_moves[:MAX_CANDIDATE_MOVES]:
            try:
                valid_candidates.append(_uci_action_id(board, move))
            except ValueError:
                continue
        if valid_candidates:
            count = min(len(valid_candidates), MAX_CANDIDATE_MOVES)
            candidate_ids[:count] = torch.tensor(valid_candidates[:count], dtype=torch.long)
            candidate_mask[:count] = True
            if record.candidate_scores_cp:
                scores = torch.tensor(record.candidate_scores_cp[:count], dtype=torch.float32)
                candidate_weights[:count] = torch.softmax(scores / 100.0, dim=0)
            else:
                candidate_weights[:count] = 1.0 / float(count)
            if record.candidate_blunder_losses_cp:
                losses = torch.tensor(record.candidate_blunder_losses_cp[:count], dtype=torch.float32)
                candidate_blunder[:count] = losses
        else:
            candidate_ids[0] = answer_token
            candidate_weights[0] = 1.0
            candidate_mask[0] = True

        pv_ids = torch.zeros(seq_len, dtype=torch.long)
        for idx, move in enumerate(record.principal_variation[:seq_len]):
            try:
                pv_ids[idx] = _uci_action_id(board, move)
            except ValueError:
                break

        samples.append(
            {
                "observation": observation,
                "history_block": history_block,
                "answer": torch.tensor(answer_token, dtype=torch.long),
                "target": torch.tensor([answer_token], dtype=torch.long),
                "candidate_move_tokens": candidate_ids,
                "candidate_move_weights": candidate_weights,
                "candidate_move_mask": candidate_mask,
                "candidate_move_blunder_cp": candidate_blunder,
                "principal_variation_tokens": pv_ids,
                "centipawn_eval": torch.tensor(
                    float(record.candidate_scores_cp[0] if record.candidate_scores_cp else 0.0),
                    dtype=torch.float32,
                ),
                "fen": record.fen,
                "phase": record.phase,
                "source": record.source,
                "bucket": record.bucket,
                "moves": list(record.context_moves),
                "best_move": record.best_move,
                "opening_tags": list(record.opening_tags),
                "motif_tags": list(record.motif_tags),
                "result": record.result if record.result is not None else "*",
                "regime": torch.tensor(float(record.phase == "endgame"), dtype=torch.float32),
            }
        )
    return samples


def load_chess_curriculum_samples(
    root: str | Path,
    *,
    bucket: str,
    split: str,
    seq_len: int,
    max_samples: int,
) -> list[dict[str, Any]]:
    """Load one bucket worth of normalized chess curriculum samples."""
    if bucket not in CHESS_BUCKETS:
        raise ValueError(f"Unsupported chess curriculum bucket '{bucket}'.")
    split_name = "val" if split == "val" else "train"
    grouped = load_normalized_chess_records(root, max_samples_per_bucket=max_samples)
    records = grouped[bucket][split_name]
    return curriculum_samples_from_records(records, seq_len=seq_len)

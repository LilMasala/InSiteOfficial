"""Public-data helpers for Stage 3 game curriculum domains."""

from __future__ import annotations

from pathlib import Path
import re

import torch

from training.curriculum.data.public_sequence_data import (
    PROJECT_ROOT,
    candidate_files,
    read_records,
    read_text_blob,
    sequence_sample,
    split_samples,
    token_id_for_text,
)


DEFAULT_CURRICULUM_ROOT = PROJECT_ROOT / "data" / "curriculum"
RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
MAX_ENGINE_CANDIDATES = 8


def _sequence_to_sample(
    items: list[str],
    *,
    vocab_size: int,
    seq_len: int,
    regime: float = 0.0,
) -> dict[str, torch.Tensor] | None:
    """Encode a move/action sequence into one curriculum sample."""
    token_ids = [token_id_for_text(item, vocab_size) for item in items if item]
    return sequence_sample(token_ids, seq_len, regime=regime)


def _default_engine_annotation(seq_len: int) -> dict[str, torch.Tensor]:
    """Return zero-valued engine metadata so collated batches stay shape-stable."""
    return {
        "candidate_move_tokens": torch.zeros(MAX_ENGINE_CANDIDATES, dtype=torch.long),
        "candidate_move_weights": torch.zeros(MAX_ENGINE_CANDIDATES, dtype=torch.float32),
        "candidate_move_mask": torch.zeros(MAX_ENGINE_CANDIDATES, dtype=torch.bool),
        "candidate_move_blunder_cp": torch.zeros(MAX_ENGINE_CANDIDATES, dtype=torch.float32),
        "principal_variation_tokens": torch.zeros(seq_len, dtype=torch.long),
        "centipawn_eval": torch.tensor(0.0, dtype=torch.float32),
    }


def _candidate_annotation_from_record(
    record: dict,
    *,
    vocab_size: int,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    """Extract engine-ranked move metadata from a structured chess record."""
    annotation = _default_engine_annotation(seq_len)

    raw_candidates = record.get("candidate_moves") or record.get("top_moves") or record.get("engine_moves") or []
    candidate_moves: list[str] = []
    candidate_scores: list[float] = []
    candidate_blunders: list[float] = []

    if isinstance(raw_candidates, list):
        for entry in raw_candidates[:MAX_ENGINE_CANDIDATES]:
            if isinstance(entry, str) and entry.strip():
                candidate_moves.append(entry.strip())
                continue
            if isinstance(entry, dict):
                move = entry.get("move") or entry.get("uci") or entry.get("san")
                if isinstance(move, str) and move.strip():
                    candidate_moves.append(move.strip())
                    score = entry.get("score_cp", entry.get("centipawn_eval"))
                    if score is not None:
                        candidate_scores.append(float(score))
                    loss = entry.get("blunder_loss_cp", entry.get("loss_cp"))
                    if loss is not None:
                        candidate_blunders.append(float(loss))

    raw_scores = record.get("candidate_scores_cp") or record.get("top_move_scores_cp") or record.get("engine_scores_cp")
    if not candidate_scores and isinstance(raw_scores, list):
        candidate_scores = [float(score) for score in raw_scores[: len(candidate_moves)]]

    raw_blunders = (
        record.get("candidate_blunder_losses_cp")
        or record.get("top_move_blunder_losses_cp")
        or record.get("engine_blunder_losses_cp")
    )
    if not candidate_blunders and isinstance(raw_blunders, list):
        candidate_blunders = [float(loss) for loss in raw_blunders[: len(candidate_moves)]]

    if candidate_moves:
        count = min(len(candidate_moves), MAX_ENGINE_CANDIDATES)
        annotation["candidate_move_tokens"][:count] = torch.tensor(
            [token_id_for_text(move, vocab_size) for move in candidate_moves[:count]],
            dtype=torch.long,
        )
        annotation["candidate_move_mask"][:count] = True
        if candidate_scores:
            weights = torch.softmax(torch.tensor(candidate_scores[:count], dtype=torch.float32) / 100.0, dim=0)
            annotation["candidate_move_weights"][:count] = weights
        else:
            annotation["candidate_move_weights"][:count] = 1.0 / float(count)
        if candidate_blunders:
            annotation["candidate_move_blunder_cp"][:count] = torch.tensor(
                candidate_blunders[:count],
                dtype=torch.float32,
            )

    pv = record.get("principal_variation") or record.get("pv") or []
    pv_moves: list[str] = []
    if isinstance(pv, str):
        pv_moves = [move for move in pv.split() if move]
    elif isinstance(pv, list):
        pv_moves = [str(move).strip() for move in pv if str(move).strip()]
    if pv_moves:
        annotation["principal_variation_tokens"][: min(len(pv_moves), seq_len)] = torch.tensor(
            [token_id_for_text(move, vocab_size) for move in pv_moves[:seq_len]],
            dtype=torch.long,
        )

    try:
        annotation["centipawn_eval"] = torch.tensor(float(record.get("centipawn_eval", record.get("score_cp", 0.0))))
    except (TypeError, ValueError):
        pass
    return annotation


def _strip_pgn_noise(text: str) -> str:
    """Remove PGN comments, annotations, and headers."""
    text = re.sub(r"\{[^}]*\}", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\$\d+", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return text


def _moves_from_pgn(text: str) -> list[list[str]]:
    """Parse PGN text into per-game SAN move sequences."""
    cleaned = _strip_pgn_noise(text)
    blocks = [block.strip() for block in re.split(r"\n\s*\n", cleaned) if block.strip()]
    games: list[list[str]] = []
    for block in blocks:
        compact = re.sub(r"\d+\.(?:\.\.)?", " ", block)
        tokens = [token.strip() for token in compact.split() if token.strip() and token.strip() not in RESULT_TOKENS]
        if len(tokens) >= 2:
            games.append(tokens)
    if games:
        return games
    compact = re.sub(r"\d+\.(?:\.\.)?", " ", cleaned)
    tokens = [token.strip() for token in compact.split() if token.strip() and token.strip() not in RESULT_TOKENS]
    return [tokens] if len(tokens) >= 2 else []


def _moves_from_sgf(text: str) -> list[list[str]]:
    """Parse SGF text into per-game move sequences."""
    moves = re.findall(r";([BW])\[([^\]]*)\]", text)
    if len(moves) < 2:
        return []
    return [[f"{color}:{coord or 'pass'}" for color, coord in moves]]


def _record_to_sequence(record: dict, *, vocab_size: int, seq_len: int) -> dict | tuple[list[str], float] | None:
    """Extract a structured move/action sequence from one record.

    Chess records use ``moves`` (full game move list) as context and
    ``best_move`` as the answer to predict. The annotation block carries
    candidate move scores from the engine.
    """
    # Chess: moves is the game context, best_move is the supervised answer
    best_move = record.get("best_move")
    moves_list = record.get("moves")
    if isinstance(moves_list, list) and moves_list and isinstance(best_move, str) and best_move.strip():
        items = [str(m).strip() for m in moves_list if str(m).strip()]
        if not items:
            items = [best_move.strip()]
        regime = float("win" in str(record.get("result", "")).lower())
        annotation = _candidate_annotation_from_record(record, vocab_size=vocab_size, seq_len=seq_len)
        # Override answer token to be best_move specifically
        annotation["best_move_token"] = torch.tensor(
            token_id_for_text(best_move.strip(), vocab_size), dtype=torch.long
        )
        return {"items": items, "regime": regime, "annotation": annotation}

    for key in ("actions", "trajectory", "events"):
        value = record.get(key)
        if isinstance(value, list) and value:
            items: list[str] = []
            for entry in value:
                if isinstance(entry, str):
                    items.append(entry.strip())
                elif isinstance(entry, dict):
                    for nested_key in ("move", "action", "state", "text"):
                        nested = entry.get(nested_key)
                        if isinstance(nested, str) and nested.strip():
                            items.append(nested.strip())
                            break
            if len(items) >= 2:
                regime = float("win" in str(record.get("result", "")).lower())
                return {
                    "items": items,
                    "regime": regime,
                    "annotation": _candidate_annotation_from_record(record, vocab_size=vocab_size, seq_len=seq_len),
                }

    for key in ("pgn", "sgf", "record"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            if key == "sgf":
                parsed = _moves_from_sgf(value)
            else:
                parsed = _moves_from_pgn(value)
            if parsed:
                return parsed[0], float("1-0" in value)
    return None


def _stage3_root(domain_variant: str, root: Path) -> Path:
    """Return the local Stage-3 dataset root for one domain."""
    stage3_root = root / "stage3"
    if domain_variant == "chess":
        return stage3_root / "chess" / "lichess"
    if domain_variant == "go":
        return stage3_root / "go"
    if domain_variant in {"poker", "gridworld"}:
        return stage3_root / domain_variant
    return stage3_root / domain_variant


def load_public_game_samples(
    domain_variant: str,
    split: str,
    vocab_size: int,
    seq_len: int,
    max_samples: int,
    data_root: Path | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Load local public Stage-3 game samples when available."""
    root = data_root or DEFAULT_CURRICULUM_ROOT
    dataset_dir = _stage3_root(domain_variant, root)
    candidate_paths = candidate_files(dataset_dir, split=split)
    if not candidate_paths:
        return []

    samples: list[dict[str, torch.Tensor]] = []
    cap = max_samples * 2 if split != "train" else max_samples
    for path in candidate_paths:
        parsed_sequences: list[dict | tuple[list[str], float]] = []
        name = path.name.lower()
        if any(name.endswith(suffix) for suffix in (".json", ".jsonl", ".csv", ".parquet", ".json.zst", ".jsonl.zst", ".csv.zst", ".parquet.zst")):
            for record in read_records(path):
                sequence = _record_to_sequence(record, vocab_size=vocab_size, seq_len=seq_len)
                if sequence is not None:
                    parsed_sequences.append(sequence)
        else:
            blob = read_text_blob(path)
            if name.endswith(".sgf") or name.endswith(".sgf.zst"):
                parsed_sequences.extend((moves, 0.0) for moves in _moves_from_sgf(blob))
            else:
                parsed_sequences.extend((moves, 0.0) for moves in _moves_from_pgn(blob))

        for entry in parsed_sequences:
            if isinstance(entry, tuple):
                items, regime = entry
                annotation = _default_engine_annotation(seq_len)
            else:
                items = entry["items"]
                regime = entry["regime"]
                annotation = entry["annotation"]
            sample = _sequence_to_sample(items, vocab_size=vocab_size, seq_len=seq_len, regime=regime)
            if sample is None:
                continue
            sample.update(annotation)
            samples.append(sample)
            if len(samples) >= cap:
                break
        if len(samples) >= cap:
            break

    return split_samples(samples, split=split, candidate_paths=candidate_paths)[:max_samples]

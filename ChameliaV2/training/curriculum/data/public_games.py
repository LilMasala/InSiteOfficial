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


def _record_to_sequence(record: dict) -> tuple[list[str], float] | None:
    """Extract a structured move/action sequence from one record."""
    for key in ("moves", "actions", "trajectory", "events"):
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
                return items, regime

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
        parsed_sequences: list[tuple[list[str], float]] = []
        name = path.name.lower()
        if any(name.endswith(suffix) for suffix in (".json", ".jsonl", ".csv", ".parquet", ".json.zst", ".jsonl.zst", ".csv.zst", ".parquet.zst")):
            for record in read_records(path):
                sequence = _record_to_sequence(record)
                if sequence is not None:
                    parsed_sequences.append(sequence)
        else:
            blob = read_text_blob(path)
            if name.endswith(".sgf") or name.endswith(".sgf.zst"):
                parsed_sequences.extend((moves, 0.0) for moves in _moves_from_sgf(blob))
            else:
                parsed_sequences.extend((moves, 0.0) for moves in _moves_from_pgn(blob))

        for items, regime in parsed_sequences:
            sample = _sequence_to_sample(items, vocab_size=vocab_size, seq_len=seq_len, regime=regime)
            if sample is None:
                continue
            samples.append(sample)
            if len(samples) >= cap:
                break
        if len(samples) >= cap:
            break

    return split_samples(samples, split=split, candidate_paths=candidate_paths)[:max_samples]

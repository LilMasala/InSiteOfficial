"""Shared helpers for local public curriculum sequence datasets."""

from __future__ import annotations

from collections.abc import Iterable
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch

from training.curriculum.data.preprocessors import holdout_split, normalize_whitespace

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:  # pragma: no cover - optional in minimal local envs
    hf_load_dataset = None

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover - optional when reading compressed corpora
    zstd = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("validation", "valid", "val", "dev", "test", "eval", "evaluation"),
    "test": ("test", "eval", "evaluation"),
}
TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".xml",
    ".html",
    ".htm",
    ".pgn",
    ".sgf",
    ".log",
}
STRUCTURED_SUFFIXES = {".json", ".jsonl", ".csv", ".parquet"}
COMPRESSED_SUFFIXES = {".zst"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | STRUCTURED_SUFFIXES | COMPRESSED_SUFFIXES


def token_id_for_text(text: str, vocab_size: int) -> int:
    """Hash one text fragment into a stable token id."""
    normalized = normalize_whitespace(text).lower() or "<empty>"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return 1 + (int(digest[:8], 16) % max(1, vocab_size - 1))


def tokenize_text(text: str, vocab_size: int) -> list[int]:
    """Tokenize free text into hashed ids."""
    normalized = normalize_whitespace(text).lower()
    pieces = re.findall(r"[a-z0-9_]+|[^\w\s]", normalized)
    return [token_id_for_text(piece, vocab_size) for piece in pieces]


def pad_token_ids(token_ids: list[int], seq_len: int) -> torch.Tensor:
    """Pad a token-id sequence to ``seq_len``."""
    output = torch.zeros(seq_len, dtype=torch.long)
    if token_ids:
        truncated = token_ids[:seq_len]
        output[: len(truncated)] = torch.tensor(truncated, dtype=torch.long)
    return output


def sequence_sample(
    token_ids: list[int],
    seq_len: int,
    *,
    regime: float = 0.0,
) -> dict[str, torch.Tensor] | None:
    """Build one shifted-sequence supervision sample."""
    if len(token_ids) < 2:
        return None
    input_values = token_ids[:seq_len]
    target_values = token_ids[1 : seq_len + 1]
    if not target_values:
        return None
    answer = target_values[min(len(target_values), seq_len) - 1]
    return {
        "tokens": pad_token_ids(input_values, seq_len),
        "target": pad_token_ids(target_values, seq_len),
        "answer": torch.tensor(answer, dtype=torch.long),
        "regime": torch.tensor(float(regime), dtype=torch.float32),
    }


def split_matches(path: Path, split: str) -> bool:
    """Return whether a path name strongly suggests the requested split."""
    aliases = SPLIT_ALIASES.get(split, (split,))
    lowered = str(path).lower()
    return any(alias in lowered for alias in aliases)


def _supported_path(path: Path) -> bool:
    """Return whether a path looks like a supported data file."""
    lowered = path.name.lower()
    if path.suffix.lower() in TEXT_SUFFIXES | STRUCTURED_SUFFIXES:
        return True
    return any(lowered.endswith(f"{suffix}.zst") for suffix in TEXT_SUFFIXES | STRUCTURED_SUFFIXES)


def candidate_files(dataset_dir: Path, split: str, keywords: Iterable[str] = ()) -> list[Path]:
    """List candidate raw data files for one dataset directory."""
    if not dataset_dir.exists():
        return []
    files = sorted(path for path in dataset_dir.rglob("*") if path.is_file() and _supported_path(path))
    keyword_tuple = tuple(keywords)
    if keyword_tuple:
        files = [path for path in files if any(keyword in str(path).lower() for keyword in keyword_tuple)]
    if not files:
        return []
    split_specific = [path for path in files if split_matches(path, split)]
    return split_specific or files


def _flatten_json_payload(payload: Any) -> list[dict[str, Any]]:
    """Flatten nested JSON payloads into a record list."""
    if isinstance(payload, list):
        records: list[dict[str, Any]] = []
        for item in payload:
            records.extend(_flatten_json_payload(item))
        return records
    if isinstance(payload, dict):
        split_like = {"train", "validation", "valid", "val", "test", "dev"}
        if set(payload.keys()).issubset(split_like):
            records: list[dict[str, Any]] = []
            for value in payload.values():
                records.extend(_flatten_json_payload(value))
            return records
        return [payload]
    return []


def _decompress_zst_bytes(path: Path) -> bytes:
    """Read and decompress a Zstandard-compressed file."""
    if zstd is None:
        return b""
    with path.open("rb") as handle:
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(handle.read())


def read_text_blob(path: Path) -> str:
    """Read one text-like file, handling optional Zstandard compression."""
    if path.suffix.lower() == ".zst":
        return _decompress_zst_bytes(path).decode("utf-8", errors="ignore")
    return path.read_text(errors="ignore")


def read_json_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSON or JSONL-like path into record dictionaries."""
    name = path.name.lower()
    if name.endswith(".jsonl") or name.endswith(".jsonl.zst"):
        records: list[dict[str, Any]] = []
        for line in read_text_blob(path).splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            records.extend(_flatten_json_payload(json.loads(stripped)))
        return records
    return _flatten_json_payload(json.loads(read_text_blob(path)))


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    """Read a CSV path into record dictionaries."""
    text = read_text_blob(path)
    rows = csv.DictReader(text.splitlines())
    return [dict(row) for row in rows]


def read_parquet_records(path: Path) -> list[dict[str, Any]]:
    """Read a parquet path via Hugging Face datasets when available."""
    if hf_load_dataset is None:
        return []
    dataset = hf_load_dataset("parquet", data_files=str(path), split="train")
    return [dict(row) for row in dataset]


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read one supported structured file into record dictionaries."""
    name = path.name.lower()
    if name.endswith(".json") or name.endswith(".json.zst") or name.endswith(".jsonl") or name.endswith(".jsonl.zst"):
        return read_json_records(path)
    if name.endswith(".csv") or name.endswith(".csv.zst"):
        return read_csv_records(path)
    if name.endswith(".parquet") or name.endswith(".parquet.zst"):
        return read_parquet_records(path)
    return []


def text_segments(text: str, *, min_tokens: int = 8) -> list[str]:
    """Split a raw text blob into coarse text segments."""
    normalized = normalize_whitespace(text)
    if not normalized:
        return []
    parts = re.split(r"(?:\n\s*\n|(?<=[.!?])\s{2,})", normalized)
    segments = [part.strip() for part in parts if len(tokenize_text(part, vocab_size=1024)) >= min_tokens]
    return segments or ([normalized] if len(tokenize_text(normalized, vocab_size=1024)) >= min_tokens else [])


def split_samples(
    samples: list[dict[str, torch.Tensor]],
    split: str,
    candidate_paths: list[Path],
) -> list[dict[str, torch.Tensor]]:
    """Apply deterministic holdout splitting when explicit split files are absent."""
    if split == "train":
        return samples
    if candidate_paths and any(split_matches(path, split) for path in candidate_paths):
        return samples
    train_samples, heldout_samples = holdout_split(samples, fraction=0.2)
    if split in {"val", "test"}:
        return heldout_samples
    return train_samples

"""Lightweight preprocessors for curriculum data staging."""

from __future__ import annotations

from typing import TypeVar


T = TypeVar("T")


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace in a text sample."""
    return " ".join(text.split())


def chunk_text(text: str, max_chars: int = 512) -> list[str]:
    """Chunk long text into fixed-size character windows."""
    normalized = normalize_whitespace(text)
    return [normalized[index : index + max_chars] for index in range(0, len(normalized), max_chars)]


def holdout_split(items: list[T], fraction: float = 0.2) -> tuple[list[T], list[T]]:
    """Create a deterministic train/held-out split."""
    cutoff = int(len(items) * (1.0 - fraction))
    return items[:cutoff], items[cutoff:]

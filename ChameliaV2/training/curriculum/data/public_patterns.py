"""Public-data helpers for Stage 2 pattern and structure domains."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import torch

from training.curriculum.data.preprocessors import holdout_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CURRICULUM_ROOT = PROJECT_ROOT / "data" / "curriculum"


def _pad_sequence(values: list[int], seq_len: int) -> torch.Tensor:
    """Pad an integer token sequence to fixed length.

    Args:
        values: Raw integer tokens.
        seq_len: Output length N.

    Returns:
        Tensor of shape ``[N]``.
    """
    output = torch.zeros(seq_len, dtype=torch.long)
    truncated = values[:seq_len]
    if truncated:
        output[: len(truncated)] = torch.tensor(truncated, dtype=torch.long)
    return output


def _flatten_grid(grid: list[list[int]]) -> list[int]:
    """Flatten an ARC grid with row separators.

    Args:
        grid: Grid of shape ``[H][W]`` with small integer colors.

    Returns:
        Flat token list.
    """
    flattened: list[int] = []
    for row in grid:
        flattened.extend(32 + int(value) for value in row)
        flattened.append(16)
    return flattened[:-1] if flattened else []


def _arc_json_paths(arc_root: Path, split: str) -> list[Path]:
    """Return candidate ARC task paths.

    Args:
        arc_root: ARC dataset root.
        split: Requested split.

    Returns:
        Sorted JSON file list.
    """
    if not arc_root.exists():
        return []
    paths = sorted(path for path in arc_root.rglob("*.json"))
    if not paths:
        return []
    if split == "train":
        train_paths = [path for path in paths if "train" in str(path).lower()]
        return train_paths or paths
    heldout_paths = [path for path in paths if any(tag in str(path).lower() for tag in ("eval", "test", "validation"))]
    return heldout_paths or paths


def load_arc_samples(
    split: str,
    seq_len: int,
    max_samples: int,
    data_root: Path | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Load ARC train-pair samples from a cloned ARC-AGI repo.

    Args:
        split: Requested split.
        seq_len: Sequence length N.
        max_samples: Maximum sample count.
        data_root: Optional curriculum data root.

    Returns:
        Normalized sample list.
    """
    root = data_root or DEFAULT_CURRICULUM_ROOT
    arc_root = root / "stage2" / "arc_agi_2"
    samples: list[dict[str, torch.Tensor]] = []
    paths = _arc_json_paths(arc_root, split=split)
    for path in paths:
        payload = json.loads(path.read_text())
        for example in payload.get("train", []):
            input_grid = example.get("input")
            output_grid = example.get("output")
            if not isinstance(input_grid, list) or not isinstance(output_grid, list):
                continue
            input_tokens = _flatten_grid(input_grid)
            output_tokens = _flatten_grid(output_grid)
            color_count = len({value for row in input_grid for value in row})
            samples.append(
                {
                    "tokens": _pad_sequence(input_tokens, seq_len),
                    "target": _pad_sequence(output_tokens, seq_len),
                    "regime": torch.tensor(float(color_count), dtype=torch.float32),
                }
            )
    if split in {"val", "test"} and paths and not any(
        any(tag in str(path).lower() for tag in ("eval", "test", "validation")) for path in paths
    ):
        _train_samples, heldout_samples = holdout_split(samples, fraction=0.2)
        return heldout_samples[:max_samples]
    return samples[:max_samples]


def load_oeis_samples(
    split: str,
    seq_len: int,
    max_samples: int,
    data_root: Path | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Load OEIS-like integer sequences from local text files.

    Args:
        split: Requested split.
        seq_len: Sequence length N.
        max_samples: Maximum sample count.
        data_root: Optional curriculum data root.

    Returns:
        Normalized sample list.
    """
    root = data_root or DEFAULT_CURRICULUM_ROOT
    oeis_root = root / "stage2" / "oeis"
    if not oeis_root.exists():
        return []

    records: list[list[int]] = []
    for path in sorted(oeis_root.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(errors="ignore")
        for line in text.splitlines():
            if line.count(",") < 2:
                continue
            matches = re.findall(r"-?\d+", line)
            if len(matches) < 4:
                continue
            records.append([int(value) for value in matches[: max(4, seq_len + 1)]])
            if len(records) >= max_samples * 2:
                break
        if len(records) >= max_samples * 2:
            break

    if not records:
        return []

    train_records, heldout_records = holdout_split(records, fraction=0.2)
    selected = heldout_records if split in {"val", "test"} else train_records
    samples: list[dict[str, torch.Tensor]] = []
    for values in selected[:max_samples]:
        input_values = [256 + (value % 2048) for value in values[:-1]]
        target_values = [256 + (value % 2048) for value in values[1:]]
        samples.append(
            {
                "tokens": _pad_sequence(input_values, seq_len),
                "target": _pad_sequence(target_values, seq_len),
                "regime": torch.tensor(float(len(set(values[:-1]))), dtype=torch.float32),
            }
        )
    return samples


def load_public_pattern_samples(
    domain_variant: str,
    split: str,
    seq_len: int,
    max_samples: int,
    data_root: Path | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Load public Stage-2 samples when available.

    Args:
        domain_variant: Stage-2 domain variant.
        split: Requested split.
        seq_len: Sequence length N.
        max_samples: Maximum sample count.
        data_root: Optional curriculum data root.

    Returns:
        Normalized sample list.
    """
    if domain_variant == "arc_tasks":
        return load_arc_samples(split=split, seq_len=seq_len, max_samples=max_samples, data_root=data_root)
    if domain_variant == "oeis_sequences":
        return load_oeis_samples(split=split, seq_len=seq_len, max_samples=max_samples, data_root=data_root)
    return []

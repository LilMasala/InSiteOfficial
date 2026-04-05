"""Public-data helpers for Stage 4 collaboration curriculum domains."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from training.curriculum.data.public_sequence_data import (
    PROJECT_ROOT,
    candidate_files,
    read_records,
    read_text_blob,
    sequence_sample,
    split_samples,
    token_id_for_text,
    tokenize_text,
)


DEFAULT_CURRICULUM_ROOT = PROJECT_ROOT / "data" / "curriculum"


def _flatten_value(value: Any) -> list[str]:
    """Flatten a nested collaboration payload into string tokens."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        items: list[str] = []
        for entry in value:
            items.extend(_flatten_value(entry))
        return items
    if isinstance(value, dict):
        items: list[str] = []
        for key in ("text", "message", "action", "state", "goal", "observation", "content", "value"):
            if key in value:
                items.extend(_flatten_value(value[key]))
        return items
    return []


def _record_to_sample(record: dict, *, vocab_size: int, seq_len: int) -> dict[str, torch.Tensor] | None:
    """Convert one collaboration trace record into a supervision sample."""
    parts: list[str] = []
    for key in ("task", "goal", "agent_a", "agent_b", "messages", "trajectory", "events", "history"):
        if key in record:
            flattened = _flatten_value(record[key])
            if flattened:
                parts.append(f"{key}: {' '.join(flattened)}")
    if not parts:
        return None

    answer_text = ""
    for key in ("target", "solution", "joint_action", "outcome", "label"):
        value = record.get(key)
        flattened = _flatten_value(value)
        if flattened:
            answer_text = " ".join(flattened)
            break

    token_ids = tokenize_text(" ".join(parts), vocab_size)
    sample = sequence_sample(token_ids, seq_len, regime=float(bool(answer_text)))
    if sample is None:
        return None
    if answer_text:
        answer_token = token_id_for_text(answer_text, vocab_size)
        sample["answer"] = torch.tensor(answer_token, dtype=torch.long)
        sample["target"][-1] = answer_token
    return sample


def _samples_from_text_blob(
    text: str,
    *,
    vocab_size: int,
    seq_len: int,
    cap: int,
) -> list[dict[str, torch.Tensor]]:
    """Build collaboration samples from raw text transcripts."""
    samples: list[dict[str, torch.Tensor]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        sample = sequence_sample(tokenize_text(stripped, vocab_size), seq_len, regime=0.0)
        if sample is None:
            continue
        samples.append(sample)
        if len(samples) >= cap:
            return samples
    return samples


def load_public_collaboration_samples(
    split: str,
    vocab_size: int,
    seq_len: int,
    max_samples: int,
    data_root: Path | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Load local Stage-4 collaboration traces when available."""
    root = data_root or DEFAULT_CURRICULUM_ROOT
    dataset_dir = root / "stage4" / "generated"
    candidate_paths = candidate_files(dataset_dir, split=split)
    if not candidate_paths:
        return []

    samples: list[dict[str, torch.Tensor]] = []
    cap = max_samples * 2 if split != "train" else max_samples
    for path in candidate_paths:
        name = path.name.lower()
        if any(name.endswith(suffix) for suffix in (".json", ".jsonl", ".csv", ".parquet", ".json.zst", ".jsonl.zst", ".csv.zst", ".parquet.zst")):
            for record in read_records(path):
                sample = _record_to_sample(record, vocab_size=vocab_size, seq_len=seq_len)
                if sample is None:
                    continue
                samples.append(sample)
                if len(samples) >= cap:
                    break
        else:
            samples.extend(
                _samples_from_text_blob(
                    read_text_blob(path),
                    vocab_size=vocab_size,
                    seq_len=seq_len,
                    cap=cap - len(samples),
                )
            )
        if len(samples) >= cap:
            break

    return split_samples(samples, split=split, candidate_paths=candidate_paths)[:max_samples]

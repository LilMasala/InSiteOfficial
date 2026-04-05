"""Public-data helpers for Stage 0 language curriculum domains."""

from __future__ import annotations

from pathlib import Path

import torch

from training.curriculum.data.public_sequence_data import (
    PROJECT_ROOT,
    candidate_files,
    read_records,
    read_text_blob,
    sequence_sample,
    split_samples,
    text_segments,
    tokenize_text,
)


DEFAULT_CURRICULUM_ROOT = PROJECT_ROOT / "data" / "curriculum"


def _dataset_plan() -> list[tuple[str, tuple[str, ...]]]:
    """Return the Stage-0 local dataset search plan."""
    return [
        ("wikipedia", tuple()),
        ("gutenberg", tuple()),
        ("xnli", tuple()),
        ("pmc_open_access", tuple()),
        ("the_stack", ("python", "julia")),
    ]


def _record_to_language_text(record: dict) -> str:
    """Convert one structured record into a language-training text span."""
    sentence1 = str(record.get("sentence1", record.get("premise", ""))).strip()
    sentence2 = str(record.get("sentence2", record.get("hypothesis", ""))).strip()
    label = str(record.get("label", "")).strip()
    if sentence1 and sentence2:
        parts = [f"Premise: {sentence1}", f"Hypothesis: {sentence2}"]
        if label:
            parts.append(f"Label: {label}")
        return " ".join(parts)

    for key in (
        "text",
        "content",
        "body",
        "article",
        "document",
        "passage",
        "summary",
        "code",
        "completion",
        "response",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _samples_from_structured_records(
    records: list[dict],
    *,
    vocab_size: int,
    seq_len: int,
    cap: int,
) -> list[dict[str, torch.Tensor]]:
    """Build Stage-0 samples from structured records."""
    samples: list[dict[str, torch.Tensor]] = []
    for record in records:
        text = _record_to_language_text(record)
        if not text:
            continue
        token_ids = tokenize_text(text, vocab_size)
        for start in range(0, max(1, len(token_ids) - 1), max(1, seq_len // 2)):
            sample = sequence_sample(token_ids[start:], seq_len, regime=float(len(text) > 256))
            if sample is None:
                continue
            samples.append(sample)
            if len(samples) >= cap:
                return samples
    return samples


def _samples_from_text_blob(
    text: str,
    *,
    vocab_size: int,
    seq_len: int,
    cap: int,
) -> list[dict[str, torch.Tensor]]:
    """Build Stage-0 samples from raw text."""
    samples: list[dict[str, torch.Tensor]] = []
    for segment in text_segments(text):
        token_ids = tokenize_text(segment, vocab_size)
        for start in range(0, max(1, len(token_ids) - 1), max(1, seq_len // 2)):
            sample = sequence_sample(token_ids[start:], seq_len, regime=float(len(segment) > 256))
            if sample is None:
                continue
            samples.append(sample)
            if len(samples) >= cap:
                return samples
    return samples


def load_public_language_samples(
    split: str,
    vocab_size: int,
    seq_len: int,
    max_samples: int,
    data_root: Path | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Load local public Stage-0 language samples when available."""
    root = data_root or DEFAULT_CURRICULUM_ROOT
    stage0_root = root / "stage0"
    all_samples: list[dict[str, torch.Tensor]] = []

    for dataset_name, keywords in _dataset_plan():
        dataset_dir = stage0_root / dataset_name
        candidate_paths = candidate_files(dataset_dir, split=split, keywords=keywords)
        if not candidate_paths:
            continue

        dataset_samples: list[dict[str, torch.Tensor]] = []
        cap = max_samples * 2 if split != "train" else max_samples
        for path in candidate_paths:
            name = path.name.lower()
            if any(name.endswith(suffix) for suffix in (".json", ".jsonl", ".csv", ".parquet", ".json.zst", ".jsonl.zst", ".csv.zst", ".parquet.zst")):
                dataset_samples.extend(
                    _samples_from_structured_records(
                        read_records(path),
                        vocab_size=vocab_size,
                        seq_len=seq_len,
                        cap=cap - len(dataset_samples),
                    )
                )
            else:
                dataset_samples.extend(
                    _samples_from_text_blob(
                        read_text_blob(path),
                        vocab_size=vocab_size,
                        seq_len=seq_len,
                        cap=cap - len(dataset_samples),
                    )
                )
            if len(dataset_samples) >= cap:
                break

        selected = split_samples(dataset_samples, split=split, candidate_paths=candidate_paths)
        if selected:
            all_samples.extend(selected[: max_samples - len(all_samples)])
        if len(all_samples) >= max_samples:
            return all_samples

    return all_samples

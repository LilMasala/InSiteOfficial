"""Public-data helpers for Stage 1 reasoning domains."""

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
except ImportError:  # pragma: no cover - optional on minimal local envs
    hf_load_dataset = None


MAX_REASONING_CHOICES = 8
SUPPORTED_SUFFIXES = {".json", ".jsonl", ".csv", ".parquet"}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CURRICULUM_ROOT = PROJECT_ROOT / "data" / "curriculum"
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("validation", "valid", "val", "dev", "test"),
    "test": ("test", "eval", "evaluation", "dev"),
}


def _token_id_for_text(text: str, vocab_size: int) -> int:
    """Hash a text fragment into one stable token id.

    Args:
        text: Raw text fragment.
        vocab_size: Target vocabulary size V.

    Returns:
        Integer token id in ``[1, V - 1]``.
    """
    normalized = normalize_whitespace(text).lower() or "<empty>"
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return 1 + (int(digest[:8], 16) % max(1, vocab_size - 1))


def encode_reasoning_text(text: str, vocab_size: int, seq_len: int) -> torch.Tensor:
    """Encode a reasoning prompt into fixed-length token ids.

    Args:
        text: Prompt text.
        vocab_size: Target vocabulary size V.
        seq_len: Output length N.

    Returns:
        Tensor of shape ``[N]``.
    """
    normalized = normalize_whitespace(text).lower()
    pieces = re.findall(r"[a-z0-9_]+|[^\w\s]", normalized)
    token_ids = [_token_id_for_text(piece, vocab_size) for piece in pieces[:seq_len]]
    encoded = torch.zeros(seq_len, dtype=torch.long)
    if token_ids:
        encoded[: len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
    return encoded


def _default_target(answer_text: str, vocab_size: int, seq_len: int) -> torch.Tensor:
    """Encode the target answer text.

    Args:
        answer_text: Canonical answer text.
        vocab_size: Target vocabulary size V.
        seq_len: Output length N.

    Returns:
        Tensor of shape ``[N]``.
    """
    return encode_reasoning_text(answer_text, vocab_size=vocab_size, seq_len=seq_len)


def _choice_tensor(num_choices: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build padded choice-token and choice-mask tensors.

    Args:
        num_choices: Number of active choices C.
        vocab_size: Target vocabulary size V.

    Returns:
        Tuple of padded choice tokens ``[MAX_REASONING_CHOICES]`` and mask ``[MAX_REASONING_CHOICES]``.
    """
    choice_tokens = torch.zeros(MAX_REASONING_CHOICES, dtype=torch.long)
    choice_mask = torch.zeros(MAX_REASONING_CHOICES, dtype=torch.bool)
    for index in range(min(num_choices, MAX_REASONING_CHOICES)):
        choice_tokens[index] = _token_id_for_text(chr(ord("A") + index), vocab_size)
        choice_mask[index] = True
    return choice_tokens, choice_mask


def _split_matches(path: Path, split: str) -> bool:
    """Return whether a path name strongly suggests a split.

    Args:
        path: Candidate data file.
        split: Requested split name.

    Returns:
        Boolean match flag.
    """
    aliases = SPLIT_ALIASES.get(split, (split,))
    lowered = str(path).lower()
    return any(alias in lowered for alias in aliases)


def _candidate_files(dataset_dir: Path, split: str, keywords: tuple[str, ...]) -> list[Path]:
    """List candidate raw data files for one dataset directory.

    Args:
        dataset_dir: Dataset root.
        split: Requested split.
        keywords: Optional path keywords to filter by filename stem (not full path).

    Returns:
        Sorted file paths, excluding hidden cache directories.
    """
    if not dataset_dir.exists():
        return []
    files = sorted(
        path for path in dataset_dir.rglob("*")
        if path.suffix.lower() in SUPPORTED_SUFFIXES
        and ".cache" not in path.parts
    )
    if keywords:
        # Match keywords against the file stem only, not the full path, so
        # agieval/test/lsat-ar.jsonl matches keyword "lsat-ar" but not "test".
        files = [
            path for path in files
            if any(keyword in path.stem.lower() for keyword in keywords)
        ]
    if not files:
        return []
    split_specific = [path for path in files if _split_matches(path, split)]
    return split_specific or files


def _flatten_json_payload(payload: Any) -> list[dict[str, Any]]:
    """Flatten nested JSON payloads into a record list.

    Args:
        payload: Arbitrary JSON-decoded object.

    Returns:
        List of dictionary records.
    """
    if isinstance(payload, list):
        records: list[dict[str, Any]] = []
        for item in payload:
            records.extend(_flatten_json_payload(item))
        return records
    if isinstance(payload, dict):
        if any(isinstance(value, (str, int, float, bool, list, dict, type(None))) for value in payload.values()):
            split_like = {"train", "validation", "valid", "val", "test", "dev"}
            if set(payload.keys()).issubset(split_like):
                records: list[dict[str, Any]] = []
                for value in payload.values():
                    records.extend(_flatten_json_payload(value))
                return records
            return [payload]
    return []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON records.

    Args:
        path: Input path.

    Returns:
        List of dictionary records.
    """
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        decoded = json.loads(stripped)
        records.extend(_flatten_json_payload(decoded))
    return records


def _read_json(path: Path) -> list[dict[str, Any]]:
    """Read JSON records.

    Args:
        path: Input path.

    Returns:
        List of dictionary records.
    """
    return _flatten_json_payload(json.loads(path.read_text()))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    """Read CSV rows into dictionaries.

    Args:
        path: Input path.

    Returns:
        List of dictionary records.
    """
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    """Read a parquet file through Hugging Face datasets if available.

    Args:
        path: Input path.

    Returns:
        List of dictionary records.
    """
    if hf_load_dataset is None:
        return []
    dataset = hf_load_dataset("parquet", data_files=str(path), split="train")
    return [dict(row) for row in dataset]


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Read one data file into dictionary records.

    Args:
        path: Input path.

    Returns:
        List of records.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".json":
        return _read_json(path)
    if suffix == ".csv":
        return _read_csv(path)
    if suffix == ".parquet":
        return _read_parquet(path)
    return []


def _extract_text(record: dict[str, Any], keys: Iterable[str]) -> str:
    """Extract the first non-empty text field from a record.

    Args:
        record: Source record.
        keys: Candidate key names.

    Returns:
        Extracted text, or an empty string.
    """
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
            return " ".join(item.strip() for item in value if item.strip())
    return ""


def _extract_choices(record: dict[str, Any]) -> list[str]:
    """Extract multiple-choice options from a record.

    Args:
        record: Source record.

    Returns:
        Choice text list.
    """
    for key in ("options", "choices", "endings", "candidates"):
        value = record.get(key)
        if isinstance(value, list) and value:
            choices: list[str] = []
            for item in value:
                if isinstance(item, str):
                    choices.append(item.strip())
                elif isinstance(item, dict):
                    text = _extract_text(item, ("text", "option", "label", "value"))
                    if text:
                        choices.append(text)
            if choices:
                return choices
    return []


def _extract_label_index(record: dict[str, Any], num_choices: int) -> int | None:
    """Extract a zero-based correct-choice index.

    Args:
        record: Source record.
        num_choices: Number of available choices.

    Returns:
        Choice index or ``None``.
    """
    label = record.get("label", record.get("gold", record.get("correct", record.get("target", record.get("answer")))))
    if isinstance(label, int):
        return label if 0 <= label < num_choices else None
    if isinstance(label, str):
        stripped = label.strip()
        if stripped.isdigit():
            idx = int(stripped)
            return idx if 0 <= idx < num_choices else None
        if len(stripped) == 1 and stripped.upper() in "ABCDEFGH":
            idx = ord(stripped.upper()) - ord("A")
            return idx if 0 <= idx < num_choices else None
    return None


def _extract_answer_text(record: dict[str, Any]) -> str:
    """Extract a canonical answer string.

    Args:
        record: Source record.

    Returns:
        Canonical answer text.
    """
    answer = _extract_text(record, ("answer", "solution", "output", "completion", "response", "final_answer", "explanation"))
    if answer:
        match = re.findall(r"####\s*([^\n]+)", answer)
        if match:
            return match[-1].strip()
        boxed = re.findall(r"\\boxed\{([^}]+)\}", answer)
        if boxed:
            return boxed[-1].strip()
        return answer
    return ""


def _normalize_mcq_record(
    record: dict[str, Any],
    vocab_size: int,
    seq_len: int,
) -> dict[str, torch.Tensor] | None:
    """Normalize a multiple-choice reasoning record.

    Args:
        record: Source record.
        vocab_size: Vocabulary size V.
        seq_len: Sequence length N.

    Returns:
        Normalized sample dictionary or ``None``.
    """
    choices = _extract_choices(record)
    if len(choices) > MAX_REASONING_CHOICES:
        choices = choices[:MAX_REASONING_CHOICES]
    label_idx = _extract_label_index(record, len(choices))
    question = _extract_text(record, ("question", "query", "problem", "instruction"))
    context = _extract_text(record, ("passage", "article", "context", "premise", "premises"))
    if not question or not choices or label_idx is None:
        return None
    prompt_parts = []
    if context:
        prompt_parts.append(f"Context: {context}")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append(
        "Choices: "
        + " ".join(f"{chr(ord('A') + idx)}. {choice}" for idx, choice in enumerate(choices[:MAX_REASONING_CHOICES]))
    )
    prompt = " ".join(prompt_parts)
    target_text = f"Answer: {chr(ord('A') + label_idx)}"
    choice_tokens, choice_mask = _choice_tensor(len(choices), vocab_size)
    answer_token = choice_tokens[label_idx].clone()
    return {
        "tokens": encode_reasoning_text(prompt, vocab_size=vocab_size, seq_len=seq_len),
        "target": _default_target(target_text, vocab_size=vocab_size, seq_len=seq_len),
        "answer": answer_token.long(),
        "choice_tokens": choice_tokens.long(),
        "choice_mask": choice_mask,
        "correct_choice": torch.tensor(label_idx, dtype=torch.long),
    }


def _normalize_label_record(
    record: dict[str, Any],
    vocab_size: int,
    seq_len: int,
) -> dict[str, torch.Tensor] | None:
    """Normalize an entailment/label-style reasoning record.

    Handles ProofWriter (theory+question, answer=True/False),
    FOLIO (premises+conclusion+label), and generic entailment datasets.
    True/False answers are converted to binary MCQ (A=True, B=False).

    Args:
        record: Source record.
        vocab_size: Vocabulary size V.
        seq_len: Sequence length N.

    Returns:
        Normalized sample dictionary or ``None``.
    """
    # ProofWriter: theory (facts+rules) + question
    theory = _extract_text(record, ("theory",))
    question = _extract_text(record, ("question", "query", "prompt"))
    context = _extract_text(record, ("premise", "premises", "context", "facts"))
    hypothesis = _extract_text(record, ("hypothesis", "conclusion", "claim", "statement"))
    label = record.get("label", record.get("answer", record.get("gold")))
    if isinstance(label, (int, float)):
        label_text = str(int(label))
    elif isinstance(label, str):
        label_text = label.strip()
    else:
        label_text = ""
    if not label_text:
        return None

    prompt_parts = []
    if theory:
        prompt_parts.append(f"Facts: {theory}")
    elif context:
        prompt_parts.append(f"Facts: {context}")
    if hypothesis:
        prompt_parts.append(f"Hypothesis: {hypothesis}")
    if question:
        prompt_parts.append(f"Question: {question}")
    if not prompt_parts:
        return None

    # Convert True/False answers to binary MCQ (A=True, B=False)
    normalized_label = label_text.lower()
    if normalized_label in ("true", "false", "yes", "no", "0", "1", "entailment", "contradiction", "neutral"):
        binary_choices = ["True", "False"]
        if normalized_label in ("true", "yes", "1", "entailment"):
            correct_idx = 0
        else:
            correct_idx = 1
        choice_tokens, choice_mask = _choice_tensor(2, vocab_size)
        answer_token = choice_tokens[correct_idx].clone()
        prompt = (
            " ".join(prompt_parts)
            + " Choices: A. True B. False"
        )
        return {
            "tokens": encode_reasoning_text(prompt, vocab_size=vocab_size, seq_len=seq_len),
            "target": _default_target(f"Answer: {chr(ord('A') + correct_idx)}", vocab_size=vocab_size, seq_len=seq_len),
            "answer": answer_token.long(),
            "choice_tokens": choice_tokens.long(),
            "choice_mask": choice_mask,
            "correct_choice": torch.tensor(correct_idx, dtype=torch.long),
        }

    # Generic label — hash into vocab as-is
    target_text = f"Label: {label_text}"
    return {
        "tokens": encode_reasoning_text(" ".join(prompt_parts), vocab_size=vocab_size, seq_len=seq_len),
        "target": _default_target(target_text, vocab_size=vocab_size, seq_len=seq_len),
        "answer": torch.tensor(_token_id_for_text(label_text, vocab_size), dtype=torch.long),
        "choice_tokens": torch.zeros(MAX_REASONING_CHOICES, dtype=torch.long),
        "choice_mask": torch.zeros(MAX_REASONING_CHOICES, dtype=torch.bool),
        "correct_choice": torch.tensor(-1, dtype=torch.long),
    }


def _extract_numeric_value(text: str) -> float | None:
    """Try to parse a numeric value from an answer string.

    Args:
        text: Answer text.

    Returns:
        Float value or ``None`` if not numeric.
    """
    cleaned = text.strip().replace(",", "").replace("$", "").replace("%", "")
    # Handle fractions like "3/4"
    fraction_match = re.match(r"^(-?\d+)\s*/\s*(\d+)$", cleaned)
    if fraction_match:
        num, den = int(fraction_match.group(1)), int(fraction_match.group(2))
        return num / den if den != 0 else None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _numeric_distractors(correct: float, n: int, seed: int) -> list[float]:
    """Generate plausible numeric distractors for a math answer.

    Args:
        correct: Correct numerical answer.
        n: Number of distractors to generate.
        seed: Deterministic seed.

    Returns:
        List of ``n`` distinct distractor values.
    """
    import random as _random
    rng = _random.Random(seed)
    distractors: list[float] = []
    abs_val = abs(correct) if correct != 0 else 1.0
    perturbations = [
        correct + abs_val * 0.5,
        correct - abs_val * 0.5,
        correct * 2.0,
        correct + abs_val,
        correct - abs_val,
        correct * 0.5,
        correct + 1,
        correct - 1,
        correct + 10,
        correct * 3,
    ]
    for candidate in perturbations:
        if candidate != correct and candidate not in distractors:
            distractors.append(candidate)
        if len(distractors) >= n:
            break
    while len(distractors) < n:
        distractors.append(correct + rng.randint(1, 100) * (1 if rng.random() > 0.5 else -1))
    return distractors[:n]


def _format_numeric(value: float) -> str:
    """Format a numeric value compactly.

    Args:
        value: Float value.

    Returns:
        String representation.
    """
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"


def _normalize_open_answer_record(
    record: dict[str, Any],
    vocab_size: int,
    seq_len: int,
) -> dict[str, torch.Tensor] | None:
    """Normalize an open-ended QA or instruction-following record.

    Numeric answers (e.g. GSM8K) are converted to 4-choice MCQ by generating
    3 plausible numerical distractors. This collapses the answer space from
    vocab_size unique hashes to 4 stable choice tokens (hashes of A/B/C/D),
    making the task learnable without a generative decoder.

    Args:
        record: Source record.
        vocab_size: Vocabulary size V.
        seq_len: Sequence length N.

    Returns:
        Normalized sample dictionary or ``None``.
    """
    question = _extract_text(record, ("question", "problem", "query", "instruction", "input"))
    answer_text = _extract_answer_text(record)
    if not question or not answer_text:
        return None

    numeric_val = _extract_numeric_value(answer_text)
    if numeric_val is not None:
        # Build a deterministic 4-choice MCQ from the numeric answer
        seed = int(hashlib.sha256(question.encode()).hexdigest()[:8], 16)
        distractors = _numeric_distractors(numeric_val, n=3, seed=seed)
        choices_vals = [numeric_val] + distractors[:3]
        # Shuffle deterministically so correct answer isn't always choice A
        shuffled_indices = list(range(4))
        import random as _random
        _random.Random(seed + 1).shuffle(shuffled_indices)
        choices_vals = [choices_vals[i] for i in shuffled_indices]
        correct_idx = shuffled_indices.index(0)  # index 0 was the correct answer
        choices_text = [_format_numeric(v) for v in choices_vals]
        choice_tokens, choice_mask = _choice_tensor(4, vocab_size)
        answer_token = choice_tokens[correct_idx].clone()
        prompt = (
            f"Question: {question} Choices: "
            + " ".join(f"{chr(ord('A') + i)}. {txt}" for i, txt in enumerate(choices_text))
        )
        return {
            "tokens": encode_reasoning_text(prompt, vocab_size=vocab_size, seq_len=seq_len),
            "target": _default_target(f"Answer: {chr(ord('A') + correct_idx)}", vocab_size=vocab_size, seq_len=seq_len),
            "answer": answer_token.long(),
            "choice_tokens": choice_tokens.long(),
            "choice_mask": choice_mask,
            "correct_choice": torch.tensor(correct_idx, dtype=torch.long),
        }

    # Non-numeric, non-MCQ open answer (e.g. symbolic math like \frac{1}{2},
    # or free-text code/reasoning answers from open_platypus).
    # Hashing these into a unique vocab token produces an unlearnable target —
    # the model can never predict a random hash, so these examples contribute
    # only noise to the loss without improving accuracy.  Skip them.
    return None


def _normalize_reasoning_record(
    record: dict[str, Any],
    vocab_size: int,
    seq_len: int,
) -> dict[str, torch.Tensor] | None:
    """Normalize one raw reasoning record using best-effort heuristics.

    Args:
        record: Source record.
        vocab_size: Vocabulary size V.
        seq_len: Sequence length N.

    Returns:
        Normalized sample dictionary or ``None``.
    """
    for normalizer in (_normalize_mcq_record, _normalize_label_record, _normalize_open_answer_record):
        normalized = normalizer(record, vocab_size=vocab_size, seq_len=seq_len)
        if normalized is not None:
            return normalized
    return None


def _records_for_dataset(
    dataset_dir: Path,
    split: str,
    keywords: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Load raw records from a local dataset snapshot.

    Args:
        dataset_dir: Dataset root.
        split: Requested split.
        keywords: Optional path keywords.

    Returns:
        Raw record list.
    """
    candidate_paths = _candidate_files(dataset_dir, split=split, keywords=keywords)
    records: list[dict[str, Any]] = []
    for path in candidate_paths:
        records.extend(_read_records(path))
    if not candidate_paths or split == "train":
        return records
    if candidate_paths and not any(_split_matches(path, split) for path in candidate_paths):
        train_records, heldout_records = holdout_split(records, fraction=0.2)
        return heldout_records if split in {"val", "test"} else train_records
    return records


def _dataset_plan_for_variant(domain_variant: str) -> list[tuple[str, tuple[str, ...]]]:
    """Return local source directories and optional keyword filters for a domain.

    logiqa2 only ships a .py loader (no actual data files), so all variants
    that would use it fall back to agieval/logiqa-en instead.

    Args:
        domain_variant: Stage-1 domain variant.

    Returns:
        Dataset plan list.
    """
    if domain_variant == "lsat":
        # lsat-ar + lsat-lr + lsat-rc cover the three LSAT sections
        return [("agieval", ("lsat-ar", "lsat-lr", "lsat-rc"))]
    if domain_variant == "gre":
        # SAT reasoning + LogiQA English is the closest proxy for GRE verbal
        return [("agieval", ("sat-en", "logiqa-en")), ("folio", tuple())]
    if domain_variant == "formal_logic":
        # ProofWriter (585k deductive chains) + FOLIO (FOL entailment) + LogiQA
        return [("proofwriter", tuple()), ("folio", tuple()), ("agieval", ("logiqa-en",))]
    if domain_variant == "math_competition":
        # GSM8K (grade-school) + all Hendrycks MATH subtypes
        return [("gsm8k", tuple()), ("hendrycks_math", tuple())]
    if domain_variant == "code_reasoning":
        return [("open_platypus", tuple())]
    if domain_variant == "mcat_cars":
        # LSAT reading comprehension + SAT English passages
        return [("agieval", ("lsat-rc", "sat-en"))]
    return []


def load_public_reasoning_samples(
    domain_variant: str,
    split: str,
    vocab_size: int,
    seq_len: int,
    max_samples: int,
    data_root: Path | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Load normalized public reasoning samples for one Stage-1 variant.

    Args:
        domain_variant: Stage-1 domain variant.
        split: Requested split.
        vocab_size: Vocabulary size V.
        seq_len: Sequence length N.
        max_samples: Maximum sample count.
        data_root: Optional curriculum data root.

    Returns:
        Normalized sample list.
    """
    root = data_root or DEFAULT_CURRICULUM_ROOT
    stage1_root = root / "stage1"
    samples: list[dict[str, torch.Tensor]] = []
    for dataset_name, keywords in _dataset_plan_for_variant(domain_variant):
        dataset_dir = stage1_root / dataset_name
        for record in _records_for_dataset(dataset_dir, split=split, keywords=keywords):
            normalized = _normalize_reasoning_record(record, vocab_size=vocab_size, seq_len=seq_len)
            if normalized is None:
                continue
            samples.append(normalized)
            if len(samples) >= max_samples:
                return samples
    return samples

"""Annotation feature helpers for the protein DTI domain."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


@dataclass
class AnnotationVocab:
    """Deterministic annotation vocabulary with hash fallback."""

    token_to_id: dict[str, int]
    size: int

    def encode(self, token: str) -> int:
        """Return the integer id for one annotation token."""
        resolved = self.token_to_id.get(token)
        if resolved is not None:
            return resolved
        if self.size <= 1:
            return 0
        return 1 + (_stable_hash(token) % (self.size - 1))

    def encode_many(self, tokens: list[str], max_items: int) -> torch.Tensor:
        """Encode a token list into a padded integer tensor."""
        indices = [self.encode(token) for token in tokens[:max_items]]
        if len(indices) < max_items:
            indices.extend([0] * (max_items - len(indices)))
        return torch.tensor(indices, dtype=torch.long)

    def save_json(self, path: str | Path) -> None:
        """Persist the vocabulary to JSON."""
        payload = {
            "size": self.size,
            "token_to_id": self.token_to_id,
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "AnnotationVocab":
        """Load a vocabulary from JSON."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        token_to_id = {str(key): int(value) for key, value in payload["token_to_id"].items()}
        return cls(token_to_id=token_to_id, size=int(payload["size"]))


def build_vocab(tokens: list[str], *, max_size: int) -> AnnotationVocab:
    """Build a small deterministic vocabulary from annotation ids."""
    if max_size < 1:
        raise ValueError("max_size must be positive.")
    counts = Counter(token for token in tokens if token)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    token_to_id = {
        token: index
        for index, (token, _) in enumerate(ordered[: max(0, max_size - 1)], start=1)
    }
    return AnnotationVocab(token_to_id=token_to_id, size=max_size)


def build_annotation_vocabs_from_db(
    db_path: str | Path,
    *,
    go_vocab_size: int = 50_000,
    cath_vocab_size: int = 10_000,
) -> tuple[AnnotationVocab, AnnotationVocab]:
    """Build GO and CATH vocabularies from the metadata database."""
    conn = sqlite3.connect(db_path)
    try:
        go_tokens = [str(row[0]) for row in conn.execute("SELECT go_id FROM protein_go_terms")]
        cath_tokens = [str(row[0]) for row in conn.execute("SELECT cath_id FROM protein_cath")]
    finally:
        conn.close()
    return (
        build_vocab(go_tokens, max_size=go_vocab_size),
        build_vocab(cath_tokens, max_size=cath_vocab_size),
    )


def write_feature_hdf5(
    db_path: str | Path,
    output_path: str | Path,
    *,
    go_vocab_size: int = 50_000,
    cath_vocab_size: int = 10_000,
    max_go_terms: int = 64,
    max_cath_ids: int = 16,
) -> dict[str, Any]:
    """Write padded GO/CATH feature matrices to HDF5."""
    import h5py  # type: ignore[import-untyped]

    go_vocab, cath_vocab = build_annotation_vocabs_from_db(
        db_path,
        go_vocab_size=go_vocab_size,
        cath_vocab_size=cath_vocab_size,
    )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    proteins = [str(row[0]) for row in conn.execute("SELECT uniprot_id FROM proteins ORDER BY uniprot_id")]
    go_matrix = torch.zeros(len(proteins), max_go_terms, dtype=torch.long)
    cath_matrix = torch.zeros(len(proteins), max_cath_ids, dtype=torch.long)
    for index, uniprot_id in enumerate(proteins):
        go_ids = [
            str(row["go_id"])
            for row in conn.execute(
                "SELECT go_id FROM protein_go_terms WHERE uniprot_id = ? ORDER BY go_id",
                (uniprot_id,),
            )
        ]
        cath_ids = [
            str(row["cath_id"])
            for row in conn.execute(
                "SELECT cath_id FROM protein_cath WHERE uniprot_id = ? ORDER BY cath_id",
                (uniprot_id,),
            )
        ]
        go_matrix[index] = go_vocab.encode_many(go_ids, max_go_terms)
        cath_matrix[index] = cath_vocab.encode_many(cath_ids, max_cath_ids)
    conn.close()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as handle:
        handle.create_dataset("proteins/uniprot_ids", data=[value.encode("utf-8") for value in proteins])
        handle.create_dataset("proteins/go_ids", data=go_matrix.numpy())
        handle.create_dataset("proteins/cath_ids", data=cath_matrix.numpy())
        handle.attrs["go_vocab_size"] = go_vocab.size
        handle.attrs["cath_vocab_size"] = cath_vocab.size

    return {
        "protein_count": len(proteins),
        "go_vocab_size": go_vocab.size,
        "cath_vocab_size": cath_vocab.size,
        "output_path": str(output),
    }

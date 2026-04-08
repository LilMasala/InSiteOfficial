"""Optional LanceDB assessment for larger-scale vector storage."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
import torch.nn.functional as F


try:
    import lancedb
except ImportError:  # pragma: no cover - optional dependency.
    lancedb = None


@dataclass(frozen=True)
class BackendAssessment:
    """Benchmark result for one vector backend."""

    backend: str
    available: bool
    latency_ms: float | None
    note: str


def assess_vector_backends(
    skill_embeddings: torch.Tensor,
    queries: torch.Tensor,
    *,
    root: str,
) -> list[BackendAssessment]:
    """Benchmark FAISS-adjacent storage candidates against simple cosine search."""
    normalized_skills = F.normalize(skill_embeddings.detach().float(), dim=-1)
    normalized_queries = F.normalize(queries.detach().float(), dim=-1)
    started = time.perf_counter()
    _ = normalized_queries @ normalized_skills.T
    tensor_latency = (time.perf_counter() - started) * 1000.0
    assessments = [
        BackendAssessment(
            backend="tensor",
            available=True,
            latency_ms=tensor_latency,
            note="Torch cosine-search baseline.",
        )
    ]
    if lancedb is None:
        assessments.append(
            BackendAssessment(
                backend="lancedb",
                available=False,
                latency_ms=None,
                note="Package not installed; assessment scaffold only.",
            )
        )
        return assessments
    started = time.perf_counter()
    db = lancedb.connect(root)
    table = db.create_table(
        "skills",
        data=[
            {"skill_id": idx, "vector": vector.tolist()}
            for idx, vector in enumerate(normalized_skills)
        ],
        mode="overwrite",
    )
    for query in normalized_queries:
        _ = table.search(query.tolist()).limit(3).to_list()
    lance_latency = (time.perf_counter() - started) * 1000.0
    assessments.append(
        BackendAssessment(
            backend="lancedb",
            available=True,
            latency_ms=lance_latency,
            note="Cold-path assessment for production storage planning.",
        )
    )
    return assessments

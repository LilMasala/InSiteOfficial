"""Semantic memory for distilled belief records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.chamelia.cognitive.storage import (
    CognitiveStorage,
    _import_lancedb,
    _import_pyarrow,
    deserialize_tensor,
)


@dataclass(frozen=True)
class BeliefRecord:
    """Persistent semantic belief."""

    belief_id: int
    embedding: torch.Tensor
    confidence: float
    provenance: frozenset[int]
    update_count: int
    domain_name: str
    description: str | None = None
    extras: dict[str, Any] | None = None


class SemanticMemory:
    """Domain-gated semantic memory with optional LanceDB acceleration."""

    def __init__(self, root: str | Path, embed_dim: int, use_lancedb: bool | None = None) -> None:
        self.storage = CognitiveStorage(root)
        self.embed_dim = int(embed_dim)
        if use_lancedb is None:
            use_lancedb = _import_lancedb() is not None and _import_pyarrow() is not None
        self._use_lancedb = bool(use_lancedb)
        self._belief_db: Any | None = None
        self._belief_table: Any | None = None
        if self._use_lancedb:
            self._ensure_lancedb()

    def _ensure_lancedb(self) -> None:
        lancedb = _import_lancedb()
        pa = _import_pyarrow()
        if lancedb is None or pa is None:
            self._use_lancedb = False
            return
        self._belief_db = lancedb.connect(self.storage.paths.lancedb_path)
        table_name = "beliefs_index"
        table_names = self._belief_db.list_tables()
        if hasattr(table_names, "tables"):
            table_names = table_names.tables
        existing = {str(name) for name in table_names}
        if table_name in existing:
            self._belief_table = self._belief_db.open_table(table_name)
            return
        schema = pa.schema(
            [
                pa.field("belief_id", pa.int64()),
                pa.field("domain_name", pa.string()),
                pa.field("vector", lancedb.vector(self.embed_dim)),
            ]
        )
        self._belief_table = self._belief_db.create_table(table_name, schema=schema, mode="create")

    def _rows_to_records(self, rows: list[Any]) -> list[BeliefRecord]:
        records: list[BeliefRecord] = []
        for row in rows:
            embedding = deserialize_tensor(row["embedding"], row["embedding_shape"])
            if embedding is None:
                continue
            provenance = frozenset(int(item) for item in json.loads(str(row["provenance_json"] or "[]")))
            extras = json.loads(str(row["extras"] or "{}"))
            records.append(
                BeliefRecord(
                    belief_id=int(row["belief_id"]),
                    embedding=embedding.float(),
                    confidence=float(row["confidence"]),
                    provenance=provenance,
                    update_count=int(row["update_count"]),
                    domain_name=str(row["domain_name"]),
                    description=row["description"],
                    extras=extras,
                )
            )
        return records

    def _upsert_lancedb_vector(self, belief_id: int, domain_name: str, embedding: torch.Tensor) -> None:
        if not self._use_lancedb or self._belief_table is None:
            return
        payload = [{"belief_id": int(belief_id), "domain_name": domain_name, "vector": embedding.detach().float().view(-1).cpu().tolist()}]
        try:
            self._belief_table.delete(f"belief_id = {int(belief_id)}")
        except Exception:
            pass
        self._belief_table.add(payload)

    def add_or_update_belief(
        self,
        *,
        embedding: torch.Tensor,
        domain_name: str,
        provenance: set[int] | tuple[int, ...] | list[int],
        description: str | None = None,
        corroborated: bool = True,
        similarity_threshold: float = 0.92,
    ) -> BeliefRecord:
        existing = self.retrieve(embedding, domain_name=domain_name, k=1)
        provenance_set = {int(item) for item in provenance}
        if existing and F.cosine_similarity(
            F.normalize(existing[0].embedding.view(1, -1), dim=-1),
            F.normalize(embedding.detach().float().view(1, -1), dim=-1),
            dim=-1,
        ).item() >= similarity_threshold:
            record = existing[0]
            prior_confidence = float(record.confidence)
            target = 1.0 if corroborated else 0.0
            step = (1.0 - prior_confidence) if corroborated else prior_confidence
            new_confidence = max(0.0, min(1.0, prior_confidence + (0.25 * step * (target - prior_confidence))))
            blend = 0.85 if corroborated and prior_confidence >= 0.7 else 0.55
            merged_embedding = F.normalize(
                (blend * record.embedding.detach().float()) + ((1.0 - blend) * embedding.detach().float()),
                dim=-1,
            )
            merged_provenance = set(record.provenance) | provenance_set
            update_count = int(record.update_count) + 1
            self.storage.update_belief(
                belief_id=record.belief_id,
                embedding=merged_embedding,
                confidence=new_confidence,
                provenance=merged_provenance,
                update_count=update_count,
                description=description or record.description,
                extras=record.extras,
            )
            self._upsert_lancedb_vector(record.belief_id, domain_name, merged_embedding)
            return BeliefRecord(
                belief_id=record.belief_id,
                embedding=merged_embedding,
                confidence=new_confidence,
                provenance=frozenset(merged_provenance),
                update_count=update_count,
                domain_name=domain_name,
                description=description or record.description,
                extras=record.extras,
            )
        confidence = 0.6 if corroborated else 0.4
        belief_id = self.storage.insert_belief(
            domain_name=domain_name,
            embedding=F.normalize(embedding.detach().float(), dim=-1),
            confidence=confidence,
            provenance=provenance_set,
            update_count=1,
            description=description,
        )
        self._upsert_lancedb_vector(belief_id, domain_name, embedding)
        return self.get_belief(belief_id)

    def get_belief(self, belief_id: int) -> BeliefRecord:
        rows = [row for row in self.storage.fetch_beliefs() if int(row["belief_id"]) == int(belief_id)]
        records = self._rows_to_records(rows)
        if not records:
            raise KeyError(f"belief_id={belief_id} not found")
        return records[0]

    def retrieve(self, query: torch.Tensor, *, domain_name: str, k: int = 4) -> list[BeliefRecord]:
        rows = self.storage.fetch_beliefs(domain_name=domain_name)
        records = self._rows_to_records(rows)
        if not records:
            return []
        query_norm = F.normalize(query.detach().float().view(1, -1), dim=-1)
        embeddings = torch.stack([F.normalize(record.embedding.view(-1), dim=-1) for record in records], dim=0)
        scores = (query_norm @ embeddings.T).squeeze(0)
        topk = scores.topk(min(int(k), scores.numel()))
        return [records[int(index)] for index in topk.indices.tolist()]

    def retrieval_tokens(self, query: torch.Tensor, *, domain_name: str, k: int = 4) -> torch.Tensor | None:
        records = self.retrieve(query, domain_name=domain_name, k=k)
        if not records:
            return None
        return torch.stack([record.embedding for record in records], dim=0)

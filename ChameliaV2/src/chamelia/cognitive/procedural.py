"""Procedural memory backed by SQLite metadata and a hot FAISS skill index."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.chamelia.cognitive.representation import (
    ContrastiveSparseRepresentation,
    IsotropicSkillCodec,
)
from src.chamelia.cognitive.storage import (
    CognitiveStorage,
    deserialize_codes,
    deserialize_tensor,
)

_FAISS_MODULE: Any | None = None
_LANCEDB_MODULE: Any | None = None
_PYARROW_MODULE: Any | None = None


def _import_faiss() -> Any | None:
    """Import FAISS lazily so environments can fall back cleanly when needed."""
    global _FAISS_MODULE
    if _FAISS_MODULE is not None:
        return _FAISS_MODULE
    if platform.system() == "Darwin":
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import faiss as imported_faiss
    except ImportError:  # pragma: no cover - dependency is optional at import time.
        return None
    _FAISS_MODULE = imported_faiss
    return _FAISS_MODULE


def _import_lancedb() -> Any | None:
    global _LANCEDB_MODULE
    if _LANCEDB_MODULE is not None:
        return _LANCEDB_MODULE
    try:
        import lancedb as imported_lancedb
    except ImportError:  # pragma: no cover - dependency is optional at import time.
        return None
    _LANCEDB_MODULE = imported_lancedb
    return _LANCEDB_MODULE


def _import_pyarrow() -> Any | None:
    global _PYARROW_MODULE
    if _PYARROW_MODULE is not None:
        return _PYARROW_MODULE
    try:
        import pyarrow as imported_pyarrow
    except ImportError:  # pragma: no cover - dependency is optional at import time.
        return None
    _PYARROW_MODULE = imported_pyarrow
    return _PYARROW_MODULE


@dataclass(frozen=True)
class SkillRecord:
    """Stored procedural skill."""

    skill_id: int
    embedding: torch.Tensor
    retrieval_vector: torch.Tensor
    action_path: torch.Tensor
    confidence: float
    source_episodes: tuple[int, ...]
    constraints: dict[str, Any]
    name: str | None = None
    description: str | None = None
    domain_name: str | None = None
    symbolic_program: tuple[str, ...] | None = None
    trigger_weights: dict[str, float] | None = None
    deprecated_by: int | None = None
    compressed_codes: torch.Tensor | None = None
    storage_format: str = "dense"
    extras: dict[str, Any] | None = None


@dataclass(frozen=True)
class RetrievedSkill:
    """Skill plus retrieval metadata."""

    record: SkillRecord
    similarity: float
    score: float
    trigger_weight: float


class TensorSkillIndex:
    """Pure-torch cosine-similarity fallback for small skill libraries."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.skill_ids: list[int] = []
        self.embeddings = torch.empty(0, dim, dtype=torch.float32)

    def add(self, skill_id: int, embedding: torch.Tensor) -> None:
        normalized = F.normalize(embedding.detach().float().view(1, -1), dim=-1)
        self.skill_ids.append(int(skill_id))
        self.embeddings = torch.cat([self.embeddings, normalized.cpu()], dim=0)

    def search(self, query: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
        if not self.skill_ids:
            return [], []
        query_norm = F.normalize(query.detach().float().view(1, -1), dim=-1).cpu()
        scores = (query_norm @ self.embeddings.T).squeeze(0)
        topk = scores.topk(min(k, scores.numel()))
        return (
            [self.skill_ids[int(index)] for index in topk.indices.tolist()],
            [float(value) for value in topk.values.tolist()],
        )

    def save(self, path: Path, meta_path: Path) -> None:
        np.save(path.with_suffix(".npy"), self.embeddings.numpy())
        meta_path.write_text(json.dumps(self.skill_ids))

    @classmethod
    def load(cls, dim: int, path: Path, meta_path: Path) -> "TensorSkillIndex":
        index = cls(dim=dim)
        tensor_path = path.with_suffix(".npy")
        if not tensor_path.exists() or not meta_path.exists():
            return index
        index.embeddings = torch.from_numpy(np.load(tensor_path)).float()
        index.skill_ids = [int(item) for item in json.loads(meta_path.read_text())]
        return index


class FaissSkillIndex:
    """FAISS inner-product index with explicit skill-id sidecar metadata."""

    def __init__(self, dim: int) -> None:
        faiss = _import_faiss()
        if faiss is None:  # pragma: no cover - exercised when dependency is missing.
            raise RuntimeError("faiss is not installed")
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.skill_ids: list[int] = []

    def add(self, skill_id: int, embedding: torch.Tensor) -> None:
        vector = F.normalize(embedding.detach().float().view(1, -1), dim=-1).cpu().numpy()
        self.index.add(vector.astype(np.float32))
        self.skill_ids.append(int(skill_id))

    def search(self, query: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
        if not self.skill_ids:
            return [], []
        vector = F.normalize(query.detach().float().view(1, -1), dim=-1).cpu().numpy()
        scores, indices = self.index.search(vector.astype(np.float32), min(k, len(self.skill_ids)))
        mapped_ids: list[int] = []
        mapped_scores: list[float] = []
        for index, score in zip(indices[0].tolist(), scores[0].tolist(), strict=False):
            if index < 0:
                continue
            mapped_ids.append(self.skill_ids[index])
            mapped_scores.append(float(score))
        return mapped_ids, mapped_scores

    def save(self, path: Path, meta_path: Path) -> None:
        faiss = _import_faiss()
        if faiss is None:
            raise RuntimeError("faiss is not installed")
        faiss.write_index(self.index, str(path))
        meta_path.write_text(json.dumps(self.skill_ids))

    @classmethod
    def load(cls, dim: int, path: Path, meta_path: Path) -> "FaissSkillIndex":
        faiss = _import_faiss()
        if faiss is None:
            raise RuntimeError("faiss is not installed")
        loaded = cls(dim=dim)
        if path.exists():
            loaded.index = faiss.read_index(str(path))
        if meta_path.exists():
            loaded.skill_ids = [int(item) for item in json.loads(meta_path.read_text())]
        return loaded


class LanceSkillIndex:
    """LanceDB-backed vector store for large procedural libraries."""

    def __init__(self, root: Path, dim: int, table_name: str = "skills_index") -> None:
        lancedb = _import_lancedb()
        pa = _import_pyarrow()
        if lancedb is None or pa is None:  # pragma: no cover - exercised when dependency is missing.
            raise RuntimeError("lancedb and pyarrow are required for the LanceDB backend")
        self.dim = dim
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.table_name = table_name
        self.db = lancedb.connect(self.root)
        table_names = set(getattr(self.db.list_tables(), "tables", []))
        self.table_exists = table_name in table_names
        if self.table_exists:
            self.table = self.db.open_table(table_name)
        else:
            schema = pa.schema(
                [
                    pa.field("skill_id", pa.int64()),
                    pa.field("vector", lancedb.vector(dim)),
                ]
            )
            self.table = self.db.create_table(table_name, schema=schema, mode="create")

    def add(self, skill_id: int, embedding: torch.Tensor) -> None:
        vector = embedding.detach().float().view(-1).cpu().tolist()
        self.table.add([{"skill_id": int(skill_id), "vector": vector}])

    def search(self, query: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
        if self.table.count_rows() == 0:
            return [], []
        results = self.table.search(query.detach().float().view(-1).cpu().tolist()).limit(k).to_list()
        skill_ids: list[int] = []
        similarities: list[float] = []
        for row in results:
            distance = float(row.get("_distance", 0.0))
            skill_ids.append(int(row["skill_id"]))
            similarities.append(1.0 / (1.0 + max(distance, 0.0)))
        return skill_ids, similarities

    def save(self, _path: Path, _meta_path: Path) -> None:
        if self.table.count_rows() == 0:
            return
        try:
            self.table.create_index(metric="cosine", replace=True, index_type="IVF_FLAT")
        except Exception:
            return


class ProceduralMemory:
    """Hot procedural memory with SQLite metadata and FAISS retrieval."""

    def __init__(
        self,
        root: str | Path,
        skill_dim: int,
        *,
        device: str = "cpu",
        use_faiss: bool = True,
        use_lancedb: bool | None = None,
        csr_encoder: ContrastiveSparseRepresentation | None = None,
        codec: IsotropicSkillCodec | None = None,
    ) -> None:
        self.skill_dim = skill_dim
        self.device = device
        self.storage = CognitiveStorage(root)
        self.csr_encoder = csr_encoder
        self.codec = codec
        self.records: dict[int, SkillRecord] = {}
        self.retrieval_dim = (
            int(csr_encoder.output_dim) if csr_encoder is not None else int(skill_dim)
        )
        if use_lancedb is None:
            use_lancedb = _import_lancedb() is not None
        if use_lancedb and _import_lancedb() is not None and _import_pyarrow() is not None:
            self.index: TensorSkillIndex | FaissSkillIndex | LanceSkillIndex = LanceSkillIndex(
                root=self.storage.paths.lancedb_path,
                dim=self.retrieval_dim,
            )
        elif use_faiss and _import_faiss() is not None:
            self.index = FaissSkillIndex(dim=self.retrieval_dim)
        else:
            self.index = TensorSkillIndex(dim=self.retrieval_dim)
        self._load_existing_skills()

    def _index_vector(self, vector: torch.Tensor) -> torch.Tensor:
        candidate = vector.detach().float()
        target_device = torch.device(self.device)
        if self.csr_encoder is not None:
            try:
                target_device = next(self.csr_encoder.parameters()).device
            except StopIteration:
                target_device = torch.device(self.device)
        candidate = candidate.to(target_device)
        if candidate.dim() == 1:
            candidate = candidate.unsqueeze(0)
        if self.csr_encoder is not None:
            with torch.no_grad():
                candidate = self.csr_encoder(candidate)
        return candidate.squeeze(0).cpu()

    def _compress_embedding(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, str]:
        dense_embedding = embedding.detach().float().cpu().view(-1)
        if self.codec is None:
            return dense_embedding, None, "dense"
        codec_device = dense_embedding.device
        try:
            codec_device = next(self.codec.parameters()).device
        except StopIteration:
            codec_device = dense_embedding.device
        with torch.no_grad():
            codec_input = dense_embedding.unsqueeze(0).to(codec_device)
            codes = self.codec.encode_codes(codec_input).squeeze(0).cpu()
            reconstructed = self.codec.decode(codes.unsqueeze(0).to(codec_device)).squeeze(0).cpu()
        storage_format = f"isotropic_vq_{self.codec.num_tokens}"
        return reconstructed, codes, storage_format

    def _load_existing_skills(self) -> None:
        rows = self.storage.fetch_skills()
        if not rows:
            return
        if isinstance(self.index, FaissSkillIndex) and self.storage.paths.skill_index_path.exists():
            self.index = FaissSkillIndex.load(
                dim=self.retrieval_dim,
                path=self.storage.paths.skill_index_path,
                meta_path=self.storage.paths.skill_index_meta_path,
            )
        elif isinstance(self.index, TensorSkillIndex):
            self.index = TensorSkillIndex.load(
                dim=self.retrieval_dim,
                path=self.storage.paths.skill_index_path,
                meta_path=self.storage.paths.skill_index_meta_path,
            )
        indexed_ids = set(getattr(self.index, "skill_ids", []))
        skip_rebuild = isinstance(self.index, LanceSkillIndex) and self.index.table_exists
        for row in rows:
            dense_embedding = deserialize_tensor(row["embedding"], row["embedding_shape"])
            codes = deserialize_codes(row["embedding_codes"], row["embedding_codes_shape"])
            action_path = deserialize_tensor(row["action_path"], row["action_shape"])
            retrieval_vector = deserialize_tensor(row["retrieval_vector"], row["retrieval_shape"])
            if dense_embedding is None or action_path is None:
                continue
            if codes is not None and self.codec is not None:
                with torch.no_grad():
                    embedding = self.codec.decode(codes.unsqueeze(0)).squeeze(0).cpu()
            else:
                embedding = dense_embedding
            if retrieval_vector is None:
                retrieval_vector = self._index_vector(embedding)
            skill_id = int(row["skill_id"])
            record = SkillRecord(
                skill_id=skill_id,
                embedding=embedding,
                retrieval_vector=retrieval_vector,
                action_path=action_path,
                confidence=float(row["confidence"]),
                source_episodes=self.storage.fetch_skill_sources(skill_id),
                constraints=self.storage.fetch_skill_constraints(skill_id),
                name=str(row["name"]) if row["name"] is not None else None,
                description=(
                    str(row["description"]) if row["description"] is not None else None
                ),
                domain_name=str(row["domain_name"]) if row["domain_name"] is not None else None,
                symbolic_program=(
                    tuple(json.loads(str(row["symbolic_program"])))
                    if row["symbolic_program"] is not None
                    else None
                ),
                trigger_weights=json.loads(str(row["trigger_weights"] or "{}")),
                deprecated_by=(
                    int(row["deprecated_by"]) if row["deprecated_by"] is not None else None
                ),
                compressed_codes=codes,
                storage_format=str(row["storage_format"] or "dense"),
                extras=json.loads(str(row["extras"] or "{}")),
            )
            self.records[skill_id] = record
            if not skip_rebuild and skill_id not in indexed_ids:
                self.index.add(skill_id, record.retrieval_vector)

    def add_skill(
        self,
        embedding: torch.Tensor,
        action_path: torch.Tensor,
        *,
        source_episodes: tuple[int, ...] = (),
        constraints: dict[str, Any] | None = None,
        confidence: float = 1.0,
        name: str | None = None,
        description: str | None = None,
        domain_name: str | None = None,
        symbolic_program: tuple[str, ...] | None = None,
        trigger_weights: dict[str, float] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> SkillRecord:
        runtime_embedding, compressed_codes, storage_format = self._compress_embedding(embedding)
        retrieval_vector = self._index_vector(runtime_embedding)
        skill_id = self.storage.insert_skill(
            embedding=runtime_embedding,
            embedding_codes=compressed_codes,
            retrieval_vector=retrieval_vector,
            storage_format=storage_format,
            action_path=action_path.detach().float().cpu(),
            confidence=confidence,
            source_episodes=source_episodes,
            constraints=constraints,
            name=name,
            description=description,
            domain_name=domain_name,
            symbolic_program=symbolic_program,
            trigger_weights=trigger_weights,
            extras=extras,
        )
        record = SkillRecord(
            skill_id=skill_id,
            embedding=runtime_embedding,
            retrieval_vector=retrieval_vector,
            action_path=action_path.detach().float().cpu(),
            confidence=float(confidence),
            source_episodes=tuple(int(value) for value in source_episodes),
            constraints=constraints or {},
            name=name,
            description=description,
            domain_name=domain_name,
            symbolic_program=symbolic_program,
            trigger_weights=trigger_weights or {},
            compressed_codes=compressed_codes,
            storage_format=storage_format,
            extras=extras or {},
        )
        self.records[skill_id] = record
        self.index.add(skill_id, retrieval_vector)
        return record

    def get_skill(self, skill_id: int) -> SkillRecord | None:
        return self.records.get(int(skill_id))

    def retrieve(
        self,
        query: torch.Tensor,
        *,
        k: int = 4,
        domain_name: str | None = None,
        min_confidence: float = 0.0,
        trigger_weights: dict[int, float] | None = None,
    ) -> list[RetrievedSkill]:
        skill_ids, similarities = self.index.search(self._index_vector(query), k)
        retrieved: list[RetrievedSkill] = []
        for skill_id, similarity in zip(skill_ids, similarities, strict=False):
            record = self.records.get(skill_id)
            if record is None or record.deprecated_by is not None:
                continue
            if record.confidence < min_confidence:
                continue
            trigger_weight = 1.0
            if trigger_weights is not None:
                trigger_weight = float(trigger_weights.get(skill_id, trigger_weight))
            elif domain_name is not None and record.trigger_weights is not None:
                trigger_weight = float(record.trigger_weights.get(domain_name, trigger_weight))
            score = float(similarity) * float(record.confidence) * trigger_weight
            retrieved.append(
                RetrievedSkill(
                    record=record,
                    similarity=float(similarity),
                    score=score,
                    trigger_weight=trigger_weight,
                )
            )
        retrieved.sort(key=lambda item: item.score, reverse=True)
        return retrieved[:k]

    def update_confidence(self, skill_id: int, confidence: float) -> None:
        record = self.records.get(int(skill_id))
        if record is None:
            raise KeyError(f"Unknown skill_id={skill_id}")
        updated = SkillRecord(
            skill_id=record.skill_id,
            embedding=record.embedding,
            retrieval_vector=record.retrieval_vector,
            action_path=record.action_path,
            confidence=float(confidence),
            source_episodes=record.source_episodes,
            constraints=record.constraints,
            name=record.name,
            description=record.description,
            domain_name=record.domain_name,
            symbolic_program=record.symbolic_program,
            trigger_weights=record.trigger_weights,
            deprecated_by=record.deprecated_by,
            compressed_codes=record.compressed_codes,
            storage_format=record.storage_format,
            extras=record.extras,
        )
        self.records[int(skill_id)] = updated
        self.storage.update_skill(int(skill_id), confidence=float(confidence))

    def deprecate(self, skill_id: int, replacement_id: int) -> None:
        record = self.records.get(int(skill_id))
        if record is None:
            raise KeyError(f"Unknown skill_id={skill_id}")
        updated = SkillRecord(
            skill_id=record.skill_id,
            embedding=record.embedding,
            retrieval_vector=record.retrieval_vector,
            action_path=record.action_path,
            confidence=record.confidence,
            source_episodes=record.source_episodes,
            constraints=record.constraints,
            name=record.name,
            description=record.description,
            domain_name=record.domain_name,
            symbolic_program=record.symbolic_program,
            trigger_weights=record.trigger_weights,
            deprecated_by=int(replacement_id),
            compressed_codes=record.compressed_codes,
            storage_format=record.storage_format,
            extras=record.extras,
        )
        self.records[int(skill_id)] = updated
        self.storage.update_skill(int(skill_id), deprecated_by=int(replacement_id))

    def save(self) -> None:
        self.index.save(
            self.storage.paths.skill_index_path,
            self.storage.paths.skill_index_meta_path,
        )

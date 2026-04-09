"""Persistence helpers for procedural memory, clusters, and archived episodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any

import numpy as np
import torch

_LANCEDB_MODULE: Any | None = None
_PYARROW_MODULE: Any | None = None


def _import_lancedb() -> Any | None:
    global _LANCEDB_MODULE
    if _LANCEDB_MODULE is not None:
        return _LANCEDB_MODULE
    try:
        import lancedb as imported_lancedb
    except ImportError:  # pragma: no cover - optional dependency at import time.
        return None
    _LANCEDB_MODULE = imported_lancedb
    return _LANCEDB_MODULE


def _import_pyarrow() -> Any | None:
    global _PYARROW_MODULE
    if _PYARROW_MODULE is not None:
        return _PYARROW_MODULE
    try:
        import pyarrow as imported_pyarrow
    except ImportError:  # pragma: no cover - optional dependency at import time.
        return None
    _PYARROW_MODULE = imported_pyarrow
    return _PYARROW_MODULE


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _ensure_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ensure_jsonable(item) for item in value]
    return value


def serialize_tensor(value: torch.Tensor | np.ndarray | None) -> tuple[bytes | None, str | None]:
    """Serialize a tensor into a SQLite-friendly blob plus JSON shape metadata."""
    if value is None:
        return None, None
    if torch.is_tensor(value):
        array = value.detach().cpu().float().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    return array.tobytes(), json.dumps(list(array.shape))


def deserialize_tensor(blob: bytes | None, shape_json: str | None) -> torch.Tensor | None:
    """Deserialize a tensor blob produced by :func:`serialize_tensor`."""
    if blob is None or shape_json is None:
        return None
    shape = tuple(int(dim) for dim in json.loads(shape_json))
    array = np.frombuffer(blob, dtype=np.float32).copy().reshape(shape)
    return torch.from_numpy(array)


def serialize_codes(value: torch.Tensor | np.ndarray | None) -> tuple[bytes | None, str | None]:
    """Serialize discrete codes into a compact int32 blob plus JSON shape metadata."""
    if value is None:
        return None, None
    if torch.is_tensor(value):
        array = value.detach().cpu().to(torch.int32).numpy()
    else:
        array = np.asarray(value, dtype=np.int32)
    return array.tobytes(), json.dumps(list(array.shape))


def deserialize_codes(blob: bytes | None, shape_json: str | None) -> torch.Tensor | None:
    """Deserialize an int32 code tensor produced by :func:`serialize_codes`."""
    if blob is None or shape_json is None:
        return None
    shape = tuple(int(dim) for dim in json.loads(shape_json))
    array = np.frombuffer(blob, dtype=np.int32).copy().reshape(shape)
    return torch.from_numpy(array).long()


@dataclass(frozen=True)
class StoragePaths:
    """Resolved filesystem locations for cognitive-architecture persistence."""

    root: Path
    sqlite_path: Path
    skill_index_path: Path
    skill_index_meta_path: Path
    lancedb_path: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "StoragePaths":
        resolved = Path(root).expanduser().resolve()
        return cls(
            root=resolved,
            sqlite_path=resolved / "cognitive.sqlite3",
            skill_index_path=resolved / "skills.faiss",
            skill_index_meta_path=resolved / "skills.faiss.meta.json",
            lancedb_path=resolved / "lancedb",
        )


class CognitiveStorage:
    """SQLite + LanceDB storage scaffold for the cognitive architecture."""

    def __init__(self, root: str | Path) -> None:
        self.paths = StoragePaths.from_root(root)
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.lancedb_path.mkdir(parents=True, exist_ok=True)
        self._episode_db: Any | None = None
        self._episode_table: Any | None = None
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.paths.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _managed_connection(self) -> Any:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._managed_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skills (
                    skill_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    description TEXT,
                    confidence REAL NOT NULL,
                    deprecated_by INTEGER,
                    domain_name TEXT,
                    symbolic_program TEXT,
                    embedding BLOB NOT NULL,
                    embedding_shape TEXT NOT NULL,
                    embedding_codes BLOB,
                    embedding_codes_shape TEXT,
                    retrieval_vector BLOB,
                    retrieval_shape TEXT,
                    storage_format TEXT NOT NULL DEFAULT 'dense',
                    action_path BLOB NOT NULL,
                    action_shape TEXT NOT NULL,
                    trigger_weights TEXT NOT NULL DEFAULT '{}',
                    extras TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._ensure_column(conn, "skills", "embedding_codes", "BLOB")
            self._ensure_column(conn, "skills", "embedding_codes_shape", "TEXT")
            self._ensure_column(conn, "skills", "retrieval_vector", "BLOB")
            self._ensure_column(conn, "skills", "retrieval_shape", "TEXT")
            self._ensure_column(
                conn,
                "skills",
                "storage_format",
                "TEXT NOT NULL DEFAULT 'dense'",
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_sources (
                    skill_id INTEGER NOT NULL,
                    episode_id INTEGER NOT NULL,
                    PRIMARY KEY (skill_id, episode_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS skill_constraints (
                    skill_id INTEGER NOT NULL,
                    constraint_key TEXT NOT NULL,
                    constraint_value TEXT NOT NULL,
                    PRIMARY KEY (skill_id, constraint_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS domain_clusters (
                    cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain_name TEXT NOT NULL,
                    centroid BLOB NOT NULL,
                    centroid_shape TEXT NOT NULL,
                    count INTEGER NOT NULL,
                    trigger_weights TEXT NOT NULL DEFAULT '{}',
                    adapter_payload BLOB,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        column_name: str,
        definition: str,
    ) -> None:
        existing = {
            str(row["name"])
            for row in conn.execute(f"PRAGMA table_info({table_name})")
        }
        if column_name in existing:
            return
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")

    def insert_skill(
        self,
        *,
        embedding: torch.Tensor,
        embedding_codes: torch.Tensor | None,
        retrieval_vector: torch.Tensor | None,
        storage_format: str,
        action_path: torch.Tensor,
        confidence: float,
        source_episodes: tuple[int, ...],
        constraints: dict[str, Any] | None,
        name: str | None,
        description: str | None,
        domain_name: str | None,
        symbolic_program: tuple[str, ...] | None,
        trigger_weights: dict[str, float] | None,
        extras: dict[str, Any] | None,
    ) -> int:
        embedding_blob, embedding_shape = serialize_tensor(embedding)
        embedding_codes_blob, embedding_codes_shape = serialize_codes(embedding_codes)
        retrieval_blob, retrieval_shape = serialize_tensor(retrieval_vector)
        action_blob, action_shape = serialize_tensor(action_path)
        if embedding_blob is None or embedding_shape is None:
            raise ValueError("embedding is required")
        if action_blob is None or action_shape is None:
            raise ValueError("action_path is required")
        created_at = _now_iso()
        with self._managed_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO skills (
                    name, description, confidence, domain_name, symbolic_program,
                    embedding, embedding_shape, embedding_codes, embedding_codes_shape,
                    retrieval_vector, retrieval_shape, storage_format, action_path, action_shape,
                    trigger_weights, extras, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    description,
                    float(confidence),
                    domain_name,
                    json.dumps(list(symbolic_program)) if symbolic_program is not None else None,
                    embedding_blob,
                    embedding_shape,
                    embedding_codes_blob,
                    embedding_codes_shape,
                    retrieval_blob,
                    retrieval_shape,
                    storage_format,
                    action_blob,
                    action_shape,
                    json.dumps(trigger_weights or {}),
                    json.dumps(_ensure_jsonable(extras or {})),
                    created_at,
                    created_at,
                ),
            )
            skill_id = int(cursor.lastrowid)
            if constraints:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO skill_constraints (skill_id, constraint_key, constraint_value)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (skill_id, key, json.dumps(_ensure_jsonable(value)))
                        for key, value in constraints.items()
                    ],
                )
            if source_episodes:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO skill_sources (skill_id, episode_id)
                    VALUES (?, ?)
                    """,
                    [(skill_id, int(episode_id)) for episode_id in source_episodes],
                )
        return skill_id

    def update_skill(
        self,
        skill_id: int,
        *,
        confidence: float | None = None,
        deprecated_by: int | None = None,
        name: str | None = None,
        description: str | None = None,
        trigger_weights: dict[str, float] | None = None,
        extras: dict[str, Any] | None = None,
    ) -> None:
        assignments: list[str] = ["updated_at = ?"]
        values: list[Any] = [_now_iso()]
        if confidence is not None:
            assignments.append("confidence = ?")
            values.append(float(confidence))
        if deprecated_by is not None:
            assignments.append("deprecated_by = ?")
            values.append(int(deprecated_by))
        if name is not None:
            assignments.append("name = ?")
            values.append(name)
        if description is not None:
            assignments.append("description = ?")
            values.append(description)
        if trigger_weights is not None:
            assignments.append("trigger_weights = ?")
            values.append(json.dumps(trigger_weights))
        if extras is not None:
            assignments.append("extras = ?")
            values.append(json.dumps(_ensure_jsonable(extras)))
        values.append(int(skill_id))
        with self._managed_connection() as conn:
            conn.execute(
                f"UPDATE skills SET {', '.join(assignments)} WHERE skill_id = ?",
                values,
            )

    def fetch_skills(self) -> list[sqlite3.Row]:
        with self._managed_connection() as conn:
            return list(conn.execute("SELECT * FROM skills ORDER BY skill_id ASC"))

    def fetch_skill_sources(self, skill_id: int) -> tuple[int, ...]:
        with self._managed_connection() as conn:
            rows = list(
                conn.execute(
                    "SELECT episode_id FROM skill_sources WHERE skill_id = ? ORDER BY episode_id ASC",
                    (int(skill_id),),
                )
            )
        return tuple(int(row["episode_id"]) for row in rows)

    def fetch_skill_constraints(self, skill_id: int) -> dict[str, Any]:
        with self._managed_connection() as conn:
            rows = list(
                conn.execute(
                    """
                    SELECT constraint_key, constraint_value
                    FROM skill_constraints
                    WHERE skill_id = ?
                    ORDER BY constraint_key ASC
                    """,
                    (int(skill_id),),
                )
            )
        return {
            str(row["constraint_key"]): json.loads(str(row["constraint_value"]))
            for row in rows
        }

    def upsert_cluster(
        self,
        *,
        cluster_id: int | None,
        domain_name: str,
        centroid: torch.Tensor,
        count: int,
        trigger_weights: dict[str, float] | None = None,
        adapter_payload: bytes | None = None,
    ) -> int:
        centroid_blob, centroid_shape = serialize_tensor(centroid)
        if centroid_blob is None or centroid_shape is None:
            raise ValueError("centroid is required")
        timestamp = _now_iso()
        with self._managed_connection() as conn:
            if cluster_id is None:
                cursor = conn.execute(
                    """
                    INSERT INTO domain_clusters (
                        domain_name, centroid, centroid_shape, count,
                        trigger_weights, adapter_payload, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        domain_name,
                        centroid_blob,
                        centroid_shape,
                        int(count),
                        json.dumps(trigger_weights or {}),
                        adapter_payload,
                        timestamp,
                        timestamp,
                    ),
                )
                return int(cursor.lastrowid)
            conn.execute(
                """
                UPDATE domain_clusters
                SET domain_name = ?, centroid = ?, centroid_shape = ?, count = ?,
                    trigger_weights = ?, adapter_payload = ?, updated_at = ?
                WHERE cluster_id = ?
                """,
                (
                    domain_name,
                    centroid_blob,
                    centroid_shape,
                    int(count),
                    json.dumps(trigger_weights or {}),
                    adapter_payload,
                    timestamp,
                    int(cluster_id),
                ),
            )
        return int(cluster_id)

    def fetch_clusters(self) -> list[sqlite3.Row]:
        with self._managed_connection() as conn:
            return list(conn.execute("SELECT * FROM domain_clusters ORDER BY cluster_id ASC"))

    def _list_lancedb_tables(self, db: Any) -> list[str]:
        table_names = db.list_tables()
        if hasattr(table_names, "tables"):
            return [str(name) for name in table_names.tables]
        return [str(name) for name in table_names]

    def _get_episode_table(self) -> Any:
        if self._episode_table is not None:
            return self._episode_table
        lancedb = _import_lancedb()
        pa = _import_pyarrow()
        if lancedb is None or pa is None:  # pragma: no cover - exercised when deps are missing.
            raise RuntimeError("lancedb and pyarrow are required for episode archival.")
        if self._episode_db is None:
            self._episode_db = lancedb.connect(self.paths.lancedb_path)
        table_name = "episodes_archive"
        schema = pa.schema(
            [
                pa.field("record_id", pa.int64()),
                pa.field("domain_name", pa.string()),
                pa.field("key_vector", pa.list_(pa.float32())),
                pa.field("payload_json", pa.string()),
                pa.field("created_at", pa.string()),
                pa.field("updated_at", pa.string()),
            ]
        )
        if table_name in self._list_lancedb_tables(self._episode_db):
            self._episode_table = self._episode_db.open_table(table_name)
        else:
            self._episode_table = self._episode_db.create_table(
                table_name,
                schema=schema,
                mode="create",
            )
        return self._episode_table

    def archive_episode(self, record_id: int, payload: dict[str, Any]) -> None:
        try:
            table = self._get_episode_table()
        except RuntimeError:
            return
        timestamp = _now_iso()
        key = payload.get("key")
        key_vector: list[float] = []
        if torch.is_tensor(key):
            key_vector = key.detach().cpu().float().reshape(-1).tolist()
        row = {
            "record_id": int(record_id),
            "domain_name": str(payload.get("domain_name") or ""),
            "key_vector": key_vector,
            "payload_json": json.dumps(_ensure_jsonable(payload)),
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        try:
            table.delete(f"record_id = {int(record_id)}")
        except Exception:
            pass
        table.add([row])

    def fetch_archived_episode(self, record_id: int) -> dict[str, Any] | None:
        table = self._get_episode_table()
        for row in table.to_arrow().to_pylist():
            if int(row["record_id"]) == int(record_id):
                return json.loads(str(row["payload_json"]))
        return None

    def fetch_archived_episodes(
        self,
        *,
        domain_name: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        table = self._get_episode_table()
        rows = table.to_arrow().to_pylist()
        archived: list[dict[str, Any]] = []
        for row in rows:
            if domain_name is not None and str(row.get("domain_name") or "") != domain_name:
                continue
            archived.append(json.loads(str(row["payload_json"])))
            if limit is not None and len(archived) >= int(limit):
                break
        return archived

    def close(self) -> None:
        """Release optional LanceDB archival handles before interpreter shutdown."""
        for handle_name in ("_episode_table", "_episode_db"):
            handle = getattr(self, handle_name, None)
            if handle is None:
                continue
            close_fn = getattr(handle, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            setattr(self, handle_name, None)

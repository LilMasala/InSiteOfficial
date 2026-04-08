"""Domain clustering and lightweight LoRA adapter management."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.chamelia.cognitive.storage import CognitiveStorage, deserialize_tensor


@dataclass
class AttentionLoRAAdapter:
    """Rank-r low-rank deltas for one ``nn.MultiheadAttention`` module."""

    in_proj_a: torch.Tensor
    in_proj_b: torch.Tensor
    out_proj_a: torch.Tensor
    out_proj_b: torch.Tensor

    @property
    def rank(self) -> int:
        return self.in_proj_a.shape[0]

    def delta_in_proj(self) -> torch.Tensor:
        return self.in_proj_b @ self.in_proj_a

    def delta_out_proj(self) -> torch.Tensor:
        return self.out_proj_b @ self.out_proj_a


@dataclass
class DomainCluster:
    """Online Dirichlet-process-style domain cluster."""

    cluster_id: int
    domain_name: str
    centroid: torch.Tensor
    count: int
    trigger_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DomainRoute:
    """Routing decision for a latent state."""

    cluster_id: int
    confidence: float
    spawned_new: bool


class LoRAAdapterBank:
    """Apply and remove rank-8 attention adapters per domain cluster."""

    def __init__(
        self,
        model: nn.Module,
        *,
        rank: int = 8,
    ) -> None:
        self.model = model
        self.rank = rank
        self.attention_modules = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, nn.MultiheadAttention)
        }
        self.adapters: dict[int, dict[str, AttentionLoRAAdapter]] = {}
        self.active_cluster_id: int | None = None

    def ensure_cluster(self, cluster_id: int) -> None:
        if cluster_id in self.adapters:
            return
        cluster_adapters: dict[str, AttentionLoRAAdapter] = {}
        for name, module in self.attention_modules.items():
            embed_dim = int(module.embed_dim)
            in_rows = int(module.in_proj_weight.shape[0])
            cluster_adapters[name] = AttentionLoRAAdapter(
                in_proj_a=torch.zeros(self.rank, embed_dim),
                in_proj_b=torch.zeros(in_rows, self.rank),
                out_proj_a=torch.zeros(self.rank, embed_dim),
                out_proj_b=torch.zeros(embed_dim, self.rank),
            )
        self.adapters[int(cluster_id)] = cluster_adapters

    def snapshot_from_difference(
        self,
        *,
        cluster_id: int,
        reference_model: nn.Module,
        tuned_model: nn.Module,
    ) -> None:
        self.ensure_cluster(cluster_id)
        reference_modules = {
            name: module
            for name, module in reference_model.named_modules()
            if isinstance(module, nn.MultiheadAttention)
        }
        tuned_modules = {
            name: module
            for name, module in tuned_model.named_modules()
            if isinstance(module, nn.MultiheadAttention)
        }
        for name, adapter in self.adapters[int(cluster_id)].items():
            ref = reference_modules[name]
            tuned = tuned_modules[name]
            in_diff = (tuned.in_proj_weight - ref.in_proj_weight).detach().float()
            out_diff = (tuned.out_proj.weight - ref.out_proj.weight).detach().float()
            in_u, in_s, in_vh = torch.linalg.svd(in_diff, full_matrices=False)
            out_u, out_s, out_vh = torch.linalg.svd(out_diff, full_matrices=False)
            rank = min(self.rank, in_s.shape[0], out_s.shape[0])
            adapter.in_proj_b = in_u[:, :rank] * in_s[:rank].unsqueeze(0)
            adapter.in_proj_a = in_vh[:rank, :]
            adapter.out_proj_b = out_u[:, :rank] * out_s[:rank].unsqueeze(0)
            adapter.out_proj_a = out_vh[:rank, :]

    def clear_active(self) -> None:
        if self.active_cluster_id is None:
            return
        adapters = self.adapters.get(self.active_cluster_id, {})
        for name, module in self.attention_modules.items():
            adapter = adapters.get(name)
            if adapter is None:
                continue
            module.in_proj_weight.data.sub_(adapter.delta_in_proj().to(module.in_proj_weight))
            module.out_proj.weight.data.sub_(adapter.delta_out_proj().to(module.out_proj.weight))
        self.active_cluster_id = None

    def apply_cluster(self, cluster_id: int) -> None:
        cluster_id = int(cluster_id)
        if self.active_cluster_id == cluster_id:
            return
        self.clear_active()
        self.ensure_cluster(cluster_id)
        for name, module in self.attention_modules.items():
            adapter = self.adapters[cluster_id][name]
            module.in_proj_weight.data.add_(adapter.delta_in_proj().to(module.in_proj_weight))
            module.out_proj.weight.data.add_(adapter.delta_out_proj().to(module.out_proj.weight))
        self.active_cluster_id = cluster_id

    def serialize_cluster(self, cluster_id: int) -> bytes:
        self.ensure_cluster(cluster_id)
        payload: dict[str, list[list[float]]] = {}
        for name, adapter in self.adapters[int(cluster_id)].items():
            payload[name] = {
                "in_proj_a": adapter.in_proj_a.tolist(),
                "in_proj_b": adapter.in_proj_b.tolist(),
                "out_proj_a": adapter.out_proj_a.tolist(),
                "out_proj_b": adapter.out_proj_b.tolist(),
            }
        return json.dumps(payload).encode("utf-8")

    def load_cluster(self, cluster_id: int, payload: bytes | None) -> None:
        self.ensure_cluster(cluster_id)
        if payload is None:
            return
        decoded = json.loads(payload.decode("utf-8"))
        for name, adapter_payload in decoded.items():
            self.adapters[int(cluster_id)][name] = AttentionLoRAAdapter(
                in_proj_a=torch.tensor(adapter_payload["in_proj_a"], dtype=torch.float32),
                in_proj_b=torch.tensor(adapter_payload["in_proj_b"], dtype=torch.float32),
                out_proj_a=torch.tensor(adapter_payload["out_proj_a"], dtype=torch.float32),
                out_proj_b=torch.tensor(adapter_payload["out_proj_b"], dtype=torch.float32),
            )


class DomainIndex:
    """Nonparametric domain routing with persisted centroids and trigger weights."""

    def __init__(
        self,
        storage_root: str | Any,
        *,
        concentration: float = 0.75,
        similarity_scale: float = 6.0,
        spawn_threshold: float = 0.45,
        adapter_bank: LoRAAdapterBank | None = None,
    ) -> None:
        self.storage = CognitiveStorage(storage_root)
        self.concentration = concentration
        self.similarity_scale = similarity_scale
        self.spawn_threshold = spawn_threshold
        self.adapter_bank = adapter_bank
        self.clusters: dict[int, DomainCluster] = {}
        self._load()

    def _load(self) -> None:
        for row in self.storage.fetch_clusters():
            centroid = deserialize_tensor(row["centroid"], row["centroid_shape"])
            if centroid is None:
                continue
            cluster = DomainCluster(
                cluster_id=int(row["cluster_id"]),
                domain_name=str(row["domain_name"]),
                centroid=centroid.float(),
                count=int(row["count"]),
                trigger_weights=json.loads(str(row["trigger_weights"] or "{}")),
            )
            self.clusters[cluster.cluster_id] = cluster
            if self.adapter_bank is not None:
                self.adapter_bank.load_cluster(cluster.cluster_id, row["adapter_payload"])

    def _cluster_scores(
        self,
        z: torch.Tensor,
        domain_name: str,
    ) -> tuple[list[int], torch.Tensor]:
        candidate_ids: list[int] = []
        scores: list[torch.Tensor] = []
        for cluster_id, cluster in self.clusters.items():
            if cluster.domain_name != domain_name:
                continue
            candidate_ids.append(cluster_id)
            sim = F.cosine_similarity(z.unsqueeze(0), cluster.centroid.unsqueeze(0), dim=-1)
            score = (self.similarity_scale * sim) + torch.log(
                torch.tensor(float(cluster.count), device=z.device).clamp_min(1.0)
            )
            scores.append(score)
        if not scores:
            return candidate_ids, torch.empty(0, device=z.device)
        return candidate_ids, torch.stack(scores).squeeze(-1)

    def route(self, z: torch.Tensor, domain_name: str) -> DomainRoute:
        z = z.detach().float().cpu()
        candidate_ids, scores = self._cluster_scores(z, domain_name)
        new_cluster_score = torch.tensor(float(self.concentration), dtype=torch.float32).log()
        if scores.numel() == 0:
            cluster_id = self._spawn_cluster(z, domain_name)
            return DomainRoute(cluster_id=cluster_id, confidence=1.0, spawned_new=True)
        best_score, best_idx = scores.max(dim=0)
        best_cluster_id = candidate_ids[int(best_idx.item())]
        best_cluster = self.clusters[best_cluster_id]
        best_similarity = F.cosine_similarity(
            z.unsqueeze(0),
            best_cluster.centroid.unsqueeze(0),
            dim=-1,
        )
        if float(best_similarity.item()) < self.spawn_threshold and float(new_cluster_score.item()) >= float(best_score.item() - 1.0):
            cluster_id = self._spawn_cluster(z, domain_name)
            return DomainRoute(cluster_id=cluster_id, confidence=1.0, spawned_new=True)
        self._update_cluster(best_cluster_id, z)
        return DomainRoute(
            cluster_id=best_cluster_id,
            confidence=float(best_similarity.item()),
            spawned_new=False,
        )

    def _spawn_cluster(self, z: torch.Tensor, domain_name: str) -> int:
        cluster_id = self.storage.upsert_cluster(
            cluster_id=None,
            domain_name=domain_name,
            centroid=z,
            count=1,
            trigger_weights={},
            adapter_payload=(
                self.adapter_bank.serialize_cluster(0)
                if self.adapter_bank is not None and 0 in self.adapter_bank.adapters
                else None
            ),
        )
        self.clusters[cluster_id] = DomainCluster(
            cluster_id=cluster_id,
            domain_name=domain_name,
            centroid=z.clone(),
            count=1,
            trigger_weights={},
        )
        if self.adapter_bank is not None:
            self.adapter_bank.ensure_cluster(cluster_id)
        return cluster_id

    def _update_cluster(self, cluster_id: int, z: torch.Tensor) -> None:
        cluster = self.clusters[int(cluster_id)]
        updated_count = cluster.count + 1
        updated_centroid = ((cluster.centroid * cluster.count) + z) / updated_count
        cluster.count = updated_count
        cluster.centroid = updated_centroid
        adapter_payload = None
        if self.adapter_bank is not None:
            adapter_payload = self.adapter_bank.serialize_cluster(cluster.cluster_id)
        self.storage.upsert_cluster(
            cluster_id=cluster.cluster_id,
            domain_name=cluster.domain_name,
            centroid=cluster.centroid,
            count=cluster.count,
            trigger_weights=cluster.trigger_weights,
            adapter_payload=adapter_payload,
        )

    def record_skill_trigger(
        self,
        cluster_id: int,
        skill_id: int,
        weight: float = 1.0,
    ) -> None:
        cluster = self.clusters[int(cluster_id)]
        current = float(cluster.trigger_weights.get(str(skill_id), 0.0))
        cluster.trigger_weights[str(skill_id)] = current + float(weight)
        adapter_payload = None
        if self.adapter_bank is not None:
            adapter_payload = self.adapter_bank.serialize_cluster(cluster.cluster_id)
        self.storage.upsert_cluster(
            cluster_id=cluster.cluster_id,
            domain_name=cluster.domain_name,
            centroid=cluster.centroid,
            count=cluster.count,
            trigger_weights=cluster.trigger_weights,
            adapter_payload=adapter_payload,
        )

    def get_trigger_weights(self, cluster_id: int) -> dict[int, float]:
        cluster = self.clusters[int(cluster_id)]
        return {int(key): float(value) for key, value in cluster.trigger_weights.items()}

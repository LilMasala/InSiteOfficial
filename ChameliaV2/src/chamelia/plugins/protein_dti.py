"""Protein-drug interaction domain plugin for tokenizer-driven Chamelia."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.chamelia.domains.protein_dti.dataset import ProteinDTIDataset
from src.chamelia.domains.protein_dti.paths import default_data_dir, default_db_path
from src.chamelia.domains.protein_dti.tokenizer import (
    ProteinDTIBatch,
    ProteinDTIObservation,
    ProteinDrugTokenizer,
)

from .base import AbstractDomain


class ProteinDTIDomain(AbstractDomain):
    """Protein DTI domain with pairwise ranking as the primary learning signal."""

    def __init__(
        self,
        *,
        db_path: str | None = None,
        data_base_dir: str | None = None,
        embed_dim: int = 512,
        max_candidate_drugs: int = 20,
        action_dim: int | None = None,
        affinity_type: str = "Kd",
        split: str = "train",
        split_strategy: str = "protein_family",
        ranking_margin: float = 0.1,
        seed: int = 42,
    ) -> None:
        resolved_action_dim = int(action_dim or max_candidate_drugs)
        if resolved_action_dim < max_candidate_drugs:
            raise ValueError("action_dim must be at least max_candidate_drugs for protein_dti.")
        self.embed_dim = int(embed_dim)
        self.max_candidate_drugs = int(max_candidate_drugs)
        self.action_dim = resolved_action_dim
        self.affinity_type = str(affinity_type)
        self.ranking_margin = float(ranking_margin)
        self._tokenizer = ProteinDrugTokenizer(
            embed_dim=embed_dim,
            max_candidate_drugs=max_candidate_drugs,
        )
        self.dataset = ProteinDTIDataset(
            db_path=db_path or str(default_db_path()),
            data_base_dir=data_base_dir or str(default_data_dir()),
            split=split,
            split_strategy=split_strategy,
            affinity_type=affinity_type,
            max_candidate_drugs=max_candidate_drugs,
            seed=seed,
        )
        self._last_candidate_ids: list[list[str]] = []
        self._last_candidate_counts: list[int] = []

    def get_tokenizer(self) -> ProteinDrugTokenizer:
        return self._tokenizer

    def get_action_dim(self) -> int:
        return self.action_dim

    def sample_observation(self) -> ProteinDTIObservation | None:
        """Sample one observation from the domain dataset."""
        return self.dataset.sample_episode()

    def _as_batch(self, observation: Any) -> ProteinDTIBatch:
        if isinstance(observation, ProteinDTIBatch):
            return observation
        if isinstance(observation, ProteinDTIObservation):
            return ProteinDTIBatch(observations=[observation])
        if isinstance(observation, list) and all(
            isinstance(item, ProteinDTIObservation) for item in observation
        ):
            return ProteinDTIBatch(observations=list(observation))
        raise TypeError("ProteinDTIDomain expects ProteinDTIObservation or ProteinDTIBatch input.")

    def decode_action(self, action_vec: torch.Tensor) -> Any:
        if action_vec.dim() == 1:
            action_vec = action_vec.unsqueeze(0)
        decoded: list[dict[str, Any]] = []
        for batch_idx in range(action_vec.shape[0]):
            candidate_count = (
                self._last_candidate_counts[batch_idx]
                if batch_idx < len(self._last_candidate_counts)
                else min(self.max_candidate_drugs, action_vec.shape[1])
            )
            candidate_ids = (
                self._last_candidate_ids[batch_idx]
                if batch_idx < len(self._last_candidate_ids)
                else [f"candidate_{index}" for index in range(candidate_count)]
            )
            scores = action_vec[batch_idx, :candidate_count]
            ranked_indices = scores.argsort(dim=-1, descending=True)
            decoded.append(
                {
                    "candidate_ids": candidate_ids,
                    "scores": scores.detach().cpu(),
                    "ranked_indices": ranked_indices.detach().cpu(),
                    "ranked_candidate_ids": [candidate_ids[index] for index in ranked_indices.tolist()],
                }
            )
        return decoded[0] if len(decoded) == 1 else decoded

    def get_intrinsic_cost_fns(self) -> list[tuple[Any, float]]:
        def pairwise_ranking_cost(
            _z: torch.Tensor,
            action: torch.Tensor,
            domain_state: dict[str, Any],
        ) -> torch.Tensor:
            candidate_mask = domain_state["candidate_mask"].to(action.device)
            pairwise_preferences = domain_state["pairwise_preferences"].to(action.device)
            pairwise_weights = domain_state["pairwise_weights"].to(action.device)
            scores = action[:, : self.max_candidate_drugs]
            score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
            pairwise_loss = F.softplus(-score_diff) * pairwise_preferences.float() * pairwise_weights
            denominator = (pairwise_preferences.float() * pairwise_weights).sum(dim=(1, 2)).clamp_min(1.0)
            ranking_loss = pairwise_loss.sum(dim=(1, 2)) / denominator
            overflow_scores = action[:, self.max_candidate_drugs :]
            if overflow_scores.numel() == 0:
                return ranking_loss
            overflow_penalty = overflow_scores.pow(2).mean(dim=-1)
            valid_fraction = candidate_mask.float().mean(dim=-1)
            return ranking_loss + (1.0 - valid_fraction) * 0.05 * overflow_penalty

        return [(pairwise_ranking_cost, 1.0)]

    def get_domain_state(self, observation: Any) -> dict[str, Any]:
        batch = self._as_batch(observation)
        batch_size = len(batch.observations)
        candidate_mask = torch.zeros(batch_size, self.max_candidate_drugs, dtype=torch.bool)
        affinity_values = torch.zeros(batch_size, self.max_candidate_drugs, dtype=torch.float32)
        pairwise_preferences = torch.zeros(
            batch_size,
            self.max_candidate_drugs,
            self.max_candidate_drugs,
            dtype=torch.bool,
        )
        pairwise_weights = torch.zeros(
            batch_size,
            self.max_candidate_drugs,
            self.max_candidate_drugs,
            dtype=torch.float32,
        )
        candidate_counts = torch.zeros(batch_size, dtype=torch.long)
        self._last_candidate_ids = []
        self._last_candidate_counts = []
        uniprot_ids: list[str] = []

        for batch_idx, sample in enumerate(batch.observations):
            candidate_count = min(len(sample.candidate_ids), self.max_candidate_drugs)
            uniprot_ids.append(sample.uniprot_id)
            self._last_candidate_ids.append(list(sample.candidate_ids[:candidate_count]))
            self._last_candidate_counts.append(candidate_count)
            candidate_counts[batch_idx] = candidate_count
            candidate_mask[batch_idx, :candidate_count] = True
            if candidate_count == 0:
                continue
            sample_affinities = torch.tensor(
                sample.affinity_values[:candidate_count],
                dtype=torch.float32,
            )
            affinity_values[batch_idx, :candidate_count] = sample_affinities
            diff = sample_affinities.unsqueeze(1) - sample_affinities.unsqueeze(0)
            preferences = diff > self.ranking_margin
            weights = diff.abs()
            preferences.fill_diagonal_(False)
            weights.fill_diagonal_(0.0)
            pairwise_preferences[batch_idx, :candidate_count, :candidate_count] = preferences
            pairwise_weights[batch_idx, :candidate_count, :candidate_count] = weights

        return {
            "uniprot_ids": uniprot_ids,
            "candidate_ids": list(self._last_candidate_ids),
            "candidate_counts": candidate_counts,
            "candidate_mask": candidate_mask,
            "legal_actions_mask": candidate_mask,
            "affinity_values": affinity_values,
            "pairwise_preferences": pairwise_preferences,
            "pairwise_weights": pairwise_weights,
            "raw_observation": batch,
            "affinity_type": self.affinity_type,
        }

    def get_persistable_domain_state(self, domain_state: dict) -> dict[str, Any] | None:
        return {
            "uniprot_ids": list(domain_state.get("uniprot_ids", [])),
            "candidate_ids": list(domain_state.get("candidate_ids", [])),
            "affinity_type": str(domain_state.get("affinity_type", self.affinity_type)),
        }

    def compute_regime_embedding(self, domain_state: dict) -> torch.Tensor | None:
        affinity_values = domain_state.get("affinity_values")
        candidate_mask = domain_state.get("candidate_mask")
        if affinity_values is None or candidate_mask is None:
            return None
        values = affinity_values.float()
        mask = candidate_mask.float()
        summed = (values * mask).sum(dim=-1, keepdim=True)
        counts = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean_affinity = summed / counts
        spread = (((values - mean_affinity) * mask) ** 2).sum(dim=-1, keepdim=True) / counts
        return torch.cat([mean_affinity, spread.sqrt()], dim=-1)

    def simulate_delayed_outcome(
        self,
        action_vec: torch.Tensor,
        domain_state: dict,
    ) -> dict[str, torch.Tensor | ProteinDTIBatch] | None:
        cost_fn = self.get_intrinsic_cost_fns()[0][0]
        realized = cost_fn(torch.empty(action_vec.shape[0], 0), action_vec, domain_state).detach()
        return {
            "outcome_observation": domain_state["raw_observation"],
            "realized_intrinsic_cost": realized,
        }

    @property
    def domain_name(self) -> str:
        return "protein_dti"

    @property
    def vocab_size(self) -> int:
        return 0

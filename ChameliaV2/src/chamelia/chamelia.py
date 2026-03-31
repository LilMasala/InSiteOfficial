"""Top-level Chamelia assembly module."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.chamelia.actor import Actor
from src.chamelia.configurator import Configurator
from src.chamelia.cost import CostModule
from src.chamelia.hjepa_adapter import forward_hjepa
from src.chamelia.memory import EpisodeRecord, LatentMemory
from src.chamelia.plugins.base import AbstractDomain
from src.models.hjepa import HJEPA


class Chamelia(nn.Module):
    """Assembled Chamelia system: HJEPA + Configurator + Actor + Cost + Memory."""

    def __init__(
        self,
        hjepa: HJEPA,
        configurator: Configurator,
        actor: Actor,
        cost: CostModule,
        memory: LatentMemory,
        domain: AbstractDomain,
        embed_dim: int = 512,
        action_dim: int = 64,
        num_ctx_tokens: int = 16,
    ) -> None:
        """Initialize the assembled Chamelia model.

        Args:
            hjepa: Backbone HJEPA-compatible model.
            configurator: Configurator module emitting [B, C, D].
            actor: Actor module emitting [B, A].
            cost: Cost module returning [B] scalar costs.
            memory: Latent episodic memory.
            domain: Registered domain plugin.
            embed_dim: Shared latent dimension D.
            action_dim: Actor action dimension A.
            num_ctx_tokens: Number of configurator context tokens C.

        Returns:
            None.
        """
        super().__init__()
        self.hjepa = hjepa
        self.configurator = configurator
        self.actor = actor
        self.cost = cost
        self.memory = memory
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.num_ctx_tokens = num_ctx_tokens
        self._pending_record_indices: list[int] = []
        self._step_counter = 0
        self.set_domain(domain)

    def set_domain(self, domain: AbstractDomain) -> None:
        """Attach a runtime domain plugin and register its tokenizer if trainable.

        Args:
            domain: Active runtime domain.

        Returns:
            None.
        """
        self.domain = domain
        tokenizer = domain.get_tokenizer()
        if isinstance(tokenizer, nn.Module):
            self.domain_tokenizer = tokenizer

    def get_domain_tokenizer(self) -> nn.Module | None:
        """Return the registered domain tokenizer module if present.

        Args:
            None.

        Returns:
            Tokenizer module or ``None``.
        """
        tokenizer = getattr(self, "domain_tokenizer", None)
        return tokenizer if isinstance(tokenizer, nn.Module) else None

    def _extract_level_features(self, hjepa_outputs: dict) -> list[torch.Tensor]:
        """Extract per-level FPN features from HJEPA output.

        Args:
            hjepa_outputs: HJEPA output dict containing target_features [B, 197, D].

        Returns:
            List of level feature tensors [B, N_i, D].
        """
        target_features = hjepa_outputs["target_features"]
        patch_features = target_features[:, 1:, :]
        level_features = self.hjepa._apply_fpn(patch_features, is_prediction=False)
        return level_features

    def _get_scene_summary(self, hjepa_outputs: dict) -> torch.Tensor:
        """Extract the CLS scene summary vector.

        Args:
            hjepa_outputs: HJEPA output dict containing target_features [B, 197, D].

        Returns:
            Scene summary tensor [B, D].
        """
        return hjepa_outputs["target_features"][:, 0, :]

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        domain_state: dict,
        actor_mode: str = "mode2",
        store_to_memory: bool = True,
        input_kind: str = "auto",
    ) -> dict[str, Any]:
        """Run the full Chamelia pipeline.

        Args:
            tokens: Input tensor passed to HJEPA. This may be images [B, C, H, W] or
                pre-embedded tokens [B, N, D] depending on ``input_kind``.
            mask: Binary patch mask [B, N].
            domain_state: Opaque domain-state dict.
            actor_mode: Actor mode string, "mode1" or "mode2".
            store_to_memory: Whether to store the current episode in latent memory.
            input_kind: ``image``, ``embedded_tokens``, or ``auto``.

        Returns:
            Dict containing:
                - action: Domain-decoded action object(s)
                - action_vec: [B, A]
                - ctx_tokens: [B, C, D]
                - cost: dict with [B] tensors
                - z: [B, D]
                - hjepa_out: raw HJEPA output dict
        """
        hjepa_out = forward_hjepa(self.hjepa, tokens, mask, input_kind=input_kind)
        z = self._get_scene_summary(hjepa_out)
        level_feats = self._extract_level_features(hjepa_out)

        retrieved_keys, episodes = self.memory.retrieve(z)
        _ = episodes

        ctx_tokens = self.configurator(
            hjepa_outputs={"target_features_per_level": level_feats},
            memory_keys=retrieved_keys.to(z.device) if retrieved_keys is not None else None,
        )
        action_vec = self.actor(z, ctx_tokens, mode=actor_mode)
        action = self.domain.decode_action(action_vec)
        cost_out = self.cost(z, action_vec, ctx_tokens, domain_state)

        if store_to_memory:
            self._pending_record_indices = []
            for batch_idx in range(z.shape[0]):
                record = EpisodeRecord(
                    key=z.detach()[batch_idx],
                    action=action_vec.detach()[batch_idx],
                    ctx_tokens=ctx_tokens.detach()[batch_idx],
                    ic_at_decision=float(cost_out["ic"][batch_idx].item()),
                    ic_realized=None,
                    tc_predicted=float(cost_out["tc"][batch_idx].item()),
                    outcome_key=None,
                    step=self._step_counter,
                    domain_name=self.domain.domain_name,
                )
                self._pending_record_indices.append(self.memory.store(record))
        else:
            self._pending_record_indices = []

        self._step_counter += 1
        return {
            "action": action,
            "action_vec": action_vec,
            "ctx_tokens": ctx_tokens,
            "cost": cost_out,
            "z": z,
            "hjepa_out": hjepa_out,
        }

    def fill_outcome(
        self,
        ic_realized: float | torch.Tensor | list[float],
        outcome_observation: Any,
    ) -> None:
        """Fill delayed outcome into memory.

        Args:
            ic_realized: Realized intrinsic cost scalar, list, or tensor [B].
            outcome_observation: Raw observation to tokenize and encode. May be batched.

        Returns:
            None.
        """
        if not self._pending_record_indices:
            return

        tokenizer = self.domain.get_tokenizer()
        with torch.no_grad():
            outcome_tokenized = tokenizer(outcome_observation)
            outcome_tokens = outcome_tokenized.tokens
            outcome_mask = torch.zeros(
                outcome_tokens.shape[0],
                outcome_tokens.shape[1],
                device=outcome_tokens.device,
                dtype=torch.float32,
            )
            outcome_hjepa = forward_hjepa(
                self.hjepa,
                outcome_tokens,
                mask=outcome_mask,
                input_kind="embedded_tokens",
            )
            outcome_z = self._get_scene_summary(outcome_hjepa)

        realized_tensor = torch.as_tensor(ic_realized, dtype=torch.float32).flatten()
        if realized_tensor.numel() == 1 and outcome_z.shape[0] > 1:
            realized_tensor = realized_tensor.repeat(outcome_z.shape[0])

        batch_count = min(len(self._pending_record_indices), outcome_z.shape[0], realized_tensor.shape[0])
        for batch_idx in range(batch_count):
            self.memory.fill_outcome(
                self._pending_record_indices[batch_idx],
                ic_realized=float(realized_tensor[batch_idx].item()),
                outcome_key=outcome_z.detach()[batch_idx],
            )
        self._pending_record_indices = []

    def train_critic_from_memory(self) -> torch.Tensor | None:
        """Build a critic loss from memory-stored realized outcomes.

        Args:
            None.

        Returns:
            Scalar tensor [] critic loss, or None if memory has no realized outcomes.
        """
        keys, ics = self.memory.get_critic_training_pairs()
        if keys is None or ics is None:
            return None

        device = next(self.parameters()).device
        B = keys.shape[0]
        dummy_ctx = torch.zeros(B, self.num_ctx_tokens, self.embed_dim, device=device)
        predicted = self.cost.trainable_critic(keys.to(device), dummy_ctx)
        return self.cost.trainable_critic.compute_critic_loss(predicted, ics.to(device))

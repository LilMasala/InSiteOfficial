"""Shared batch contracts for curriculum training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CurriculumBatch:
    """Standardized curriculum batch emitted by stage/domain dataloaders."""

    domain_name: str
    raw_inputs: Any
    tokens: torch.Tensor | None
    embedded_tokens: torch.Tensor | None
    input_mask: torch.Tensor
    targets: dict[str, Any]
    domain_state: dict[str, Any]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate the minimal curriculum batch contract."""
        if self.tokens is None and self.embedded_tokens is None:
            raise ValueError("CurriculumBatch requires either tokens or embedded_tokens.")
        if self.tokens is not None and self.tokens.dim() < 2:
            raise ValueError(f"tokens must have at least shape [B, N], got {tuple(self.tokens.shape)}.")
        if self.embedded_tokens is not None and self.embedded_tokens.dim() != 3:
            raise ValueError(
                f"embedded_tokens must have shape [B, N, D], got {tuple(self.embedded_tokens.shape)}."
            )
        if self.input_mask.dim() != 2:
            raise ValueError(f"input_mask must have shape [B, N], got {tuple(self.input_mask.shape)}.")

    @property
    def batch_size(self) -> int:
        """Return batch size."""
        if self.tokens is not None:
            return int(self.tokens.shape[0])
        return int(self.embedded_tokens.shape[0])  # type: ignore[union-attr]

    @property
    def seq_len(self) -> int:
        """Return sequence length."""
        if self.tokens is not None:
            return int(self.tokens.shape[1])
        return int(self.embedded_tokens.shape[1])  # type: ignore[union-attr]

    def to_device(self, device: torch.device | str) -> "CurriculumBatch":
        """Move tensor members onto a device.

        Args:
            device: Target device.

        Returns:
            A new ``CurriculumBatch`` with tensors moved to the target device.
        """
        target_device = torch.device(device)
        moved_targets = {
            key: value.to(target_device) if isinstance(value, torch.Tensor) else value
            for key, value in self.targets.items()
        }
        moved_state = {
            key: value.to(target_device) if isinstance(value, torch.Tensor) else value
            for key, value in self.domain_state.items()
        }
        return CurriculumBatch(
            domain_name=self.domain_name,
            raw_inputs=self.raw_inputs,
            tokens=self.tokens.to(target_device) if self.tokens is not None else None,
            embedded_tokens=(
                self.embedded_tokens.to(target_device) if self.embedded_tokens is not None else None
            ),
            input_mask=self.input_mask.to(target_device),
            targets=moved_targets,
            domain_state=moved_state,
            metadata=dict(self.metadata),
        )


@dataclass
class ChameliaStepBatch:
    """Normalized model-step batch passed into Chamelia."""

    domain_name: str
    model_inputs: torch.Tensor
    input_mask: torch.Tensor
    input_kind: str
    targets: dict[str, Any]
    domain_state: dict[str, Any]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate the normalized model-step batch."""
        if self.input_kind not in {"image", "token_ids", "embedded_tokens"}:
            raise ValueError(f"Unsupported input_kind '{self.input_kind}'.")
        if self.input_mask.dim() != 2:
            raise ValueError(
                f"input_mask must have shape [B, N], got {tuple(self.input_mask.shape)}."
            )

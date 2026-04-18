"""Action geometry contracts and structured action containers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class ActionKind(str, Enum):
    """Supported action families."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    GENERATIVE = "generative"


@dataclass(frozen=True)
class ActionSpec:
    """Immutable description of the active domain action space."""

    kind: ActionKind
    continuous_dim: int | None = None
    discrete_vocab_size: int | None = None
    generative_chunk_dim: int | None = None

    def __post_init__(self) -> None:
        active_fields = [
            self.continuous_dim is not None,
            self.discrete_vocab_size is not None,
            self.generative_chunk_dim is not None,
        ]
        if sum(active_fields) != 1:
            raise ValueError("ActionSpec must define exactly one action payload.")
        width = self.primary_width
        if width <= 0:
            raise ValueError("ActionSpec widths must be positive integers.")
        if self.kind == ActionKind.CONTINUOUS and self.continuous_dim is None:
            raise ValueError("Continuous ActionSpec requires continuous_dim.")
        if self.kind == ActionKind.DISCRETE and self.discrete_vocab_size is None:
            raise ValueError("Discrete ActionSpec requires discrete_vocab_size.")
        if self.kind == ActionKind.GENERATIVE and self.generative_chunk_dim is None:
            raise ValueError("Generative ActionSpec requires generative_chunk_dim.")

    @classmethod
    def continuous(cls, dim: int) -> "ActionSpec":
        return cls(kind=ActionKind.CONTINUOUS, continuous_dim=int(dim))

    @classmethod
    def discrete(cls, vocab_size: int) -> "ActionSpec":
        return cls(kind=ActionKind.DISCRETE, discrete_vocab_size=int(vocab_size))

    @classmethod
    def generative(cls, chunk_dim: int) -> "ActionSpec":
        return cls(kind=ActionKind.GENERATIVE, generative_chunk_dim=int(chunk_dim))

    @property
    def primary_width(self) -> int:
        if self.kind == ActionKind.CONTINUOUS:
            return int(self.continuous_dim or 0)
        if self.kind == ActionKind.DISCRETE:
            return int(self.discrete_vocab_size or 0)
        return int(self.generative_chunk_dim or 0)


@dataclass(frozen=True)
class ActionPath:
    """Structured action path container keyed by ActionSpec."""

    action_spec: ActionSpec
    continuous: torch.Tensor | None = None
    discrete_ids: torch.Tensor | None = None
    discrete_logits: torch.Tensor | None = None
    generative_chunks: torch.Tensor | None = None
    generated_tokens: torch.Tensor | None = None

    def __post_init__(self) -> None:
        payload = self.primary
        if payload is None:
            raise ValueError("ActionPath requires a payload matching action_spec.")

    @property
    def primary(self) -> torch.Tensor | None:
        if self.action_spec.kind == ActionKind.CONTINUOUS:
            return self.continuous
        if self.action_spec.kind == ActionKind.DISCRETE:
            return self.discrete_ids if self.discrete_ids is not None else self.discrete_logits
        return self.generative_chunks

    @property
    def device(self) -> torch.device:
        payload = self.primary
        if payload is None:
            return torch.device("cpu")
        return payload.device

    @property
    def dtype(self) -> torch.dtype:
        payload = self.primary
        if payload is None:
            return torch.float32
        return payload.dtype

    def as_tensor(self) -> torch.Tensor:
        """Return a dense tensor view compatible with legacy code."""
        if self.action_spec.kind == ActionKind.CONTINUOUS:
            if self.continuous is None:
                raise ValueError("Continuous ActionPath is missing continuous payload.")
            return self.continuous
        if self.action_spec.kind == ActionKind.DISCRETE:
            if self.discrete_ids is not None:
                return self.discrete_ids.float().unsqueeze(-1)
            if self.discrete_logits is None:
                raise ValueError("Discrete ActionPath is missing ids/logits payload.")
            return self.discrete_logits.argmax(dim=-1, keepdim=True).float()
        if self.generative_chunks is None:
            raise ValueError("Generative ActionPath is missing chunk payload.")
        return self.generative_chunks

    def first_action_tensor(self) -> torch.Tensor:
        tensor = self.as_tensor()
        if tensor.dim() >= 2:
            return tensor[..., 0, :]
        return tensor

    def detach(self) -> "ActionPath":
        return ActionPath(
            action_spec=self.action_spec,
            continuous=None if self.continuous is None else self.continuous.detach(),
            discrete_ids=None if self.discrete_ids is None else self.discrete_ids.detach(),
            discrete_logits=None if self.discrete_logits is None else self.discrete_logits.detach(),
            generative_chunks=(
                None if self.generative_chunks is None else self.generative_chunks.detach()
            ),
            generated_tokens=(
                None if self.generated_tokens is None else self.generated_tokens.detach()
            ),
        )

    def to(self, *args: Any, **kwargs: Any) -> "ActionPath":
        return ActionPath(
            action_spec=self.action_spec,
            continuous=None if self.continuous is None else self.continuous.to(*args, **kwargs),
            discrete_ids=None if self.discrete_ids is None else self.discrete_ids.to(*args, **kwargs),
            discrete_logits=(
                None if self.discrete_logits is None else self.discrete_logits.to(*args, **kwargs)
            ),
            generative_chunks=(
                None if self.generative_chunks is None else self.generative_chunks.to(*args, **kwargs)
            ),
            generated_tokens=(
                None if self.generated_tokens is None else self.generated_tokens.to(*args, **kwargs)
            ),
        )


def coerce_action_path(
    actions: torch.Tensor | ActionPath,
    action_spec: ActionSpec | None = None,
) -> ActionPath:
    """Normalize either a tensor or ActionPath into an ActionPath."""
    if isinstance(actions, ActionPath):
        return actions
    spec = action_spec or ActionSpec.continuous(actions.shape[-1])
    if spec.kind == ActionKind.CONTINUOUS:
        return ActionPath(action_spec=spec, continuous=actions)
    if spec.kind == ActionKind.DISCRETE:
        discrete_ids = actions.long()
        if discrete_ids.dim() > 0 and discrete_ids.shape[-1] == 1:
            discrete_ids = discrete_ids.squeeze(-1)
        return ActionPath(action_spec=spec, discrete_ids=discrete_ids)
    return ActionPath(action_spec=spec, generative_chunks=actions)

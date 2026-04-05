"""Base abstractions for curriculum domains and masking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.chamelia.plugins.base import AbstractDomain
from src.chamelia.tokenizers import BoardTokenizer, SequenceTokenizer
from training.curriculum.batch import CurriculumBatch
from training.curriculum.cost_schedule import CostLevel, MaturingIntrinsicCost


class MaskingStrategy(ABC):
    """Abstract masking strategy used by curriculum domains."""

    @abstractmethod
    def apply(self, tokens: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask a batch of tokens.

        Args:
            tokens: Token tensor of shape [B, N].
            level: Current curriculum level.

        Returns:
            Tuple of masked tokens [B, N] and binary mask [B, N].
        """


class CurriculumDomain(ABC):
    """Abstract domain interface for curriculum training."""

    @abstractmethod
    def domain_name(self) -> str:
        """Return the human-readable domain identifier."""

    @abstractmethod
    def stage(self) -> int:
        """Return the curriculum stage index."""

    @abstractmethod
    def get_cost_schedule(self) -> list[CostLevel]:
        """Return the list of developmental cost levels."""

    @abstractmethod
    def get_data_loader(self, level: int, split: str) -> DataLoader:
        """Return a dataloader for a stage/level/split."""

    @abstractmethod
    def get_masking_strategy(self, level: int) -> MaskingStrategy:
        """Return the masking strategy active at this level."""

    @abstractmethod
    def run_advancement_probe(self, model: Any, level: int) -> dict[str, float]:
        """Run the current advancement probe and return metrics."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Return the domain action dimension."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the domain vocabulary size."""

    def build_runtime_domain(self, embed_dim: int) -> AbstractDomain | None:
        """Optionally build a Chamelia runtime domain plugin.

        Args:
            embed_dim: Target embedding dimension.

        Returns:
            Runtime domain plugin or ``None`` if the domain has no wired runtime path yet.
        """
        _ = embed_dim
        return None

    def build_curriculum_batch(
        self,
        raw_batch: dict[str, Any],
        split: str,
    ) -> CurriculumBatch:
        """Convert a collated raw batch into a standardized curriculum batch.

        Args:
            raw_batch: Collated tensor dictionary.
            split: Dataset split identifier.

        Returns:
            ``CurriculumBatch``.
        """
        tokens = raw_batch["tokens"]
        input_mask = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.float32)
        targets = {key: value for key, value in raw_batch.items() if key != "tokens"}
        domain_state = dict(raw_batch)
        return CurriculumBatch(
            domain_name=self.domain_name(),
            raw_inputs=tokens.clone(),
            tokens=tokens,
            embedded_tokens=None,
            input_mask=input_mask,
            targets=targets,
            domain_state=domain_state,
            metadata={"split": split, "stage": self.stage(), "level": self.cost.current_level},
        )


def _default_advancement_probe(
    probe_results: dict[str, Any],
    thresholds: dict[str, float],
) -> bool:
    """Check that all named metrics meet their thresholds."""
    return all(float(probe_results.get(key, float("-inf"))) >= value for key, value in thresholds.items())


class TensorSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Tiny synthetic dataset used to scaffold curriculum domain wiring."""

    def __init__(
        self,
        samples: list[dict[str, torch.Tensor]],
    ) -> None:
        """Initialize the synthetic dataset.

        Args:
            samples: List of sample dictionaries with compatible tensor shapes.
        """
        self.samples = samples

    def __len__(self) -> int:
        """Return number of synthetic samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one sample dictionary."""
        return self.samples[index]


@dataclass
class DomainSpec:
    """Static metadata for a curriculum domain."""

    name: str
    stage_idx: int
    action_dim: int
    vocab_size: int
    batch_size: int = 16
    seq_len: int = 32
    dataset_size: int = 64


class BaseCurriculumDomain(CurriculumDomain):
    """Convenience base class providing a usable curriculum-domain skeleton."""

    def __init__(
        self,
        spec: DomainSpec,
        masking_strategy: MaskingStrategy,
        cost_schedule: list[CostLevel],
        probe_fn: Callable[[Any, int], dict[str, float]],
        sample_builder: Callable[[int, str, DomainSpec], list[dict[str, torch.Tensor]]],
    ) -> None:
        """Initialize a scaffold curriculum domain.

        Args:
            spec: Static metadata for the domain.
            masking_strategy: Masking strategy used for all levels.
            cost_schedule: Developmental cost schedule.
            probe_fn: Probe callback for advancement and graduation metrics.
            sample_builder: Callable that emits synthetic samples for loaders.
        """
        self.spec = spec
        self._masking_strategy = masking_strategy
        self._cost_schedule = cost_schedule
        self.cost = MaturingIntrinsicCost(cost_schedule, spec.name)
        self._probe_fn = probe_fn
        self._sample_builder = sample_builder

    def _collate_samples(self, samples: list[dict[str, Any]], split: str) -> CurriculumBatch:
        """Collate raw sample dictionaries into a standardized batch.

        Args:
            samples: Raw sample list.
            split: Dataset split.

        Returns:
            ``CurriculumBatch`` for the active domain.
        """
        collated: dict[str, Any] = {}
        keys = samples[0].keys()
        for key in keys:
            values = [sample[key] for sample in samples]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values, dim=0)
            else:
                collated[key] = values
        return self.build_curriculum_batch(collated, split)

    def _current_cost_terms(self) -> list[tuple[callable, float]]:
        """Return the current level's cost functions and weights."""
        return self.cost.cost_schedule[self.cost.current_level].cost_fns

    def build_sequence_runtime_domain(self, embed_dim: int) -> AbstractDomain:
        """Build a generic sequence-based runtime plugin."""
        return SequenceRuntimeDomain(owner=self, embed_dim=embed_dim)

    def build_board_runtime_domain(self, embed_dim: int) -> AbstractDomain:
        """Build a generic board/grid runtime plugin."""
        return BoardRuntimeDomain(owner=self, embed_dim=embed_dim)

    def domain_name(self) -> str:
        """Return the domain name."""
        return self.spec.name

    def stage(self) -> int:
        """Return the stage index."""
        return self.spec.stage_idx

    def get_cost_schedule(self) -> list[CostLevel]:
        """Return the configured cost schedule."""
        return self._cost_schedule

    def get_data_loader(self, level: int, split: str) -> DataLoader:
        """Build a synthetic dataloader for the requested split.

        Args:
            level: Active cost level.
            split: Dataset split name such as ``train`` or ``val``.

        Returns:
            Dataloader yielding dict batches whose ``tokens`` field is [B, N].
        """
        seed_material = f"{self.spec.name}:{self.spec.stage_idx}:{level}:{split}:{self.spec.dataset_size}"
        seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16) % (2**31)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            samples = self._sample_builder(level, split, self.spec)
        dataset = TensorSequenceDataset(samples)
        generator = torch.Generator()
        generator.manual_seed(seed + 1)
        return DataLoader(
            dataset,
            batch_size=self.spec.batch_size,
            shuffle=(split == "train"),
            generator=generator,
            collate_fn=lambda batch: self._collate_samples(batch, split),
        )

    def get_masking_strategy(self, level: int) -> MaskingStrategy:
        """Return the active masking strategy."""
        _ = level
        return self._masking_strategy

    def run_advancement_probe(self, model: Any, level: int) -> dict[str, float]:
        """Run the domain's advancement probe."""
        return self._probe_fn(model, level)

    @property
    def action_dim(self) -> int:
        """Return the synthetic action dimension."""
        return self.spec.action_dim

    @property
    def vocab_size(self) -> int:
        """Return the synthetic vocabulary size."""
        return self.spec.vocab_size


def build_threshold_probe(metric_name: str) -> Callable[[Any, int], dict[str, float]]:
    """Build a simple probe callback keyed on the active level.

    Args:
        metric_name: Primary metric to emit.

    Returns:
        Callable emitting deterministic placeholder metrics for a given level.
    """

    def probe_fn(model: Any, level: int) -> dict[str, float]:
        _ = model
        base = min(0.99, 0.55 + 0.1 * level)
        return {
            metric_name: base,
            "consistency": min(0.99, base + 0.05),
            "generalization": min(0.99, base + 0.03),
        }

    return probe_fn


def make_level(
    level: int,
    description: str,
    cost_fns: list[tuple[callable, float]],
    thresholds: dict[str, float],
    min_episodes: int,
) -> CostLevel:
    """Construct a standardized ``CostLevel``."""
    return CostLevel(
        level=level,
        description=description,
        cost_fns=cost_fns,
        advancement_probe=_default_advancement_probe,
        advancement_threshold=thresholds,
        min_episodes_at_level=min_episodes,
    )


class SequenceRuntimeDomain(AbstractDomain):
    """Generic runtime domain plugin for sequence-like curriculum domains."""

    def __init__(self, owner: BaseCurriculumDomain, embed_dim: int) -> None:
        self.owner = owner
        self._tokenizer = SequenceTokenizer(
            vocab_size=owner.vocab_size,
            embed_dim=embed_dim,
            max_seq_len=owner.spec.seq_len,
            domain_name=f"{owner.domain_name()}_runtime",
            pad_token_id=0,
        )

    def get_tokenizer(self) -> SequenceTokenizer:
        """Return the sequence tokenizer."""
        return self._tokenizer

    def get_action_dim(self) -> int:
        """Return the runtime action dimension."""
        return self.owner.action_dim

    def decode_action(self, action_vec: torch.Tensor) -> Any:
        """Decode the actor output."""
        if self.get_action_dim() == self.vocab_size:
            return action_vec.argmax(dim=-1)
        return action_vec

    def get_intrinsic_cost_fns(self) -> list[tuple[Callable, float]]:
        """Return the current level's fixed intrinsic costs."""
        return self.owner._current_cost_terms()

    def get_domain_state(self, observation: Any) -> dict:
        """Construct an opaque domain state."""
        tokens = observation if torch.is_tensor(observation) else torch.tensor(observation)
        return {"tokens": tokens}

    def prepare_bridge_observation(self, observation: Any) -> Any:
        """Normalize bridge payloads into tokenizer-ready sequence tokens."""
        if isinstance(observation, dict) and "tokens" in observation:
            observation = observation["tokens"]
        return observation if torch.is_tensor(observation) else torch.tensor(observation, dtype=torch.long)

    def compute_regime_embedding(self, domain_state: dict) -> torch.Tensor | None:
        """Return no explicit regime embedding by default."""
        _ = domain_state
        return None

    @property
    def domain_name(self) -> str:
        """Return the domain name."""
        return self.owner.domain_name()

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        return self.owner.vocab_size


class BoardRuntimeDomain(AbstractDomain):
    """Generic runtime domain plugin for board/grid curriculum domains."""

    def __init__(self, owner: BaseCurriculumDomain, embed_dim: int) -> None:
        self.owner = owner
        self._tokenizer = BoardTokenizer(
            vocab_size=owner.vocab_size,
            embed_dim=embed_dim,
            max_seq_len=owner.spec.seq_len,
            domain_name=f"{owner.domain_name()}_runtime",
            pad_token_id=0,
        )

    def get_tokenizer(self) -> BoardTokenizer:
        """Return the board tokenizer."""
        return self._tokenizer

    def get_action_dim(self) -> int:
        """Return the runtime action dimension."""
        return self.owner.action_dim

    def decode_action(self, action_vec: torch.Tensor) -> Any:
        """Decode actor output for board-like domains."""
        return action_vec.argmax(dim=-1) if action_vec.dim() == 2 else action_vec

    def get_intrinsic_cost_fns(self) -> list[tuple[Callable, float]]:
        """Return the current level's fixed intrinsic costs."""
        return self.owner._current_cost_terms()

    def get_domain_state(self, observation: Any) -> dict:
        """Construct an opaque board-state payload."""
        tokens = observation if torch.is_tensor(observation) else torch.tensor(observation)
        return {"tokens": tokens}

    def prepare_bridge_observation(self, observation: Any) -> Any:
        """Normalize bridge payloads into tokenizer-ready board tokens."""
        if isinstance(observation, dict) and "tokens" in observation:
            observation = observation["tokens"]
        return observation if torch.is_tensor(observation) else torch.tensor(observation, dtype=torch.long)

    def compute_regime_embedding(self, domain_state: dict) -> torch.Tensor | None:
        """Board domains do not expose a regime embedding by default."""
        _ = domain_state
        return None

    @property
    def domain_name(self) -> str:
        """Return the domain name."""
        return self.owner.domain_name()

    @property
    def vocab_size(self) -> int:
        """Return board vocabulary size."""
        return self.owner.vocab_size

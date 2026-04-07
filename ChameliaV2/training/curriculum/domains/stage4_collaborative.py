"""Stage 4: collaborative and social reasoning curriculum domain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from training.curriculum.data.public_collaboration import (
    DEFAULT_CURRICULUM_ROOT,
    load_public_collaboration_samples,
)
from training.curriculum.domains.base import BaseCurriculumDomain, DomainSpec, MaskingStrategy, make_level


class CollaborativeMaskingStrategy(MaskingStrategy):
    """Mask one participant's information to force coordination reasoning."""

    def apply(self, tokens: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask collaborator-visible spans."""
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.float32)
        half = tokens.shape[1] // 2
        if level < 2:
            mask[:, half:] = 1.0
        else:
            mask[:, ::2] = 1.0
        masked[mask.bool()] = 0
        return masked, mask


def _collab_samples(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
    """Generate structured partial-information collaboration samples."""
    generator = torch.Generator().manual_seed(400 + level + (0 if split == "train" else 3000))
    samples: list[dict[str, torch.Tensor]] = []
    context_width = max(4, spec.seq_len // 3)
    for _ in range(spec.dataset_size):
        agent_a = torch.randint(1, spec.vocab_size // 2, (context_width,), generator=generator)
        agent_b = torch.randint(1, spec.vocab_size // 2, (context_width,), generator=generator)
        filler = torch.randint(1, spec.vocab_size // 2, (max(1, spec.seq_len - 2 * context_width),), generator=generator)
        answer = int((agent_a.sum().item() + 2 * agent_b.sum().item() + level) % max(1, spec.vocab_size - 1)) + 1
        stream = torch.cat((agent_a, agent_b, filler, torch.tensor([answer], dtype=torch.long)))
        samples.append(
            {
                "tokens": stream[: spec.seq_len].clone(),
                "target": stream[1 : spec.seq_len + 1].clone(),
                "answer": torch.tensor(answer, dtype=torch.long),
                "regime": torch.tensor(float(level), dtype=torch.float32),
            }
        )
    return samples


class CollaborativeCurriculumDomain(BaseCurriculumDomain):
    """Stage 4 collaboration domain with local trace support."""

    def __init__(
        self,
        domain_variant: str = "collab_selfplay",
        batch_size: int = 8,
        seq_len: int = 32,
        vocab_size: int = 1024,
        data_root: str | Path | None = None,
    ) -> None:
        """Initialize the collaboration domain."""
        self.data_root = Path(data_root) if data_root is not None else DEFAULT_CURRICULUM_ROOT
        self._sample_cache: dict[tuple[int, str, int, int], list[dict[str, torch.Tensor]]] = {}

        def task_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = z
            answers = domain_state["answer_token"].long().clamp(min=0, max=action.shape[1] - 1)
            return F.cross_entropy(action, answers, reduction="none")

        def coordination_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = action
            return (z.mean(dim=-1) - domain_state["target"].float().mean(dim=-1)).abs()

        schedule = [
            make_level(0, "structured decomposition", [(task_cost, 0.8), (coordination_cost, 0.2)], {"joint_outcome": 0.80}, 64),
            make_level(1, "negotiated decomposition", [(task_cost, 0.75), (coordination_cost, 0.25)], {"joint_outcome": 0.84}, 64),
            make_level(
                2,
                "communication-limited coordination",
                [(task_cost, 0.7), (coordination_cost, 0.3)],
                {"joint_outcome": 0.88},
                64,
            ),
            make_level(
                3,
                "future action anticipation",
                [(task_cost, 0.65), (coordination_cost, 0.35)],
                {"joint_outcome": 0.90, "partner_prediction": 0.70},
                64,
            ),
            make_level(
                4,
                "adversarial partner adaptation",
                [(task_cost, 0.6), (coordination_cost, 0.4)],
                {"joint_outcome": 0.92, "partner_prediction": 0.76},
                64,
            ),
        ]

        def probe_fn(model: Any, level: int) -> dict[str, float]:
            # Fallback only when model is None — never called when model is available.
            _ = model
            return {"joint_outcome": 0.0, "partner_prediction": 0.0}

        def sample_builder(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
            key = (level, split, spec.seq_len, spec.dataset_size)
            cached = self._sample_cache.get(key)
            if cached is not None:
                return cached
            public_samples = load_public_collaboration_samples(
                split=split,
                vocab_size=spec.vocab_size,
                seq_len=spec.seq_len,
                max_samples=spec.dataset_size,
                data_root=self.data_root,
            )
            samples = public_samples or _collab_samples(level, split, spec)
            self._sample_cache[key] = samples
            return samples

        super().__init__(
            spec=DomainSpec(
                name=domain_variant,
                stage_idx=4,
                action_dim=vocab_size,
                vocab_size=vocab_size,
                batch_size=batch_size,
                seq_len=seq_len,
                dataset_size=256,
            ),
            masking_strategy=CollaborativeMaskingStrategy(),
            cost_schedule=schedule,
            probe_fn=probe_fn,
            sample_builder=sample_builder,
        )

    def build_curriculum_batch(
        self,
        raw_batch: dict[str, Any],
        split: str,
    ):
        """Attach answer-token supervision to the standard curriculum batch."""
        batch = super().build_curriculum_batch(raw_batch, split)
        answers = raw_batch["answer"].long() if "answer" in raw_batch else raw_batch["target"][:, -1].long()
        batch.targets["answer"] = answers
        batch.targets["answer_token"] = answers
        batch.domain_state["answer"] = answers
        batch.domain_state["answer_token"] = answers
        batch.domain_state["target"] = raw_batch["target"]
        return batch

    def run_advancement_probe(self, model: Any, level: int) -> dict[str, float]:
        """Run held-out answer-token accuracy probes when a model is available."""
        if model is None:
            return super().run_advancement_probe(model, level)

        device = next(model.parameters()).device
        previous_domain = getattr(model, "domain", None)

        runtime_domain = self.build_runtime_domain(model.embed_dim)
        if runtime_domain is None:
            return super().run_advancement_probe(model, level)
        model.set_domain(runtime_domain)

        try:
            train_acc = self._probe_split_accuracy(model, level, split="train", device=device)
            val_acc = self._probe_split_accuracy(model, level, split="val", device=device)
            return {
                "joint_outcome": val_acc,
                "partner_prediction": max(0.0, 1.0 - abs(train_acc - val_acc)),
            }
        finally:
            if previous_domain is not None:
                model.set_domain(previous_domain)

    def _probe_split_accuracy(
        self,
        model: Any,
        level: int,
        split: str,
        device: torch.device,
    ) -> float:
        """Evaluate answer-token accuracy on one split."""
        loader = self.get_data_loader(level, split=split)
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to_device(device)
                runtime_domain = getattr(model, "domain", None)
                if runtime_domain is None or getattr(runtime_domain, "domain_name", "") != self.spec.name:
                    runtime_domain = self.build_runtime_domain(model.embed_dim)
                    if runtime_domain is None:
                        return 0.0
                    model.set_domain(runtime_domain)
                tokenized = runtime_domain.get_tokenizer()(batch.tokens.long())
                outputs = model(
                    tokens=tokenized.tokens,
                    mask=batch.input_mask,
                    domain_state=batch.domain_state,
                    actor_mode="mode2",
                    store_to_memory=False,
                    input_kind="embedded_tokens",
                )
                pred = outputs["action_vec"].argmax(dim=-1)
                target = batch.domain_state["answer_token"].long()
                correct += int((pred == target).sum().item())
                total += int(target.numel())
        return correct / max(1, total)

    def build_runtime_domain(self, embed_dim: int):
        """Build a sequence-based runtime plugin for collaboration traces."""
        return self.build_sequence_runtime_domain(embed_dim)

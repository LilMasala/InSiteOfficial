"""Stage 0: language foundation curriculum domain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from training.curriculum.data.public_sequence_data import token_id_for_text

from training.curriculum.data.public_language import (
    DEFAULT_CURRICULUM_ROOT,
    load_public_language_samples,
)
from training.curriculum.domains.base import (
    BaseCurriculumDomain,
    DomainSpec,
    MaskingStrategy,
    build_threshold_probe,
    make_level,
)


class LanguageMaskingStrategy(MaskingStrategy):
    """Language masking schedule from local token masking to discourse masking."""

    def apply(self, tokens: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply curriculum-aware language masking."""
        ratio = min(0.15 + 0.1 * level, 0.6)
        mask = torch.rand(tokens.shape, device=tokens.device) < ratio
        masked = tokens.clone()
        masked[mask] = 0
        return masked, mask.to(dtype=torch.float32)


def _language_samples(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
    """Create synthetic shifted-token language samples as a fallback."""
    generator = torch.Generator().manual_seed(100 + level + (0 if split == "train" else 1000))
    samples: list[dict[str, torch.Tensor]] = []
    for _ in range(spec.dataset_size):
        stream = torch.randint(1, spec.vocab_size, (spec.seq_len + 1,), generator=generator)
        samples.append(
            {
                "tokens": stream[:-1].clone(),
                "target": stream[1:].clone(),
                "answer": stream[-1].clone(),
                "regime": torch.tensor(float(level), dtype=torch.float32),
            }
        )
    return samples


def _sentence_sample(
    text: str,
    *,
    seq_len: int,
    vocab_size: int,
    regime: float,
) -> dict[str, torch.Tensor]:
    """Build one synthetic language sample with a controllable final answer token."""
    pieces = text.split()
    token_ids = [token_id_for_text(piece, vocab_size) for piece in pieces]
    if not token_ids:
        token_ids = [token_id_for_text("<empty>", vocab_size)]
    while len(token_ids) < seq_len + 1:
        token_ids.extend(token_ids[: max(1, seq_len + 1 - len(token_ids))])
    token_ids = token_ids[: seq_len + 1]
    stream = torch.tensor(token_ids, dtype=torch.long)
    return {
        "tokens": stream[:-1].clone(),
        "target": stream[1:].clone(),
        "answer": stream[-1].clone(),
        "regime": torch.tensor(float(regime), dtype=torch.float32),
    }


def _structured_language_samples(
    level: int,
    split: str,
    spec: DomainSpec,
) -> list[dict[str, torch.Tensor]]:
    """Create level-aware synthetic language tasks with strong predictive structure.

    The lower Stage-0 levels need a learnable curriculum, not a hard open-ended
    hashed-next-token task over arbitrary public text. These templates preserve the
    sequence-supervision interface while giving the model progressively harder language
    regularities to master.
    """
    generator = torch.Generator().manual_seed(9100 + (97 * level) + (0 if split == "train" else 1000))
    samples: list[dict[str, torch.Tensor]] = []
    dataset_size = max(spec.dataset_size, 4096)

    subjects = [
        "doctor", "artist", "teacher", "pilot", "chemist", "farmer", "writer", "engineer",
        "nurse", "runner", "baker", "singer", "captain", "student", "dancer", "scientist",
    ]
    colors = [
        "amber", "blue", "crimson", "gold", "green", "indigo", "ivory", "jade",
        "lavender", "maroon", "navy", "orange", "silver", "teal", "violet", "white",
    ]
    animals = [
        "otter", "falcon", "panda", "rabbit", "tiger", "whale", "yak", "zebra",
        "badger", "cougar", "dolphin", "eagle", "fox", "gecko", "heron", "ibis",
    ]
    actions = [
        "gather", "notice", "prefer", "carry", "paint", "follow", "assemble", "study",
        "measure", "protect", "repair", "record", "sort", "trace", "watch", "weigh",
    ]
    places = [
        "archive", "garden", "harbor", "kitchen", "library", "market", "museum", "studio",
        "tower", "workshop", "forest", "station", "valley", "village", "school", "clinic",
    ]

    for sample_idx in range(dataset_size):
        idx = int(torch.randint(0, len(subjects), (1,), generator=generator).item())
        alt = int(torch.randint(0, len(subjects), (1,), generator=generator).item())
        subj = subjects[idx]
        color = colors[idx]
        animal = animals[idx]
        action = actions[idx]
        place = places[idx]
        alt_subj = subjects[alt]
        alt_color = colors[alt]
        alt_animal = animals[alt]
        alt_action = actions[alt]
        alt_place = places[alt]

        if level == 0:
            text = (
                f"{subj} likes {color} . "
                f"{subj} likes {color} . "
                f"{subj} likes {color} . "
                f"{subj} likes"
            )
            regime = 0.0
        elif level == 1:
            text = (
                f"{subj} keeps {animal} in the {place} . "
                f"{alt_subj} keeps {alt_animal} in the {alt_place} . "
                f"{subj} keeps {animal} in the"
            )
            regime = 0.2
        elif level == 2:
            verdict = "support" if idx % 2 == 0 else "reject"
            evidence = "stable" if idx % 2 == 0 else "fragile"
            text = (
                f"report topic {subj} evidence {evidence} summary {verdict} . "
                f"analysis topic {subj} evidence {evidence} summary"
            )
            regime = 0.5
        elif level == 3:
            text = (
                f"{subj} will {action} the {animal} before dawn . "
                f"later {subj} will {action} the {animal} before noon . "
                f"finally {subj} will {action} the"
            )
            regime = 0.8
        else:
            relation = "same" if idx % 2 == 0 else "different"
            answer = color if relation == "same" else alt_color
            text = (
                f"memo compare {subj} with {alt_subj} relation {relation} . "
                f"reference {subj} color {color} . "
                f"reference {alt_subj} color {alt_color} . "
                f"conclusion color"
            )
            if sample_idx % 2 == 1:
                text = (
                    f"summary compare {alt_subj} with {subj} relation {relation} . "
                    f"reference {alt_subj} color {alt_color} . "
                    f"reference {subj} color {color} . "
                    f"conclusion color"
                )
            # Make the answer token explicit and deterministically dependent on long-range context.
            text = f"{text} {answer}"
            regime = 1.0

        samples.append(
            _sentence_sample(
                text,
                seq_len=spec.seq_len,
                vocab_size=spec.vocab_size,
                regime=regime,
            )
        )

    return samples


class LanguageCurriculumDomain(BaseCurriculumDomain):
    """Stage 0 language curriculum with local-corpus support."""

    def __init__(
        self,
        batch_size: int = 16,
        seq_len: int = 64,
        vocab_size: int = 8192,
        data_root: str | Path | None = None,
    ) -> None:
        """Initialize the language foundation domain."""
        self.data_root = Path(data_root) if data_root is not None else DEFAULT_CURRICULUM_ROOT
        self._sample_cache: dict[tuple[int, str, int, int], list[dict[str, torch.Tensor]]] = {}

        def next_token_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = z
            answers = domain_state["answer_token"].long().clamp(min=0, max=action.shape[1] - 1)
            return F.cross_entropy(action, answers, reduction="none")

        def representation_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = action
            return (z.mean(dim=-1) - domain_state["target"].float().mean(dim=-1)).abs()

        schedule = [
            make_level(0, "masked next-token prediction", [(next_token_cost, 0.8), (representation_cost, 0.2)], {"token_accuracy": 0.55}, 512),
            make_level(
                1,
                "sentence continuity",
                [(next_token_cost, 0.75), (representation_cost, 0.25)],
                {"token_accuracy": 0.65, "consistency": 0.7},
                512,
            ),
            make_level(
                2,
                "document semantics",
                [(next_token_cost, 0.7), (representation_cost, 0.3)],
                {"token_accuracy": 0.75, "generalization": 0.72},
                1024,
            ),
            make_level(
                3,
                "long-context abstraction",
                [(next_token_cost, 0.65), (representation_cost, 0.35)],
                {"token_accuracy": 0.82, "generalization": 0.78},
                1024,
            ),
            make_level(
                4,
                "cross-domain transfer",
                [(next_token_cost, 0.6), (representation_cost, 0.4)],
                {"token_accuracy": 0.88, "generalization": 0.82},
                1024,
            ),
        ]

        def sample_builder(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
            key = (level, split, spec.seq_len, spec.dataset_size)
            cached = self._sample_cache.get(key)
            if cached is not None:
                return cached
            target_sample_count = max(spec.dataset_size, 4096)
            structured_samples = _structured_language_samples(level, split, spec)
            public_samples = []
            if level >= 3:
                public_samples = load_public_language_samples(
                    split=split,
                    vocab_size=spec.vocab_size,
                    seq_len=spec.seq_len,
                    max_samples=target_sample_count,
                    data_root=self.data_root,
                )
            if public_samples:
                synth_fraction = 0.75 if level >= 4 else 0.5
                synth_count = min(len(structured_samples), int(target_sample_count * synth_fraction))
                public_count = max(0, target_sample_count - synth_count)
                samples = structured_samples[:synth_count] + public_samples[:public_count]
            else:
                samples = structured_samples or _language_samples(level, split, spec)
            self._sample_cache[key] = samples
            return samples

        super().__init__(
            spec=DomainSpec(
                name="language",
                stage_idx=0,
                action_dim=vocab_size,
                vocab_size=vocab_size,
                batch_size=batch_size,
                seq_len=seq_len,
                dataset_size=4096,
            ),
            masking_strategy=LanguageMaskingStrategy(),
            cost_schedule=schedule,
            probe_fn=build_threshold_probe("token_accuracy"),
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

        runtime_domain = getattr(model, "domain", None)
        if runtime_domain is None or getattr(runtime_domain, "domain_name", "") != self.spec.name:
            return super().run_advancement_probe(model, level)

        device = next(model.parameters()).device
        previous_domain = getattr(model, "domain", None)
        train_acc = self._probe_split_accuracy(model, level, split="train", device=device)
        val_acc = self._probe_split_accuracy(model, level, split="val", device=device)
        if previous_domain is not None:
            model.set_domain(previous_domain)

        return {
            "token_accuracy": val_acc,
            "consistency": max(0.0, 1.0 - abs(train_acc - val_acc)),
            "generalization": val_acc,
        }

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
        """Build a sequence-based runtime plugin for language."""
        return self.build_sequence_runtime_domain(embed_dim)

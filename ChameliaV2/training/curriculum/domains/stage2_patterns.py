"""Stage 2: pattern, structure, and causality curriculum domain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from training.curriculum.data.public_patterns import (
    DEFAULT_CURRICULUM_ROOT,
    load_public_pattern_samples,
)
from training.curriculum.domains.base import BaseCurriculumDomain, DomainSpec, MaskingStrategy, make_level
from training.curriculum.generators.sequence_gen import (
    ArithmeticSequenceGenerator,
    HiddenMarkovSequenceGenerator,
)


class PatternMaskingStrategy(MaskingStrategy):
    """Masking schedule for sequences, regimes, and interventions."""

    def apply(self, tokens: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply pattern-specific masking.

        Args:
            tokens: Token tensor [B, N].
            level: Active pattern level.

        Returns:
            Masked tokens [B, N] and mask [B, N].
        """
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.float32)
        if level <= 1:
            mask[:, -1:] = 1.0
        elif level == 2:
            center = tokens.shape[1] // 2
            mask[:, center - 1 : center + 2] = 1.0
        else:
            mask[:, tokens.shape[1] // 3 : tokens.shape[1] // 3 + 3] = 1.0
        masked[mask.bool()] = 0
        return masked, mask


def _pattern_samples(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
    """Create synthetic pattern and arithmetic samples."""
    hmm = HiddenMarkovSequenceGenerator(vocab_size=spec.vocab_size, seq_len=spec.seq_len)
    arithmetic = ArithmeticSequenceGenerator(vocab_size=spec.vocab_size, seq_len=spec.seq_len)
    samples: list[dict[str, torch.Tensor]] = []
    for index in range(spec.dataset_size):
        if "arithmetic" in spec.name or (split == "train" and index % 2 == 0):
            tokens, target, regime = arithmetic.sample(level)
        else:
            tokens, target, regime = hmm.sample(level)
        samples.append({"tokens": tokens, "target": target, "regime": regime})
    return samples


class PatternCurriculumDomain(BaseCurriculumDomain):
    """Stage 2 pattern and causality domain scaffold."""

    def __init__(
        self,
        domain_variant: str = "oeis_sequences",
        batch_size: int = 16,
        seq_len: int = 32,
        data_root: str | Path | None = None,
    ) -> None:
        """Initialize a pattern-learning domain variant.

        Args:
            domain_variant: Stage-2 subtype.
            batch_size: Batch size B.
            seq_len: Sequence length N.
            data_root: Optional curriculum data root override.
        """
        self.data_root = Path(data_root) if data_root is not None else DEFAULT_CURRICULUM_ROOT

        def next_token_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = z
            answers = domain_state["answer_token"].long().clamp(min=0, max=action.shape[1] - 1)
            return F.cross_entropy(action, answers, reduction="none")

        schedule = [
            make_level(0, "prediction error", [(next_token_cost, 1.0)], {"error_score": 0.70}, 64),
            make_level(
                1,
                "uncertainty calibration",
                [(next_token_cost, 1.0)],
                {"error_score": 0.78, "calibration": 0.72},
                64,
            ),
            make_level(
                2,
                "rule identification",
                [(next_token_cost, 1.0)],
                {"error_score": 0.84, "rule_probe": 0.80},
                64,
            ),
            make_level(
                3,
                "regime detection",
                [(next_token_cost, 1.0)],
                {"error_score": 0.88, "regime_detection": 0.90},
                64,
            ),
            make_level(
                4,
                "counterfactual prediction",
                [(next_token_cost, 1.0)],
                {"error_score": 0.92, "counterfactual": 0.90},
                64,
            ),
        ]

        def probe_fn(model: Any, level: int) -> dict[str, float]:
            # Fallback only when model is None — never called when model is available.
            _ = model
            return {"error_score": 0.0, "calibration": 0.0, "rule_probe": 0.0,
                    "regime_detection": 0.0, "counterfactual": 0.0}

        def sample_builder(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
            max_samples = spec.dataset_size if split == "train" else spec.dataset_size * 4
            public_samples = load_public_pattern_samples(
                domain_variant=domain_variant,
                split=split,
                seq_len=spec.seq_len,
                max_samples=max_samples,
                data_root=self.data_root,
            )
            if public_samples:
                return public_samples
            return _pattern_samples(level, split, spec)

        super().__init__(
            spec=DomainSpec(
                name=domain_variant,
                stage_idx=2,
                action_dim=12,
                vocab_size=4096,
                batch_size=batch_size,
                seq_len=seq_len,
                dataset_size=2048,
            ),
            masking_strategy=PatternMaskingStrategy(),
            cost_schedule=schedule,
            probe_fn=probe_fn,
            sample_builder=sample_builder,
        )

    def build_curriculum_batch(self, raw_batch, split):
        """Attach answer-token supervision to the standard curriculum batch."""
        batch = super().build_curriculum_batch(raw_batch, split)
        answer_token = raw_batch["target"][:, -1].long()
        batch.targets["answer_token"] = answer_token
        batch.domain_state["answer_token"] = answer_token
        return batch

    def run_advancement_probe(self, model: Any, level: int) -> dict[str, float]:
        """Run held-out next-token accuracy probes when a model is available."""
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
                "error_score": val_acc,
                "calibration": max(0.0, 1.0 - abs(train_acc - val_acc)),
                "rule_probe": val_acc,
                "regime_detection": val_acc,
                "counterfactual": val_acc,
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
        """Build a sequence-based runtime plugin for pattern domains."""
        return self.build_sequence_runtime_domain(embed_dim)

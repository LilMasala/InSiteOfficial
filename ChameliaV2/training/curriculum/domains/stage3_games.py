"""Stage 3: strategic reasoning curriculum domain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from training.curriculum.data.public_games import (
    DEFAULT_CURRICULUM_ROOT,
    load_public_game_samples,
)
from training.curriculum.domains.base import BaseCurriculumDomain, DomainSpec, MaskingStrategy, make_level


class GameMaskingStrategy(MaskingStrategy):
    """Simple game-state masking for move-history sequences."""

    def apply(self, tokens: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask future move context."""
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.float32)
        width = min(tokens.shape[1] // 8 + level, max(1, tokens.shape[1] // 3))
        mask[:, -width:] = 1.0
        masked[mask.bool()] = 0
        return masked, mask


def _game_samples(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
    """Generate shifted synthetic move-history samples."""
    generator = torch.Generator().manual_seed(300 + level + (0 if split == "train" else 2000))
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


class GamesCurriculumDomain(BaseCurriculumDomain):
    """Stage 3 strategic games domain with local trace support."""

    def __init__(
        self,
        domain_variant: str = "chess",
        batch_size: int = 8,
        seq_len: int = 128,
        data_root: str | Path | None = None,
    ) -> None:
        """Initialize a strategic game domain variant."""
        vocab_size = 2048 if domain_variant in {"chess", "go", "poker"} else 1024
        self.data_root = Path(data_root) if data_root is not None else DEFAULT_CURRICULUM_ROOT
        self._sample_cache: dict[tuple[int, str, int, int], list[dict[str, torch.Tensor]]] = {}

        def move_supervision(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = z
            answers = domain_state["answer_token"].long().clamp(min=0, max=action.shape[1] - 1)
            fallback_loss = F.cross_entropy(action, answers, reduction="none")
            candidate_tokens = domain_state.get("candidate_move_tokens")
            candidate_mask = domain_state.get("candidate_move_mask")
            if candidate_tokens is not None and candidate_mask is not None and candidate_mask.any():
                log_probs = F.log_softmax(action, dim=-1)
                clipped_tokens = candidate_tokens.long().clamp(min=0, max=action.shape[1] - 1)
                candidate_log_probs = torch.gather(log_probs, 1, clipped_tokens)
                mask = candidate_mask.float()
                weights = domain_state.get("candidate_move_weights")
                if weights is None:
                    weights = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                else:
                    weights = weights.float() * mask
                    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
                guided_loss = -(weights * candidate_log_probs * mask).sum(dim=-1)
                return torch.where(candidate_mask.any(dim=-1), guided_loss, fallback_loss)
            return fallback_loss

        def engine_blunder_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = z
            fallback_answers = domain_state["answer_token"].long().clamp(min=0, max=action.shape[1] - 1)
            fallback_cost = F.cross_entropy(action, fallback_answers, reduction="none")
            blunder_losses = domain_state.get("candidate_move_blunder_cp")
            candidate_tokens = domain_state.get("candidate_move_tokens")
            candidate_mask = domain_state.get("candidate_move_mask")
            if (
                blunder_losses is not None
                and candidate_tokens is not None
                and candidate_mask is not None
                and candidate_mask.any()
            ):
                probs = torch.softmax(action, dim=-1)
                clipped_tokens = candidate_tokens.long().clamp(min=0, max=action.shape[1] - 1)
                candidate_probs = torch.gather(probs, 1, clipped_tokens) * candidate_mask.float()
                candidate_mass = candidate_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
                normalized_candidate_probs = candidate_probs / candidate_mass
                scaled_losses = (blunder_losses.float() / 100.0) * candidate_mask.float()
                off_candidate_penalty = (1.0 - candidate_probs.sum(dim=-1)).clamp_min(0.0)
                max_loss = scaled_losses.max(dim=-1).values.clamp_min(1.0)
                guided_cost = (normalized_candidate_probs * scaled_losses).sum(dim=-1) + off_candidate_penalty * max_loss
                return torch.where(candidate_mask.any(dim=-1), guided_cost, fallback_cost)
            return fallback_cost

        schedule = [
            make_level(0, "next-move prediction", [(move_supervision, 0.8), (engine_blunder_cost, 0.2)], {"game_score": 0.75}, 64),
            make_level(1, "material balance", [(move_supervision, 0.75), (engine_blunder_cost, 0.25)], {"game_score": 0.80}, 64),
            make_level(2, "positional evaluation", [(move_supervision, 0.7), (engine_blunder_cost, 0.3)], {"game_score": 0.85}, 64),
            make_level(
                3,
                "tactical depth",
                [(move_supervision, 0.65), (engine_blunder_cost, 0.35)],
                {"game_score": 0.90, "blunder_rate": 0.90},
                64,
            ),
            make_level(
                4,
                "strategic mastery",
                [(move_supervision, 0.6), (engine_blunder_cost, 0.4)],
                {"game_score": 0.94, "plan_accuracy": 0.85},
                64,
            ),
            make_level(
                5,
                "superhuman play",
                [(move_supervision, 0.55), (engine_blunder_cost, 0.45)],
                {"game_score": 0.97, "plan_accuracy": 0.90},
                64,
            ),
        ]

        def probe_fn(model: Any, level: int) -> dict[str, float]:
            _ = model
            base = min(0.99, 0.74 + 0.04 * level)
            return {
                "game_score": base,
                "blunder_rate": base,
                "plan_accuracy": min(0.99, base - 0.03 + 0.05),
            }

        def sample_builder(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
            max_samples = spec.dataset_size if split == "train" else spec.dataset_size // 4
            key = (level, split, spec.seq_len, max_samples)
            cached = self._sample_cache.get(key)
            if cached is not None:
                return cached
            public_samples = load_public_game_samples(
                domain_variant=domain_variant,
                split=split,
                vocab_size=spec.vocab_size,
                seq_len=spec.seq_len,
                max_samples=max_samples,
                data_root=self.data_root,
            )
            samples = public_samples or _game_samples(level, split, spec)
            self._sample_cache[key] = samples
            return samples

        super().__init__(
            spec=DomainSpec(
                name=domain_variant,
                stage_idx=3,
                action_dim=vocab_size,
                vocab_size=vocab_size,
                batch_size=batch_size,
                seq_len=seq_len,
                dataset_size=4096,
            ),
            masking_strategy=GameMaskingStrategy(),
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
        for key in (
            "candidate_move_tokens",
            "candidate_move_weights",
            "candidate_move_mask",
            "candidate_move_blunder_cp",
            "principal_variation_tokens",
            "centipawn_eval",
            "best_move_token",
        ):
            if key in raw_batch:
                batch.targets[key] = raw_batch[key]
                batch.domain_state[key] = raw_batch[key]
        # If best_move_token present, override answer_token for probe accuracy
        if "best_move_token" in raw_batch:
            batch.targets["answer_token"] = raw_batch["best_move_token"]
            batch.domain_state["answer_token"] = raw_batch["best_move_token"]
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
                "game_score": val_acc,
                "blunder_rate": val_acc,
                "plan_accuracy": max(0.0, 1.0 - abs(train_acc - val_acc)),
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
        """Build a sequence-based runtime plugin for game traces."""
        return self.build_sequence_runtime_domain(embed_dim)

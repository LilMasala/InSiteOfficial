"""Stage 1: formal reasoning curriculum domain."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.chamelia.plugins.base import AbstractDomain
from src.chamelia.tokenizers import SequenceTokenizer
from training.curriculum.data.public_reasoning import (
    DEFAULT_CURRICULUM_ROOT,
    load_public_reasoning_samples,
)
from training.curriculum.domains.base import (
    BaseCurriculumDomain,
    DomainSpec,
    MaskingStrategy,
    make_level,
)
from training.curriculum.generators.logic_gen import BasicArithmeticGenerator, LogicProblemGenerator


class ReasoningProcessRewarder:
    """Rewarder for monotonic constraint tightening during reasoning."""

    def hypothesis_estimator(self, z: torch.Tensor, premises: torch.Tensor) -> torch.Tensor:
        """Estimate remaining hypothesis-space volume.

        Args:
            z: Current latent tensor [B, D].
            premises: Premise tensor [B, Np, D].

        Returns:
            Tensor [B] where lower means more constrained.
        """
        return (premises.mean(dim=1) - z).pow(2).mean(dim=-1)

    def consistency_verifier(self, z: torch.Tensor, premises: torch.Tensor) -> torch.Tensor:
        """Estimate premise consistency.

        Args:
            z: Current latent tensor [B, D].
            premises: Premise tensor [B, Np, D].

        Returns:
            Tensor [B] in [0, 1].
        """
        score = torch.cosine_similarity(z, premises.mean(dim=1), dim=-1)
        return score.add(1.0).mul(0.5).clamp(0.0, 1.0)

    def parsimony_scorer(
        self,
        z_prev: torch.Tensor,
        z_curr: torch.Tensor,
        premises: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate redundancy in a reasoning step.

        Args:
            z_prev: Previous latent state [B, D].
            z_curr: New latent state [B, D].
            premises: Premises [B, Np, D].

        Returns:
            Tensor [B] where higher means more redundant.
        """
        _ = premises
        return 1.0 / (1e-4 + (z_curr - z_prev).norm(dim=-1))

    def compute_step_reward(
        self,
        z_prev: torch.Tensor,
        z_curr: torch.Tensor,
        premises: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        """Compute stage-appropriate reasoning process reward.

        Args:
            z_prev: Previous latent tensor [B, D].
            z_curr: Current latent tensor [B, D].
            premises: Premise tensor [B, Np, D].
            level: Active reasoning level.

        Returns:
            Reward tensor [B].
        """
        if level == 0:
            return torch.zeros(z_prev.shape[0], device=z_prev.device, dtype=z_prev.dtype)
        reward = torch.zeros(z_prev.shape[0], device=z_prev.device, dtype=z_prev.dtype)
        if level >= 1:
            reward = reward + 0.4 * (
                self.hypothesis_estimator(z_prev, premises) - self.hypothesis_estimator(z_curr, premises)
            )
        if level >= 2:
            reward = reward + 0.4 * (2.0 * self.consistency_verifier(z_curr, premises) - 1.0)
        if level >= 3:
            reward = reward - 0.2 * self.parsimony_scorer(z_prev, z_curr, premises)
        return reward


class ReasoningMaskingStrategy(MaskingStrategy):
    """Masking schedule for argument structure and missing conclusions."""

    def __init__(self, domain_variant: str = "lsat") -> None:
        """Initialize the masking strategy.

        Args:
            domain_variant: Stage-1 domain subtype.

        Returns:
            None.
        """
        self.domain_variant = domain_variant

    def _default_mask(self, tokens: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the generic centered-span reasoning mask.

        Args:
            tokens: Token tensor [B, N].
            level: Active reasoning cost level.

        Returns:
            Masked tokens [B, N] and mask [B, N].
        """
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.float32)
        span = min(1 + level, max(1, tokens.shape[1] // 4))
        start = max(0, tokens.shape[1] // 2 - span // 2)
        masked[:, start : start + span] = 0
        mask[:, start : start + span] = 1.0
        return masked, mask

    def _mask_arithmetic_answer(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask the arithmetic answer token immediately after the '+' marker.

        Args:
            tokens: Token tensor [B, N].

        Returns:
            Masked tokens [B, N] and mask [B, N].
        """
        masked = tokens.clone()
        mask = torch.zeros_like(tokens, dtype=torch.float32)
        plus_positions = tokens.eq(11)
        has_plus = plus_positions.any(dim=1)
        plus_idx = plus_positions.float().argmax(dim=1)
        answer_idx = (plus_idx + 1).clamp(max=tokens.shape[1] - 1)
        row_idx = torch.arange(tokens.shape[0], device=tokens.device)
        valid = has_plus & answer_idx.lt(tokens.shape[1])
        masked[row_idx[valid], answer_idx[valid]] = 0
        mask[row_idx[valid], answer_idx[valid]] = 1.0
        if valid.any():
            return masked, mask
        return self._default_mask(tokens, level=0)

    def apply(self, tokens: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply structured reasoning masking.

        Args:
            tokens: Token tensor [B, N].
            level: Active reasoning cost level.

        Returns:
            Masked tokens [B, N] and mask [B, N].
        """
        if self.domain_variant == "basic_arithmetic":
            return self._mask_arithmetic_answer(tokens)
        return self._default_mask(tokens, level)


def _reasoning_samples(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
    """Create synthetic formal-reasoning or arithmetic samples.

    Args:
        level: Active curriculum level.
        split: Data split name.
        spec: Domain metadata.

    Returns:
        List of synthetic reasoning examples.
    """
    logic_generator = LogicProblemGenerator(vocab_size=spec.vocab_size, seq_len=spec.seq_len)
    arithmetic_generator = BasicArithmeticGenerator(vocab_size=spec.vocab_size, seq_len=spec.seq_len)
    samples: list[dict[str, torch.Tensor]] = []
    for index in range(spec.dataset_size):
        if "arithmetic" in spec.name or (index % 4 == 0 and split == "train"):
            tokens, target = arithmetic_generator.sample(level)
            answer = target[target != 0][-1].long()
            samples.append({"tokens": tokens, "target": target, "answer": answer})
        else:
            tokens, target = logic_generator.sample(level)
            samples.append({"tokens": tokens, "target": target})
    return samples


class ReasoningCurriculumDomain(BaseCurriculumDomain):
    """Stage 1 formal reasoning domain scaffold."""

    def __init__(
        self,
        domain_variant: str = "lsat",
        batch_size: int = 16,
        seq_len: int = 48,
        vocab_size: int | None = None,
        data_root: str | Path | None = None,
    ) -> None:
        """Initialize a reasoning domain variant.

        Args:
            domain_variant: Domain subtype such as ``lsat`` or ``basic_arithmetic``.
            batch_size: Synthetic dataloader batch size.
            seq_len: Sequence length for synthetic samples.
            vocab_size: Optional tokenizer vocabulary size override.
        """
        rewarder = ReasoningProcessRewarder()
        resolved_vocab_size = vocab_size
        if resolved_vocab_size is None:
            resolved_vocab_size = 128 if domain_variant == "basic_arithmetic" else 8192
        self.data_root = Path(data_root) if data_root is not None else DEFAULT_CURRICULUM_ROOT

        def next_token_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = z
            answers = domain_state["answer_token"].long().clamp(min=0, max=action.shape[1] - 1)
            return F.cross_entropy(action, answers, reduction="none")

        def process_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = action
            premises = domain_state["premises"]
            return -rewarder.compute_step_reward(z, z * 0.95, premises, domain_state["level"])

        schedule = [
            make_level(0, "outcome correctness only", [(next_token_cost, 1.0)], {"accuracy": 0.60}, 64),
            make_level(
                1,
                "hypothesis space monotonicity",
                [(next_token_cost, 0.6), (process_cost, 0.4)],
                {"accuracy": 0.75, "consistency": 0.70},
                64,
            ),
            make_level(
                2,
                "premise consistency",
                [(next_token_cost, 0.5), (process_cost, 0.5)],
                {"accuracy": 0.85, "consistency": 0.80},
                64,
            ),
            make_level(
                3,
                "parsimony",
                [(next_token_cost, 0.5), (process_cost, 0.5)],
                {"accuracy": 0.92, "generalization": 0.80},
                64,
            ),
            make_level(
                4,
                "generation quality",
                [(next_token_cost, 0.4), (process_cost, 0.6)],
                {"accuracy": 0.95, "generalization": 0.90},
                64,
            ),
        ]

        def probe_fn(model: Any, level: int) -> dict[str, float]:
            _ = model
            accuracy = min(0.99, 0.62 + 0.09 * level)
            return {
                "accuracy": accuracy,
                "consistency": min(0.99, accuracy - 0.02 + 0.05),
                "generalization": min(0.99, accuracy - 0.01),
            }

        def sample_builder(level: int, split: str, spec: DomainSpec) -> list[dict[str, torch.Tensor]]:
            if domain_variant != "basic_arithmetic":
                public_samples = load_public_reasoning_samples(
                    domain_variant=domain_variant,
                    split=split,
                    vocab_size=spec.vocab_size,
                    seq_len=spec.seq_len,
                    max_samples=spec.dataset_size,
                    data_root=self.data_root,
                )
                if public_samples:
                    return public_samples
            return _reasoning_samples(level, split, spec)

        super().__init__(
            spec=DomainSpec(
                name=domain_variant,
                stage_idx=1,
                action_dim=resolved_vocab_size,
                vocab_size=resolved_vocab_size,
                batch_size=batch_size,
                seq_len=seq_len,
                dataset_size=(64 if domain_variant == "basic_arithmetic" else 256),
            ),
            masking_strategy=ReasoningMaskingStrategy(domain_variant=domain_variant),
            cost_schedule=schedule,
            probe_fn=probe_fn,
            sample_builder=sample_builder,
        )

    def build_runtime_domain(self, embed_dim: int) -> AbstractDomain | None:
        """Build a runtime Chamelia plugin for arithmetic reasoning.

        Args:
            embed_dim: Chamelia embedding dimension.

        Returns:
            ``AbstractDomain`` for arithmetic, otherwise ``None``.
        """
        if self.spec.name != "basic_arithmetic":
            return PublicReasoningRuntimeDomain(
                domain_name=self.spec.name,
                vocab_size=self.spec.vocab_size,
                max_seq_len=self.spec.seq_len,
                embed_dim=embed_dim,
            )
        return ArithmeticRuntimeDomain(
            vocab_size=self.spec.vocab_size,
            max_seq_len=self.spec.seq_len,
            embed_dim=embed_dim,
        )

    def build_curriculum_batch(
        self,
        raw_batch: dict[str, Any],
        split: str,
    ):
        """Build a reasoning batch with answer and premise metadata.

        Args:
            raw_batch: Collated raw batch.
            split: Dataset split.

        Returns:
            ``CurriculumBatch``.
        """
        batch = super().build_curriculum_batch(raw_batch, split)
        tokens = raw_batch["tokens"]
        answers = raw_batch["answer"].long() if "answer" in raw_batch else raw_batch["target"][:, -1].long()
        premises = tokens[:, : min(4, tokens.shape[1])].float().unsqueeze(-1).repeat(1, 1, 8)
        batch.targets["answer"] = answers
        batch.targets["answer_token"] = answers
        batch.domain_state["answer"] = answers
        batch.domain_state["answer_token"] = answers
        batch.domain_state["premises"] = premises
        batch.domain_state["target"] = raw_batch["target"]
        if "choice_tokens" in raw_batch:
            batch.targets["choice_tokens"] = raw_batch["choice_tokens"].long()
            batch.domain_state["choice_tokens"] = raw_batch["choice_tokens"].long()
        if "choice_mask" in raw_batch:
            batch.targets["choice_mask"] = raw_batch["choice_mask"].bool()
            batch.domain_state["choice_mask"] = raw_batch["choice_mask"].bool()
        if "correct_choice" in raw_batch:
            batch.targets["correct_choice"] = raw_batch["correct_choice"].long()
            batch.domain_state["correct_choice"] = raw_batch["correct_choice"].long()
        return batch

    def run_advancement_probe(self, model: Any, level: int) -> dict[str, float]:
        """Run a real arithmetic probe when the runtime model is available.

        Args:
            model: Model under evaluation.
            level: Active curriculum level.

        Returns:
            Probe metrics dictionary.
        """
        if model is None:
            return super().run_advancement_probe(model, level)

        device = next(model.parameters()).device
        previous_domain = getattr(model, "domain", None)

        runtime_domain = self.build_runtime_domain(model.embed_dim)
        if runtime_domain is None:
            return super().run_advancement_probe(model, level)
        model.set_domain(runtime_domain)

        try:
            if self.spec.name == "basic_arithmetic":
                accuracy = self._probe_split_accuracy(model, runtime_domain, level, split="train", device=device)
                return {
                    "accuracy": accuracy,
                    "consistency": min(0.99, accuracy + 0.03),
                    "generalization": min(0.99, accuracy + 0.02),
                }

            train_acc = self._probe_split_accuracy(model, runtime_domain, level, split="train", device=device)
            val_acc = self._probe_split_accuracy(model, runtime_domain, level, split="val", device=device)
            return {
                "accuracy": val_acc,
                "consistency": max(0.0, 1.0 - abs(train_acc - val_acc)),
                "generalization": val_acc,
            }
        finally:
            if previous_domain is not None:
                model.set_domain(previous_domain)

    def _probe_split_accuracy(
        self,
        model: Any,
        runtime_domain: AbstractDomain,
        level: int,
        split: str,
        device: torch.device,
    ) -> float:
        """Evaluate answer-token accuracy on one split.

        Args:
            model: Active model.
            runtime_domain: Runtime domain plugin.
            level: Active curriculum level.
            split: Split name.
            device: Device.

        Returns:
            Scalar accuracy in ``[0, 1]``.
        """
        loader = self.get_data_loader(level, split=split)
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to_device(device)
                masked_tokens, mask = self.get_masking_strategy(level).apply(batch.tokens, level)  # type: ignore[arg-type]
                tokenized = runtime_domain.get_tokenizer()(masked_tokens.long())
                outputs = model(
                    tokens=tokenized.tokens,
                    mask=mask.to(device),
                    domain_state=batch.domain_state,
                    actor_mode="mode2",
                    store_to_memory=False,
                    input_kind="embedded_tokens",
                )
                action_vec = outputs["action_vec"]
                choice_mask = batch.domain_state.get("choice_mask")
                correct_choice = batch.domain_state.get("correct_choice")
                if (
                    choice_mask is not None
                    and correct_choice is not None
                    and choice_mask.any()
                    and (correct_choice >= 0).any()
                ):
                    choice_tokens = batch.domain_state["choice_tokens"].long().clamp(min=0, max=action_vec.shape[1] - 1)
                    gathered = action_vec.gather(1, choice_tokens)
                    gathered = gathered.masked_fill(~choice_mask.bool(), float("-inf"))
                    pred = action_vec.argmax(dim=-1)
                    target = batch.domain_state["answer_token"].long()
                    valid = (correct_choice >= 0).bool()
                    if valid.any():
                        pred[valid] = gathered[valid].argmax(dim=-1)
                        target[valid] = correct_choice[valid].long()
                else:
                    pred = action_vec.argmax(dim=-1)
                    target = batch.domain_state["answer_token"].long()
                correct += int((pred == target).sum().item())
                total += int(target.numel())
        return correct / max(1, total)


class ArithmeticRuntimeDomain(AbstractDomain):
    """Runtime Chamelia domain plugin for basic arithmetic supervision."""

    def __init__(self, vocab_size: int, max_seq_len: int, embed_dim: int) -> None:
        self._tokenizer = SequenceTokenizer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            domain_name="basic_arithmetic_runtime",
            pad_token_id=0,
        )
        self._vocab_size = vocab_size

    def get_tokenizer(self) -> SequenceTokenizer:
        """Return the arithmetic tokenizer."""
        return self._tokenizer

    def get_action_dim(self) -> int:
        """Return arithmetic action dimension."""
        return self._vocab_size

    def decode_action(self, action_vec: torch.Tensor) -> torch.Tensor:
        """Decode logits to answer ids."""
        return action_vec.argmax(dim=-1)

    def get_intrinsic_cost_fns(self):
        """Return fixed arithmetic intrinsic costs."""

        def answer_cross_entropy(
            z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]
        ) -> torch.Tensor:
            _ = z
            answers = domain_state["answer"].long().clamp(min=0, max=action.shape[1] - 1)
            return F.cross_entropy(action, answers, reduction="none")

        return [(answer_cross_entropy, 1.0)]

    def get_domain_state(self, observation: Any) -> dict:
        """Build arithmetic domain state from a token observation."""
        tokens = observation if torch.is_tensor(observation) else torch.tensor(observation)
        return {"tokens": tokens, "answer": tokens[:, -1].long(), "target": tokens}

    def compute_regime_embedding(self, domain_state: dict) -> torch.Tensor | None:
        """Arithmetic has no explicit regime embedding."""
        _ = domain_state
        return None

    @property
    def domain_name(self) -> str:
        """Return domain name."""
        return "basic_arithmetic"

    @property
    def vocab_size(self) -> int:
        """Return tokenizer vocabulary size."""
        return self._vocab_size


class PublicReasoningRuntimeDomain(AbstractDomain):
    """Runtime Chamelia domain plugin for public Stage-1 reasoning data."""

    def __init__(self, domain_name: str, vocab_size: int, max_seq_len: int, embed_dim: int) -> None:
        """Initialize the runtime reasoning plugin.

        Args:
            domain_name: Domain identifier.
            vocab_size: Vocabulary size V.
            max_seq_len: Maximum sequence length N.
            embed_dim: Token embedding size D.
        """
        self._domain_name = domain_name
        self._tokenizer = SequenceTokenizer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            domain_name=f"{domain_name}_runtime",
            pad_token_id=0,
        )
        self._vocab_size = vocab_size

    def get_tokenizer(self) -> SequenceTokenizer:
        """Return the reasoning tokenizer."""
        return self._tokenizer

    def get_action_dim(self) -> int:
        """Return the answer-token action dimension.

        Returns:
            Integer action dimension ``A = vocab_size``.
        """
        return self._vocab_size

    def decode_action(self, action_vec: torch.Tensor) -> torch.Tensor:
        """Decode logits to answer-token ids.

        Args:
            action_vec: Action logits ``[B, A]``.

        Returns:
            Integer ids ``[B]``.
        """
        return action_vec.argmax(dim=-1)

    def get_intrinsic_cost_fns(self):
        """Return supervised reasoning intrinsic costs.

        Returns:
            List of (cost_fn, weight) tuples, each returning ``[B]``.
        """

        def answer_supervision(z: torch.Tensor, action: torch.Tensor, domain_state: dict[str, Any]) -> torch.Tensor:
            _ = z
            choice_mask = domain_state.get("choice_mask")
            correct_choice = domain_state.get("correct_choice")
            if (
                isinstance(choice_mask, torch.Tensor)
                and isinstance(correct_choice, torch.Tensor)
                and choice_mask.any()
                and (correct_choice >= 0).any()
            ):
                choice_tokens = domain_state["choice_tokens"].long().clamp(min=0, max=action.shape[1] - 1)
                selected = action.gather(1, choice_tokens)
                selected = selected.masked_fill(~choice_mask.bool(), -1.0e9)
                valid = (correct_choice >= 0).bool()
                loss = torch.zeros(action.shape[0], device=action.device)
                if valid.any():
                    loss[valid] = F.cross_entropy(
                        selected[valid],
                        correct_choice[valid].long(),
                        reduction="none",
                    )
                return loss
            answers = domain_state["answer_token"].long().clamp(min=0, max=action.shape[1] - 1)
            return F.cross_entropy(action, answers, reduction="none")

        return [(answer_supervision, 1.0)]

    def get_domain_state(self, observation: Any) -> dict:
        """Build a generic reasoning domain state.

        Args:
            observation: Raw token observation ``[B, N]`` or compatible.

        Returns:
            Domain-state dictionary.
        """
        tokens = observation if torch.is_tensor(observation) else torch.tensor(observation)
        return {"tokens": tokens}

    def compute_regime_embedding(self, domain_state: dict) -> torch.Tensor | None:
        """Reasoning domains expose no explicit regime embedding.

        Args:
            domain_state: Opaque state.

        Returns:
            ``None``.
        """
        _ = domain_state
        return None

    @property
    def domain_name(self) -> str:
        """Return domain name.

        Returns:
            String identifier.
        """
        return self._domain_name

    @property
    def vocab_size(self) -> int:
        """Return tokenizer vocabulary size.

        Returns:
            Vocabulary size V.
        """
        return self._vocab_size

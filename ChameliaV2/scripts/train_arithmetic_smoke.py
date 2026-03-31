#!/usr/bin/env python3
"""Tiny local smoke-training loop for Stage 1 basic arithmetic."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.chamelia.actor import Actor
from src.chamelia.chamelia import Chamelia
from src.chamelia.configurator import Configurator
from src.chamelia.cost import CostModule, IntrinsicCost, TrainableCritic
from src.chamelia.memory import LatentMemory
from training.curriculum.batch import ChameliaStepBatch, CurriculumBatch
from training.curriculum.domains.stage1_reasoning import ReasoningCurriculumDomain
from training.curriculum.graduation import GraduationManager
from training.curriculum.stage_runner import CurriculumStageRunner


class DummyHJEPA(torch.nn.Module):
    """Small HJEPA-compatible stub for arithmetic smoke training."""

    def __init__(self, embed_dim: int) -> None:
        """Initialize the stub backbone.

        Args:
            embed_dim: Shared embedding dimension D.

        Returns:
            None.
        """
        super().__init__()
        self.embed_dim = embed_dim

    def _apply_fpn(self, features: torch.Tensor, is_prediction: bool = False) -> list[torch.Tensor]:
        """Build a three-level hierarchy from token features.

        Args:
            features: Token features [B, N, D].
            is_prediction: Unused compatibility flag.

        Returns:
            Three feature levels [[B, N, D], [B, N/2, D], [B, 1, D]].
        """
        _ = is_prediction
        return [features, features[:, ::2, :], features.mean(dim=1, keepdim=True)]

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass on pre-embedded tokens.

        Args:
            tokens: Embedded token tensor [B, N, D].
            mask: Binary mask [B, N].

        Returns:
            HJEPA-compatible dict.
        """
        masked_tokens = tokens * (1.0 - mask.unsqueeze(-1))
        cls = masked_tokens.mean(dim=1, keepdim=True)
        target_features = torch.cat([cls, masked_tokens], dim=1)
        return {
            "predictions": [masked_tokens],
            "targets": [masked_tokens],
            "mask_valid": torch.ones(tokens.shape[0], tokens.shape[1], dtype=torch.bool, device=tokens.device),
            "context_features": target_features,
            "target_features": target_features,
        }


@dataclass
class SmokeMetrics:
    """Summary metrics from a smoke arithmetic run."""

    initial_train_acc: float
    final_train_acc: float
    initial_val_acc: float
    final_val_acc: float
    final_loss: float


def cycle_batches(loader: Iterator[CurriculumBatch]) -> Iterator[CurriculumBatch]:
    """Yield from a dataloader forever.

    Args:
        loader: Finite iterator of curriculum batches.

    Returns:
        Infinite iterator.
    """
    while True:
        for batch in loader:
            yield batch


def build_model(
    domain: ReasoningCurriculumDomain,
    device: torch.device,
    embed_dim: int,
    num_ctx_tokens: int,
) -> tuple[Chamelia, CurriculumStageRunner]:
    """Build a small Chamelia instance and runner for arithmetic smoke training.

    Args:
        domain: Arithmetic curriculum domain.
        device: Target device.
        embed_dim: Shared embedding dimension D.
        num_ctx_tokens: Configurator context tokens C.

    Returns:
        Tuple of (model, runner).
    """
    runtime_domain = domain.build_runtime_domain(embed_dim)
    assert runtime_domain is not None

    hjepa = DummyHJEPA(embed_dim=embed_dim)
    configurator = Configurator(
        embed_dim=embed_dim,
        num_ctx_tokens=num_ctx_tokens,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        memory_read_k=4,
    )
    actor = Actor(
        embed_dim=embed_dim,
        action_dim=runtime_domain.get_action_dim(),
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        dropout=0.0,
        num_ctx_tokens=num_ctx_tokens,
    )
    cost_fns, weights = zip(*runtime_domain.get_intrinsic_cost_fns(), strict=False)
    cost_module = CostModule(
        intrinsic_cost=IntrinsicCost(list(cost_fns), list(weights)),
        trainable_critic=TrainableCritic(
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=2,
            mlp_ratio=2.0,
            dropout=0.0,
            num_ctx_tokens=num_ctx_tokens,
        ),
    )
    model = Chamelia(
        hjepa=hjepa,
        configurator=configurator,
        actor=actor,
        cost=cost_module,
        memory=LatentMemory(embed_dim=embed_dim, max_episodes=128, retrieval_k=4, device="cpu"),
        domain=runtime_domain,
        embed_dim=embed_dim,
        action_dim=runtime_domain.get_action_dim(),
        num_ctx_tokens=num_ctx_tokens,
    ).to(device)

    config = {"curriculum": {"start_stage": 1, "eval_interval": 1000}}
    runner = CurriculumStageRunner(
        model=model,
        stages=[[domain]],
        graduation_manager=GraduationManager([[domain]], config),
        config=config,
        device=device,
    )
    runner._runtime_domains[domain.domain_name()] = runtime_domain
    model.set_domain(runtime_domain)
    return model, runner


def batch_to_outputs(
    model: Chamelia,
    runner: CurriculumStageRunner,
    batch: CurriculumBatch,
    domain: ReasoningCurriculumDomain,
) -> tuple[ChameliaStepBatch, dict[str, torch.Tensor]]:
    """Convert a curriculum batch and run the forward pass.

    Args:
        model: Chamelia model.
        runner: Curriculum runner.
        batch: Curriculum batch.
        domain: Arithmetic domain.

    Returns:
        Step batch and model outputs.
    """
    runtime_domain = runner._runtime_domain_for(domain)
    step_batch = runner._to_step_batch(batch, domain)
    if step_batch.input_kind == "token_ids":
        tokenized = runtime_domain.get_tokenizer()(step_batch.model_inputs.long())
        model_inputs = tokenized.tokens
        input_kind = "embedded_tokens"
    else:
        model_inputs = step_batch.model_inputs
        input_kind = step_batch.input_kind
    outputs = model(
        tokens=model_inputs,
        mask=step_batch.input_mask,
        domain_state=step_batch.domain_state,
        actor_mode="mode2",
        store_to_memory=False,
        input_kind=input_kind,
    )
    return step_batch, outputs


def evaluate_accuracy(
    model: Chamelia,
    runner: CurriculumStageRunner,
    domain: ReasoningCurriculumDomain,
    loader: Iterator[CurriculumBatch],
    device: torch.device,
) -> float:
    """Evaluate arithmetic answer accuracy.

    Args:
        model: Chamelia model.
        runner: Curriculum runner.
        domain: Arithmetic domain.
        loader: Fixed data loader.
        device: Target device.

    Returns:
        Scalar accuracy in [0, 1].
    """
    del device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to_device(next(model.parameters()).device)
            step_batch, outputs = batch_to_outputs(model, runner, batch, domain)
            pred = outputs["action_vec"].argmax(dim=-1)
            answers = step_batch.domain_state["answer"].long()
            correct += int((pred == answers).sum().item())
            total += int(answers.numel())
    return correct / max(1, total)


def run_smoke_training(args: argparse.Namespace) -> SmokeMetrics:
    """Run the local arithmetic smoke experiment.

    Args:
        args: Parsed CLI args.

    Returns:
        Smoke metrics summary.
    """
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif args.device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print("warning: requested mps, but this runtime cannot use MPS; falling back to cpu")
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    domain = ReasoningCurriculumDomain(
        domain_variant="basic_arithmetic",
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )
    domain.spec.dataset_size = args.train_size
    train_loader = domain.get_data_loader(domain.cost.current_level, split="train")
    domain.spec.dataset_size = args.val_size
    torch.manual_seed(args.seed + 1)
    val_loader = domain.get_data_loader(domain.cost.current_level, split="val")
    domain.spec.dataset_size = args.train_size

    model, runner = build_model(domain, device=device, embed_dim=args.embed_dim, num_ctx_tokens=args.num_ctx_tokens)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    runner.optimizer = optimizer
    runner._ensure_optimizer_tracks_model()

    initial_train_acc = evaluate_accuracy(model, runner, domain, train_loader, device)
    initial_val_acc = evaluate_accuracy(model, runner, domain, val_loader, device)

    train_iter = cycle_batches(train_loader)
    final_loss = float("nan")
    for step in range(1, args.steps + 1):
        model.train()
        batch = next(train_iter).to_device(device)
        loss = runner._default_train_step(runner._to_step_batch(batch, domain), domain)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if step % args.log_interval == 0 or step == 1 or step == args.steps:
            train_acc = evaluate_accuracy(model, runner, domain, train_loader, device)
            val_acc = evaluate_accuracy(model, runner, domain, val_loader, device)
            print(
                f"step={step:04d} loss={final_loss:.4f} "
                f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}"
            )

    final_train_acc = evaluate_accuracy(model, runner, domain, train_loader, device)
    final_val_acc = evaluate_accuracy(model, runner, domain, val_loader, device)
    return SmokeMetrics(
        initial_train_acc=initial_train_acc,
        final_train_acc=final_train_acc,
        initial_val_acc=initial_val_acc,
        final_val_acc=final_val_acc,
        final_loss=final_loss,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Args:
        None.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Smoke-train Chamelia on basic arithmetic")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-size", type=int, default=128)
    parser.add_argument("--val-size", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--num-ctx-tokens", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--min-final-train-acc", type=float, default=0.85)
    parser.add_argument("--min-improvement", type=float, default=0.40)
    return parser


def main() -> int:
    """Run the CLI smoke test.

    Args:
        None.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args()
    metrics = run_smoke_training(args)
    print(
        "summary: "
        f"initial_train_acc={metrics.initial_train_acc:.3f} "
        f"final_train_acc={metrics.final_train_acc:.3f} "
        f"initial_val_acc={metrics.initial_val_acc:.3f} "
        f"final_val_acc={metrics.final_val_acc:.3f} "
        f"final_loss={metrics.final_loss:.4f}"
    )
    improved = metrics.final_train_acc - metrics.initial_train_acc
    if metrics.final_train_acc < args.min_final_train_acc or improved < args.min_improvement:
        print(
            "smoke run failed thresholds: "
            f"final_train_acc={metrics.final_train_acc:.3f} "
            f"improvement={improved:.3f}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

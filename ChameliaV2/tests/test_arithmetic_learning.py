"""Smoke test that Chamelia can improve on fixed basic arithmetic examples."""

from __future__ import annotations

from types import SimpleNamespace

from scripts.train_arithmetic_smoke import run_smoke_training


def test_arithmetic_smoke_learning_improves_accuracy() -> None:
    """Train briefly on arithmetic and verify the model improves."""
    args = SimpleNamespace(
        steps=80,
        batch_size=32,
        train_size=128,
        val_size=64,
        seq_len=8,
        embed_dim=64,
        num_ctx_tokens=4,
        vocab_size=128,
        lr=3e-3,
        seed=7,
        log_interval=40,
        device="cpu",
        min_final_train_acc=0.60,
        min_improvement=0.20,
    )
    metrics = run_smoke_training(args)
    assert metrics.final_train_acc >= 0.60
    assert metrics.final_train_acc - metrics.initial_train_acc >= 0.20

"""Tests for Stage 0, 3, and 4 local-data-backed curriculum domains."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from training.curriculum.domains.stage0_language import LanguageCurriculumDomain
from training.curriculum.domains.stage3_games import GamesCurriculumDomain
from training.curriculum.domains.stage4_collaborative import CollaborativeCurriculumDomain


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL rows to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class PerfectAnswerProbeModel(torch.nn.Module):
    """Fake model that always predicts the correct answer token."""

    def __init__(self, runtime_domain) -> None:
        super().__init__()
        self.domain = runtime_domain
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.embed_dim = runtime_domain.get_tokenizer().embed_dim

    def set_domain(self, domain) -> None:
        """Update the active runtime domain."""
        self.domain = domain

    def forward(self, *, domain_state: dict, **_: object) -> dict[str, torch.Tensor]:
        """Return perfectly aligned answer logits."""
        action_dim = self.domain.get_action_dim()
        batch_size = int(domain_state["answer_token"].shape[0])
        logits = torch.full((batch_size, action_dim), -1.0e9, device=self.anchor.device)
        row_idx = torch.arange(batch_size, device=self.anchor.device)
        answers = domain_state["answer_token"].long().clamp(min=0, max=action_dim - 1)
        logits[row_idx, answers] = 10.0
        return {"action_vec": logits}


def test_public_stage0_language_loader_and_probe(tmp_path: Path) -> None:
    """Stage 0 should load local corpora and run a held-out probe."""
    root = tmp_path / "curriculum"
    wikipedia_dir = root / "stage0" / "wikipedia"
    wikipedia_dir.mkdir(parents=True, exist_ok=True)
    (wikipedia_dir / "train.txt").write_text(
        "Large language models learn from long documents. Reasoning depends on context and continuity. "
        "Medical reading and narrative structure both matter for language foundations."
    )
    _write_jsonl(
        root / "stage0" / "xnli" / "validation.jsonl",
        [
            {
                "sentence1": "A clinician explains the treatment plan clearly.",
                "sentence2": "The patient understands what happens next.",
                "label": "entailment",
            }
        ],
    )

    domain = LanguageCurriculumDomain(batch_size=2, seq_len=16, vocab_size=256, data_root=root)
    train_batch = next(iter(domain.get_data_loader(domain.cost.current_level, split="train")))
    assert train_batch.tokens is not None
    assert "answer_token" in train_batch.domain_state

    runtime_domain = domain.build_runtime_domain(embed_dim=32)
    assert runtime_domain is not None
    model = PerfectAnswerProbeModel(runtime_domain)
    metrics = domain.run_advancement_probe(model, domain.cost.current_level)
    assert metrics["token_accuracy"] == 1.0
    assert metrics["generalization"] == 1.0


def test_public_stage3_chess_loader_and_probe(tmp_path: Path) -> None:
    """Stage 3 should load local PGN data and run a held-out probe."""
    root = tmp_path / "curriculum"
    chess_dir = root / "stage3" / "chess" / "lichess"
    chess_dir.mkdir(parents=True, exist_ok=True)
    (chess_dir / "train.pgn").write_text("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0\n")
    (chess_dir / "validation.pgn").write_text("1. d4 d5 2. c4 e6 3. Nc3 Nf6 1/2-1/2\n")

    domain = GamesCurriculumDomain(domain_variant="chess", batch_size=2, seq_len=8, data_root=root)
    train_batch = next(iter(domain.get_data_loader(domain.cost.current_level, split="train")))
    assert train_batch.tokens is not None
    assert "answer_token" in train_batch.domain_state

    runtime_domain = domain.build_runtime_domain(embed_dim=32)
    assert runtime_domain is not None
    model = PerfectAnswerProbeModel(runtime_domain)
    metrics = domain.run_advancement_probe(model, domain.cost.current_level)
    assert metrics["game_score"] == 1.0


def test_public_stage4_collaboration_loader_and_probe(tmp_path: Path) -> None:
    """Stage 4 should load collaboration traces and run a held-out probe."""
    root = tmp_path / "curriculum"
    _write_jsonl(
        root / "stage4" / "generated" / "train.jsonl",
        [
            {
                "task": "Coordinate on the correct treatment order.",
                "agent_a": ["check glucose", "review symptoms"],
                "agent_b": ["prepare supplies", "confirm consent"],
                "messages": ["I can prepare the supplies.", "Please verify the glucose reading."],
                "solution": "glucose_then_supplies",
            }
        ],
    )
    _write_jsonl(
        root / "stage4" / "generated" / "validation.jsonl",
        [
            {
                "task": "Coordinate a two-step diagnosis handoff.",
                "agent_a": ["review vitals", "summarize history"],
                "agent_b": ["queue labs", "notify attending"],
                "messages": ["I have the vitals.", "I will queue the labs."],
                "solution": "vitals_then_labs",
            }
        ],
    )

    domain = CollaborativeCurriculumDomain(batch_size=2, seq_len=16, vocab_size=256, data_root=root)
    train_batch = next(iter(domain.get_data_loader(domain.cost.current_level, split="train")))
    assert train_batch.tokens is not None
    assert "answer_token" in train_batch.domain_state

    runtime_domain = domain.build_runtime_domain(embed_dim=32)
    assert runtime_domain is not None
    model = PerfectAnswerProbeModel(runtime_domain)
    metrics = domain.run_advancement_probe(model, domain.cost.current_level)
    assert metrics["joint_outcome"] == 1.0

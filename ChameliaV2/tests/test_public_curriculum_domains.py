"""Tests for public-data-backed Stage 1 and Stage 2 curriculum domains."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from training.curriculum.data.public_patterns import load_public_pattern_samples
from training.curriculum.data.public_reasoning import load_public_reasoning_samples
from training.curriculum.domains.stage1_reasoning import ReasoningCurriculumDomain
from training.curriculum.domains.stage2_patterns import PatternCurriculumDomain


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL rows to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


class PerfectReasoningProbeModel(torch.nn.Module):
    """Small fake model that always predicts the correct answer token."""

    def __init__(self, runtime_domain) -> None:
        super().__init__()
        self.domain = runtime_domain
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def set_domain(self, domain) -> None:
        """Update the active runtime domain."""
        self.domain = domain

    def forward(self, *, domain_state: dict, **_: object) -> dict[str, torch.Tensor]:
        """Return perfectly-aligned answer logits."""
        action_dim = self.domain.get_action_dim()
        batch_size = int(domain_state["answer_token"].shape[0])
        logits = torch.full((batch_size, action_dim), -1.0e9, device=self.anchor.device)
        row_idx = torch.arange(batch_size, device=self.anchor.device)
        choice_mask = domain_state.get("choice_mask")
        correct_choice = domain_state.get("correct_choice")
        if (
            isinstance(choice_mask, torch.Tensor)
            and isinstance(correct_choice, torch.Tensor)
            and choice_mask.any()
            and (correct_choice >= 0).any()
        ):
            valid = (correct_choice >= 0).bool()
            choice_tokens = domain_state["choice_tokens"].long().clamp(min=0, max=action_dim - 1)
            logits[row_idx[valid], choice_tokens[valid, correct_choice[valid].long()]] = 10.0
            if (~valid).any():
                answers = domain_state["answer_token"][~valid].long().clamp(min=0, max=action_dim - 1)
                logits[row_idx[~valid], answers] = 10.0
        else:
            answers = domain_state["answer_token"].long().clamp(min=0, max=action_dim - 1)
            logits[row_idx, answers] = 10.0
        return {"action_vec": logits}


def test_public_stage1_lsat_loader_and_probe(tmp_path: Path) -> None:
    """Load local AGIEval-style LSAT data and run a real held-out probe."""
    root = tmp_path / "curriculum"
    _write_jsonl(
        root / "stage1" / "agieval" / "lsat_train.jsonl",
        [
            {
                "passage": "All mammals breathe air.",
                "question": "Which statement follows?",
                "options": ["Fish breathe water.", "Whales breathe air.", "Birds are fish.", "None."],
                "label": "B",
            },
            {
                "passage": "Planets orbit stars.",
                "question": "What orbits stars?",
                "options": ["Planets", "Clouds", "Stones", "Cars"],
                "label": "A",
            },
        ],
    )
    _write_jsonl(
        root / "stage1" / "agieval" / "lsat_validation.jsonl",
        [
            {
                "passage": "Triangles have three sides.",
                "question": "Which is true?",
                "options": ["Triangles have four sides.", "Triangles have three sides.", "Squares are triangles.", "None."],
                "label": "B",
            }
        ],
    )

    domain = ReasoningCurriculumDomain(
        domain_variant="lsat",
        batch_size=2,
        seq_len=32,
        vocab_size=256,
        data_root=root,
    )
    train_batch = next(iter(domain.get_data_loader(domain.cost.current_level, split="train")))
    assert train_batch.tokens is not None
    assert "choice_tokens" in train_batch.domain_state
    assert train_batch.domain_state["choice_mask"].shape[-1] == 8

    runtime_domain = domain.build_runtime_domain(embed_dim=32)
    assert runtime_domain is not None
    model = PerfectReasoningProbeModel(runtime_domain)
    metrics = domain.run_advancement_probe(model, domain.cost.current_level)
    assert metrics["accuracy"] == 1.0
    assert metrics["generalization"] == 1.0


def test_public_stage1_open_answer_and_logic_sources(tmp_path: Path) -> None:
    """Normalize GSM8K-like open answers and FOLIO-style logic labels."""
    root = tmp_path / "curriculum"
    _write_jsonl(
        root / "stage1" / "gsm8k" / "train.jsonl",
        [{"question": "If Alice has 40 apples and gives away 3, how many remain?", "answer": "Let's think. #### 37"}],
    )
    _write_jsonl(
        root / "stage1" / "folio" / "validation.jsonl",
        [{"premises": ["Birds can fly.", "Penguins are birds."], "hypothesis": "Penguins can fly.", "label": "contradiction"}],
    )

    math_samples = load_public_reasoning_samples(
        domain_variant="math_competition",
        split="train",
        vocab_size=256,
        seq_len=24,
        max_samples=4,
        data_root=root,
    )
    assert len(math_samples) == 1
    assert int(math_samples[0]["answer"].item()) > 0
    assert not math_samples[0]["choice_mask"].any().item()

    logic_samples = load_public_reasoning_samples(
        domain_variant="formal_logic",
        split="val",
        vocab_size=256,
        seq_len=24,
        max_samples=4,
        data_root=root,
    )
    assert len(logic_samples) == 1
    assert int(logic_samples[0]["answer"].item()) > 0


def test_public_stage1_probe_handles_mixed_batch_and_truncates_long_choices(tmp_path: Path) -> None:
    """Probe scoring should handle mixed MCQ/open-answer batches and clip long option lists."""
    root = tmp_path / "curriculum"
    _write_jsonl(
        root / "stage1" / "agieval" / "lsat_train.jsonl",
        [
            {
                "passage": "Only the first eight options should be kept.",
                "question": "Which option is correct?",
                "options": [f"Option {idx}" for idx in range(10)],
                "label": "H",
            }
        ],
    )
    _write_jsonl(
        root / "stage1" / "agieval" / "lsat_validation.jsonl",
        [
            {
                "passage": "Mammals are warm-blooded.",
                "question": "Which option is correct?",
                "options": ["Fish are mammals.", "Mammals are warm-blooded."],
                "label": "B",
            }
        ],
    )
    _write_jsonl(
        root / "stage1" / "gsm8k" / "validation.jsonl",
        [{"question": "If there are 5 birds and 4 fly away, how many remain?", "answer": "#### 1"}],
    )

    lsat_samples = load_public_reasoning_samples(
        domain_variant="lsat",
        split="train",
        vocab_size=256,
        seq_len=24,
        max_samples=4,
        data_root=root,
    )
    assert len(lsat_samples) == 1
    assert int(lsat_samples[0]["correct_choice"].item()) == 7
    assert int(lsat_samples[0]["choice_mask"].sum().item()) == 8

    lsat_domain = ReasoningCurriculumDomain(
        domain_variant="lsat",
        batch_size=2,
        seq_len=24,
        vocab_size=256,
        data_root=root,
    )
    math_domain = ReasoningCurriculumDomain(
        domain_variant="math_competition",
        batch_size=2,
        seq_len=24,
        vocab_size=256,
        data_root=root,
    )
    runtime_domain = lsat_domain.build_runtime_domain(embed_dim=32)
    assert runtime_domain is not None

    mcq_batch = next(iter(lsat_domain.get_data_loader(lsat_domain.cost.current_level, split="val")))
    open_batch = next(iter(math_domain.get_data_loader(math_domain.cost.current_level, split="val")))
    mixed_state = dict(mcq_batch.domain_state)
    for key in ("answer_token", "choice_tokens", "choice_mask", "correct_choice", "target"):
        mixed_state[key] = torch.cat((mcq_batch.domain_state[key], open_batch.domain_state[key]), dim=0)

    model = PerfectReasoningProbeModel(runtime_domain)
    original_get_data_loader = lsat_domain.get_data_loader
    def _mixed_loader(level: int, split: str = "train") -> list:
        _ = level, split
        return [
            mcq_batch.__class__(
                domain_name=mcq_batch.domain_name,
                raw_inputs=torch.cat((mcq_batch.raw_inputs, open_batch.raw_inputs), dim=0),
                tokens=torch.cat((mcq_batch.tokens, open_batch.tokens), dim=0),
                embedded_tokens=None,
                input_mask=torch.cat((mcq_batch.input_mask, open_batch.input_mask), dim=0),
                targets={
                    key: torch.cat((mcq_batch.targets[key], open_batch.targets[key]), dim=0)
                    for key in ("answer_token", "choice_tokens", "choice_mask", "correct_choice")
                },
                domain_state=mixed_state,
                metadata=dict(mcq_batch.metadata),
            )
        ]

    lsat_domain.get_data_loader = _mixed_loader  # type: ignore[method-assign]
    try:
        metrics = lsat_domain._probe_split_accuracy(  # noqa: SLF001 - direct regression coverage for mixed batches
            model=model,
            runtime_domain=runtime_domain,
            level=lsat_domain.cost.current_level,
            split="val",
            device=model.anchor.device,
        )
    finally:
        lsat_domain.get_data_loader = original_get_data_loader  # type: ignore[method-assign]
    assert metrics == 1.0

    logits = model(domain_state=mixed_state)["action_vec"]
    choice_tokens = mixed_state["choice_tokens"].long().clamp(min=0, max=logits.shape[1] - 1)
    gathered = logits.gather(1, choice_tokens).masked_fill(~mixed_state["choice_mask"].bool(), float("-inf"))
    pred = logits.argmax(dim=-1)
    target = mixed_state["answer_token"].long()
    valid = (mixed_state["correct_choice"] >= 0).bool()
    pred[valid] = gathered[valid].argmax(dim=-1)
    target[valid] = mixed_state["correct_choice"][valid].long()
    assert torch.equal(pred, target)


def test_public_stage2_arc_and_oeis_loaders(tmp_path: Path) -> None:
    """Load ARC tasks and OEIS-like sequences from local public-data roots."""
    root = tmp_path / "curriculum"
    arc_dir = root / "stage2" / "arc_agi_2" / "train"
    arc_dir.mkdir(parents=True, exist_ok=True)
    for index in range(5):
        (arc_dir / f"task_{index}.json").write_text(
            json.dumps(
                {
                    "train": [
                        {
                            "input": [[index, 0], [0, index]],
                            "output": [[0, index], [index, 0]],
                        }
                    ]
                }
            )
        )
    oeis_dir = root / "stage2" / "oeis"
    oeis_dir.mkdir(parents=True, exist_ok=True)
    (oeis_dir / "stripped.txt").write_text("1, 1, 2, 3, 5, 8\n2, 4, 8, 16, 32, 64\n")

    arc_domain = PatternCurriculumDomain(domain_variant="arc_tasks", batch_size=2, seq_len=32, data_root=root)
    arc_train = next(iter(arc_domain.get_data_loader(arc_domain.cost.current_level, split="train")))
    assert arc_train.tokens is not None
    assert arc_train.tokens.shape[-1] == 32

    arc_val_samples = load_public_pattern_samples(
        domain_variant="arc_tasks",
        split="val",
        seq_len=32,
        max_samples=8,
        data_root=root,
    )
    assert arc_val_samples

    oeis_samples = load_public_pattern_samples(
        domain_variant="oeis_sequences",
        split="train",
        seq_len=16,
        max_samples=8,
        data_root=root,
    )
    assert len(oeis_samples) == 1
    assert int(oeis_samples[0]["tokens"].sum().item()) > 0

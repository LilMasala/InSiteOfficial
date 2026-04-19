"""Tests for the structured Stage-3 chess curriculum domain."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import yaml

from scripts.train_chamelia import build_stage_domains
from training.curriculum.domains.stage3_chess import ChessCurriculumDomain


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(__import__("json").dumps(row) + "\n" for row in rows))


def _write_puzzles_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "PuzzleId",
                "FEN",
                "Moves",
                "Rating",
                "RatingDeviation",
                "Popularity",
                "NbPlays",
                "Themes",
                "GameUrl",
                "OpeningTags",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_pgn(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(lines) + "\n")


def _stage3_root(tmp_path: Path) -> Path:
    return tmp_path / "curriculum" / "stage3" / "chess" / "lichess"


def _seed_chess_curriculum(tmp_path: Path) -> Path:
    root = _stage3_root(tmp_path)
    _write_jsonl(
        root / "train.jsonl",
        [
            {
                "moves": ["e2e4", "e7e5", "g1f3"],
                "fen": "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
                "best_move": "g1f3",
                "candidate_moves": ["g1f3", "f1c4", "d2d4"],
                "candidate_scores_cp": [42, 18, -30],
                "candidate_blunder_losses_cp": [0, 24, 72],
                "principal_variation": ["g1f3", "b8c6", "f1b5"],
                "centipawn_eval": 42,
                "phase": "opening",
                "source": "opening_book",
                "opening_tags": ["Italian_Game"],
            }
        ],
    )
    _write_jsonl(
        root / "validation.jsonl",
        [
            {
                "moves": ["e2e4", "c7c5", "g1f3", "d7d6"],
                "fen": "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
                "best_move": "d2d4",
                "candidate_moves": ["d2d4", "f1b5", "c2c3"],
                "candidate_scores_cp": [35, 8, -10],
                "candidate_blunder_losses_cp": [0, 27, 45],
                "principal_variation": ["d2d4", "c5d4"],
                "centipawn_eval": 35,
                "phase": "opening",
                "source": "opening_book",
                "opening_tags": ["Sicilian_Defense"],
            }
        ],
    )
    _write_puzzles_csv(
        root / "puzzles.csv",
        [
            {
                "PuzzleId": "fork1",
                "FEN": "r3r1k1/p4ppp/2p2n2/1p6/3P1qb1/2NQR3/PPB2PP1/R1B3K1 w - - 5 18",
                "Moves": "e3g3 e8e1 g1h2",
                "Rating": "2671",
                "RatingDeviation": "105",
                "Popularity": "87",
                "NbPlays": "325",
                "Themes": "advantage attraction fork middlegame sacrifice",
                "GameUrl": "https://lichess.org/gyFeQsOE#35",
                "OpeningTags": "French_Defense French_Defense_Exchange_Variation",
            },
            {
                "PuzzleId": "end1",
                "FEN": "8/8/3k4/8/3K4/8/4P3/8 w - - 0 1",
                "Moves": "d4e4 d6e6 e4d4",
                "Rating": "1200",
                "RatingDeviation": "80",
                "Popularity": "90",
                "NbPlays": "100",
                "Themes": "endgame pawnEndgame",
                "GameUrl": "https://lichess.org/example",
                "OpeningTags": "",
            },
        ],
    )
    _write_pgn(
        root / "train.pgn",
        [
            '[Event "Short"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0',
            '[Event "Full"]\n[Result "1-0"]\n\n1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 Nbd7 7. Rc1 c6 8. Bd3 dxc4 9. Bxc4 1-0',
        ],
    )
    return tmp_path / "curriculum"


def test_chess_curriculum_loads_structured_buckets(tmp_path: Path) -> None:
    curriculum_root = _seed_chess_curriculum(tmp_path)
    domain = ChessCurriculumDomain(batch_size=1, seq_len=8, data_root=curriculum_root)

    opening_batch = next(iter(domain.get_data_loader(level=0, split="train")))
    assert opening_batch.tokens.shape == (1, 8, 8, 111)
    assert opening_batch.domain_state["bucket"][0] == "openings"
    assert opening_batch.domain_state["source"][0] == "opening_book"

    puzzle_batch = next(iter(domain.get_data_loader(level=1, split="train")))
    assert puzzle_batch.domain_state["bucket"][0] == "puzzles"
    assert "fork" in puzzle_batch.domain_state["motif_tags"][0]

    endgame_batch = next(iter(domain.get_data_loader(level=2, split="train")))
    assert endgame_batch.domain_state["bucket"][0] == "endgames"
    assert endgame_batch.domain_state["phase"][0] == "endgame"

    short_game_batch = next(iter(domain.get_data_loader(level=4, split="train")))
    assert short_game_batch.domain_state["bucket"][0] == "short_games"
    assert short_game_batch.domain_state["source"][0] == "short_game"


def test_chess_curriculum_probe_and_stockfish_fallback(tmp_path: Path) -> None:
    curriculum_root = _seed_chess_curriculum(tmp_path)
    domain = ChessCurriculumDomain(
        batch_size=1,
        seq_len=8,
        data_root=curriculum_root,
        curriculum_config={"stockfish_path": "/missing/stockfish"},
    )
    zero_metrics = domain.run_advancement_probe(None, level=3)
    assert zero_metrics["middlegame_accuracy"] == 0.0
    runtime = domain.build_runtime_domain(embed_dim=16)
    opponent = domain._opponent_for_level(3, runtime)  # noqa: SLF001
    assert opponent.name.startswith("builtin_")


def test_build_stage_domains_uses_chess_curriculum_domain(tmp_path: Path) -> None:
    curriculum_root = _seed_chess_curriculum(tmp_path)
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "curriculum_hjepa_single_gpu.yaml").read_text()
    )
    stages = build_stage_domains(
        curriculum_config=config,
        selected_stages=[3],
        selected_domains=["chess"],
        data_root=str(curriculum_root),
    )
    assert len(stages) == 1
    assert len(stages[0]) == 1
    assert isinstance(stages[0][0], ChessCurriculumDomain)

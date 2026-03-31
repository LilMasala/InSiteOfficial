"""Static curriculum data-source metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataSourceSpec:
    """Metadata for one curriculum data source."""

    stage: int
    name: str
    description: str
    estimated_size_gb: float


def stage_source_specs() -> dict[int, list[DataSourceSpec]]:
    """Return stage-indexed source metadata from the curriculum spec."""
    return {
        0: [
            DataSourceSpec(0, "wikipedia_multilingual", "Wikipedia in 15 languages", 50.0),
            DataSourceSpec(0, "books", "Long-form public-domain books", 5.0),
            DataSourceSpec(0, "medical_text", "Medical text and case narratives", 5.0),
        ],
        1: [
            DataSourceSpec(1, "lsat", "LSAT reasoning corpus", 1.0),
            DataSourceSpec(1, "math_competition", "AMC/AIME and related math reasoning", 0.5),
            DataSourceSpec(1, "basic_arithmetic", "Simple arithmetic expressions and traces", 0.1),
        ],
        2: [
            DataSourceSpec(2, "oeis", "Integer sequences and formulas", 2.0),
            DataSourceSpec(2, "arc", "ARC abstraction tasks", 0.01),
            DataSourceSpec(2, "synthetic_regimes", "Generated HMM and arithmetic pattern data", 0.1),
        ],
        3: [
            DataSourceSpec(3, "chess", "Lichess games and puzzles plus Stockfish", 102.0),
            DataSourceSpec(3, "poker", "Solver-backed poker traces", 10.0),
        ],
        4: [DataSourceSpec(4, "collaborative_selfplay", "Self-play coordination tasks", 1.0)],
        5: [
            DataSourceSpec(5, "clinical_reasoning", "Medical vignettes and care dialogues", 2.0),
            DataSourceSpec(5, "synthetic_patients", "Synthetic physiology trajectories", 1.0),
        ],
    }

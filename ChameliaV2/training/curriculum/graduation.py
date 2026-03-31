"""Graduation manager for curriculum stages and domains."""

from __future__ import annotations

from typing import Any

from training.curriculum.domains.base import CurriculumDomain


class GraduationManager:
    """Manage advancement and graduation checks across curriculum stages."""

    def __init__(self, stages: list[list[CurriculumDomain]], config: dict[str, Any]) -> None:
        """Initialize the graduation manager.

        Args:
            stages: Stages represented as lists of domains.
            config: Curriculum configuration dictionary.
        """
        self.stages = stages
        self.config = config
        self.probe_history: dict[int, list[dict[str, Any]]] = {}

    def run_stage_probe(self, model: Any, stage_idx: int) -> dict[str, dict[str, float]]:
        """Run probes for every domain in the target stage.

        Args:
            model: Model under evaluation.
            stage_idx: Stage index.

        Returns:
            Nested mapping from domain name to probe metrics.
        """
        results: dict[str, dict[str, float]] = {}
        for domain in self.stages[stage_idx]:
            results[domain.domain_name()] = domain.run_advancement_probe(model, domain.cost.current_level)
        self.probe_history.setdefault(stage_idx, []).append(results)
        return results

    def check_advancement(self, probe_results: dict[str, float], stage_idx: int) -> tuple[bool, str]:
        """Check whether a domain should advance based on current probe results.

        Args:
            probe_results: Probe results for a single domain.
            stage_idx: Stage index for reporting only.

        Returns:
            Tuple of ``(should_advance, reason)``.
        """
        _ = stage_idx
        passed = all(value >= 0.75 for value in probe_results.values())
        reason = "thresholds_met" if passed else "insufficient_probe_metrics"
        return passed, reason

    def check_stage_graduation(self, model: Any, stage_idx: int) -> tuple[bool, dict[str, dict[str, float]]]:
        """Run the full graduation check for one stage.

        Args:
            model: Model under evaluation.
            stage_idx: Stage index.

        Returns:
            Tuple of ``(passed, metrics)``.
        """
        metrics = self.run_stage_probe(model, stage_idx)
        passed = all(
            domain.cost.current_level >= len(domain.get_cost_schedule()) - 1
            and all(value >= 0.85 for value in metrics[domain.domain_name()].values())
            for domain in self.stages[stage_idx]
        )
        return passed, metrics

    def get_status_report(self) -> str:
        """Return a concise human-readable status report."""
        parts: list[str] = []
        for stage_idx, domains in enumerate(self.stages):
            domain_status = ", ".join(
                f"{domain.domain_name()}:L{domain.cost.current_level}" for domain in domains
            )
            parts.append(f"stage {stage_idx}: {domain_status}")
        return " | ".join(parts)

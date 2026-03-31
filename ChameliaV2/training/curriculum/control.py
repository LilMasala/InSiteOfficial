"""Adaptive control logic for curriculum training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from training.curriculum.domains.base import CurriculumDomain


@dataclass(frozen=True)
class EvalPoint:
    """One evaluation snapshot used by the adaptive controller."""

    step: int
    mean_loss: float
    stage_score: float
    metrics: dict[str, dict[str, float]]


@dataclass(frozen=True)
class ControllerDecision:
    """Action selected by the adaptive controller."""

    action: str
    reason: str
    new_budget_steps: int | None = None


class AdaptiveTrainingController:
    """Bounded controller for extend/retune/fail stage decisions."""

    def __init__(
        self,
        *,
        extension_factor: float = 1.5,
        min_score_delta: float = 0.01,
        min_score_slope: float = 1e-4,
        eval_window: int = 3,
    ) -> None:
        """Initialize the controller.

        Args:
            extension_factor: Budget multiplier when a stage is still improving.
            min_score_delta: Minimum recent score gain treated as meaningful.
            min_score_slope: Minimum positive slope treated as promising.
            eval_window: Number of recent eval points to inspect.
        """
        self.extension_factor = extension_factor
        self.min_score_delta = min_score_delta
        self.min_score_slope = min_score_slope
        self.eval_window = eval_window

    def stage_score(
        self,
        stage_domains: list[CurriculumDomain],
        metrics: dict[str, dict[str, float]],
    ) -> float:
        """Summarize stage progress into a scalar in [0, 1].

        Args:
            stage_domains: Domains in the active stage.
            metrics: Latest probe metrics by domain.

        Returns:
            Scalar stage progress score.
        """
        if not stage_domains:
            return 0.0

        domain_scores: list[float] = []
        for domain in stage_domains:
            level_den = max(1, len(domain.get_cost_schedule()) - 1)
            level_progress = domain.cost.current_level / level_den
            domain_metrics = metrics.get(domain.domain_name(), {})
            metric_values = list(domain_metrics.values())
            if metric_values:
                metric_progress = sum(min(max(value / 0.85, 0.0), 1.0) for value in metric_values) / len(
                    metric_values
                )
            else:
                metric_progress = 0.0
            domain_scores.append(0.5 * level_progress + 0.5 * metric_progress)
        return sum(domain_scores) / len(domain_scores)

    def metric_score(self, metrics: dict[str, dict[str, float]]) -> float:
        """Summarize raw probe metrics into a scalar in [0, 1].

        Args:
            metrics: Latest probe metrics by domain.

        Returns:
            Scalar metric-only progress score.
        """
        if not metrics:
            return 0.0
        domain_scores: list[float] = []
        for domain_metrics in metrics.values():
            metric_values = list(domain_metrics.values())
            if not metric_values:
                domain_scores.append(0.0)
                continue
            normalized = [min(max(value / 0.85, 0.0), 1.0) for value in metric_values]
            domain_scores.append(sum(normalized) / len(normalized))
        return sum(domain_scores) / len(domain_scores)

    def is_promising(
        self,
        history: list[EvalPoint],
        *,
        total_steps: int,
        max_total_stage_steps: int,
    ) -> bool:
        """Return whether recent progress justifies extending the budget.

        Args:
            history: Eval history for the active stage.
            total_steps: Current number of optimization steps for this stage.
            max_total_stage_steps: Hard cap on stage steps.

        Returns:
            Boolean promising flag.
        """
        if len(history) < 2:
            return True

        recent = history[-self.eval_window :]
        steps = [point.step for point in recent]
        scores = [point.stage_score for point in recent]
        metric_scores = [self.metric_score(point.metrics) for point in recent]
        losses = [point.mean_loss for point in recent]
        if any(not math.isfinite(loss) for loss in losses):
            return False

        step_span = max(1, steps[-1] - steps[0])
        slope = (scores[-1] - scores[0]) / step_span
        recent_gain = scores[-1] - scores[0]
        metric_slope = (metric_scores[-1] - metric_scores[0]) / step_span
        metric_gain = metric_scores[-1] - metric_scores[0]
        if (
            slope <= self.min_score_slope
            and recent_gain <= self.min_score_delta
            and metric_slope <= self.min_score_slope
            and metric_gain <= self.min_score_delta
        ):
            return False

        remaining = max_total_stage_steps - total_steps
        if remaining <= 0:
            return False
        current_progress = max(scores[-1], metric_scores[-1])
        if current_progress >= 1.0:
            return True

        projected_steps = (1.0 - current_progress) / max(max(slope, metric_slope), self.min_score_slope)
        return projected_steps <= remaining

    def decide(
        self,
        *,
        history: list[EvalPoint],
        stage_passed: bool,
        total_steps: int,
        current_budget_steps: int,
        max_total_stage_steps: int,
        extensions_used: int,
        max_extensions: int,
        retunes_used: int,
        max_retunes: int,
    ) -> ControllerDecision:
        """Choose the next controller action.

        Args:
            history: Eval history for this stage.
            stage_passed: Whether the stage graduated.
            total_steps: Steps already spent on this stage.
            current_budget_steps: Current stage budget.
            max_total_stage_steps: Hard cap for the stage.
            extensions_used: Extensions already consumed.
            max_extensions: Extension limit.
            retunes_used: Retunes already consumed.
            max_retunes: Retune limit.

        Returns:
            Controller decision.
        """
        if stage_passed:
            return ControllerDecision(action="graduate", reason="graduation_gate_met")

        if total_steps < current_budget_steps:
            return ControllerDecision(action="continue", reason="within_budget")

        if extensions_used < max_extensions and self.is_promising(
            history,
            total_steps=total_steps,
            max_total_stage_steps=max_total_stage_steps,
        ):
            new_budget = min(
                max_total_stage_steps,
                max(current_budget_steps + 1, int(current_budget_steps * self.extension_factor)),
            )
            return ControllerDecision(
                action="extend",
                reason="metrics_still_improving",
                new_budget_steps=new_budget,
            )

        if retunes_used < max_retunes:
            return ControllerDecision(action="retune", reason="budget_exhausted_without_pass")

        return ControllerDecision(action="fail", reason="budget_and_retunes_exhausted")

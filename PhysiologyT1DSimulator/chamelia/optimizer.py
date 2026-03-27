"""Block 7: Optimization Engine — action search with safety constraints.

Receives a state, a baseline action, a user profile, and access to the model
zoo. Returns a recommended action that maximises risk-adjusted reward subject
to safety constraints — or 'no recommendation' if nothing beats the baseline
confidently.

Components:
    - Action Space: TherapySchedule (piecewise-constant ISF/CR/basal)
    - Search Strategy Tiers: Grid (real), Bayesian (stub), RL (stub)
    - Reward Function: user-weighted TIR/low/high/var + change penalty + uncertainty penalty
    - Constraint System: pre-filter + post-evaluation safety check
    - RecommendationPackage: output format
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from chamelia.models.base import PredictionEnvelope, PredictorCard
    from t1d_sim.agency import UserAgencyProfile


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

@dataclass
class TherapyAction:
    """A therapy parameter set: ISF/CR/basal multipliers.

    For the POC, a single set of global multipliers. The architecture supports
    piecewise-constant schedules (different params for time-of-day intervals)
    via the `intervals` field.
    """
    isf_multiplier: float = 1.0
    cr_multiplier: float = 1.0
    basal_multiplier: float = 1.0
    intervals: list[dict] | None = None  # Future: [{start_h, end_h, isf, cr, basal}, ...]

    def to_array(self) -> np.ndarray:
        return np.array([self.isf_multiplier, self.cr_multiplier, self.basal_multiplier])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "TherapyAction":
        a = np.asarray(arr).flatten()
        return cls(
            isf_multiplier=float(a[0]),
            cr_multiplier=float(a[1]) if len(a) > 1 else 1.0,
            basal_multiplier=float(a[2]) if len(a) > 2 else 1.0,
        )


@dataclass
class BehaviorPlan:
    """Stub action type for future behavioral interventions."""
    items: list[dict] = field(default_factory=list)
    # Each item: {type, start_h, duration_min, intensity, burden_estimate, constraints}


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------

@dataclass
class ObjectiveWeights:
    """User-controlled objective weights for the reward function.

    reward = w_tir × TIR - w_low × %low - w_high × %high - w_var × BG_var + w_stab × stability
           + w_mood × predicted_mood_delta - w_burden × rec_burden - w_anxiety × anxiety_cost
    """
    w_tir: float = 1.0
    w_low: float = 2.0    # Higher weight because lows are dangerous
    w_high: float = 0.5
    w_var: float = 0.1
    w_stab: float = 0.3
    w_mood: float = 0.3          # Mood delta weight
    w_burden: float = 0.1        # Recommendation burden penalty
    w_anxiety: float = 0.2       # Change anxiety cost
    conservativeness: float = 0.5  # 0 = aggressive, 1 = very conservative


def compute_reward(
    predicted_outcomes: dict[str, float],
    baseline_outcomes: dict[str, float],
    action: TherapyAction,
    baseline_action: TherapyAction,
    weights: ObjectiveWeights,
    prediction_uncertainty: float = 0.0,
    mood_valence: float = 0.0,
    change_anxiety: float = 0.0,
    n_prior_recs: int = 0,
) -> float:
    """Compute the risk-adjusted reward for a proposed action.

    Components:
        1. Outcome reward: weighted sum of predicted metrics
        2. Change penalty: proportional to deviation from current settings
        3. Uncertainty penalty: prefers confident recommendations

    Args:
        predicted_outcomes: {tir, percent_low, percent_high, mean_bg, bg_var}
        baseline_outcomes:  Same keys for the baseline/current action.
        action:             Proposed TherapyAction.
        baseline_action:    Current TherapyAction.
        weights:            User's objective weights.
        prediction_uncertainty: Mean prediction interval width.

    Returns:
        Scalar reward value (higher = better).
    """
    # Outcome reward.
    tir = predicted_outcomes.get("tir", 0.65)
    pct_low = predicted_outcomes.get("percent_low", 0.05)
    pct_high = predicted_outcomes.get("percent_high", 0.30)
    bg_var = predicted_outcomes.get("bg_var", 1000.0)

    # Stability: TIR improvement over baseline.
    baseline_tir = baseline_outcomes.get("tir", 0.65)
    stability = max(0.0, tir - baseline_tir)

    reward = (
        weights.w_tir * tir
        - weights.w_low * pct_low
        - weights.w_high * pct_high
        - weights.w_var * (bg_var / 1000.0)  # Normalise variance
        + weights.w_stab * stability
    )

    # Change penalty: cost proportional to deviation, scaled by conservativeness.
    delta = action.to_array() - baseline_action.to_array()
    change_magnitude = float(np.linalg.norm(delta))
    change_penalty = weights.conservativeness * change_magnitude * 2.0
    reward -= change_penalty

    # Uncertainty penalty: prefer confident recommendations.
    uncertainty_penalty = 0.3 * prediction_uncertainty
    reward -= uncertainty_penalty

    # Mood-integrated components (Section 6.1)
    # Mood delta: prefer gentler recs when mood is fragile
    if mood_valence < -0.1:
        mood_penalty = weights.w_mood * abs(mood_valence) * change_magnitude
        reward -= mood_penalty

    # Change anxiety cost: first-ever rec carries high anxiety cost
    if change_anxiety > 0 and change_magnitude > 0:
        novelty_factor = max(0.1, 1.0 - 0.05 * n_prior_recs)
        anxiety_cost = weights.w_anxiety * change_anxiety * novelty_factor * change_magnitude
        reward -= anxiety_cost

    # Recommendation burden: penalize frequent large changes
    burden_cost = weights.w_burden * change_magnitude
    reward -= burden_cost

    return float(reward)


# ---------------------------------------------------------------------------
# Constraint System
# ---------------------------------------------------------------------------

@dataclass
class ConstraintConfig:
    """Configuration for the action space constraint system.

    Pre-filter (before evaluation):
        - Aggressiveness bounds (max deviation from current)
        - Locked time windows (do not change certain periods)
        - Minimum step size

    Post-evaluation (after model scoring):
        - Hard safety via ConfidenceModule
    """
    max_isf_deviation: float = 0.15     # ±15% from current
    max_cr_deviation: float = 0.15
    max_basal_deviation: float = 0.10
    min_step_size: float = 0.02         # Ignore changes < 2%
    locked_hours: list[tuple[int, int]] = field(default_factory=list)  # [(start_h, end_h), ...]


def apply_prefilter(
    candidates: list[TherapyAction],
    baseline: TherapyAction,
    constraints: ConstraintConfig,
    aggressiveness: float = 0.5,
) -> list[TherapyAction]:
    """Pre-filter candidates based on aggressiveness and constraints.

    Args:
        candidates:      List of candidate actions to filter.
        baseline:        Current therapy settings.
        constraints:     Constraint config.
        aggressiveness:  From UserAgencyProfile (0–1). Scales max deviation.

    Returns:
        Filtered list of candidates that pass all pre-filters.
    """
    scale = 0.5 + aggressiveness  # 0.5–1.5x the configured max deviation
    max_isf = constraints.max_isf_deviation * scale
    max_cr = constraints.max_cr_deviation * scale
    max_basal = constraints.max_basal_deviation * scale

    filtered = []
    for c in candidates:
        d_isf = abs(c.isf_multiplier - baseline.isf_multiplier)
        d_cr = abs(c.cr_multiplier - baseline.cr_multiplier)
        d_basal = abs(c.basal_multiplier - baseline.basal_multiplier)

        # Aggressiveness bounds.
        if d_isf > max_isf or d_cr > max_cr or d_basal > max_basal:
            continue

        # Minimum step size.
        total_change = d_isf + d_cr + d_basal
        if total_change < constraints.min_step_size and total_change > 0:
            continue

        filtered.append(c)

    return filtered


# ---------------------------------------------------------------------------
# Recommendation Package
# ---------------------------------------------------------------------------

class RecommendationDecision:
    """Decision types for the optimizer output."""
    RECOMMEND = "recommend"
    HOLD = "hold"
    INSUFFICIENT = "insufficient_confidence"


@dataclass
class RecommendationPackage:
    """Output of the optimisation engine.

    Contains the primary recommendation, alternatives, baseline prediction,
    and the overall decision.
    """
    decision: str  # RecommendationDecision value
    primary: TherapyAction | None = None
    primary_predicted_outcomes: dict[str, float] = field(default_factory=dict)
    primary_confidence: float = 0.0
    primary_improvement_vs_baseline: float = 0.0
    primary_reward: float = 0.0
    explanation: str = ""
    risk_assessment: str = ""
    framing: str = "tentative"  # tentative, reinforcing, gentle_reminder, celebrating
    alternatives: list[TherapyAction] = field(default_factory=list)
    baseline_prediction: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Optimizer Base Class
# ---------------------------------------------------------------------------

class BaseOptimizer(ABC):
    """Abstract base for all optimiser tiers."""

    @abstractmethod
    def search(
        self,
        features: np.ndarray,
        baseline_action: TherapyAction,
        models: dict[str, "PredictorCard"],
        weights: ObjectiveWeights,
        constraints: ConstraintConfig,
        aggressiveness: float = 0.5,
    ) -> RecommendationPackage:
        """Search the action space for the best recommendation.

        Args:
            features:        Current feature vector.
            baseline_action: Current therapy settings.
            models:          model_id → PredictorCard (active zoo models).
            weights:         User's objective weights.
            constraints:     Action space constraints.
            aggressiveness:  From UserAgencyProfile.

        Returns:
            RecommendationPackage with the decision and recommended action.
        """


# ---------------------------------------------------------------------------
# Tier 1: Grid Search With Pruning
# ---------------------------------------------------------------------------

class GridSearchOptimizer(BaseOptimizer):
    """Grid search over discretised action space with pruning.

    Discretises ISF/CR/basal around current settings, evaluates each candidate
    via the model zoo, and returns the best risk-adjusted action.

    Easy to implement, easy to debug; the bottleneck is zoo evaluation speed
    (fast for XGBoost, tolerable for small neural models).
    """

    def __init__(
        self,
        isf_steps: int = 5,
        cr_steps: int = 3,
        basal_steps: int = 3,
        top_k_joint: int = 10,
    ) -> None:
        self._isf_steps = isf_steps
        self._cr_steps = cr_steps
        self._basal_steps = basal_steps
        self._top_k = top_k_joint

    def search(
        self,
        features: np.ndarray,
        baseline_action: TherapyAction,
        models: dict[str, "PredictorCard"],
        weights: ObjectiveWeights,
        constraints: ConstraintConfig,
        aggressiveness: float = 0.5,
    ) -> RecommendationPackage:
        """Grid search with pruning.

        Strategy:
            1. Generate grid of candidates centred on baseline.
            2. Pre-filter by constraints.
            3. Evaluate each candidate with all zoo models.
            4. Score by risk-adjusted reward.
            5. Return best + alternatives.
        """
        # Step 1: Generate grid.
        scale = 0.5 + aggressiveness
        isf_range = np.linspace(
            baseline_action.isf_multiplier - constraints.max_isf_deviation * scale,
            baseline_action.isf_multiplier + constraints.max_isf_deviation * scale,
            self._isf_steps,
        )
        cr_range = np.linspace(
            baseline_action.cr_multiplier - constraints.max_cr_deviation * scale,
            baseline_action.cr_multiplier + constraints.max_cr_deviation * scale,
            self._cr_steps,
        )
        basal_range = np.linspace(
            baseline_action.basal_multiplier - constraints.max_basal_deviation * scale,
            baseline_action.basal_multiplier + constraints.max_basal_deviation * scale,
            self._basal_steps,
        )

        candidates = [
            TherapyAction(
                isf_multiplier=float(np.clip(isf, 0.70, 1.35)),
                cr_multiplier=float(np.clip(cr, 0.70, 1.35)),
                basal_multiplier=float(np.clip(bas, 0.75, 1.25)),
            )
            for isf in isf_range
            for cr in cr_range
            for bas in basal_range
        ]

        # Always include "hold" (baseline).
        candidates.append(TherapyAction(
            isf_multiplier=baseline_action.isf_multiplier,
            cr_multiplier=baseline_action.cr_multiplier,
            basal_multiplier=baseline_action.basal_multiplier,
        ))

        # Step 2: Pre-filter.
        candidates = apply_prefilter(candidates, baseline_action, constraints, aggressiveness)

        if not candidates:
            return RecommendationPackage(
                decision=RecommendationDecision.HOLD,
                explanation="No candidates passed pre-filter constraints.",
                baseline_prediction=self._evaluate_action(features, baseline_action, models),
            )

        # Step 3+4: Evaluate and score each candidate.
        baseline_outcomes = self._evaluate_action(features, baseline_action, models)
        scored: list[tuple[float, TherapyAction, dict[str, float], float]] = []

        for c in candidates:
            outcomes = self._evaluate_action(features, c, models)
            uncertainty = self._estimate_uncertainty(features, c, models)
            reward = compute_reward(
                outcomes, baseline_outcomes, c, baseline_action,
                weights, uncertainty,
            )
            scored.append((reward, c, outcomes, uncertainty))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Step 5: Check if best candidate beats baseline.
        best_reward, best_action, best_outcomes, best_uncertainty = scored[0]
        baseline_reward = compute_reward(
            baseline_outcomes, baseline_outcomes,
            baseline_action, baseline_action, weights, 0.0,
        )
        improvement = best_reward - baseline_reward

        if improvement <= 0 or best_action.to_array().tolist() == baseline_action.to_array().tolist():
            return RecommendationPackage(
                decision=RecommendationDecision.HOLD,
                explanation="Current settings look good — no change recommended.",
                baseline_prediction=baseline_outcomes,
                primary_improvement_vs_baseline=0.0,
            )

        # Build alternatives (next 2 distinct candidates).
        alternatives = []
        for r, a, o, u in scored[1:]:
            if len(alternatives) >= 2:
                break
            if not np.allclose(a.to_array(), best_action.to_array(), atol=0.01):
                alternatives.append(a)

        # Explanation.
        delta = best_action.to_array() - baseline_action.to_array()
        parts = []
        names = ["ISF", "CR", "Basal"]
        for i, name in enumerate(names):
            if abs(delta[i]) > 0.01:
                direction = "increase" if delta[i] > 0 else "decrease"
                parts.append(f"{name} {direction} by {abs(delta[i]):.1%}")
        explanation = "; ".join(parts) if parts else "Minor adjustment"

        tir_improvement = best_outcomes.get("tir", 0) - baseline_outcomes.get("tir", 0)
        risk = "low" if best_outcomes.get("percent_low", 0) < 0.03 else "moderate"

        return RecommendationPackage(
            decision=RecommendationDecision.RECOMMEND,
            primary=best_action,
            primary_predicted_outcomes=best_outcomes,
            primary_confidence=float(np.clip(1.0 - best_uncertainty, 0.0, 1.0)),
            primary_improvement_vs_baseline=improvement,
            primary_reward=best_reward,
            explanation=explanation,
            risk_assessment=f"Hypo risk: {risk}. Predicted TIR change: {tir_improvement:+.1%}",
            alternatives=alternatives,
            baseline_prediction=baseline_outcomes,
        )

    def _evaluate_action(
        self,
        features: np.ndarray,
        action: TherapyAction,
        models: dict[str, "PredictorCard"],
    ) -> dict[str, float]:
        """Evaluate an action across all zoo models and aggregate."""
        predictions: list[dict[str, float]] = []
        action_arr = action.to_array()

        for model_id, model in models.items():
            try:
                env = model.predict(features, action=action_arr)
                p = np.asarray(env.point)
                if p.size >= 4:
                    predictions.append({
                        "percent_low": float(p.flat[0]),
                        "percent_high": float(p.flat[1]),
                        "tir": float(p.flat[2]),
                        "mean_bg": float(p.flat[3]),
                    })
                elif p.size == 1:
                    predictions.append({"tir": float(p.flat[0])})
            except Exception:
                continue

        if not predictions:
            return {"tir": 0.65, "percent_low": 0.05, "percent_high": 0.30, "mean_bg": 150.0}

        # Aggregate: mean across models.
        keys = set()
        for p in predictions:
            keys.update(p.keys())
        result = {}
        for k in keys:
            vals = [p[k] for p in predictions if k in p]
            result[k] = float(np.mean(vals))
        return result

    def _estimate_uncertainty(
        self,
        features: np.ndarray,
        action: TherapyAction,
        models: dict[str, "PredictorCard"],
    ) -> float:
        """Estimate prediction uncertainty across zoo models."""
        widths: list[float] = []
        action_arr = action.to_array()

        for model_id, model in models.items():
            try:
                env = model.predict(features, action=action_arr)
                interval = np.abs(np.asarray(env.upper) - np.asarray(env.lower))
                widths.append(float(np.mean(interval)))
            except Exception:
                continue

        return float(np.mean(widths)) if widths else 1.0


# ---------------------------------------------------------------------------
# Tier 2: Bayesian Optimization (Stub)
# ---------------------------------------------------------------------------

class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimisation over the action→reward surface.

    Fits a GP surrogate over (action → predicted reward) and uses an
    acquisition function (Expected Improvement or Constrained EI).

    Activation conditions:
        - Zoo evaluations become expensive (neural models in ensemble)
        - Action space is large (piecewise schedules with many intervals)
        - Grid search is taking too long per cycle

    Status: STUB — raises NotImplementedError.
    """

    def search(
        self,
        features: np.ndarray,
        baseline_action: TherapyAction,
        models: dict[str, "PredictorCard"],
        weights: ObjectiveWeights,
        constraints: ConstraintConfig,
        aggressiveness: float = 0.5,
    ) -> RecommendationPackage:
        raise NotImplementedError(
            "BayesianOptimizer is not yet implemented. "
            "Activation conditions: zoo evaluations become expensive "
            "(neural models in ensemble) or action space is large "
            "(piecewise schedules). Use GridSearchOptimizer for now."
        )


# ---------------------------------------------------------------------------
# Tier 3: Offline RL Policy (Stub)
# ---------------------------------------------------------------------------

class RLPolicyOptimizer(BaseOptimizer):
    """Offline RL policy: state → action via Conservative Q-Learning.

    Trained on the trajectory dataset from the forked simulator.
    Advantage is speed (single forward pass); disadvantage is data
    requirements and difficulty with constraints.

    Runs alongside search-based optimizer as a warm start, not replacement.

    Activation conditions:
        - Large trajectory dataset from forked sim (>10K trajectories)
        - Validated policy via off-policy evaluation
        - Grid search warm-start shows RL policy is competitive

    Status: STUB — raises NotImplementedError.
    """

    def search(
        self,
        features: np.ndarray,
        baseline_action: TherapyAction,
        models: dict[str, "PredictorCard"],
        weights: ObjectiveWeights,
        constraints: ConstraintConfig,
        aggressiveness: float = 0.5,
    ) -> RecommendationPackage:
        raise NotImplementedError(
            "RLPolicyOptimizer is not yet implemented. "
            "Activation conditions: large trajectory dataset from forked sim "
            "(>10K trajectories), validated policy via off-policy evaluation. "
            "Use GridSearchOptimizer for now."
        )

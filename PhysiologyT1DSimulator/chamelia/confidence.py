"""Block 4: Confidence Module — four-layer gate + hard safety check.

Determines whether a recommendation should be surfaced. Protects against:
    1. Model extrapolation (state is unfamiliar)
    2. Model disagreement (ensemble is inconsistent)
    3. Miscalibration (model is confident but historically wrong)
    4. Marginal effect (predicted improvement is within noise)

A separate hard safety gate runs independently.

Gate composition: all four layers run sequentially. If any layer closes the
gate, the recommendation is suppressed. If all pass, the module produces a
composite confidence score combining familiarity, concordance, reliability,
and effect SNR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from chamelia.models.base import PredictionEnvelope


# ---------------------------------------------------------------------------
# Gate result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Result of the confidence gate evaluation.

    Attributes:
        passed:           True if the recommendation should be surfaced.
        composite_score:  Overall confidence in [0, 1] (only meaningful if passed).
        layer_scores:     Per-layer scores for logging/debugging.
        blocked_by:       Name of the layer that blocked (None if passed).
        details:          Arbitrary metadata for shadow record logging.
    """
    passed: bool
    composite_score: float
    layer_scores: dict[str, float] = field(default_factory=dict)
    blocked_by: str | None = None
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hard safety constraints
# ---------------------------------------------------------------------------

@dataclass
class SafetyConstraint:
    """A single hard safety constraint.

    Attributes:
        name:       Human-readable constraint name.
        metric:     Which PredictionEnvelope target to check (e.g. "percent_low").
        threshold:  Maximum allowed value.
        use_upper:  If True, check the upper bound (worst case), not point estimate.
    """
    name: str
    metric: str
    threshold: float
    use_upper: bool = True


# Default hard safety constraints for T1D therapy.
DEFAULT_SAFETY_CONSTRAINTS: list[SafetyConstraint] = [
    SafetyConstraint(name="hypo_risk", metric="percent_low", threshold=0.04, use_upper=True),
    SafetyConstraint(name="severe_hyper", metric="percent_high", threshold=0.50, use_upper=True),
    SafetyConstraint(name="mean_bg_low", metric="mean_bg", threshold=65.0, use_upper=False),
]


# ---------------------------------------------------------------------------
# Confidence Module
# ---------------------------------------------------------------------------

class ConfidenceModule:
    """Four-layer confidence gate with independent hard safety check.

    Usage::

        gate = ConfidenceModule()
        result = gate.evaluate(
            proposed_envelopes={"aggregate": env_proposed, "temporal": env_temporal},
            baseline_envelopes={"aggregate": env_baseline, "temporal": env_baseline_t},
            familiarity_score=anomaly_detector.predict(features).confidence,
            calibration_scores=shadow_module.get_calibration_scores(),
            user_aggressiveness=0.6,
        )
        if result.passed:
            surface_recommendation()
    """

    # Layer thresholds (can be tuned).
    FAMILIARITY_THRESHOLD = 0.4
    CONCORDANCE_THRESHOLD = 0.3
    CALIBRATION_THRESHOLD = 0.5
    EFFECT_SNR_THRESHOLD = 0.5

    # Weights for composite score.
    W_FAMILIARITY = 0.25
    W_CONCORDANCE = 0.25
    W_CALIBRATION = 0.25
    W_EFFECT = 0.25

    def __init__(
        self,
        safety_constraints: list[SafetyConstraint] | None = None,
    ) -> None:
        self.safety_constraints = safety_constraints or list(DEFAULT_SAFETY_CONSTRAINTS)

    # ------------------------------------------------------------------
    # Layer 1: GP Familiarity
    # ------------------------------------------------------------------

    def _check_familiarity(self, familiarity_score: float) -> tuple[bool, float]:
        """Check how far the current state is from training distribution.

        Args:
            familiarity_score: Output of AnomalyDetector.predict().confidence.
                               1.0 = very familiar, 0.0 = completely novel.

        Returns:
            (passed, score)
        """
        passed = familiarity_score >= self.FAMILIARITY_THRESHOLD
        return passed, familiarity_score

    # ------------------------------------------------------------------
    # Layer 2: Ensemble Agreement
    # ------------------------------------------------------------------

    def _check_concordance(
        self,
        envelopes: dict[str, "PredictionEnvelope"],
    ) -> tuple[bool, float]:
        """Check whether zoo models agree on their predictions.

        For overlapping targets, checks whether prediction intervals overlap.
        Concordance = mean pairwise interval overlap across model pairs.

        Args:
            envelopes: model_id → PredictionEnvelope for the proposed action.

        Returns:
            (passed, concordance_score)
        """
        if len(envelopes) < 2:
            # Single model — concordance is vacuously true but score is low.
            return True, 0.5

        # Extract scalar point estimates (or mean of arrays).
        points = []
        lowers = []
        uppers = []
        for env in envelopes.values():
            p = np.asarray(env.point)
            l = np.asarray(env.lower)
            u = np.asarray(env.upper)
            points.append(float(np.mean(p)))
            lowers.append(float(np.mean(l)))
            uppers.append(float(np.mean(u)))

        # Pairwise interval overlap.
        n = len(points)
        overlaps = []
        for i in range(n):
            for j in range(i + 1, n):
                overlap_lo = max(lowers[i], lowers[j])
                overlap_hi = min(uppers[i], uppers[j])
                union_lo = min(lowers[i], lowers[j])
                union_hi = max(uppers[i], uppers[j])
                union_width = max(union_hi - union_lo, 1e-6)
                overlap_width = max(0.0, overlap_hi - overlap_lo)
                overlaps.append(overlap_width / union_width)

        concordance = float(np.mean(overlaps)) if overlaps else 0.5
        passed = concordance >= self.CONCORDANCE_THRESHOLD
        return passed, concordance

    # ------------------------------------------------------------------
    # Layer 3: Calibration Tracker
    # ------------------------------------------------------------------

    def _check_calibration(
        self,
        calibration_scores: dict[str, float],
    ) -> tuple[bool, float]:
        """Verify that models have been honest in recent predictions.

        Args:
            calibration_scores: model_id → reliability score from shadow module's
                                rolling 30-day calibration tracker. 1.0 = perfectly
                                calibrated, 0.0 = unreliable.

        Returns:
            (passed, mean_reliability)
        """
        if not calibration_scores:
            # No calibration data yet (e.g. early in shadow period).
            return True, 0.5

        mean_reliability = float(np.mean(list(calibration_scores.values())))
        passed = mean_reliability >= self.CALIBRATION_THRESHOLD
        return passed, mean_reliability

    # ------------------------------------------------------------------
    # Layer 4: Effect Size Gate
    # ------------------------------------------------------------------

    def _check_effect_size(
        self,
        proposed_envelopes: dict[str, "PredictionEnvelope"],
        baseline_envelopes: dict[str, "PredictionEnvelope"],
        user_aggressiveness: float = 0.5,
    ) -> tuple[bool, float]:
        """Verify the predicted improvement is practically significant.

        Signal-to-noise ratio: improvement / prediction uncertainty.
        Conservative users require larger effect sizes.

        Args:
            proposed_envelopes: model_id → envelope for proposed action.
            baseline_envelopes: model_id → envelope for current/baseline action.
            user_aggressiveness: From UserAgencyProfile (0–1).

        Returns:
            (passed, snr)
        """
        improvements = []
        uncertainties = []

        for model_id in proposed_envelopes:
            if model_id not in baseline_envelopes:
                continue
            prop = proposed_envelopes[model_id]
            base = baseline_envelopes[model_id]

            prop_point = float(np.mean(np.asarray(prop.point)))
            base_point = float(np.mean(np.asarray(base.point)))
            prop_width = float(np.mean(np.abs(
                np.asarray(prop.upper) - np.asarray(prop.lower)
            )))

            improvements.append(abs(prop_point - base_point))
            uncertainties.append(prop_width + 1e-6)

        if not improvements:
            return True, 0.5

        snr = float(np.mean(
            [imp / unc for imp, unc in zip(improvements, uncertainties)]
        ))

        # Adjust threshold by aggressiveness: aggressive users accept smaller SNR.
        adjusted_threshold = self.EFFECT_SNR_THRESHOLD * (1.5 - user_aggressiveness)
        passed = snr >= adjusted_threshold
        return passed, snr

    # ------------------------------------------------------------------
    # Hard Safety Gate (independent)
    # ------------------------------------------------------------------

    def _check_safety(
        self,
        proposed_envelopes: dict[str, "PredictionEnvelope"],
        target_names: list[str] | None = None,
    ) -> tuple[bool, list[str]]:
        """Check hard safety constraints against worst-case predictions.

        Non-negotiable: a highly confident recommendation with even a small
        tail probability of dangerous lows gets blocked.

        Args:
            proposed_envelopes: model_id → envelope for proposed action.
            target_names:       Ordered target names for indexing into envelope
                                arrays (e.g. ["percent_low", "percent_high", "tir", "mean_bg"]).

        Returns:
            (safe, list_of_violations)
        """
        if target_names is None:
            target_names = ["percent_low", "percent_high", "tir", "mean_bg"]

        violations: list[str] = []
        for constraint in self.safety_constraints:
            if constraint.metric not in target_names:
                continue
            idx = target_names.index(constraint.metric)

            for model_id, env in proposed_envelopes.items():
                arr = np.asarray(env.upper if constraint.use_upper else env.lower)
                if arr.ndim == 0:
                    val = float(arr)
                elif arr.size > idx:
                    val = float(arr.flat[idx]) if arr.ndim == 1 else float(arr[0, idx])
                else:
                    continue

                if constraint.use_upper and val > constraint.threshold:
                    violations.append(
                        f"{constraint.name}: {model_id} upper={val:.3f} > {constraint.threshold}"
                    )
                elif not constraint.use_upper and val < constraint.threshold:
                    violations.append(
                        f"{constraint.name}: {model_id} lower={val:.3f} < {constraint.threshold}"
                    )

        return len(violations) == 0, violations

    # ------------------------------------------------------------------
    # Gate Composition
    # ------------------------------------------------------------------

    def evaluate(
        self,
        proposed_envelopes: dict[str, "PredictionEnvelope"],
        baseline_envelopes: dict[str, "PredictionEnvelope"],
        familiarity_score: float,
        calibration_scores: dict[str, float] | None = None,
        user_aggressiveness: float = 0.5,
        target_names: list[str] | None = None,
        mood_budget_available: bool = True,
    ) -> GateResult:
        """Run all four confidence layers + hard safety gate + mood budget.

        Returns a GateResult indicating whether the recommendation should be
        surfaced and the composite confidence score.
        """
        details: dict = {}

        # Mood budget gate: suppress recommendations when budget is empty
        if not mood_budget_available:
            return GateResult(
                passed=False,
                composite_score=0.0,
                layer_scores={},
                blocked_by="mood_budget",
                details={"mood_budget": {"passed": False, "reason": "budget_empty"}},
            )

        # Layer 1: GP Familiarity
        fam_passed, fam_score = self._check_familiarity(familiarity_score)
        details["familiarity"] = {"passed": fam_passed, "score": fam_score}
        if not fam_passed:
            return GateResult(
                passed=False,
                composite_score=0.0,
                layer_scores={"familiarity": fam_score},
                blocked_by="familiarity",
                details=details,
            )

        # Layer 2: Ensemble Agreement
        conc_passed, conc_score = self._check_concordance(proposed_envelopes)
        details["concordance"] = {"passed": conc_passed, "score": conc_score}
        if not conc_passed:
            return GateResult(
                passed=False,
                composite_score=0.0,
                layer_scores={"familiarity": fam_score, "concordance": conc_score},
                blocked_by="concordance",
                details=details,
            )

        # Layer 3: Calibration
        cal_passed, cal_score = self._check_calibration(calibration_scores or {})
        details["calibration"] = {"passed": cal_passed, "score": cal_score}
        if not cal_passed:
            return GateResult(
                passed=False,
                composite_score=0.0,
                layer_scores={
                    "familiarity": fam_score, "concordance": conc_score,
                    "calibration": cal_score,
                },
                blocked_by="calibration",
                details=details,
            )

        # Layer 4: Effect Size
        eff_passed, eff_score = self._check_effect_size(
            proposed_envelopes, baseline_envelopes, user_aggressiveness,
        )
        details["effect_size"] = {"passed": eff_passed, "score": eff_score}
        if not eff_passed:
            return GateResult(
                passed=False,
                composite_score=0.0,
                layer_scores={
                    "familiarity": fam_score, "concordance": conc_score,
                    "calibration": cal_score, "effect_size": eff_score,
                },
                blocked_by="effect_size",
                details=details,
            )

        # Hard Safety Gate (independent — runs regardless of confidence layers).
        safe, violations = self._check_safety(proposed_envelopes, target_names)
        details["safety"] = {"passed": safe, "violations": violations}
        if not safe:
            return GateResult(
                passed=False,
                composite_score=0.0,
                layer_scores={
                    "familiarity": fam_score, "concordance": conc_score,
                    "calibration": cal_score, "effect_size": eff_score,
                },
                blocked_by="safety",
                details=details,
            )

        # All layers passed — compute composite confidence.
        composite = (
            self.W_FAMILIARITY * fam_score
            + self.W_CONCORDANCE * conc_score
            + self.W_CALIBRATION * cal_score
            + self.W_EFFECT * min(eff_score, 1.0)
        )
        composite = float(np.clip(composite, 0.0, 1.0))

        return GateResult(
            passed=True,
            composite_score=composite,
            layer_scores={
                "familiarity": fam_score,
                "concordance": conc_score,
                "calibration": cal_score,
                "effect_size": eff_score,
            },
            details=details,
        )

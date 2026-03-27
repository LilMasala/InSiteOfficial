"""Block 6: Meta-Controller / Chameleon — structural decisions about the modelling stack.

Detects drift, routes decisions to the most trustworthy models, and triggers
adaptation when performance degrades.

Components:
    - Drift Detection: feature drift, outcome drift, action drift, regime change
    - Model Trust Routing: per-model trust weights from shadow accuracy
    - Adaptation Escalation Ladder: reweight → fine-tune → retrain → expand
    - Action Family Selection: therapy vs behaviour routing
    - Model Registry: structured catalog of every model trained
    - Pressure-Based Retraining: accumulates signals, triggers retrain at threshold
    - Candidate Validation: shadow-validates new models before promotion
    - Cross-Patient Intelligence: cohort detection and transfer confidence
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Drift Detection
# ---------------------------------------------------------------------------

class DriftType(Enum):
    """Types of distribution drift the meta-controller monitors."""
    FEATURE = "feature"    # Covariate shift — features look different
    OUTCOME = "outcome"    # Concept shift — feature→outcome mapping changed
    ACTION = "action"      # Behavioral shift — user response changed
    REGIME = "regime"      # Structural break — fundamentally new state


@dataclass
class DriftSignal:
    """Result of a drift detection check."""
    drift_type: DriftType
    detected: bool
    severity: float  # 0.0 = none, 1.0 = severe
    details: dict = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )


class DriftDetector:
    """Monitors four types of distribution drift.

    Feature Drift (Covariate Shift):
        Track rolling statistics (mean, var, quantiles) of each feature vs
        a reference window. PSI or KS tests per feature.

    Outcome Drift (Concept Shift):
        The feature→outcome relationship changed. Detected via CUSUM on
        prediction residuals from the shadow module.

    Action Drift (Behavioral Shift):
        User response to recommendations changed. Rolling acceptance rate,
        compliance gap, revert frequency.

    Regime Change (Structural Break):
        Step-function changes: pregnancy, new medication, timezone move.
        Detected via simultaneous multi-feature shifts.
    """

    # PSI threshold: >0.25 = significant drift (standard rule of thumb).
    PSI_THRESHOLD = 0.25
    # CUSUM threshold for outcome drift.
    CUSUM_THRESHOLD = 5.0
    # Acceptance rate change threshold.
    ACTION_DRIFT_THRESHOLD = 0.20
    # Number of features simultaneously drifting for regime change.
    REGIME_MIN_FEATURES = 5

    def __init__(self, reference_window: int = 30) -> None:
        self._reference_window = reference_window
        self._feature_reference: np.ndarray | None = None
        self._residual_cusum: float = 0.0
        self._acceptance_history: list[float] = []

    def set_reference(self, X_ref: np.ndarray) -> None:
        """Set the reference feature distribution for drift comparison."""
        self._feature_reference = np.asarray(X_ref, dtype=float)

    # ------------------------------------------------------------------
    # Feature Drift (PSI-based)
    # ------------------------------------------------------------------

    def check_feature_drift(self, X_current: np.ndarray) -> DriftSignal:
        """Check for covariate shift using Population Stability Index."""
        if self._feature_reference is None:
            return DriftSignal(DriftType.FEATURE, False, 0.0)

        X_cur = np.asarray(X_current, dtype=float)
        n_features = min(X_cur.shape[1], self._feature_reference.shape[1])

        psi_values = []
        for col in range(n_features):
            ref_col = self._feature_reference[:, col]
            cur_col = X_cur[:, col]
            psi = self._compute_psi(ref_col, cur_col)
            psi_values.append(psi)

        mean_psi = float(np.mean(psi_values))
        n_drifted = sum(1 for p in psi_values if p > self.PSI_THRESHOLD)
        severity = float(np.clip(mean_psi / self.PSI_THRESHOLD, 0.0, 1.0))

        return DriftSignal(
            drift_type=DriftType.FEATURE,
            detected=mean_psi > self.PSI_THRESHOLD,
            severity=severity,
            details={
                "mean_psi": mean_psi,
                "n_drifted_features": n_drifted,
                "psi_per_feature": psi_values[:10],  # First 10 for logging
            },
        )

    @staticmethod
    def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Compute Population Stability Index between two distributions."""
        ref_finite = reference[np.isfinite(reference)]
        cur_finite = current[np.isfinite(current)]
        if ref_finite.size < 10 or cur_finite.size < 10:
            return 0.0

        # Use reference quantiles as bin edges.
        edges = np.quantile(ref_finite, np.linspace(0, 1, n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        # Remove duplicates.
        edges = np.unique(edges)
        if len(edges) < 3:
            return 0.0

        ref_hist = np.histogram(ref_finite, bins=edges)[0].astype(float)
        cur_hist = np.histogram(cur_finite, bins=edges)[0].astype(float)

        # Normalise to proportions, with small epsilon to avoid log(0).
        eps = 1e-6
        ref_prop = ref_hist / ref_hist.sum() + eps
        cur_prop = cur_hist / cur_hist.sum() + eps

        psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
        return max(0.0, psi)

    # ------------------------------------------------------------------
    # Outcome Drift (CUSUM on residuals)
    # ------------------------------------------------------------------

    def check_outcome_drift(self, residuals: np.ndarray) -> DriftSignal:
        """Check for concept shift using CUSUM on prediction residuals.

        Args:
            residuals: Recent prediction errors (predicted - actual).
        """
        res = np.asarray(residuals, dtype=float)
        res = res[np.isfinite(res)]
        if res.size < 5:
            return DriftSignal(DriftType.OUTCOME, False, 0.0)

        # One-sided CUSUM (upward shifts in residual magnitude).
        target = 0.0
        slack = 0.5 * float(np.std(res))
        cusum = 0.0
        max_cusum = 0.0
        for r in res:
            cusum = max(0.0, cusum + abs(r) - target - slack)
            max_cusum = max(max_cusum, cusum)

        self._residual_cusum = max_cusum
        severity = float(np.clip(max_cusum / self.CUSUM_THRESHOLD, 0.0, 1.0))

        return DriftSignal(
            drift_type=DriftType.OUTCOME,
            detected=max_cusum > self.CUSUM_THRESHOLD,
            severity=severity,
            details={"cusum": max_cusum, "mean_abs_residual": float(np.mean(np.abs(res)))},
        )

    # ------------------------------------------------------------------
    # Action Drift (behavioral shift)
    # ------------------------------------------------------------------

    def check_action_drift(
        self,
        recent_acceptance_rate: float,
        historical_acceptance_rate: float,
    ) -> DriftSignal:
        """Check if user response to recommendations has changed.

        Significant shifts signal a need to update the user model or
        increase conservativeness.
        """
        delta = abs(recent_acceptance_rate - historical_acceptance_rate)
        severity = float(np.clip(delta / self.ACTION_DRIFT_THRESHOLD, 0.0, 1.0))

        return DriftSignal(
            drift_type=DriftType.ACTION,
            detected=delta > self.ACTION_DRIFT_THRESHOLD,
            severity=severity,
            details={
                "recent_rate": recent_acceptance_rate,
                "historical_rate": historical_acceptance_rate,
                "delta": delta,
            },
        )

    # ------------------------------------------------------------------
    # Regime Change (structural break)
    # ------------------------------------------------------------------

    def check_regime_change(self, X_current: np.ndarray) -> DriftSignal:
        """Detect step-function changes that invalidate prior learning.

        Response: potentially reset shadow graduation clock and enter
        rapid-relearning phase.
        """
        if self._feature_reference is None:
            return DriftSignal(DriftType.REGIME, False, 0.0)

        X_cur = np.asarray(X_current, dtype=float)
        n_features = min(X_cur.shape[1], self._feature_reference.shape[1])

        # Count features with PSI > threshold.
        n_drifted = 0
        for col in range(n_features):
            psi = self._compute_psi(self._feature_reference[:, col], X_cur[:, col])
            if psi > self.PSI_THRESHOLD:
                n_drifted += 1

        detected = n_drifted >= self.REGIME_MIN_FEATURES
        severity = float(np.clip(n_drifted / max(n_features, 1), 0.0, 1.0))

        return DriftSignal(
            drift_type=DriftType.REGIME,
            detected=detected,
            severity=severity,
            details={"n_drifted_features": n_drifted, "total_features": n_features},
        )


# ---------------------------------------------------------------------------
# Adaptation Escalation Ladder
# ---------------------------------------------------------------------------

class EscalationLevel(IntEnum):
    """Ordered escalation steps, least to most disruptive."""
    NONE = 0
    REWEIGHT = 1        # Adjust ensemble weights
    FINE_TUNE = 2       # Fine-tune existing models on recent data
    RETRAIN = 3         # Full retrain from scratch
    EXPAND = 4          # Add new model to the zoo


@dataclass
class EscalationAction:
    """Action to take at a given escalation level."""
    level: EscalationLevel
    triggered_by: str  # What drift signal triggered this
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Model Registry Entry
# ---------------------------------------------------------------------------

@dataclass
class ModelRegistryEntry:
    """Structured metadata for a trained model in the zoo."""
    model_id: str
    version: str
    architecture: str  # "xgboost", "transformer", "feedforward", "gp"
    target: str
    training_date: str = ""
    data_window: str = ""  # "days 0-60" or "2025-01-01 to 2025-03-01"
    hyperparameters: dict = field(default_factory=dict)
    validation_metrics: dict = field(default_factory=dict)
    trust_weight: float = 1.0
    status: str = "active"  # active, standby, deprecated, retraining
    drift_sensitivity: float = 0.5
    regime_tags: list[str] = field(default_factory=list)

    def to_row(self) -> tuple:
        """Return a tuple for the model_registry table."""
        return (
            self.model_id, self.version, self.architecture, self.target,
            self.training_date, self.data_window,
            json.dumps(self.hyperparameters),
            json.dumps(self.validation_metrics),
            self.trust_weight, self.status, self.drift_sensitivity,
            json.dumps(self.regime_tags),
        )


# ---------------------------------------------------------------------------
# Meta-Controller
# ---------------------------------------------------------------------------

class MetaController:
    """Sits above the zoo and makes structural decisions about the modelling stack.

    Responsibilities:
        1. Drift detection (feature, outcome, action, regime)
        2. Model trust routing (per-model weights from shadow accuracy)
        3. Adaptation escalation (reweight → fine-tune → retrain → expand)
        4. Action family selection (therapy vs behaviour)
        5. Model registry management
    """

    # Escalation: days to wait before escalating to next level.
    ESCALATION_PATIENCE = 7

    def __init__(self, learning_mode: str = "hybrid") -> None:
        self._drift_detector = DriftDetector()
        self._registry: dict[str, ModelRegistryEntry] = {}
        self._trust_weights: dict[str, float] = {}
        self._current_escalation = EscalationLevel.NONE
        self._days_at_current_level = 0
        self._state = MetaControllerState(global_learning_mode=learning_mode)

    @property
    def drift_detector(self) -> DriftDetector:
        return self._drift_detector

    @property
    def registry(self) -> dict[str, ModelRegistryEntry]:
        return dict(self._registry)

    @property
    def trust_weights(self) -> dict[str, float]:
        return dict(self._trust_weights)

    # ------------------------------------------------------------------
    # Model Registry
    # ------------------------------------------------------------------

    def register_model(self, entry: ModelRegistryEntry) -> None:
        """Add or update a model in the registry."""
        self._registry[entry.model_id] = entry
        if entry.model_id not in self._trust_weights:
            self._trust_weights[entry.model_id] = entry.trust_weight

    def deactivate_model(self, model_id: str) -> None:
        """Set a model's status to deprecated."""
        if model_id in self._registry:
            self._registry[model_id].status = "deprecated"
            self._trust_weights[model_id] = 0.0

    def get_active_models(self) -> list[ModelRegistryEntry]:
        """Return all models with status 'active'."""
        return [e for e in self._registry.values() if e.status == "active"]

    # ------------------------------------------------------------------
    # Trust Routing
    # ------------------------------------------------------------------

    def update_trust_weights(
        self,
        calibration_scores: dict[str, float],
        shadow_accuracies: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Update per-model trust weights from shadow module feedback.

        Trust = weighted combination of:
            - Shadow accuracy (prediction quality)
            - Calibration quality (uncertainty honesty)
            - Regime compatibility (is model trained on relevant data?)
            - Staleness (how recent is training data?)

        Args:
            calibration_scores: model_id → reliability from shadow module.
            shadow_accuracies:  model_id → recent accuracy metric.

        Returns:
            Updated trust weights.
        """
        for model_id, entry in self._registry.items():
            if entry.status != "active":
                self._trust_weights[model_id] = 0.0
                continue

            cal = calibration_scores.get(model_id, 0.5)
            acc = (shadow_accuracies or {}).get(model_id, 0.5)

            # Weighted combination.
            trust = 0.4 * cal + 0.4 * acc + 0.2 * entry.trust_weight
            self._trust_weights[model_id] = float(np.clip(trust, 0.0, 1.0))

        return dict(self._trust_weights)

    def get_top_k_models(self, k: int = 3) -> list[str]:
        """Return model IDs of the top-K most trusted active models."""
        active = {
            mid: w for mid, w in self._trust_weights.items()
            if self._registry.get(mid, ModelRegistryEntry("", "", "", "")).status == "active"
        }
        sorted_models = sorted(active.items(), key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in sorted_models[:k]]

    # ------------------------------------------------------------------
    # Drift Check (daily cycle)
    # ------------------------------------------------------------------

    def run_drift_check(
        self,
        X_current: np.ndarray | None = None,
        residuals: np.ndarray | None = None,
        recent_acceptance_rate: float | None = None,
        historical_acceptance_rate: float | None = None,
    ) -> list[DriftSignal]:
        """Run all drift detectors and return signals.

        Called once per daily cycle (Section 10, step 2).
        """
        signals: list[DriftSignal] = []

        if X_current is not None:
            signals.append(self._drift_detector.check_feature_drift(X_current))
            signals.append(self._drift_detector.check_regime_change(X_current))

        if residuals is not None:
            signals.append(self._drift_detector.check_outcome_drift(residuals))

        if recent_acceptance_rate is not None and historical_acceptance_rate is not None:
            signals.append(self._drift_detector.check_action_drift(
                recent_acceptance_rate, historical_acceptance_rate,
            ))

        return signals

    # ------------------------------------------------------------------
    # Adaptation Escalation
    # ------------------------------------------------------------------

    def escalate(self, drift_signals: list[DriftSignal]) -> EscalationAction:
        """Determine the appropriate adaptation response to drift signals.

        Follows an escalation ladder, trying the least disruptive intervention
        first. Each step requires a trigger condition before escalating.

        Escalation ladder:
            1. Reweight: Adjust ensemble trust weights
            2. Fine-tune: Fine-tune existing models on recent data
            3. Retrain: Full retrain from scratch
            4. Expand: Add new model to the zoo
        """
        active_signals = [s for s in drift_signals if s.detected]
        if not active_signals:
            self._days_at_current_level = 0
            self._current_escalation = EscalationLevel.NONE
            return EscalationAction(EscalationLevel.NONE, "no_drift")

        max_severity = max(s.severity for s in active_signals)
        trigger = active_signals[0].drift_type.value

        # Check if any signal is a regime change.
        regime_signals = [s for s in active_signals if s.drift_type == DriftType.REGIME]
        if regime_signals:
            return EscalationAction(
                EscalationLevel.RETRAIN, "regime_change",
                {"regime_signal": regime_signals[0].details},
            )

        # Normal escalation ladder.
        self._days_at_current_level += 1

        if self._current_escalation < EscalationLevel.REWEIGHT:
            self._current_escalation = EscalationLevel.REWEIGHT
            self._days_at_current_level = 0
            return EscalationAction(EscalationLevel.REWEIGHT, trigger)

        if (
            self._current_escalation == EscalationLevel.REWEIGHT
            and self._days_at_current_level >= self.ESCALATION_PATIENCE
        ):
            self._current_escalation = EscalationLevel.FINE_TUNE
            self._days_at_current_level = 0
            return EscalationAction(EscalationLevel.FINE_TUNE, trigger)

        if (
            self._current_escalation == EscalationLevel.FINE_TUNE
            and self._days_at_current_level >= self.ESCALATION_PATIENCE
        ):
            self._current_escalation = EscalationLevel.RETRAIN
            self._days_at_current_level = 0
            return EscalationAction(EscalationLevel.RETRAIN, trigger)

        if (
            self._current_escalation == EscalationLevel.RETRAIN
            and self._days_at_current_level >= self.ESCALATION_PATIENCE
            and max_severity > 0.8
        ):
            self._current_escalation = EscalationLevel.EXPAND
            self._days_at_current_level = 0
            return EscalationAction(EscalationLevel.EXPAND, trigger)

        return EscalationAction(self._current_escalation, trigger)

    # ------------------------------------------------------------------
    # Action Family Selection
    # ------------------------------------------------------------------

    def select_action_family(
        self,
        feature_importances: dict[str, float] | None = None,
    ) -> str:
        """Decide whether current situation calls for therapy or behaviour.

        Based on outcome attribution: if the dominant BG signal correlates
        with miscalibrated settings, route to therapy. If it correlates with
        behavioral patterns, route to behaviour.

        Initially rule-based; can be learned over time.

        Returns:
            "therapy" or "behavior"
        """
        if feature_importances is None:
            return "therapy"  # Default to therapy path.

        therapy_features = {
            "isf_multiplier", "cr_multiplier", "basal_multiplier",
            "bg_avg", "bg_tir", "bg_percent_low", "bg_percent_high",
        }
        behavior_features = {
            "exercise_minutes", "ex_min_last3h", "sleep_prev_total_min",
            "kcal_active", "mood_valence", "mood_arousal",
        }

        therapy_importance = sum(
            v for k, v in feature_importances.items() if k in therapy_features
        )
        behavior_importance = sum(
            v for k, v in feature_importances.items() if k in behavior_features
        )

        return "therapy" if therapy_importance >= behavior_importance else "behavior"

    # ------------------------------------------------------------------
    # Pressure-Based Retraining (Section 4.1)
    # ------------------------------------------------------------------

    def accumulate_pressure(
        self,
        new_data_rows: int = 0,
        intervention_data_rows: int = 0,
        rolling_win_rate: float = 0.5,
        prev_win_rate: float = 0.5,
        drift_signals: list[DriftSignal] | None = None,
        intervention_triple_count: int = 0,
        causal_delta: float = 0.0,
    ) -> float:
        """Accumulate retraining pressure from multiple signals.

        Called once per global day. Returns updated pressure score.
        """
        # Data staleness pressure
        data_pressure = 0.01 * new_data_rows + 0.03 * intervention_data_rows
        self._state.pressure_score += data_pressure

        # Performance degradation pressure
        win_rate_drop = max(0.0, prev_win_rate - rolling_win_rate)
        if win_rate_drop > 0.02:
            self._state.pressure_score += 2.0 * win_rate_drop
        if win_rate_drop > 0.10:
            self._state.pressure_score += 5.0 * win_rate_drop

        # Drift alarm pressure
        if drift_signals:
            for sig in drift_signals:
                if sig.detected:
                    self._state.drift_alarm_count += 1
                    if sig.drift_type == DriftType.REGIME:
                        self._state.pressure_score += 3.0 * sig.severity
                    else:
                        self._state.pressure_score += 1.5 * sig.severity

        # Intervention data milestone pressure
        milestones = [100, 500, 1000, 5000]
        prev_count = self._state.intervention_triple_count
        self._state.intervention_triple_count = intervention_triple_count
        for m in milestones:
            if prev_count < m <= intervention_triple_count:
                self._state.pressure_score += 2.0

        # Negative causal delta pressure (strongest signal)
        if causal_delta < -0.005:
            self._state.pressure_score += 4.0 * abs(causal_delta)

        # Decay during sustained good performance
        if rolling_win_rate > 0.65 and causal_delta > 0:
            self._state.pressure_score *= 0.95

        self._state.rolling_population_win_rate = rolling_win_rate
        self._state.rolling_causal_delta = causal_delta
        return self._state.pressure_score

    def check_retrain_trigger(self) -> bool:
        """Check if accumulated pressure exceeds the retraining threshold.

        Threshold is low early (retrain eagerly) and rises later.
        """
        days_since = self._state.current_day - self._state.last_retrain_day
        # Early: threshold = 3.0, rising to 8.0 over 90 days
        threshold = 3.0 + 5.0 * min(1.0, days_since / 90.0)
        return self._state.pressure_score >= threshold

    def reset_pressure(self, day: int) -> None:
        """Reset pressure after a retrain."""
        self._state.pressure_score = 0.0
        self._state.last_retrain_day = day
        self._state.drift_alarm_count = 0

    # ------------------------------------------------------------------
    # Candidate Validation (Section 4.4)
    # ------------------------------------------------------------------

    def register_candidate(self, model_id: str, day: int) -> None:
        """Register a newly trained model as a candidate for validation."""
        self._state.candidate_model_id = model_id
        self._state.candidate_validation_start = day
        if model_id in self._registry:
            self._registry[model_id].status = "candidate"

    def check_candidate_validation(
        self,
        day: int,
        candidate_win_rate: float,
        active_win_rate: float,
        validation_days: int = 7,
    ) -> str:
        """Check if a candidate model should be promoted, kept, or rejected.

        Returns: "promote", "continue", or "reject".
        """
        if self._state.candidate_model_id is None:
            return "continue"

        days_validating = day - (self._state.candidate_validation_start or day)
        if days_validating < validation_days:
            return "continue"

        # Promote if candidate matches or outperforms active
        if candidate_win_rate >= active_win_rate - 0.02:
            return "promote"
        else:
            return "reject"

    def promote_candidate(self) -> str | None:
        """Promote the current candidate to active, demote previous active to standby."""
        cid = self._state.candidate_model_id
        if cid is None:
            return None

        # Demote current active models to standby
        for mid, entry in self._registry.items():
            if entry.status == "active" and mid != cid:
                entry.status = "standby"

        # Promote candidate
        if cid in self._registry:
            self._registry[cid].status = "active"

        self._state.candidate_model_id = None
        self._state.candidate_validation_start = None
        return cid

    def reject_candidate(self) -> str | None:
        """Reject the current candidate model."""
        cid = self._state.candidate_model_id
        if cid and cid in self._registry:
            self._registry[cid].status = "deprecated"
        self._state.candidate_model_id = None
        self._state.candidate_validation_start = None
        return cid

    # ------------------------------------------------------------------
    # Cross-Patient Intelligence (Section 4.5)
    # ------------------------------------------------------------------

    def update_cohort_assignments(
        self,
        patient_features: dict[str, np.ndarray],
        n_cohorts: int = 4,
    ) -> dict[str, int]:
        """Cluster patients by metabolic behavior for transfer learning.

        Simple k-means on metabolic feature vectors. Returns patient_id → cohort_id.
        """
        if not patient_features:
            return {}

        pids = list(patient_features.keys())
        X = np.array([patient_features[pid] for pid in pids])

        if X.shape[0] < n_cohorts:
            # Not enough patients to cluster
            return {pid: 0 for pid in pids}

        # Simple k-means clustering
        from scipy.cluster.vq import kmeans2
        try:
            _, labels = kmeans2(X.astype(float), n_cohorts, minit="points")
            assignments = {pid: int(label) for pid, label in zip(pids, labels)}
        except Exception:
            assignments = {pid: 0 for pid in pids}

        self._state.patient_cohort_assignments = assignments
        return assignments

    # ------------------------------------------------------------------
    # Escalation Ladder with Patience (Section 4.3)
    # ------------------------------------------------------------------

    def escalate_with_patience(self) -> EscalationAction:
        """Determine escalation action respecting patience timers.

        Steps:
            1. Reweight (wait 7 days)
            2. Fine-tune (wait 7 days)
            3. Full retrain (wait 14 days)
            4. Architecture expand (wait 21 days)
            5. Human escalation (N/A)
        """
        patience_map = {
            EscalationLevel.NONE: 0,
            EscalationLevel.REWEIGHT: 7,
            EscalationLevel.FINE_TUNE: 7,
            EscalationLevel.RETRAIN: 14,
            EscalationLevel.EXPAND: 21,
        }

        patience = patience_map.get(self._state.escalation_level, 7)

        if self._state.escalation_patience_remaining > 0:
            self._state.escalation_patience_remaining -= 1
            return EscalationAction(
                EscalationLevel(self._state.escalation_level),
                "waiting",
            )

        # Escalate to next level
        if self._state.escalation_level < EscalationLevel.EXPAND:
            self._state.escalation_level = min(
                self._state.escalation_level + 1,
                int(EscalationLevel.EXPAND),
            )
            next_patience = patience_map.get(
                EscalationLevel(self._state.escalation_level), 7,
            )
            self._state.escalation_patience_remaining = next_patience

        return EscalationAction(
            EscalationLevel(self._state.escalation_level),
            "escalated",
        )

    def reset_escalation(self) -> None:
        """Reset escalation ladder after successful intervention."""
        self._state.escalation_level = 0
        self._state.escalation_patience_remaining = 0

    # ------------------------------------------------------------------
    # Full Daily + Weekly Evaluation (Sections 8.2, 8.3)
    # ------------------------------------------------------------------

    def daily_check(
        self,
        day: int,
        new_data_rows: int = 0,
        intervention_data_rows: int = 0,
        rolling_win_rate: float = 0.5,
        prev_win_rate: float = 0.5,
        drift_signals: list[DriftSignal] | None = None,
        intervention_triple_count: int = 0,
        causal_delta: float = 0.0,
    ) -> dict[str, Any]:
        """Lightweight daily meta-controller check.

        Returns dict with pressure, retrain_triggered, candidate_status.
        """
        self._state.current_day = day

        pressure = self.accumulate_pressure(
            new_data_rows=new_data_rows,
            intervention_data_rows=intervention_data_rows,
            rolling_win_rate=rolling_win_rate,
            prev_win_rate=prev_win_rate,
            drift_signals=drift_signals,
            intervention_triple_count=intervention_triple_count,
            causal_delta=causal_delta,
        )

        retrain = self.check_retrain_trigger()

        # Check candidate validation
        candidate_status = "none"
        if self._state.candidate_model_id is not None:
            candidate_status = "validating"

        return {
            "pressure": pressure,
            "retrain_triggered": retrain,
            "candidate_status": candidate_status,
            "escalation_level": self._state.escalation_level,
            "day": day,
        }

    def weekly_evaluation(
        self,
        day: int,
        population_win_rate: float,
        causal_delta: float,
        per_patient_mood: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Full weekly meta-controller evaluation.

        Returns dict with evaluation results.
        """
        self._state.rolling_population_win_rate = population_win_rate
        self._state.rolling_causal_delta = causal_delta

        return {
            "day": day,
            "population_win_rate": population_win_rate,
            "causal_delta": causal_delta,
            "escalation_level": self._state.escalation_level,
            "candidate_model": self._state.candidate_model_id,
            "pressure_score": self._state.pressure_score,
        }


# ---------------------------------------------------------------------------
# Meta-Controller State
# ---------------------------------------------------------------------------

@dataclass
class MetaControllerState:
    """Full state of the meta-controller for serialization and restoration."""
    pressure_score: float = 0.0
    last_retrain_day: int = 0
    current_day: int = 0
    escalation_level: int = 0
    escalation_patience_remaining: int = 0
    candidate_model_id: str | None = None
    candidate_validation_start: int | None = None
    rolling_population_win_rate: float = 0.5
    rolling_causal_delta: float = 0.0
    drift_alarm_count: int = 0
    intervention_triple_count: int = 0
    patient_cohort_assignments: dict[str, int] = field(default_factory=dict)
    global_learning_mode: str = "hybrid"
    per_patient_recommendation_budget: dict[str, float] = field(default_factory=dict)

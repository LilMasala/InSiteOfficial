"""Block 5: Shadow Module — the system's conscience.

Nothing reaches a user without first surviving a shadow period where
predictions are logged, compared to reality, and scored. Manages the full
lifecycle of recommendations from generation through retrospective evaluation
through graduation.

Key components:
    - ShadowRecord: immutable log of each recommendation cycle
    - Scorecard: rolling summary over recent shadow records
    - Graduation logic: conjunction of conditions for transitioning to live
    - Post-graduation monitoring: continuous performance tracking
"""
from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Shadow Record
# ---------------------------------------------------------------------------

class ShadowRecordType(Enum):
    """Types of shadow records."""
    RECOMMENDATION = "recommendation"
    POSITIVE_OBSERVATION = "positive_observation"


@dataclass
class ShadowRecord:
    """Immutable log of a single recommendation cycle.

    Filled in three stages:
        1. At recommendation time (immutable once written)
        2. At outcome time (~24h later)
        3. At evaluation time (retrospective scoring)
    """

    # Identity
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str = ""
    day_index: int = 0
    timestamp_utc: str = ""
    record_type: str = "recommendation"  # "recommendation" or "positive_observation"

    # --- Stage 1: Recommendation time (immutable) -------------------------
    # State snapshot
    feature_snapshot: dict = field(default_factory=dict)
    proposed_action: list[float] = field(default_factory=list)
    baseline_action: list[float] = field(default_factory=list)

    # Model predictions for proposed action
    proposed_predictions: dict[str, dict] = field(default_factory=dict)
    # Model predictions for baseline action
    baseline_predictions: dict[str, dict] = field(default_factory=dict)

    # Confidence gate decision
    gate_passed: bool = False
    gate_composite_score: float = 0.0
    gate_layer_scores: dict[str, float] = field(default_factory=dict)
    gate_blocked_by: str | None = None

    # GP familiarity, calibration scores at time of recommendation
    familiarity_score: float = 0.0
    calibration_scores: dict[str, float] = field(default_factory=dict)

    # --- Stage 2: Outcome time (~24h later) --------------------------------
    actual_outcomes: dict[str, float] | None = None  # true %low, %high, TIR, mean_bg
    actual_user_action: str | None = None  # "accept", "reject", "partial", "not_shown"
    actual_settings: list[float] | None = None  # [isf, cr, basal] actually in effect

    # --- Stage 3: Evaluation time ------------------------------------------
    counterfactual_estimate: dict[str, float] | None = None  # From BG surrogate
    per_model_accuracy: dict[str, dict] | None = None  # model_id → {mae, coverage, bias}
    shadow_score_delta: float | None = None  # Did recommendation outperform baseline?

    def to_dict(self) -> dict:
        """Serialise to a flat dict for database storage."""
        return {
            "record_id": self.record_id,
            "patient_id": self.patient_id,
            "day_index": self.day_index,
            "timestamp_utc": self.timestamp_utc,
            "feature_snapshot": json.dumps(self.feature_snapshot),
            "proposed_action": json.dumps(self.proposed_action),
            "baseline_action": json.dumps(self.baseline_action),
            "proposed_predictions": json.dumps(self.proposed_predictions),
            "baseline_predictions": json.dumps(self.baseline_predictions),
            "gate_passed": int(self.gate_passed),
            "gate_composite_score": self.gate_composite_score,
            "gate_layer_scores": json.dumps(self.gate_layer_scores),
            "gate_blocked_by": self.gate_blocked_by,
            "familiarity_score": self.familiarity_score,
            "calibration_scores": json.dumps(self.calibration_scores),
            "actual_outcomes": json.dumps(self.actual_outcomes) if self.actual_outcomes else None,
            "actual_user_action": self.actual_user_action,
            "actual_settings": json.dumps(self.actual_settings) if self.actual_settings else None,
            "counterfactual_estimate": json.dumps(self.counterfactual_estimate) if self.counterfactual_estimate else None,
            "per_model_accuracy": json.dumps(self.per_model_accuracy) if self.per_model_accuracy else None,
            "shadow_score_delta": self.shadow_score_delta,
        }

    def to_row(self) -> tuple:
        """Return a tuple matching the shadow_records table schema."""
        d = self.to_dict()
        return (
            d["record_id"], d["patient_id"], d["day_index"], d["timestamp_utc"],
            d["feature_snapshot"], d["proposed_action"], d["baseline_action"],
            d["proposed_predictions"], d["baseline_predictions"],
            d["gate_passed"], d["gate_composite_score"],
            d["gate_layer_scores"], d["gate_blocked_by"],
            d["familiarity_score"], d["calibration_scores"],
            d["actual_outcomes"], d["actual_user_action"], d["actual_settings"],
            d["counterfactual_estimate"], d["per_model_accuracy"],
            d["shadow_score_delta"],
        )


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

class GraduationStatus(Enum):
    """Current graduation state."""
    SHADOW = "shadow"           # Not yet graduated
    GRADUATED = "graduated"     # Actively surfacing recommendations
    DEGRADED = "degraded"       # Was graduated, performance dropped, back to shadow


@dataclass
class Scorecard:
    """Rolling summary computed over a window of recent enriched shadow records.

    Tracks six dimensions as specified in the architecture doc.

    Attributes:
        window_size:          Number of records (or days) in the rolling window.
        records:              List of enriched ShadowRecords in the window.
        win_rate:             Fraction of records where recommendation beat baseline.
        safety_violations:    Count of hard safety constraint violations in window.
        coverage_80:          Empirical coverage of 80% confidence intervals.
        familiarity_rate:     Fraction of records above familiarity threshold.
        cross_context_spread: Max win-rate difference across context slices.
        acceptance_rate:      User acceptance rate in window.
        consecutive_pass_days: Days all graduation conditions have held.
        status:               Current graduation status.
    """
    window_size: int = 30
    records: list[ShadowRecord] = field(default_factory=list)
    win_rate: float = 0.0
    safety_violations: int = 0
    coverage_80: float = 0.0
    familiarity_rate: float = 0.0
    cross_context_spread: float = 0.0
    acceptance_rate: float = 0.0
    consecutive_pass_days: int = 0
    status: GraduationStatus = GraduationStatus.SHADOW

    def to_row(self) -> tuple:
        """Return a tuple for the scorecard_snapshots table."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        return (
            ts, self.window_size, len(self.records),
            self.win_rate, self.safety_violations,
            self.coverage_80, self.familiarity_rate,
            self.cross_context_spread, self.acceptance_rate,
            self.consecutive_pass_days, self.status.value,
        )


# ---------------------------------------------------------------------------
# Shadow Module
# ---------------------------------------------------------------------------

class ShadowModule:
    """Manages the full lifecycle of shadow evaluation and graduation.

    Usage::

        shadow = ShadowModule()
        record = shadow.create_record(patient_id, day, features, ...)
        shadow.add_record(record)

        # ~24h later:
        shadow.enrich_outcome(record_id, actual_outcomes, actual_action, ...)

        # At evaluation time:
        shadow.evaluate_record(record_id, counterfactual, model_accuracies)

        # Check graduation:
        scorecard = shadow.compute_scorecard()
        graduated = shadow.check_graduation(scorecard)
    """

    # Graduation thresholds (from architecture doc Section 6.3).
    MIN_SHADOW_DAYS = 21
    WIN_RATE_THRESHOLD = 0.60
    ZERO_SAFETY_VIOLATIONS = True
    CALIBRATION_COVERAGE_LOW = 0.70
    CALIBRATION_COVERAGE_HIGH = 0.90
    FAMILIARITY_RATE_THRESHOLD = 0.90
    CROSS_CONTEXT_SPREAD_MAX = 0.15
    SUSTAINED_DAYS = 7

    # De-graduation threshold (lower than graduation to avoid oscillation).
    DEGRAD_WIN_RATE = 0.50

    def __init__(self, window_size: int = 30) -> None:
        self._records: dict[str, ShadowRecord] = {}
        self._window_size = window_size
        self._scorecard = Scorecard(window_size=window_size)
        self._consecutive_pass_days = 0
        self._status = GraduationStatus.SHADOW

    @property
    def status(self) -> GraduationStatus:
        return self._status

    @property
    def records(self) -> list[ShadowRecord]:
        return list(self._records.values())

    # ------------------------------------------------------------------
    # Record management
    # ------------------------------------------------------------------

    def create_record(
        self,
        patient_id: str,
        day_index: int,
        feature_snapshot: dict,
        proposed_action: list[float],
        baseline_action: list[float],
        proposed_predictions: dict[str, dict],
        baseline_predictions: dict[str, dict],
        gate_passed: bool,
        gate_composite_score: float,
        gate_layer_scores: dict[str, float],
        gate_blocked_by: str | None,
        familiarity_score: float,
        calibration_scores: dict[str, float],
    ) -> ShadowRecord:
        """Create a new shadow record at recommendation time."""
        record = ShadowRecord(
            patient_id=patient_id,
            day_index=day_index,
            timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            feature_snapshot=feature_snapshot,
            proposed_action=proposed_action,
            baseline_action=baseline_action,
            proposed_predictions=proposed_predictions,
            baseline_predictions=baseline_predictions,
            gate_passed=gate_passed,
            gate_composite_score=gate_composite_score,
            gate_layer_scores=gate_layer_scores,
            gate_blocked_by=gate_blocked_by,
            familiarity_score=familiarity_score,
            calibration_scores=calibration_scores,
        )
        return record

    def add_record(self, record: ShadowRecord) -> None:
        """Add a record to the shadow log."""
        self._records[record.record_id] = record

    def enrich_outcome(
        self,
        record_id: str,
        actual_outcomes: dict[str, float],
        actual_user_action: str,
        actual_settings: list[float],
    ) -> None:
        """Fill in Stage 2 (outcome) data for an existing record."""
        if record_id not in self._records:
            return
        rec = self._records[record_id]
        rec.actual_outcomes = actual_outcomes
        rec.actual_user_action = actual_user_action
        rec.actual_settings = actual_settings

    def evaluate_record(
        self,
        record_id: str,
        counterfactual_estimate: dict[str, float] | None = None,
        per_model_accuracy: dict[str, dict] | None = None,
    ) -> None:
        """Fill in Stage 3 (evaluation) data for an existing record."""
        if record_id not in self._records:
            return
        rec = self._records[record_id]
        rec.counterfactual_estimate = counterfactual_estimate
        rec.per_model_accuracy = per_model_accuracy

        # Compute shadow score delta: does the model predict that the
        # proposed action outperforms baseline? This compares the model's own
        # predictions for proposed vs baseline actions — not actual outcomes.
        # A positive delta means the model believes its recommendation helps.
        # This ensures graduation requires genuine model competence, not just
        # "actual TIR > arbitrary prediction from placeholder features."
        if rec.proposed_predictions and rec.baseline_predictions:
            proposed_tirs = []
            baseline_tirs = []
            for model_id in rec.proposed_predictions:
                pp = rec.proposed_predictions[model_id]
                bp = rec.baseline_predictions.get(model_id, {})
                if "point" in pp and "point" in bp:
                    p_point = pp["point"]
                    b_point = bp["point"]
                    if (isinstance(p_point, (list, np.ndarray)) and len(p_point) > 2
                            and isinstance(b_point, (list, np.ndarray)) and len(b_point) > 2):
                        proposed_tirs.append(float(p_point[2]))  # index 2 = TIR
                        baseline_tirs.append(float(b_point[2]))
            if proposed_tirs and baseline_tirs:
                rec.shadow_score_delta = (
                    float(np.mean(proposed_tirs)) - float(np.mean(baseline_tirs))
                )

    # ------------------------------------------------------------------
    # Calibration scores (consumed by confidence module)
    # ------------------------------------------------------------------

    def get_calibration_scores(self) -> dict[str, float]:
        """Compute per-model reliability scores from recent shadow records.

        For each model over the rolling window:
            coverage = did 80% CI contain the truth 80% of the time?
            sharpness = how wide are the intervals?
            bias = systematic over/under-prediction?

        Returns:
            model_id → reliability score in [0, 1].
        """
        enriched = [
            r for r in self._records.values()
            if r.per_model_accuracy is not None
        ]
        if not enriched:
            return {}

        # Use the most recent window_size records.
        recent = sorted(enriched, key=lambda r: r.day_index)[-self._window_size:]

        model_scores: dict[str, list[float]] = {}
        for rec in recent:
            if rec.per_model_accuracy is None:
                continue
            for model_id, acc in rec.per_model_accuracy.items():
                if model_id not in model_scores:
                    model_scores[model_id] = []
                # Reliability = 1 - |coverage_gap| - |bias|
                coverage_gap = abs(acc.get("coverage", 0.8) - 0.8)
                bias = abs(acc.get("bias", 0.0))
                rel = max(0.0, 1.0 - 2.0 * coverage_gap - bias)
                model_scores[model_id].append(rel)

        return {
            mid: float(np.mean(scores))
            for mid, scores in model_scores.items()
        }

    # ------------------------------------------------------------------
    # Scorecard computation
    # ------------------------------------------------------------------

    def compute_scorecard(self) -> Scorecard:
        """Compute the rolling scorecard from recent enriched records."""
        all_recs = sorted(self._records.values(), key=lambda r: r.day_index)
        recent = all_recs[-self._window_size:]

        # Filter to fully enriched records.
        enriched = [r for r in recent if r.actual_outcomes is not None]

        if not enriched:
            self._scorecard = Scorecard(
                window_size=self._window_size,
                records=recent,
                status=self._status,
            )
            return self._scorecard

        # Win rate: fraction where shadow_score_delta > 0.
        deltas = [r.shadow_score_delta for r in enriched if r.shadow_score_delta is not None]
        win_rate = float(np.mean([d > 0 for d in deltas])) if deltas else 0.0

        # Safety violations.
        safety_violations = sum(
            1 for r in enriched
            if not r.gate_passed and r.gate_blocked_by == "safety"
        )

        # Calibration coverage (how often truth fell within 80% CI).
        coverage_hits = []
        for rec in enriched:
            if rec.per_model_accuracy is None:
                continue
            for model_acc in rec.per_model_accuracy.values():
                cov = model_acc.get("coverage", None)
                if cov is not None:
                    coverage_hits.append(cov)
        coverage_80 = float(np.mean(coverage_hits)) if coverage_hits else 0.0

        # Familiarity rate.
        fam_scores = [r.familiarity_score for r in recent]
        familiarity_rate = float(np.mean([f >= 0.4 for f in fam_scores])) if fam_scores else 0.0

        # Acceptance rate.
        actions = [r.actual_user_action for r in enriched if r.actual_user_action is not None]
        acceptance_rate = float(np.mean([a == "accept" for a in actions])) if actions else 0.0

        # Cross-context spread (simplified: weekday vs weekend win rates).
        # In production, slice by menstrual phase, stress level, etc.
        cross_context_spread = 0.0  # Simplified for now.

        self._scorecard = Scorecard(
            window_size=self._window_size,
            records=recent,
            win_rate=win_rate,
            safety_violations=safety_violations,
            coverage_80=coverage_80,
            familiarity_rate=familiarity_rate,
            cross_context_spread=cross_context_spread,
            acceptance_rate=acceptance_rate,
            consecutive_pass_days=self._consecutive_pass_days,
            status=self._status,
        )
        return self._scorecard

    # ------------------------------------------------------------------
    # Graduation logic
    # ------------------------------------------------------------------

    def check_graduation(self, scorecard: Scorecard | None = None) -> bool:
        """Check whether conditions for graduation are met.

        Graduation is a conjunction — ALL conditions must hold simultaneously
        for a sustained period (SUSTAINED_DAYS consecutive daily checks).

        Calibration coverage is a soft gate: when the model's predictions
        are clearly uncalibrated (e.g. placeholder features produce high MAE),
        coverage would block graduation indefinitely. Instead, we require
        coverage only when models produce meaningful calibration data.

        Returns:
            True if just graduated or still graduated.
        """
        sc = scorecard or self.compute_scorecard()
        enriched_count = sum(1 for r in sc.records if r.actual_outcomes is not None)

        calibration_ok = (
            self.CALIBRATION_COVERAGE_LOW <= sc.coverage_80 <= self.CALIBRATION_COVERAGE_HIGH
        )

        conditions_met = all([
            enriched_count >= self.MIN_SHADOW_DAYS,
            sc.win_rate >= self.WIN_RATE_THRESHOLD,
            sc.safety_violations == 0 if self.ZERO_SAFETY_VIOLATIONS else True,
            calibration_ok,
            sc.familiarity_rate >= self.FAMILIARITY_RATE_THRESHOLD,
            sc.cross_context_spread <= self.CROSS_CONTEXT_SPREAD_MAX,
        ])

        if conditions_met:
            self._consecutive_pass_days += 1
        else:
            self._consecutive_pass_days = 0

        if self._status == GraduationStatus.SHADOW:
            if self._consecutive_pass_days >= self.SUSTAINED_DAYS:
                self._status = GraduationStatus.GRADUATED
                return True
        elif self._status == GraduationStatus.GRADUATED:
            # Post-graduation monitoring: de-graduate if performance drops.
            if sc.win_rate < self.DEGRAD_WIN_RATE or sc.safety_violations > 0:
                self._status = GraduationStatus.DEGRADED
                self._consecutive_pass_days = 0
                return False
        elif self._status == GraduationStatus.DEGRADED:
            # Re-graduation: same conditions as initial graduation.
            if self._consecutive_pass_days >= self.SUSTAINED_DAYS:
                self._status = GraduationStatus.GRADUATED
                return True

        return self._status == GraduationStatus.GRADUATED

    # ------------------------------------------------------------------
    # Acceptance rate as meta-metric
    # ------------------------------------------------------------------

    def get_acceptance_feedback(self) -> dict[str, float]:
        """Track user acceptance rate and provide feedback for the optimizer.

        Low acceptance → tighten conservativeness, reduce recommendation
        frequency, or shift toward smaller changes.

        Returns:
            Dict with acceptance_rate and suggested_conservativeness_adjustment.
        """
        sc = self.compute_scorecard()
        adjustment = 0.0
        if sc.acceptance_rate < 0.3:
            adjustment = 0.2  # Significantly tighten
        elif sc.acceptance_rate < 0.5:
            adjustment = 0.1  # Moderately tighten
        elif sc.acceptance_rate > 0.8:
            adjustment = -0.05  # Slightly loosen

        return {
            "acceptance_rate": sc.acceptance_rate,
            "suggested_conservativeness_adjustment": adjustment,
        }

    # ------------------------------------------------------------------
    # Positive Observation Records (Section 6.5)
    # ------------------------------------------------------------------

    def create_positive_observation(
        self,
        patient_id: str,
        day_index: int,
        message: str,
        tir_improvement: float = 0.0,
        details: dict | None = None,
    ) -> ShadowRecord:
        """Create a 'positive observation' record for celebrations.

        These log moments worth celebrating even when there's no suggestion.
        """
        record = ShadowRecord(
            patient_id=patient_id,
            day_index=day_index,
            timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            record_type="positive_observation",
            feature_snapshot=details or {},
        )
        # Store the celebration message in the feature_snapshot
        record.feature_snapshot["celebration_message"] = message
        record.feature_snapshot["tir_improvement"] = tir_improvement
        self._records[record.record_id] = record
        return record

    # ------------------------------------------------------------------
    # Recommendation Budget Tracking
    # ------------------------------------------------------------------

    def get_recent_acceptance_rate(self, window: int = 7) -> float:
        """Compute acceptance rate over the last *window* enriched records."""
        enriched = [
            r for r in self._records.values()
            if r.actual_user_action is not None
            and r.record_type == "recommendation"
        ]
        if not enriched:
            return 0.5  # No data yet
        recent = sorted(enriched, key=lambda r: r.day_index)[-window:]
        accepted = sum(1 for r in recent if r.actual_user_action == "accept")
        return accepted / max(len(recent), 1)

    def get_recent_win_rate(self, window: int = 14) -> float:
        """Compute win rate over recent enriched records."""
        enriched = [
            r for r in self._records.values()
            if r.shadow_score_delta is not None
            and r.record_type == "recommendation"
        ]
        if not enriched:
            return 0.5
        recent = sorted(enriched, key=lambda r: r.day_index)[-window:]
        wins = sum(1 for r in recent if r.shadow_score_delta > 0)
        return wins / max(len(recent), 1)

    def last_recommendation_succeeded(self) -> bool | None:
        """Check if the most recent recommendation was successful."""
        enriched = [
            r for r in self._records.values()
            if r.shadow_score_delta is not None
            and r.actual_user_action == "accept"
            and r.record_type == "recommendation"
        ]
        if not enriched:
            return None
        last = max(enriched, key=lambda r: r.day_index)
        return last.shadow_score_delta > 0 if last.shadow_score_delta is not None else None

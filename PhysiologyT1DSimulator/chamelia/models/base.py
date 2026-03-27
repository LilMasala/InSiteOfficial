"""PredictorCard ABC and PredictionEnvelope — the universal model interface (Block 3)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PredictionEnvelope:
    """Standardised output from any model in the Chamelia zoo.

    Every model returns not just a point estimate but a distribution summary.
    The envelope carries calibrated uncertainty bounds and metadata for the
    confidence gate and shadow module to consume.

    Attributes:
        point:      Point estimate (q=0.5). Float for scalar targets, ndarray
                    for multi-output or trajectory targets.
        lower:      Lower bound (q=0.10 by convention).
        upper:      Upper bound (q=0.90 by convention).
        confidence: Self-assessed confidence score in [0, 1].
                    0.0 indicates an untrained or stub model.
                    1.0 indicates maximum calibrated confidence.
        metadata:   Arbitrary key-value annotations — feature importances,
                    nearest training neighbours, unseen-label flags, etc.
    """
    point: float | np.ndarray
    lower: float | np.ndarray
    upper: float | np.ndarray
    confidence: float
    metadata: dict = field(default_factory=dict)


class PredictorCard(ABC):
    """Abstract base class for all Chamelia models.

    The meta-controller interacts exclusively through this interface. It does
    not care whether it is talking to an XGBoost model or a transformer — it
    asks: given this state and this proposed action, what do you predict, and
    how sure are you?

    Class-level attributes (set on each concrete subclass):
        model_id:       Unique identifier string (e.g. "aggregate_v1").
        version:        Semantic version string (e.g. "1.0.0").
        target:         Target variable family being predicted
                        (e.g. "aggregate", "tir", "bg_trajectory").
        feature_schema: Ordered list of feature column names expected by
                        predict(). An empty list means the model accepts any
                        feature vector.
        action_schema:  Ordered list of action dimension names expected by
                        predict(action=...). None for observation-only models.
    """

    model_id: str
    version: str
    target: str
    feature_schema: list[str]
    action_schema: list[str] | None

    @abstractmethod
    def predict(
        self,
        features: np.ndarray,
        action: np.ndarray | None = None,
    ) -> PredictionEnvelope:
        """Generate a prediction for the given feature vector and optional action.

        Args:
            features: Shape (n_features,) for a single sample or
                      (n_samples, n_features) for a batch.
            action:   Shape (n_actions,) or (n_samples, n_actions).
                      Pass None for observation-only models.

        Returns:
            PredictionEnvelope with point/lower/upper arrays whose first
            dimension matches n_samples (or scalar for single-sample input).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model artifact to *path*."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "PredictorCard":
        """Load and return a model from a previously saved artifact at *path*."""

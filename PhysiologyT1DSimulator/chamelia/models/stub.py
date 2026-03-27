"""StubRecommender — drop-in placeholder for fork_timeline before real models exist."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from chamelia.models.base import PredictionEnvelope, PredictorCard


class StubRecommender(PredictorCard):
    """Minimal PredictorCard that always proposes a +5 % ISF bump.

    Used by fork_timeline() and tests before AggregateOutcomePredictor is
    trained.  Returns confidence=0.0 so downstream gates can detect it as a
    stub and suppress real recommendations.

    Action schema matches AggregateOutcomePredictor exactly:
        [isf_multiplier, cr_multiplier, basal_multiplier]

    predict() accepts the current action vector as *action* and proposes:
        isf_multiplier  *= 1.05   (5 % increase)
        cr_multiplier   unchanged
        basal_multiplier unchanged
    """

    model_id = "stub_v0"
    version = "0.0.0"
    target = "aggregate"
    feature_schema: list[str] = []
    action_schema: list[str] = ["isf_multiplier", "cr_multiplier", "basal_multiplier"]

    def predict(
        self,
        features: np.ndarray,
        action: np.ndarray | None = None,
    ) -> PredictionEnvelope:
        """Return a fixed +5 % ISF recommendation.

        Args:
            features: Ignored. Accepted for interface compatibility.
            action:   Current action vector [isf_mult, cr_mult, basal_mult].
                      Defaults to [1.0, 1.0, 1.0] if None.

        Returns:
            PredictionEnvelope with point/lower/upper shaped (3,), confidence=0.0.
        """
        if action is None:
            current = np.array([1.0, 1.0, 1.0], dtype=float)
        else:
            current = np.asarray(action, dtype=float).flatten()[:3]
            if current.shape[0] < 3:
                current = np.pad(current, (0, 3 - current.shape[0]), constant_values=1.0)

        proposed = current * np.array([1.05, 1.0, 1.0])
        proposed = np.clip(proposed, 0.70, 1.35)

        return PredictionEnvelope(
            point=proposed,
            lower=proposed * 0.95,
            upper=proposed * 1.05,
            confidence=0.0,
            metadata={"source": "stub"},
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=5)

    @classmethod
    def load(cls, path: str) -> "StubRecommender":
        with open(path, "rb") as f:
            return pickle.load(f)

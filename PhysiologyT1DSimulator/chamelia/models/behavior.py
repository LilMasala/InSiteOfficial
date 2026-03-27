"""Model 5: Behavior Response Model — stub for future behavioral interventions (Block 3).

Predicts how exercise timing, wind-down routines, and other behavioral
interventions affect next-day BG and HR/HRV. Same PredictorCard interface,
different action schema.

Deferred until the therapy path is working, but the zoo architecture
accommodates it from day one.
"""
from __future__ import annotations

import numpy as np

from chamelia.models.base import PredictionEnvelope, PredictorCard


class BehaviorResponseModel(PredictorCard):
    """Stub for future behavioral intervention predictions.

    Action schema covers behavioral interventions rather than therapy params.
    Raises NotImplementedError on predict()/fit() — this model is deferred
    until the therapy recommendation path is proven.

    Activation conditions:
        - Therapy path (Models 1-4, Blocks 4-7) is operational and graduated
        - Sufficient real-world behavioral data is collected
        - Behavioral outcome metrics are defined and validated
    """

    model_id = "behavior_v0"
    version = "0.0.0"
    target = "behavior_response"
    feature_schema: list[str] = []
    action_schema: list[str] = [
        "exercise_timing_h",
        "exercise_duration_min",
        "exercise_intensity",
        "wind_down_start_h",
        "meal_window_start_h",
        "meal_window_end_h",
    ]

    def predict(
        self,
        features: np.ndarray,
        action: np.ndarray | None = None,
    ) -> PredictionEnvelope:
        """Not implemented — behavioral model is deferred.

        Raises:
            NotImplementedError: Always. This model requires the therapy path
                to be operational before activation.
        """
        raise NotImplementedError(
            "BehaviorResponseModel is deferred until the therapy recommendation "
            "path (Models 1-4, Blocks 4-7) is operational and graduated. "
            "See Chamelia Architecture v1, Section 4.6."
        )

    def save(self, path: str) -> None:
        raise NotImplementedError("BehaviorResponseModel is a stub.")

    @classmethod
    def load(cls, path: str) -> "BehaviorResponseModel":
        raise NotImplementedError("BehaviorResponseModel is a stub.")

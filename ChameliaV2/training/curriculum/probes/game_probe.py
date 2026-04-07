"""Game probe scaffold."""

from __future__ import annotations

from typing import Any


class GameProbe:
    """Probe strategic quality in games."""

    def evaluate(self, model: Any, batch: dict[str, Any]) -> dict[str, float]:
        """Return placeholder game metrics."""
        _ = model
        _ = batch
        return {"game_score": 0.0, "plan_accuracy": 0.0}

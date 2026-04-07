"""Pattern probe scaffold."""

from __future__ import annotations

from typing import Any


class PatternProbe:
    """Probe rule and regime understanding."""

    def evaluate(self, model: Any, batch: dict[str, Any]) -> dict[str, float]:
        """Return placeholder pattern metrics."""
        _ = model
        _ = batch
        return {"rule_probe": 0.0, "regime_detection": 0.0, "counterfactual": 0.0}

"""Synthetic patient environment scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PatientState:
    """Toy physiological and psychosocial patient state."""

    bg: float = 140.0
    mood: float = 0.0
    engagement: float = 0.7
    trust: float = 0.7
    burnout: float = 0.1
    burden: float = 0.2


class SyntheticPatientEnv:
    """Physiology and care scaffold environment."""

    def __init__(self) -> None:
        self.state = PatientState()

    def reset(self, patient_profile: dict[str, Any] | None = None) -> dict[str, float]:
        """Reset the synthetic patient."""
        _ = patient_profile
        self.state = PatientState()
        return self._to_dict()

    def step(self, intervention: str, params: dict[str, Any]) -> tuple[dict[str, float], float, bool, dict[str, Any]]:
        """Apply an intervention and emit a toy next state."""
        _ = params
        if intervention == "aggressive_optimize":
            self.state.bg -= 15.0
            self.state.burden += 0.1
            self.state.trust -= 0.05
        elif intervention in {"stabilize", "support"}:
            self.state.bg -= 5.0
            self.state.burden -= 0.03
            self.state.trust += 0.04
        reward = -(abs(self.state.bg - 110.0) / 100.0 + self.state.burden + (1.0 - self.state.trust))
        return self._to_dict(), reward, False, {"intervention": intervention}

    def inject_event(self, event_type: str, severity: float) -> None:
        """Inject a life event into the toy patient."""
        if event_type == "illness":
            self.state.bg += 30.0 * severity
            self.state.engagement -= 0.2 * severity
        elif event_type == "acute_stress":
            self.state.bg += 20.0 * severity
            self.state.burden += 0.2 * severity

    def get_patient_narrative(self) -> str:
        """Return a short natural-language patient narrative."""
        return (
            f"Patient BG is {self.state.bg:.1f}. Trust {self.state.trust:.2f}. "
            f"Burnout {self.state.burnout:.2f}. Burden {self.state.burden:.2f}."
        )

    def _to_dict(self) -> dict[str, float]:
        """Convert state to a plain dictionary."""
        return {
            "bg": self.state.bg,
            "mood": self.state.mood,
            "engagement": self.state.engagement,
            "trust": self.state.trust,
            "burnout": self.state.burnout,
            "burden": self.state.burden,
        }

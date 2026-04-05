"""InSite/T1D bridge plugin for the repaired Chamelia substrate."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from src.chamelia.tokenizers import TimeSeriesTokenizer

from .base import AbstractDomain

_INSITE_SIGNAL_ORDER = (
    "bg_avg",
    "tir_7d",
    "pct_low_7d",
    "pct_high_7d",
    "bg_var",
    "exercise_mins",
    "cycle_phase_menstrual",
    "cycle_phase_luteal",
)


def _signal_tensor(observation: dict[str, Any]) -> torch.Tensor:
    """Project an InSite observation payload into a fixed feature tensor [1, 1, F]."""
    signals = observation.get("signals", {})
    if not isinstance(signals, dict):
        signals = {}
    values: list[float] = []
    for key in _INSITE_SIGNAL_ORDER:
        raw = signals.get(key, 0.0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = 0.0
        if key == "bg_avg":
            value = value / 250.0
        elif key in {"exercise_mins"}:
            value = min(max(value / 120.0, 0.0), 2.0)
        else:
            value = min(max(value, -2.0), 2.0)
        values.append(value)
    return torch.tensor(values, dtype=torch.float32).view(1, 1, -1)


class InSiteBridgeDomain(AbstractDomain):
    """Plugin-owned bridge domain for InSite/T1D observations."""

    def __init__(self, embed_dim: int = 64, action_dim: int = 8) -> None:
        self._action_dim = action_dim
        self._tokenizer = TimeSeriesTokenizer(
            num_features=len(_INSITE_SIGNAL_ORDER),
            embed_dim=embed_dim,
            max_seq_len=8,
            domain_name="insite_t1d",
            use_learned_pos=True,
        )

    def get_tokenizer(self) -> TimeSeriesTokenizer:
        return self._tokenizer

    def get_action_dim(self) -> int:
        return self._action_dim

    def decode_action(self, action_vec: torch.Tensor) -> Any:
        if action_vec.dim() == 1:
            action_vec = action_vec.unsqueeze(0)
        labels = (
            "hold_bias",
            "basal_adjustment",
            "correction_bias",
            "meal_bias",
            "support_intensity",
            "stability_bias",
            "probe_bias",
            "trust_preservation",
        )
        decoded: list[dict[str, float]] = []
        for row in action_vec[:, : len(labels)]:
            decoded.append({label: float(value) for label, value in zip(labels, row.tolist(), strict=False)})
        return decoded[0] if len(decoded) == 1 else decoded

    def get_intrinsic_cost_fns(self) -> list[tuple[Callable, float]]:
        def hypoglycemia_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict) -> torch.Tensor:
            _ = z
            bg = domain_state["bg_avg"].to(action.device)
            pct_low = domain_state["pct_low_7d"].to(action.device)
            aggressiveness = action[:, 1:4].abs().mean(dim=-1)
            low_risk = torch.relu((95.0 - bg) / 55.0) + pct_low
            return low_risk * (1.0 + aggressiveness)

        def hyperglycemia_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict) -> torch.Tensor:
            _ = z
            bg = domain_state["bg_avg"].to(action.device)
            pct_high = domain_state["pct_high_7d"].to(action.device)
            support = action[:, 4].abs()
            return torch.relu((bg - 160.0) / 120.0) + pct_high + 0.1 * support

        def volatility_cost(z: torch.Tensor, action: torch.Tensor, domain_state: dict) -> torch.Tensor:
            _ = z
            bg_var = domain_state["bg_var"].to(action.device)
            intervention_size = action[:, :4].abs().mean(dim=-1)
            return bg_var + 0.25 * intervention_size

        return [
            (hypoglycemia_cost, 0.5),
            (hyperglycemia_cost, 0.3),
            (volatility_cost, 0.2),
        ]

    def get_domain_state(self, observation: Any) -> dict:
        if not isinstance(observation, dict):
            raise TypeError("InSiteBridgeDomain expects a bridge observation dict.")
        signals = observation.get("signals", {})
        if not isinstance(signals, dict):
            signals = {}

        def scalar(name: str, default: float = 0.0) -> torch.Tensor:
            raw = signals.get(name, default)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = default
            return torch.tensor([value], dtype=torch.float32)

        return {
            "timestamp": torch.tensor([float(observation.get("timestamp", 0.0))], dtype=torch.float32),
            "bg_avg": scalar("bg_avg", 110.0),
            "tir_7d": scalar("tir_7d", 0.65),
            "pct_low_7d": scalar("pct_low_7d", 0.02),
            "pct_high_7d": scalar("pct_high_7d", 0.20),
            "bg_var": scalar("bg_var", 0.12),
            "exercise_mins": scalar("exercise_mins", 0.0),
            "cycle_phase_menstrual": scalar("cycle_phase_menstrual", 0.0),
            "cycle_phase_luteal": scalar("cycle_phase_luteal", 0.0),
            "signals": signals,
        }

    def get_persistable_domain_state(self, domain_state: dict) -> dict[str, Any] | None:
        signals = domain_state.get("signals", {})
        if not isinstance(signals, dict):
            signals = {}
        return {
            "signals": {str(key): float(value) for key, value in signals.items() if isinstance(value, (int, float))},
        }

    def prepare_bridge_observation(self, observation: Any) -> Any:
        if not isinstance(observation, dict):
            raise TypeError("InSite bridge observation must be a dict with timestamp/signals.")
        return _signal_tensor(observation)

    def compute_regime_embedding(self, domain_state: dict) -> torch.Tensor | None:
        features = torch.stack(
            [
                domain_state["bg_avg"],
                domain_state["tir_7d"],
                domain_state["pct_low_7d"],
                domain_state["pct_high_7d"],
                domain_state["exercise_mins"],
            ],
            dim=-1,
        )
        return features.float()

    @property
    def domain_name(self) -> str:
        return "insite_t1d"

    @property
    def vocab_size(self) -> int:
        return 0

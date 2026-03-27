"""Meal and mood logging behavior."""
from __future__ import annotations

import numpy as np


def confirm_meals(meals: list[tuple], quality: str, inactive_days: int, seed: int) -> list[tuple]:
    rng = np.random.default_rng(seed)
    base = {"good": 0.9, "mediocre": 0.65, "poor": 0.35}[quality]
    eff = base * np.exp(-0.08 * inactive_days)
    return [m for m in meals if rng.random() < eff]


def should_log_mood(quality: str, seed: int) -> bool:
    rng = np.random.default_rng(seed)
    p = {"good": 0.8, "mediocre": 0.4, "poor": 0.15}[quality]
    return bool(rng.random() < p)

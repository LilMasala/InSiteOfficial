"""Observation/noise model.

Applies sensor noise (Dexcom G6-like) and block-structured missingness
from a DayMissingness object.  The missingness is generated upstream in
missingness.py and passed in — this module never decides *what* is missing,
only applies the mask after adding realistic sensor artifacts.
"""
from __future__ import annotations

import numpy as np
from t1d_sim.missingness import DayMissingness, apply_cgm_missingness


def observe_cgm(true_bg: np.ndarray,
                dm: DayMissingness,
                patient_seed: int,
                day_index: int = 0) -> np.ndarray:
    """Apply Dexcom G6-like noise and block-structured missingness.

    Args:
        true_bg: (288,) ground truth BG in mg/dL.
        dm: DayMissingness with pre-computed cgm_mask.
        patient_seed: for reproducibility.
        day_index: day within patient history.

    Returns:
        (288,) float32 array with NaN for missing readings.
    """
    rng = np.random.default_rng(patient_seed * 10000 + day_index + 1)
    obs = true_bg.astype(float).copy()

    # Sensor lag: 5-10 min (1-2 bins)
    lag = int(rng.integers(1, 3))
    obs[lag:] = obs[:-lag]
    obs[:lag] = obs[lag]

    # Additive + multiplicative Gaussian noise (Dexcom G6 spec)
    obs = obs * (1 + rng.normal(0, 0.02, size=obs.size))
    obs = obs + rng.normal(0, 5, size=obs.size)

    # Slow calibration drift over sensor life
    drift = np.linspace(0, rng.uniform(-3, 3), obs.size)
    obs += drift

    # Clip to physiological range before applying missingness
    obs = np.clip(obs, 40.0, 450.0)

    # Apply block-structured missingness from DayMissingness
    obs = apply_cgm_missingness(obs, dm)

    return obs


def synthesize_hr(base_rhr: float,
                  stress: float,
                  exercise_minutes: float,
                  exercise_intensity: float,
                  dm: DayMissingness,
                  day_index: int,
                  patient_seed: int,
                  exercise_hour: int | None = None,
                  sleep_debt_min: float = 0.0,
                  stress_reactivity: float = 0.5) -> np.ndarray:
    """Produce 24-element hourly HR array with block-structured missingness.

    Missingness is applied from dm.watch_hourly_mask — the watch off-blocks
    determined in the causal missingness generation.
    """
    rng = np.random.default_rng(patient_seed * 10000 + day_index + 2)
    rhr_offset = min(9.0, sleep_debt_min / 60.0)
    hr = np.full(24, base_rhr + rhr_offset, dtype=float)

    # Sleep suppression (hours 0-6)
    hr[:6] *= 0.85

    # Stress elevation during waking hours (additive, parameterized)
    for h in range(6, 23):
        hr[h] += 20.0 * stress * stress_reactivity

    # Exercise spike
    if exercise_minutes > 0:
        if exercise_hour is not None:
            ex_hour = exercise_hour
        else:
            ex_hour = int(rng.integers(6, 20))
        boost = 1.5 if exercise_intensity < 0.6 else 1.75
        duration = max(1, int(exercise_minutes / 60) + 1)
        hr[ex_hour:min(24, ex_hour + duration)] *= boost
        # Recovery taper
        recovery_end = min(24, ex_hour + duration + 2)
        for i in range(ex_hour + duration, recovery_end):
            hr[i] *= (1.0 + 0.3 * (boost - 1.0) * (recovery_end - i) / 2)

    # Gaussian noise
    hr += rng.normal(0, 3, size=24)
    hr = np.clip(hr, 30.0, 220.0)

    # Apply watch-based missingness + rare single-hour dropout
    from t1d_sim.missingness import apply_hr_missingness
    hr = apply_hr_missingness(hr, dm, rng)

    return hr


def synthesize_energy(base_rhr: float,
                      basal_multiplier: float,
                      exercise_minutes: float,
                      exercise_intensity: float,
                      dm: DayMissingness,
                      day_index: int,
                      patient_seed: int,
                      exercise_hour: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (basal_energy_24h, active_energy_24h) with watch missingness."""
    rng = np.random.default_rng(patient_seed * 10000 + day_index + 3)
    basal = np.full(24, 65.0 * basal_multiplier + rng.normal(0, 3))
    active = np.zeros(24)

    if exercise_minutes > 0:
        if exercise_hour is not None:
            ex_hour = exercise_hour
        else:
            ex_hour = int(rng.integers(6, 20))
        kcal = exercise_minutes * exercise_intensity * 8.0
        active[ex_hour] = kcal

    # Apply watch-based missingness
    from t1d_sim.missingness import apply_energy_missingness
    basal, active = apply_energy_missingness(basal, active, dm)

    return basal, active

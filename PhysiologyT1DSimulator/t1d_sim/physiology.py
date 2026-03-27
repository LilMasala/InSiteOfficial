"""Physiology model wrapper."""
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING
import numpy as np

from t1d_sim.behavior import ContextState
from t1d_sim.therapy import TherapySchedule

if TYPE_CHECKING:
    from t1d_sim.feedback import PatientState


class MealProfile(str, Enum):
    """Meal absorption profile types."""

    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"


_MEAL_PROFILE_PARAMS: dict[MealProfile, tuple[int, float, float]] = {
    MealProfile.FAST: (48, 0.22, 1.20),
    MealProfile.MEDIUM: (72, 0.12, 1.00),
    MealProfile.SLOW: (96, 0.075, 0.85),
}


def _meal_absorption_kernel(profile: MealProfile) -> np.ndarray:
    """Dual-exponential meal absorption kernel in 5-min bins."""
    length, decay, gain = _MEAL_PROFILE_PARAMS[profile]
    t = np.arange(length, dtype=float)
    # Difference of exponentials: fast gastric emptying vs slower elimination tail.
    gastric = np.exp(-t * decay * 0.55)
    elimination = np.exp(-t * decay * 1.35)
    kernel = np.maximum(0.0, (gastric - elimination) * gain)
    kernel_sum = float(kernel.sum())
    if kernel_sum <= 0.0:
        return np.zeros(length, dtype=float)
    return kernel / kernel_sum


_MEAL_KERNELS: dict[MealProfile, np.ndarray] = {
    profile: _meal_absorption_kernel(profile) for profile in MealProfile
}


def _parse_meal_event(meal: tuple, rng: np.random.Generator) -> tuple:
    """Normalize meal event tuple as (time, carbs, profile)."""
    if len(meal) == 2:
        t, carbs = meal
        profile = MealProfile.MEDIUM
    else:
        t, carbs, raw_profile = meal[:3]
        if raw_profile is None:
            profile = MealProfile.MEDIUM
        elif isinstance(raw_profile, MealProfile):
            profile = raw_profile
        elif str(raw_profile) == "grazer":
            profile = MealProfile(rng.choice(
                ["fast", "medium", "slow"],
                p=[0.30, 0.45, 0.25],
            ))
        else:
            profile = MealProfile(str(raw_profile).lower())
    return t, float(carbs), profile


def apply_context_effectors(
    base_params: dict,
    ctx: ContextState,
    patient_state: "PatientState | None" = None,
    event_modifiers: dict | None = None,
) -> dict:
    """Apply context-driven parameter modulation.

    Optional feedback parameters (backward-compatible, default None):
        patient_state: drifted ISF/fitness from biweekly updates.
        event_modifiers: dict from apply_event_modifiers() for life events.
    """
    params = base_params.copy()
    ev = event_modifiers or {}

    # Note: drifted ISF from PatientState is injected via base_params["k1"/"k2"]
    # by the caller (patient.py). Drifted stress_baseline reaches here through
    # ctx.stress_baseline, set by behavior.py. So patient_state param is accepted
    # for forward compatibility but doesn't modify params directly.

    # ── Event-driven ISF modifier (medication, illness) ──
    params["k1"] *= ev.get("isf_mult_factor", 1.0)
    params["k2"] *= ev.get("isf_mult_factor", 1.0)

    if ctx.hours_since_exercise < 48:
        decay = np.exp(-ctx.hours_since_exercise / 16.0)
        max_boost = 0.40 * ctx.exercise_intensity
        boost = 1.0 + max_boost * decay
        params["k1"] *= boost
        params["k2"] *= boost
    if ctx.cycle_phase == "luteal":
        r = 1.0 - (0.03 + 0.15 * ctx.cycle_sensitivity)
        params["k1"] *= r
        params["k2"] *= r
    elif ctx.cycle_phase == "menstrual":
        # Gap 7: IS elevated during menstrual phase (Brown et al. 2015 — hypoglycemia risk)
        params["k1"] *= 1.0 + (0.03 + 0.08 * ctx.cycle_sensitivity)
        params["k2"] *= 1.0 + (0.02 + 0.05 * ctx.cycle_sensitivity)
    elif ctx.cycle_phase == "follicular":
        params["k1"] *= 1.0 + 0.02 * ctx.cycle_sensitivity
        params["k2"] *= 1.0 + 0.02 * ctx.cycle_sensitivity
    elif ctx.cycle_phase == "ovulation":
        params["k1"] *= 1.0 + 0.05 * ctx.cycle_sensitivity
        params["k2"] *= 1.0 + 0.05 * ctx.cycle_sensitivity

    # Research: Spiegel et al. (1999) Lancet — 14% SI drop per 4h restriction → λ=0.035/hr
    # Van Cauter lab: 20-30% EGP rise from overnight deprivation
    # Reference baseline: 8.5h (510 min) — used in sleep deprivation studies
    SLEEP_REF_MIN = 510.0
    sleep_deficit_h = max(0.0, (SLEEP_REF_MIN - ctx.sleep_min_last_night) / 60.0)
    lam = 0.035  # 14% SI drop per 4h = 0.035/hr
    si_reduction = min(0.30, lam * sleep_deficit_h) * (0.7 + 0.3 * ctx.stress_reactivity)
    params["k1"] *= (1.0 - si_reduction)
    params["k2"] *= (1.0 - si_reduction * 0.6)
    egp_rise = min(0.28, 0.025 * sleep_deficit_h * (0.7 + 0.3 * ctx.stress_reactivity))
    params["EGP0"] *= (1.0 + egp_rise)
    # Sleep fragmentation penalty — T1D population mean efficiency is 0.82, not 0.80
    # Research: fragmentation produces ~25% SI reduction equivalent to similar total deprivation
    if ctx.sleep_efficiency < 0.82:
        frag_penalty = 0.25 * max(0.0, (0.82 - ctx.sleep_efficiency) / 0.82)
        params["k1"] *= (1.0 - frag_penalty * 0.5)
        params["EGP0"] *= (1.0 + frag_penalty * 0.3)

    # Research tiers (from pharmacological stress studies):
    #   mild psych stress (0.1-0.3):  5-10% TDI increase
    #   acute TSST/cold (0.3-0.6):    13-30% TDI increase
    #   pharmacological-like (0.6-0.9): 40-70% TDI increase
    if ctx.stress > 0.1:
        if ctx.stress < 0.3:
            stress_egp_factor = 0.12 * (ctx.stress - 0.1)
        elif ctx.stress < 0.6:
            stress_egp_factor = 0.024 + 0.35 * (ctx.stress - 0.3)
        else:
            stress_egp_factor = 0.129 + 0.65 * (ctx.stress - 0.6)
        # Gap 6: only 30% of stress EGP effect applies to fasting state;
        # the remaining 70% is postprandial-specific (applied in simulate_day_cgm)
        fasting_effect = 1.0 + stress_egp_factor * ctx.stress_reactivity * 0.30
        params["EGP0"] *= fasting_effect
        params["k1"] *= (1.0 / (1.0 + 0.45 * (fasting_effect - 1.0)))
        # Store postprandial stress factor for simulate_day_cgm
        params["_postprandial_stress_factor"] = stress_egp_factor * ctx.stress_reactivity * 0.70
    # Chronic background stress — 10-20% persistent TDI increase
    if ctx.stress_baseline > 0.15:
        chronic = min(0.20, 0.15 * (ctx.stress_baseline - 0.15) * ctx.stress_reactivity)
        params["EGP0"] *= (1.0 + chronic)

    # ── Illness: event-driven EGP/ISF replaces binary check ──
    if ev.get("is_ill_override", False):
        params["EGP0"] *= ev.get("egp_mult", 1.40)
        # ISF already applied via isf_mult_factor above
    elif ctx.is_ill:
        params["EGP0"] *= 1.40
        params["k1"] *= 0.75

    # Gap 2: post-exercise late hypoglycemia window (MacDonald 1987)
    # 6-15h after aerobic exercise: reduced nadir EGP
    if getattr(ctx, 'post_exercise_hypo_window', False):
        params["EGP0"] *= 0.92  # 8% EGP reduction during overnight window

    return params


def simulate_day_cgm(
    base_params: dict,
    modified_params: dict,
    meals: list[tuple],
    seed: int,
    therapy_schedule: TherapySchedule | None = None,
    post_exercise_hypo_window: bool = False,
    dawn_sensitivity: float = 0.0,
) -> np.ndarray:
    """Generate 288 five-minute BG points in mg/dL."""
    rng = np.random.default_rng(seed)
    bg = np.zeros(288, dtype=float)
    bg[0] = 120.0 + rng.normal(0, 8)
    meal_effect = np.zeros(288)
    for meal in meals:
        t, carbs, profile = _parse_meal_event(meal, rng)
        idx = int((t.hour * 60 + t.minute) / 5)
        kernel = _MEAL_KERNELS[profile]
        end = min(288, idx + kernel.size)
        window = end - idx
        if window > 0:
            meal_effect[idx:end] += carbs * kernel[:window] * 9.0
    sens = modified_params["k1"] / max(1e-6, base_params["k1"])
    egp = modified_params["EGP0"] / max(1e-6, base_params["EGP0"])
    ppm_stress = modified_params.get("_postprandial_stress_factor", 0.0)
    mean_isf = therapy_schedule.weighted_mean("isf") if therapy_schedule is not None else None
    mean_cr = therapy_schedule.weighted_mean("cr") if therapy_schedule is not None else None
    mean_basal = therapy_schedule.weighted_mean("basal") if therapy_schedule is not None else None
    for i in range(1, 288):
        local_sens = sens
        local_egp = egp
        local_meal_effect = meal_effect[i]
        if therapy_schedule is not None and mean_isf and mean_cr and mean_basal:
            seg = therapy_schedule.value_at_minute(i * 5)
            isf_factor = np.clip(mean_isf / max(seg.isf, 1e-6), 0.75, 1.25)
            cr_factor = np.clip(mean_cr / max(seg.cr, 1e-6), 0.75, 1.25)
            basal_factor = np.clip(seg.basal / max(mean_basal, 1e-6), 0.75, 1.25)
            local_sens *= isf_factor
            local_meal_effect *= cr_factor
            local_egp *= 1.0 / basal_factor

        # Gap 5: Dawn phenomenon (GH surge 03:00-08:00, ~50% T1D prevalence)
        hour_of_day = (i * 5) / 60.0
        if 3.0 <= hour_of_day <= 8.0 and dawn_sensitivity > 0:
            ramp = np.sin(np.pi * (hour_of_day - 3.0) / 5.667)
            local_egp *= (1.0 + dawn_sensitivity * 0.22 * ramp)

        # Gap 6: Postprandial stress boost
        postprandial_active = meal_effect[i] > 0.5
        local_egp_stress_boost = 1.0 + ppm_stress if postprandial_active else 1.0
        local_egp *= local_egp_stress_boost

        drift = (local_egp - 1.0) * 1.8 - (local_sens - 1.0) * 1.2

        # Gap 2: post-exercise hypo window — increased noise during overnight bins
        if post_exercise_hypo_window and i < 108:  # bins 0-107 = midnight to 9am
            noise_scale = 3.8
        else:
            noise_scale = 2.8
        bg[i] = max(45.0, min(450.0, bg[i-1] + 0.06 * local_meal_effect + drift + rng.normal(0, noise_scale)))
    return bg

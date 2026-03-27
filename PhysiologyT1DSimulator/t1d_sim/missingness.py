"""
Block-structured, causally-correlated missingness model for t1d_sim.

Design principles:
  1. Physical events cause correlated missingness across signals.
     Watch-off → no HR, no energy, no exercise, possibly no sleep.
  2. Quality tiers (good/mediocre/poor) set dramatically different base rates.
  3. Engagement patterns modulate base rates over time (orthogonal to tier).
  4. CGM has its own block structure (different device from watch).

Causal chain per day:
  watch schedule → derive sleep → derive exercise capture → CGM independently
  → engagement decay scales all rates
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import numpy as np


# ---------------------------------------------------------------------------
# Engagement patterns (orthogonal to quality tier)
# ---------------------------------------------------------------------------

class EngagementPattern(Enum):
    CONSISTENT    = "consistent"    # ~30% — stable rate throughout
    EARLY_DROP    = "early_drop"    # ~25% — high initial, drops after ~3-6 weeks
    SPORADIC      = "sporadic"      # ~25% — random bursts, no trend
    LATE_REVIVAL  = "late_revival"  # ~10% — dips mid-study, picks back up
    GRADUAL_FADE  = "gradual_fade"  # ~10% — slow steady decline over full 180 days


def engagement_multiplier(pattern: EngagementPattern,
                           day_index: int,
                           rng: np.random.Generator) -> float:
    """Return engagement level (≈1.0 = baseline) for the given day.

    Values > 1 mean more engaged (fewer gaps).
    Values < 1 mean less engaged (more gaps).
    """
    d = day_index

    if pattern == EngagementPattern.CONSISTENT:
        return float(np.clip(rng.normal(1.0, 0.05), 0.8, 1.2))

    elif pattern == EngagementPattern.EARLY_DROP:
        if d < 21:
            return float(np.clip(rng.normal(1.2, 0.1), 0.9, 1.5))
        else:
            fade = np.exp(-(d - 21) / 20.0)
            floor = 0.35
            return float(np.clip(rng.normal(floor + (0.8 - floor) * fade, 0.08), 0.1, 1.0))

    elif pattern == EngagementPattern.SPORADIC:
        burst = float(np.sin(d / 7.0 * np.pi) > 0.3)
        base = 0.4 + 0.6 * burst
        return float(np.clip(rng.normal(base, 0.15), 0.0, 1.5))

    elif pattern == EngagementPattern.LATE_REVIVAL:
        dip = np.exp(-((d - 65) ** 2) / (2 * 25 ** 2))
        return float(np.clip(rng.normal(1.0 - 0.6 * dip, 0.1), 0.1, 1.2))

    elif pattern == EngagementPattern.GRADUAL_FADE:
        return float(np.clip(rng.normal(1.0 - 0.7 * (d / 180.0), 0.08), 0.05, 1.0))

    return 1.0


# ---------------------------------------------------------------------------
# Quality tier classification
# ---------------------------------------------------------------------------

class QualityTier(Enum):
    GREAT    = "great"     # logging_quality_raw > 0.9  — power users, near-zero gaps
    GOOD     = "good"      # 0.7 < logging_quality_raw <= 0.9
    MEDIOCRE = "mediocre"  # 0.35 <= logging_quality_raw <= 0.7
    POOR     = "poor"      # logging_quality_raw < 0.35


def classify_tier(logging_quality_raw: float) -> QualityTier:
    if logging_quality_raw > 0.9:
        return QualityTier.GREAT
    elif logging_quality_raw > 0.7:
        return QualityTier.GOOD
    elif logging_quality_raw >= 0.35:
        return QualityTier.MEDIOCRE
    else:
        return QualityTier.POOR


# ---------------------------------------------------------------------------
# Tier-specific rate tables
# ---------------------------------------------------------------------------

# Watch: mean hours off per day — drives HR/energy/exercise missingness.
# Target HR null rates: great 3-6%, good 12-20%, mediocre 28-42%, poor 45-65%.
WATCH_OFF_HOURS_RANGE = {
    QualityTier.GREAT:    (0.3, 1.0),    # charges during shower, rarely off otherwise
    QualityTier.GOOD:     (2.5, 4.0),
    QualityTier.MEDIOCRE: (6.5, 10.0),
    QualityTier.POOR:     (11.0, 16.0),
}

# P(overnight off-block, ~22:00-06:00) — causal root for sleep missingness.
# Target sleep miss rates: great 2-5%, good 12-18%, mediocre 35-45%, poor 55-70%.
WATCH_OVERNIGHT_OFF_P = {
    QualityTier.GREAT:    0.03,           # almost never takes watch off to bed
    QualityTier.GOOD:     0.12,
    QualityTier.MEDIOCRE: 0.33,
    QualityTier.POOR:     0.50,
}

# CGM: target fraction of 288 5-min bins missing per day.
# Target CGM null rates: great 1-2%, good 3-6%, mediocre 8-15%, poor 15-30%.
CGM_MISS_FRACTION_RANGE = {
    QualityTier.GREAT:    (0.005, 0.02),  # only sensor warmup + rare dropout
    QualityTier.GOOD:     (0.03, 0.06),
    QualityTier.MEDIOCRE: (0.08, 0.15),
    QualityTier.POOR:     (0.15, 0.30),
}

# CGM sensor change interval (days).
SENSOR_CHANGE_INTERVAL = {
    QualityTier.GREAT:    10,             # replaces on time, every time
    QualityTier.GOOD:     10,
    QualityTier.MEDIOCRE: 10,
    QualityTier.POOR:     12,             # poor users let sensor expire + delay
}

# Exercise capture probability when watch IS worn during exercise.
# Target exercise capture gap: great ~5%, good ~20%, mediocre ~45%, poor ~70%.
EXERCISE_CAPTURE_P = {
    QualityTier.GREAT:    0.99,           # always has workout app running
    QualityTier.GOOD:     0.85,           # usually starts workout, sometimes forgets
    QualityTier.MEDIOCRE: 0.65,           # often doesn't start workout tracking
    QualityTier.POOR:     0.40,           # rarely uses workout app
}


# ---------------------------------------------------------------------------
# MissingnessProfile — per-patient, sampled once at init
# ---------------------------------------------------------------------------

@dataclass
class MissingnessProfile:
    """Per-patient missingness parameters derived from logging_quality_raw."""
    tier: QualityTier

    # Watch: mean hours off per day (patient-specific draw from tier range)
    watch_off_mean_hours: float
    # P(overnight off-block) — causal root for sleep missingness
    watch_overnight_off_p: float

    # CGM: target miss fraction per day (patient-specific draw)
    cgm_miss_fraction: float
    sensor_change_interval_days: int
    compression_artifact_p: float

    # Exercise capture probability (when watch is worn during exercise)
    exercise_capture_base_p: float

    # Mood logging
    mood_events_per_day_mean: float

    # Menstrual logging
    menstrual_missing_p_per_day: float

    # Engagement pattern (orthogonal to tier)
    engagement_pattern: EngagementPattern


def make_missingness_profile(logging_quality_raw: float,
                              rng: np.random.Generator) -> MissingnessProfile:
    """Derive per-patient missingness parameters from logging_quality_raw (0-1)."""
    lq = float(np.clip(logging_quality_raw, 0, 1))
    tier = classify_tier(lq)

    # Engagement pattern — orthogonal to quality tier
    pattern_weights = [0.30, 0.25, 0.25, 0.10, 0.10]
    engagement_pattern = EngagementPattern(
        rng.choice([p.value for p in EngagementPattern], p=pattern_weights)
    )

    # Watch off hours: draw from tier range
    lo, hi = WATCH_OFF_HOURS_RANGE[tier]
    watch_off_mean = float(np.clip(rng.uniform(lo, hi), lo, hi))

    # Overnight off probability (per-patient variation around tier centre)
    overnight_p = float(np.clip(
        rng.normal(WATCH_OVERNIGHT_OFF_P[tier], 0.05), 0.02, 0.90
    ))

    # CGM miss fraction: draw from tier range
    cgm_lo, cgm_hi = CGM_MISS_FRACTION_RANGE[tier]
    cgm_frac = float(np.clip(rng.uniform(cgm_lo, cgm_hi), cgm_lo, cgm_hi))

    # Sensor change interval
    sensor_interval = int(np.clip(
        rng.normal(SENSOR_CHANGE_INTERVAL[tier], 1), 7, 15
    ))

    # Compression artifact P
    compression_p = float(np.clip(rng.normal(0.08 - 0.05 * lq, 0.02), 0.01, 0.15))

    # Exercise capture
    exercise_p = float(np.clip(
        rng.normal(EXERCISE_CAPTURE_P[tier], 0.08), 0.05, 0.98
    ))

    # Mood
    mood_mean = float(np.clip(rng.normal(0.5 + 2.5 * lq, 0.3), 0.0, 4.0))

    # Menstrual
    menstrual_p = float(np.clip(rng.normal(0.35 - 0.25 * lq, 0.05), 0.05, 0.70))

    return MissingnessProfile(
        tier=tier,
        watch_off_mean_hours=watch_off_mean,
        watch_overnight_off_p=overnight_p,
        cgm_miss_fraction=cgm_frac,
        sensor_change_interval_days=sensor_interval,
        compression_artifact_p=compression_p,
        exercise_capture_base_p=exercise_p,
        mood_events_per_day_mean=mood_mean,
        menstrual_missing_p_per_day=menstrual_p,
        engagement_pattern=engagement_pattern,
    )


# ---------------------------------------------------------------------------
# DayMissingness — per-day causal block structure
# ---------------------------------------------------------------------------

@dataclass
class DayMissingness:
    """Per-day missingness state generated from causal block structure.

    All downstream signal missingness derives from this object.
    """
    # Watch: list of (start_hour, end_hour) off-blocks
    watch_off_blocks: list[tuple[int, int]] = field(default_factory=list)

    # Derived: 24-element hourly mask (True = watch worn)
    watch_hourly_mask: np.ndarray = field(
        default_factory=lambda: np.ones(24, dtype=bool)
    )

    # CGM: list of (start_bin, end_bin) gap blocks in 5-min bins (0-287)
    cgm_gap_blocks: list[tuple[int, int]] = field(default_factory=list)

    # Derived: 288-element mask (True = CGM reading present)
    cgm_mask: np.ndarray = field(
        default_factory=lambda: np.ones(288, dtype=bool)
    )

    # Sleep — derived from whether watch was worn overnight
    sleep_missing: bool = False
    sleep_partial: bool = False   # watch put on mid-night → partial data

    # Exercise — derived from watch worn during exercise window
    exercise_captured: bool = True


def generate_day_missingness(
    profile: MissingnessProfile,
    day_index: int,
    is_weekend: bool,
    exercise_hour: int | None,
    exercise_active_bins: np.ndarray,
    rng: np.random.Generator,
) -> DayMissingness:
    """Generate causally-correlated missingness for one day.

    Causal chain:
      1. Watch off-blocks (root cause for HR, energy, exercise, sleep)
      2. Sleep missingness ← watch overnight status
      3. Exercise capture ← watch worn during exercise window
      4. CGM gaps ← independent device, own block structure
      5. All rates scaled by engagement pattern × weekend bump
    """
    dm = DayMissingness()
    eng = engagement_multiplier(profile.engagement_pattern, day_index, rng)
    # Disengagement factor: higher → more missingness
    disengage = float(np.clip(2.0 - eng, 0.1, 3.0))
    weekend_bump = 1.15 if is_weekend else 1.0

    # ---- Step 1: Watch off-blocks ----------------------------------------
    _generate_watch_schedule(dm, profile, disengage, weekend_bump, rng)

    # ---- Step 2: Sleep ← watch overnight ---------------------------------
    _derive_sleep_missingness(dm)

    # ---- Step 3: Exercise capture ← watch during exercise ----------------
    _derive_exercise_capture(dm, profile, exercise_hour, rng)

    # ---- Step 4: CGM gaps (independent of watch) -------------------------
    _generate_cgm_gaps(dm, profile, day_index, disengage, weekend_bump,
                       exercise_active_bins, rng)

    return dm


def _generate_watch_schedule(
    dm: DayMissingness,
    profile: MissingnessProfile,
    disengage: float,
    weekend_bump: float,
    rng: np.random.Generator,
) -> None:
    """Populate dm.watch_off_blocks and dm.watch_hourly_mask."""
    mask = np.ones(24, dtype=bool)

    # Target hours off for today (noisy draw around patient mean)
    target_hours = profile.watch_off_mean_hours * disengage * weekend_bump
    target_hours = float(np.clip(
        rng.normal(target_hours, max(1.0, target_hours * 0.25)), 0, 22
    ))

    # --- Overnight off-block (hours ~22-06) → causal root for sleep ---
    p_overnight = profile.watch_overnight_off_p * disengage * weekend_bump
    p_overnight = float(np.clip(p_overnight, 0, 0.95))
    overnight_off = rng.random() < p_overnight

    if overnight_off:
        off_start = int(np.clip(rng.normal(22, 1), 20, 23))
        off_end = int(np.clip(rng.normal(6, 1), 4, 8))
        # Mark hours (wraps around midnight)
        for h in range(off_start, 24):
            mask[h] = False
        for h in range(0, off_end):
            mask[h] = False
        dm.watch_off_blocks.append((off_start, off_end))

    # --- Daytime off-blocks to reach target total hours ---
    current_off = int(np.sum(~mask))
    remaining = max(0, target_hours - current_off)

    if remaining >= 1.0:
        n_blocks = int(np.clip(rng.poisson(1.5), 1, 3))
        hours_per_block = max(1.0, remaining / n_blocks)

        for _ in range(n_blocks):
            if np.sum(~mask) >= target_hours:
                break

            # Prefer typical off-times: morning charging, midday, evening
            block_type = rng.integers(0, 3)
            if block_type == 0:
                start = int(np.clip(rng.normal(7, 1), 6, 10))   # morning
            elif block_type == 1:
                start = int(np.clip(rng.normal(13, 1), 11, 15))  # midday
            else:
                start = int(np.clip(rng.normal(20, 1), 18, 22))  # evening

            length = int(np.clip(rng.normal(hours_per_block, 1), 1, 5))
            end = min(24, start + length)

            # Only mark hours that are currently on
            mask[start:end] = False
            dm.watch_off_blocks.append((start, end))

    dm.watch_hourly_mask = mask


def _derive_sleep_missingness(dm: DayMissingness) -> None:
    """Derive sleep missingness from watch overnight status.

    Apple Watch requires continuous wear during the sleep window to track a
    full session.  If the watch was off for most of the core sleep hours
    (0-5), the session can't be reconstructed → sleep_missing.  If off for
    a minority of hours, partial data may be recovered → sleep_partial.
    """
    mask = dm.watch_hourly_mask
    n_off = int(np.sum(~mask[0:6]))

    if n_off >= 3:
        # Majority of core sleep hours off → no usable sleep session
        dm.sleep_missing = True
    elif n_off >= 1:
        # Some hours off → partial/degraded sleep data
        dm.sleep_partial = True
    # else: fully worn → full sleep data


def _derive_exercise_capture(
    dm: DayMissingness,
    profile: MissingnessProfile,
    exercise_hour: int | None,
    rng: np.random.Generator,
) -> None:
    """Derive exercise capture from watch status during exercise window.

    Engagement already affects exercise capture *indirectly* via watch
    off-hours (more disengaged → more off-hours → more likely off during
    exercise).  The base capture rate reflects whether the user's watch is
    configured to detect workouts, which is a stable per-patient trait,
    not a daily engagement decision.
    """
    if exercise_hour is None or exercise_hour < 0 or exercise_hour >= 24:
        dm.exercise_captured = True  # no exercise → nothing to miss
        return

    if dm.watch_hourly_mask[exercise_hour]:
        # Watch worn — capture depends on app/setup quality (per-patient trait)
        dm.exercise_captured = bool(rng.random() < profile.exercise_capture_base_p)
    else:
        # Watch not worn during exercise → definitely not captured
        dm.exercise_captured = False


def _generate_cgm_gaps(
    dm: DayMissingness,
    profile: MissingnessProfile,
    day_index: int,
    disengage: float,
    weekend_bump: float,
    exercise_active_bins: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Generate CGM gap blocks (independent of watch — different device)."""
    cgm_mask = np.ones(288, dtype=bool)

    # Sensor warmup — every sensor_change_interval_days
    if day_index % profile.sensor_change_interval_days == 0:
        warmup_bins = int(rng.integers(18, 30))  # 1.5–2.5 hours
        cgm_mask[:warmup_bins] = False
        dm.cgm_gap_blocks.append((0, warmup_bins))

    # Target total missing bins (engagement-scaled)
    target_frac = profile.cgm_miss_fraction * disengage * weekend_bump
    target_frac = float(np.clip(target_frac, 0, 0.85))
    target_missing = int(288 * target_frac)
    current_missing = int(np.sum(~cgm_mask))
    remaining = max(0, target_missing - current_missing)

    if remaining > 0:
        # Generate 1–4 gap blocks to reach target
        n_gaps = int(np.clip(rng.poisson(2.0), 1, 4))
        bins_per_gap = max(3, remaining // n_gaps)

        for _ in range(n_gaps):
            if np.sum(~cgm_mask) >= target_missing:
                break
            gap_start = int(rng.integers(0, 270))
            gap_len = int(np.clip(
                rng.normal(bins_per_gap, max(1, bins_per_gap * 0.3)), 3, 72
            ))
            gap_end = min(288, gap_start + gap_len)
            cgm_mask[gap_start:gap_end] = False
            dm.cgm_gap_blocks.append((gap_start, gap_end))

    # Compression artifact during exercise
    if exercise_active_bins is not None and exercise_active_bins.any():
        if rng.random() < profile.compression_artifact_p:
            ex_bins = np.where(exercise_active_bins)[0]
            start = max(0, int(rng.choice(ex_bins)) - 6)
            comp_len = int(rng.integers(2, 8))
            comp_end = min(288, start + comp_len)
            cgm_mask[start:comp_end] = False
            dm.cgm_gap_blocks.append((start, comp_end))

    # Rare single-bin dropouts
    cgm_mask[rng.random(288) < 0.002] = False

    dm.cgm_mask = cgm_mask


# ---------------------------------------------------------------------------
# Application functions — apply DayMissingness to signal arrays
# ---------------------------------------------------------------------------

def apply_cgm_missingness(cgm_array: np.ndarray, dm: DayMissingness) -> np.ndarray:
    """Apply CGM block gaps to a (288,) array. Returns array with NaN for missing."""
    out = cgm_array.astype(float).copy()
    out[~dm.cgm_mask] = np.nan
    return out.astype(np.float32)


def apply_hr_missingness(hr_array: np.ndarray, dm: DayMissingness,
                          rng: np.random.Generator) -> np.ndarray:
    """Apply watch-based missingness to (24,) HR array."""
    out = hr_array.astype(float).copy()
    out[~dm.watch_hourly_mask] = np.nan
    # Rare single-hour dropout even when worn
    out[rng.random(24) < 0.005] = np.nan
    return out


def apply_energy_missingness(
    basal: np.ndarray,
    active: np.ndarray,
    dm: DayMissingness,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply watch-based missingness to (24,) energy arrays."""
    b = basal.astype(float).copy()
    a = active.astype(float).copy()
    b[~dm.watch_hourly_mask] = np.nan
    a[~dm.watch_hourly_mask] = np.nan
    return b, a


def apply_exercise_missingness(
    exercise_minutes: float,
    exercise_hour: int,
    dm: DayMissingness,
) -> np.ndarray:
    """Return (24,) exercise array with missingness applied.

    NaN for hours where watch is off (can't observe anything).
    Actual minutes only at exercise_hour if captured by watch+app.
    """
    out = np.zeros(24, dtype=float)
    if exercise_minutes > 0 and dm.exercise_captured:
        if exercise_hour is not None and 0 <= exercise_hour < 24:
            out[exercise_hour] = exercise_minutes
    # Hours where watch is off → NaN (unknown, not zero)
    result = np.where(dm.watch_hourly_mask, out, np.nan)
    return result


# ---------------------------------------------------------------------------
# Non-block-structured streams (mood and menstrual)
# ---------------------------------------------------------------------------

def effective_p(base_p: float, pattern: EngagementPattern, day_index: int,
                is_weekend: bool, rng: np.random.Generator) -> float:
    """Scale a missingness probability by engagement pattern."""
    mult = engagement_multiplier(pattern, day_index, rng)
    disengagement = float(np.clip(2.0 - mult, 0.1, 3.0))
    p = base_p * disengagement
    if is_weekend:
        p *= 1.15
    return float(np.clip(p, 0, 0.95))


def mood_event_count(profile: MissingnessProfile, day_index: int,
                     is_weekend: bool, rng: np.random.Generator) -> int:
    """How many mood events the patient logs today."""
    mult = engagement_multiplier(profile.engagement_pattern, day_index, rng)
    effective_mean = max(0.0, profile.mood_events_per_day_mean * mult)
    if is_weekend:
        effective_mean *= 0.85
    return int(rng.poisson(effective_mean))


def menstrual_is_missing(profile: MissingnessProfile, day_index: int,
                          is_weekend: bool, rng: np.random.Generator) -> bool:
    """True if no menstrual log for this day."""
    p = effective_p(profile.menstrual_missing_p_per_day, profile.engagement_pattern,
                    day_index, is_weekend, rng)
    return bool(rng.random() < p)


# ---------------------------------------------------------------------------
# Backward-compatibility aliases
# ---------------------------------------------------------------------------

def watch_worn_mask(profile: MissingnessProfile, day_index: int,
                    is_weekend: bool, rng: np.random.Generator) -> np.ndarray:
    """Legacy wrapper — generates DayMissingness and returns watch mask."""
    ex_bins = np.zeros(288, dtype=bool)
    dm = generate_day_missingness(profile, day_index, is_weekend, 17, ex_bins, rng)
    return dm.watch_hourly_mask


def sleep_is_missing(profile: MissingnessProfile, day_index: int,
                     is_weekend: bool, rng: np.random.Generator) -> bool:
    """Legacy wrapper — now derived from watch overnight status."""
    ex_bins = np.zeros(288, dtype=bool)
    dm = generate_day_missingness(profile, day_index, is_weekend, 17, ex_bins, rng)
    return dm.sleep_missing

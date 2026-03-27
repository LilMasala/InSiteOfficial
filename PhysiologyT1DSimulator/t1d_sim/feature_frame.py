"""Typed feature-frame records aligned with the InSite Firestore schema."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class FeatureFrameHourly:
    """Python mirror of InSite's hourly feature frame payload."""

    hour_start_utc: datetime

    bg_avg: float | None = None
    bg_tir: float | None = None
    bg_percent_low: float | None = None
    bg_percent_high: float | None = None
    bg_u_roc: float | None = None

    bg_delta_avg7h: float | None = None
    bg_z_avg7h: float | None = None

    hr_mean: float | None = None
    hr_delta7h: float | None = None
    hr_z7h: float | None = None
    rhr_daily: float | None = None

    kcal_active: float | None = None
    kcal_active_last3h: float | None = None
    kcal_active_last6h: float | None = None
    kcal_active_delta7h: float | None = None
    kcal_active_z7h: float | None = None

    sleep_prev_total_min: float | None = None
    sleep_debt_7d_min: float | None = None
    minutes_since_wake: int | None = None

    ex_move_min: float | None = None
    ex_exercise_min: float | None = None
    ex_min_last3h: float | None = None
    ex_hours_since: float | None = None

    days_since_period_start: int | None = None
    cycle_follicular: int | None = None
    cycle_ovulation: int | None = None
    cycle_luteal: int | None = None

    days_since_site_change: int | None = None
    site_loc_current: str | None = None
    site_loc_same_as_last: int | None = None

    mood_valence: float | None = None
    mood_arousal: float | None = None
    mood_quad_pos_pos: int | None = None
    mood_quad_pos_neg: int | None = None
    mood_quad_neg_pos: int | None = None
    mood_quad_neg_neg: int | None = None
    mood_hours_since: float | None = None

    # 0 = Sunday … 6 = Saturday (matches Chamelia / InSite convention)
    day_of_week: int | None = None

    def to_signal_dict(self) -> dict[str, Any]:
        """Return the Chamelia request payload view of this frame."""
        signals: dict[str, Any] = {}

        def put(key: str, value: float | int | str | None) -> None:
            if value is None:
                return
            signals[key] = value

        def put_percent(key: str, value: float | None) -> None:
            if value is None:
                return
            signals[key] = max(0.0, min(1.0, value / 100.0))

        put("bg_avg", self.bg_avg)
        put_percent("tir_7d", self.bg_tir)
        put_percent("pct_low_7d", self.bg_percent_low)
        put_percent("pct_high_7d", self.bg_percent_high)
        put("uroc", self.bg_u_roc)
        put("bg_delta_7h", self.bg_delta_avg7h)
        put("bg_z_7h", self.bg_z_avg7h)

        put("heart_rate", self.hr_mean)
        put("hr_delta_7h", self.hr_delta7h)
        put("hr_z_7h", self.hr_z7h)
        put("resting_hr", self.rhr_daily)

        put("active_kcal", self.kcal_active)
        put("kcal_last3h", self.kcal_active_last3h)
        put("kcal_last6h", self.kcal_active_last6h)
        put("active_kcal_delta7h", self.kcal_active_delta7h)
        put("active_kcal_z7h", self.kcal_active_z7h)

        put("sleep_total_min", self.sleep_prev_total_min)
        put("sleep_debt_7d", self.sleep_debt_7d_min)
        put("mins_since_wake", self.minutes_since_wake)

        put("move_mins", self.ex_move_min)
        put("exercise_mins", self.ex_exercise_min)
        put("exercise_last3h", self.ex_min_last3h)
        put("hours_since_exercise", self.ex_hours_since)

        put("cycle_day", self.days_since_period_start)
        put("cycle_phase_follicular", self.cycle_follicular)
        put("cycle_phase_luteal", self.cycle_luteal)
        put("cycle_phase_ovulation", self.cycle_ovulation)
        put("cycle_phase_menstrual", self._menstrual_phase_flag())

        put("days_since_change", self.days_since_site_change)
        put("site_location", self.site_loc_current)
        put("site_repeat", self.site_loc_same_as_last)

        put("valence", self.mood_valence)
        put("arousal", self.mood_arousal)
        put("quad_pos_pos", self.mood_quad_pos_pos)
        put("quad_pos_neg", self.mood_quad_pos_neg)
        put("quad_neg_pos", self.mood_quad_neg_pos)
        put("quad_neg_neg", self.mood_quad_neg_neg)
        put("hours_since_mood", self.mood_hours_since)
        put("stress_acute", self._stress_acute())
        put("day_of_week", self.day_of_week)
        return signals

    def to_firebase_dict(self) -> dict[str, Any]:
        """Return the Firestore payload shape used by InSite."""
        payload = {"hourStartUtc": self.hour_start_utc.strftime("%Y-%m-%dT%H:00:00Z")}
        for key, value in asdict(self).items():
            if key == "hour_start_utc" or value is None:
                continue
            payload[self._firebase_key(key)] = value
        return payload

    def _menstrual_phase_flag(self) -> int | None:
        if self.days_since_period_start is None or self.days_since_period_start < 0:
            return None
        return 1 if 0 <= self.days_since_period_start <= 4 else 0

    def _stress_acute(self) -> float | None:
        if self.mood_arousal is None:
            return None
        if self.mood_arousal <= 0.6:
            return 0.0
        if self.mood_valence is not None and self.mood_valence < 0:
            return 1.0
        return 0.5

    @staticmethod
    def _firebase_key(attr_name: str) -> str:
        overrides = {
            "bg_percent_low": "bg_percentLow",
            "bg_percent_high": "bg_percentHigh",
            "bg_u_roc": "bg_uRoc",
            "bg_delta_avg7h": "bg_deltaAvg7h",
            "bg_z_avg7h": "bg_zAvg7h",
            "hr_delta7h": "hr_delta7h",
            "hr_z7h": "hr_z7h",
            "rhr_daily": "rhr_daily",
            "kcal_active_last3h": "kcal_active_last3h",
            "kcal_active_last6h": "kcal_active_last6h",
            "kcal_active_delta7h": "kcal_active_delta7h",
            "kcal_active_z7h": "kcal_active_z7h",
            "sleep_prev_total_min": "sleep_prev_total_min",
            "sleep_debt_7d_min": "sleep_debt_7d_min",
            "minutes_since_wake": "minutes_since_wake",
            "ex_move_min": "ex_move_min",
            "ex_exercise_min": "ex_exercise_min",
            "ex_min_last3h": "ex_min_last3h",
            "ex_hours_since": "ex_hours_since",
            "days_since_period_start": "days_since_period_start",
            "cycle_follicular": "cycle_follicular",
            "cycle_ovulation": "cycle_ovulation",
            "cycle_luteal": "cycle_luteal",
            "days_since_site_change": "days_since_site_change",
            "site_loc_current": "site_loc_current",
            "site_loc_same_as_last": "site_loc_same_as_last",
            "mood_valence": "mood_valence",
            "mood_arousal": "mood_arousal",
            "mood_quad_pos_pos": "mood_quad_posPos",
            "mood_quad_pos_neg": "mood_quad_posNeg",
            "mood_quad_neg_pos": "mood_quad_negPos",
            "mood_quad_neg_neg": "mood_quad_negNeg",
            "mood_hours_since": "mood_hours_since",
        }
        return overrides.get(attr_name, attr_name)

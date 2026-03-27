from datetime import datetime, timezone

import numpy as np
import pandas as pd

from t1d_sim.features import build_feature_frames
from t1d_sim.physiology import MealProfile, simulate_day_cgm


def test_rolling_glycaemic_variability_metrics_nan_safe():
    hours = pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC")
    avg_bg = [100 + (i % 6) * 15 for i in range(30)]
    avg_bg[4] = np.nan
    avg_bg[12] = np.nan
    df = pd.DataFrame(
        {
            "user_id": ["u1"] * 30,
            "hour_utc": hours.strftime("%Y-%m-%dT%H:00:00Z"),
            "avg_bg": avg_bg,
            "heart_rate": np.linspace(60, 70, 30),
            "active_energy": np.ones(30),
            "exercise_minutes": np.zeros(30),
        }
    )
    out = build_feature_frames(df)
    assert {"lbgi_24h", "hbgi_24h", "mage_24h"}.issubset(out.columns)
    last = out.iloc[-1]
    assert np.isfinite(last["lbgi_24h"])
    assert np.isfinite(last["hbgi_24h"])
    assert np.isfinite(last["mage_24h"])


def test_meal_profiles_produce_different_postprandial_shapes():
    base = {"k1": 1.0, "k2": 1.0, "EGP0": 1.0}
    meal_time = datetime(2025, 1, 1, 8, 0, tzinfo=timezone.utc)
    fast = simulate_day_cgm(base, base, [(meal_time, 60, MealProfile.FAST)], seed=11)
    slow = simulate_day_cgm(base, base, [(meal_time, 60, MealProfile.SLOW)], seed=11)

    # Fast meals should rise sooner; slow meals should retain more late tail.
    meal_idx = (meal_time.hour * 60 + meal_time.minute) // 5
    early_fast = np.nanmean(fast[meal_idx + 6:meal_idx + 18])
    early_slow = np.nanmean(slow[meal_idx + 6:meal_idx + 18])
    assert early_fast > early_slow

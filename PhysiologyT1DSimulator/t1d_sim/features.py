"""Feature frame builder."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _bg_risk_indices(values: np.ndarray) -> tuple[float, float]:
    """Compute LBGI and HBGI from BG readings with NaN-safe handling."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if arr.size == 0:
        return np.nan, np.nan
    f = 1.509 * (np.log(arr) ** 1.084 - 5.381)
    risk = 10.0 * (f ** 2)
    low = risk[f < 0]
    high = risk[f > 0]
    lbgi = float(np.mean(low)) if low.size else 0.0
    hbgi = float(np.mean(high)) if high.size else 0.0
    return lbgi, hbgi


def _mage(values: np.ndarray) -> float:
    """Approximate MAGE by averaging excursion amplitudes beyond 1 SD."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() < 3:
        return np.nan
    x = arr[finite]
    thresh = float(np.nanstd(x))
    if not np.isfinite(thresh) or thresh <= 0.0:
        return 0.0

    diff = np.diff(x)
    sign = np.sign(diff)
    if sign.size == 0:
        return 0.0
    # Carry forward zero-derivative signs to stabilize turning point detection.
    for i in range(1, sign.size):
        if sign[i] == 0.0:
            sign[i] = sign[i - 1]
    turns = np.where(np.diff(sign) != 0)[0] + 1
    if turns.size < 2:
        return 0.0

    excursions = np.abs(np.diff(x[turns]))
    major = excursions[excursions >= thresh]
    return float(np.mean(major)) if major.size else 0.0


def _rolling_metrics(series: pd.Series, window: int = 24) -> pd.DataFrame:
    """Rolling 24h glycaemic variability metrics for a single user series."""
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    lbgi = np.full(n, np.nan)
    hbgi = np.full(n, np.nan)
    mage = np.full(n, np.nan)
    for i in range(n):
        start = max(0, i - window + 1)
        w = vals[start:i + 1]
        lbgi[i], hbgi[i] = _bg_risk_indices(w)
        mage[i] = _mage(w)
    return pd.DataFrame({"lbgi_24h": lbgi, "hbgi_24h": hbgi, "mage_24h": mage}, index=series.index)


def build_feature_frames(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly feature frames from merged raw hourly records."""
    df = raw.sort_values(["user_id", "hour_utc"]).copy()
    g = df.groupby("user_id", group_keys=False)
    df["bg_delta_avg_7h"] = df["avg_bg"] - g["avg_bg"].shift(24).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    df["bg_z_avg_7h"] = (df["avg_bg"] - g["avg_bg"].rolling(24*7, min_periods=24).mean().reset_index(level=0, drop=True)) / g["avg_bg"].rolling(24*7, min_periods=24).std().reset_index(level=0, drop=True)
    df["hr_delta_7h"] = df["heart_rate"] - g["heart_rate"].shift(24).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    df["hr_z_7h"] = (df["heart_rate"] - g["heart_rate"].rolling(24*7, min_periods=24).mean().reset_index(level=0, drop=True)) / g["heart_rate"].rolling(24*7, min_periods=24).std().reset_index(level=0, drop=True)
    df["kcal_active_last3h"] = g["active_energy"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    df["kcal_active_last6h"] = g["active_energy"].rolling(6, min_periods=1).sum().reset_index(level=0, drop=True)
    df["ex_min_last3h"] = g["exercise_minutes"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)

    metrics = g["avg_bg"].apply(_rolling_metrics).reset_index(level=0, drop=True)
    df[["lbgi_24h", "hbgi_24h", "mage_24h"]] = metrics[["lbgi_24h", "hbgi_24h", "mage_24h"]]
    return df

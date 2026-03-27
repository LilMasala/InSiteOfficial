import pandas as pd
from t1d_sim.features import build_feature_frames


def test_features_null_safe():
    df = pd.DataFrame({
        "user_id": ["u"]*5,
        "hour_utc": [f"2025-01-01T0{i}:00:00Z" for i in range(5)],
        "avg_bg": [100,101,102,103,104],
        "heart_rate": [60,61,62,63,64],
        "active_energy": [1,2,3,4,5],
        "exercise_minutes": [0,1,0,1,0],
    })
    out = build_feature_frames(df)
    assert "kcal_active_last3h" in out

from datetime import datetime, timezone
from t1d_sim.population import sample_population
from t1d_sim.behavior import generate_day_behavior


def test_behavior_shapes():
    p = sample_population(1, seed=3)[0]
    out = generate_day_behavior(p, datetime(2025,1,1,tzinfo=timezone.utc), 0, (0.0, 0.0))
    assert out["sleep_minutes"] >= 240
    assert len(out["meals"]) >= 1

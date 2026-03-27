from datetime import datetime, timezone
from t1d_sim.population import sample_population
from t1d_sim.patient import simulate_patient


def test_physiology_shape_range():
    p = sample_population(1, seed=6)[0]
    payload = simulate_patient(p, 3, datetime(2025,1,1,tzinfo=timezone.utc))
    assert len(payload["bg_hourly"]) == 72
    vals = [r[4] for r in payload["bg_hourly"] if r[4] is not None]
    assert len(vals) > 0
    assert min(vals) > 40

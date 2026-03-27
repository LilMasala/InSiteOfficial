import numpy as np
from t1d_sim.missingness import DayMissingness
from t1d_sim.observation import observe_cgm


def test_observation_noise_nan():
    bg = np.full(288, 120.0)
    dm = DayMissingness()  # default: all present
    obs = observe_cgm(bg, dm, patient_seed=42, day_index=1)
    assert obs.shape == (288,)
    assert np.isnan(obs).mean() < 0.2

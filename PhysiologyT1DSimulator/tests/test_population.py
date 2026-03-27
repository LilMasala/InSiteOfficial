from t1d_sim.population import sample_population


def test_population_sampled():
    ps = sample_population(5, seed=1)
    assert len(ps) == 5
    assert all(p.patient_id.startswith("sim_") for p in ps)

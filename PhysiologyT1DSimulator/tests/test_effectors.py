from t1d_sim.behavior import ContextState
from t1d_sim.physiology import apply_context_effectors


def test_effectors_directionality():
    base = {"k1": 1.0, "k2": 1.0, "EGP0": 1.0}
    luteal = ContextState("luteal", 10, 1.0, 450, 0.9, 72, 0.0, 0.1, 0.5, 0.1, 0, 0, False, 0, False)
    ex = ContextState(None, -1, 0.0, 450, 0.9, 1, 1.0, 0.1, 0.5, 0.1, 0, 0, False, 0, False)
    sleep_debt = ContextState(None, -1, 0.0, 300, 0.7, 72, 0.0, 0.1, 0.5, 0.1, 0, 0, False, 0, False)
    assert apply_context_effectors(base, luteal)["k1"] < 1.0
    assert apply_context_effectors(base, ex)["k1"] > 1.0
    assert apply_context_effectors(base, sleep_debt)["EGP0"] > 1.0


def test_sleep_deficit_magnitude():
    """5h sleep (300 min) should raise EGP by roughly 10-20%."""
    base = {"k1": 1.0, "k2": 1.0, "EGP0": 1.0}
    ctx_5h = ContextState(None, -1, 0.0, 300, 0.82, 72, 0.0, 0.1, 1.0, 0.1, 0, 0, False, 0, False)
    result = apply_context_effectors(base, ctx_5h)
    egp_ratio = result["EGP0"] / base["EGP0"]
    assert 1.08 <= egp_ratio <= 1.25, f"EGP ratio {egp_ratio:.3f} outside expected 1.08-1.25 range"


def test_stress_tiers():
    """Verify stress tiers produce distinct magnitudes."""
    base = {"k1": 1.0, "k2": 1.0, "EGP0": 1.0}

    def make_stress(s):
        return ContextState(None, -1, 0.0, 450, 0.85, 72, 0.0, s, 1.0, 0.0, 0, 0, False, 0, False)

    mild = apply_context_effectors(base, make_stress(0.2))["EGP0"]
    acute = apply_context_effectors(base, make_stress(0.5))["EGP0"]
    severe = apply_context_effectors(base, make_stress(0.8))["EGP0"]
    assert mild < acute < severe, "Stress tiers should produce increasing EGP"
    assert severe < 1.40, f"Severe stress EGP {severe:.3f} exceeds pharmacological cap"

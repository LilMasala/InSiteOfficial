"""Tests for cold-start twin calibration path."""
from __future__ import annotations
from argparse import Namespace

import pytest
import numpy as np

from scripts.create_sim_patient import build_preferences
from t1d_sim.questionnaire import ColdStartTargets
from t1d_sim.population import sample_population


def _sample_cfg(seed: int = 1):
    cfgs = sample_population(5, seed=seed)
    return cfgs[0]


class TestColdStartTargets:
    def test_defaults_are_none(self):
        t = ColdStartTargets()
        assert t.recent_tir is None
        assert t.recent_pct_low is None
        assert t.recent_pct_high is None
        assert t.window_days == 14

    def test_to_chamelia_targets_empty_when_none(self):
        d = ColdStartTargets().to_chamelia_targets()
        assert d == {}

    def test_to_chamelia_targets_clips_values(self):
        d = ColdStartTargets(recent_tir=1.5, recent_pct_low=-0.1).to_chamelia_targets()
        assert d["recent_tir"] == 1.0
        assert d["recent_pct_low"] == 0.0

    def test_to_chamelia_targets_only_present_keys(self):
        d = ColdStartTargets(recent_tir=0.70).to_chamelia_targets()
        assert "recent_tir" in d
        assert "recent_pct_low" not in d
        assert "recent_pct_high" not in d

    def test_from_patient_config_returns_plausible_values(self):
        cfg = _sample_cfg()
        t = ColdStartTargets.from_patient_config(cfg)
        assert t.recent_tir is not None
        assert 0.0 <= t.recent_tir <= 1.0
        assert t.recent_pct_low is not None
        assert 0.0 <= t.recent_pct_low <= 1.0
        assert t.recent_pct_high is not None
        assert 0.0 <= t.recent_pct_high <= 1.0

    def test_from_patient_config_sum_constraint(self):
        cfg = _sample_cfg()
        t = ColdStartTargets.from_patient_config(cfg)
        d = t.to_chamelia_targets()
        total = d.get("recent_tir", 0) + d.get("recent_pct_low", 0) + d.get("recent_pct_high", 0)
        # With noise, sum may slightly exceed 1 but shouldn't be wildly wrong
        assert total <= 1.25

    def test_from_patient_config_deterministic(self):
        cfg = _sample_cfg(seed=42)
        t1 = ColdStartTargets.from_patient_config(cfg)
        t2 = ColdStartTargets.from_patient_config(cfg)
        assert t1.recent_tir == t2.recent_tir
        assert t1.recent_pct_low == t2.recent_pct_low

    def test_high_isf_patient_has_higher_tir(self):
        cfgs = sample_population(20, seed=99)
        high_isf = [c for c in cfgs if c.isf_multiplier > 1.1]
        low_isf  = [c for c in cfgs if c.isf_multiplier < 0.9]
        if high_isf and low_isf:
            t_high = np.mean([ColdStartTargets.from_patient_config(c).recent_tir for c in high_isf])
            t_low  = np.mean([ColdStartTargets.from_patient_config(c).recent_tir for c in low_isf])
            assert t_high > t_low

    def test_window_days_preserved(self):
        t = ColdStartTargets(recent_tir=0.65, window_days=30)
        assert t.window_days == 30


class TestColdStartIntegration:
    """Integration tests: ColdStartTargets → chamelia_targets dict → can be passed to server."""

    def test_chamelia_targets_keys_are_strings(self):
        t = ColdStartTargets(recent_tir=0.70, recent_pct_low=0.05, recent_pct_high=0.25)
        d = t.to_chamelia_targets()
        for key in d:
            assert isinstance(key, str)

    def test_chamelia_targets_values_are_floats(self):
        t = ColdStartTargets(recent_tir=0.70, recent_pct_low=0.05, recent_pct_high=0.25)
        d = t.to_chamelia_targets()
        for v in d.values():
            assert isinstance(v, float)

    def test_full_roundtrip_all_fields(self):
        t = ColdStartTargets(recent_tir=0.72, recent_pct_low=0.04, recent_pct_high=0.24, window_days=30)
        d = t.to_chamelia_targets()
        assert abs(d["recent_tir"] - 0.72) < 1e-9
        assert abs(d["recent_pct_low"] - 0.04) < 1e-9
        assert abs(d["recent_pct_high"] - 0.24) < 1e-9
        assert "window_days" not in d   # metadata field, not sent to Chamelia

    def test_build_preferences_includes_synthetic_targets_by_default(self):
        cfg = _sample_cfg(seed=7)
        args = Namespace(questionnaire=None, seed=7, coldstart_targets="synthetic")
        prefs = build_preferences(cfg, args)
        assert "calibration_targets" in prefs
        assert "recent_tir" in prefs["calibration_targets"]

    def test_build_preferences_can_disable_synthetic_targets(self):
        cfg = _sample_cfg(seed=11)
        args = Namespace(questionnaire=None, seed=11, coldstart_targets="none")
        prefs = build_preferences(cfg, args)
        assert "calibration_targets" not in prefs

from datetime import datetime, timezone
import math

from scripts.create_sim_patient import SimulationState, build_run_report
from t1d_sim.feature_frame import FeatureFrameHourly
from t1d_sim.population import sample_population
from t1d_sim.simulate import SimulationCarryState, simulate_day
from t1d_sim.therapy import make_default_schedule


def test_feature_frame_signal_payload_normalizes_percentages():
    frame = FeatureFrameHourly(
        hour_start_utc=datetime(2026, 1, 1, 23, tzinfo=timezone.utc),
        bg_tir=72.5,
        bg_percent_low=4.0,
        bg_percent_high=23.5,
        mood_valence=-0.2,
        mood_arousal=0.8,
        days_since_period_start=2,
    )

    payload = frame.to_signal_dict()

    assert payload["tir_7d"] == 0.725
    assert payload["pct_low_7d"] == 0.04
    assert payload["pct_high_7d"] == 0.235
    assert payload["stress_acute"] == 1.0
    assert payload["cycle_phase_menstrual"] == 1


def test_simulate_day_produces_app_shaped_records():
    cfg = sample_population(1, seed=7)[0]
    schedule = cfg.therapy_schedule or cfg.baseline_therapy_schedule or make_default_schedule(cfg)

    result = simulate_day(
        cfg=cfg,
        schedule=schedule,
        date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        rng_seed=7,
        day_index=0,
        carry_state=SimulationCarryState(),
    )

    assert len(result.feature_frames) == 24
    assert result.decision_frame.hour_start_utc.hour == 23
    assert len(result.bg_hourly) == 24
    assert len(result.therapy_hourly) == 24
    assert result.site_daily["dateUtc"] == "2026-01-01"
    assert result.hr_daily_average["dateUtc"] == "2026-01-01"

    signals = result.decision_frame.to_signal_dict()
    assert "bg_avg" in signals
    assert "tir_7d" in signals


def test_simulate_day_uses_rolling_daily_glycemic_metrics_for_decision_frame():
    cfg = sample_population(1, seed=9)[0]
    schedule = cfg.therapy_schedule or cfg.baseline_therapy_schedule or make_default_schedule(cfg)
    carry = SimulationCarryState(
        tir_daily_7d=[42.0, 48.0, 55.0],
        pct_low_daily_7d=[3.0, 4.0, 5.0],
        pct_high_daily_7d=[55.0, 48.0, 40.0],
    )

    result = simulate_day(
        cfg=cfg,
        schedule=schedule,
        date=datetime(2026, 1, 2, tzinfo=timezone.utc),
        rng_seed=9,
        day_index=1,
        carry_state=carry,
    )

    decision = result.decision_frame
    assert decision.bg_tir is not None
    assert decision.bg_percent_low is not None
    assert decision.bg_percent_high is not None
    assert math.isclose(decision.bg_tir, sum(result.carry_state.tir_daily_7d) / len(result.carry_state.tir_daily_7d))
    assert math.isclose(
        decision.bg_percent_low,
        sum(result.carry_state.pct_low_daily_7d) / len(result.carry_state.pct_low_daily_7d),
    )
    assert math.isclose(
        decision.bg_percent_high,
        sum(result.carry_state.pct_high_daily_7d) / len(result.carry_state.pct_high_daily_7d),
    )


def test_build_run_report_exposes_realized_metrics_and_trends():
    cfg = sample_population(1, seed=11)[0]
    schedule = cfg.therapy_schedule or cfg.baseline_therapy_schedule or make_default_schedule(cfg)
    sim_state = SimulationState(cfg, schedule)

    log_entries = [
        {
            "day": day,
            "date": datetime(2026, 1, day, tzinfo=timezone.utc).isoformat(),
            "bg_avg": 140.0 - day,
            "tir_7d": tir,
            "pct_low_7d": 0.0,
            "pct_high_7d": high,
            "realized_cost": high,
            "mood_valence": 0.1 * day,
            "mood_arousal": 0.2,
            "stress_acute": 0.0,
            "graduation_status": {
                "graduated": day >= 2,
                "belief_mode": "kalman",
                "jepa_weights_loaded": False,
                "belief_entropy": 0.21,
                "familiarity": 0.74,
                "concordance": 0.69,
                "calibration": 0.77,
                "trust_level": 0.63,
                "burnout_level": 0.18,
                "no_surface_streak": 2,
            },
            "recommendation": recommendation if day == 2 else None,
            "patient_response": "accept" if day == 2 else None,
            "schedule_changed": day == 2,
            "action_kind": "scheduled" if day == 2 else None,
            "action_level": 1 if day == 2 else None,
            "action_family": "parameter_adjustment" if day == 2 else None,
            "predicted_improvement": 0.08 if day == 2 else None,
            "confidence": 0.78 if day == 2 else None,
            "confidence_breakdown": {
                "familiarity": 0.74,
                "concordance": 0.69,
                "calibration": 0.77,
                "effect_support": 0.81,
                "selection_penalty": 1.0,
                "final_confidence": 0.78,
            } if day == 2 else None,
            "predicted_outcomes": {
                "delta_tir": 0.05,
                "delta_pct_low": -0.002,
                "delta_pct_high": -0.04,
                "delta_bg_avg": -12.0,
                "delta_cost_mean": -0.03,
            } if day == 2 else None,
            "predicted_uncertainty": {
                "tir_std": 0.03,
                "pct_low_std": 0.004,
                "pct_high_std": 0.05,
                "bg_avg_std": 11.0,
                "cost_std": 0.08,
            } if day == 2 else None,
            "segment_summaries": [{"label": "6:00 AM-12:00 PM"}] if day == 2 else None,
            "structure_summaries": [] if day == 2 else None,
            "configurator_mode": "rules",
            "jepa_active": False,
            "decision_block_reason": None,
        }
        for day, tir, high, recommendation in [
            (1, 0.40, 0.40, None),
            (2, 0.42, 0.38, {"action": {"kind": "scheduled"}, "action_level": 1, "action_family": "parameter_adjustment"}),
            (3, 0.50, 0.30, None),
            (4, 0.56, 0.24, None),
            (5, 0.60, 0.20, None),
            (6, 0.62, 0.18, None),
        ]
    ]

    report = build_run_report(
        log_entries,
        sim_state,
        uid="local_test",
        email=None,
        namespace="unit-test",
        persona="athlete",
        run_id="run123",
        days_this_run=len(log_entries),
    )

    assert report["recommendation_count"] == 1
    assert report["accept_or_partial_rate"] == 1.0
    assert report["schedule_change_rate"] == 1.0
    assert report["recommendations_with_followup"] == 1
    assert report["realized_positive_outcome_rate"] == 1.0
    assert report["recommendation_success_rate"] == 1.0
    assert report["trend_series"]["tir_daily"][0] == 0.40
    assert len(report["trend_series"]["tir_rolling_14d"]) == len(log_entries)
    assert report["trend_series"]["mood_valence_daily"][0] == 0.1
    assert report["recommendation_timeline"][0]["predicted_outcomes"]["delta_tir"] == 0.05
    assert report["realized_outcome_timeline"][0]["pct_high_delta"] < 0.0
    assert report["competence_snapshot"]["familiarity"] == 0.74
    assert report["calibration_summary"]["paired_count"] == 1
    assert report["calibration_summary"]["tir_direction_match_rate"] == 1.0
    assert report["uncertainty_summary"]["count"] == 1
    assert report["uncertainty_summary"]["mean_confidence"] == 0.78
    assert report["recommendation_timeline"][0]["predicted_uncertainty"]["tir_std"] == 0.03

from t1d_sim.population import PatientConfig
from t1d_sim.therapy import (
    SegmentDelta,
    StructureEdit,
    TherapySchedule,
    TherapySegment,
    make_default_schedule,
)


def _sample_schedule() -> TherapySchedule:
    return TherapySchedule(
        segments=[
            TherapySegment("overnight", 0, 360, 48.0, 12.0, 0.75),
            TherapySegment("morning", 360, 720, 42.0, 10.0, 0.95),
            TherapySegment("afternoon", 720, 1080, 44.0, 11.0, 0.85),
            TherapySegment("evening", 1080, 1440, 46.0, 12.5, 0.72),
        ]
    )


def _sample_config() -> PatientConfig:
    return PatientConfig(
        patient_id="sim_test",
        persona="default",
        is_female=False,
        activity_propensity=0.5,
        sleep_regularity=0.5,
        sleep_total_min_mean=420.0,
        sleep_efficiency=0.82,
        sleep_schedule_offset_h=0.0,
        stress_reactivity=0.4,
        stress_baseline=0.15,
        cycle_sensitivity=0.0,
        mood_stability=0.5,
        meal_regularity=0.5,
        meal_schedule_offset_h=0.0,
        meal_size_multiplier=1.0,
        n_meals_per_day_mean=3.0,
        skips_breakfast_p=0.1,
        skips_lunch_p=0.1,
        luteal_meal_size_boost=0.0,
        luteal_mood_drop=0.0,
        exercise_intensity_mean=0.5,
        logging_quality_raw=0.7,
        fitness_level=0.5,
        base_rhr=60.0,
        isf_multiplier=1.0,
        cr_multiplier=1.0,
        basal_multiplier=1.0,
        base_patient_name="adult#001",
        seed=7,
        n_days=30,
        split="train",
    )


def test_apply_level1_action_changes_only_target_segment():
    schedule = _sample_schedule()
    edited = schedule.apply_level1_action([SegmentDelta("morning", isf_delta=0.10, basal_delta=-0.05)])

    assert edited.value_at_minute(7 * 60).isf == 42.0 * 1.10
    assert edited.value_at_minute(7 * 60).basal == 0.95 * 0.95
    assert edited.value_at_minute(13 * 60).isf == 44.0


def test_apply_structural_split_produces_valid_schedule():
    schedule = _sample_schedule()
    edited = schedule.apply_structural_proposal(
        StructureEdit(edit_type="split", target_segment_id="morning", split_at_minute=540)
    )

    assert edited.is_valid(min_duration_min=120)
    assert len(edited.segments) == 5


def test_apply_structural_merge_produces_valid_schedule():
    schedule = _sample_schedule()
    edited = schedule.apply_structural_proposal(
        StructureEdit(edit_type="merge", target_segment_id="afternoon", neighbor_segment_id="evening")
    )

    assert edited.is_valid(min_duration_min=120)
    assert len(edited.segments) == 3


def test_is_valid_rejects_gaps_and_short_segments():
    invalid_gap = TherapySchedule(
        segments=[
            TherapySegment("a", 0, 360, 45.0, 12.0, 0.8),
            TherapySegment("b", 420, 1440, 45.0, 12.0, 0.8),
        ]
    )
    invalid_short = TherapySchedule(
        segments=[
            TherapySegment("a", 0, 60, 45.0, 12.0, 0.8),
            TherapySegment("b", 60, 1440, 45.0, 12.0, 0.8),
        ]
    )

    assert not invalid_gap.is_valid(min_duration_min=120)
    assert not invalid_short.is_valid(min_duration_min=120)


def test_make_default_schedule_produces_valid_surface():
    schedule = make_default_schedule(_sample_config())

    assert schedule.is_valid(min_duration_min=120)
    assert len(schedule.segments) == 4

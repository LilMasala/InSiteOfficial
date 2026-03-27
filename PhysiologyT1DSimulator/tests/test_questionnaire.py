from __future__ import annotations

from t1d_sim.population import apply_cross_parameter_interactions
from t1d_sim.questionnaire import (
    Aggressiveness,
    BedtimeCategory,
    DietType,
    ExerciseFreq,
    ExerciseType,
    FitnessLevel,
    QuestionnaireAnswers,
    ScheduleType,
    SleepConsistency,
    StressLevel,
    TrustLevel,
    physical_priors_from_twins,
    population_defaults,
    questionnaire_to_agency_priors,
    questionnaire_to_patientconfig_priors,
    sample_twins_from_priors,
)


def test_all_none_answers_produce_valid_patient_configs():
    priors = questionnaire_to_patientconfig_priors(QuestionnaireAnswers())
    twins = sample_twins_from_priors(priors, n=10, seed=7)

    assert len(twins) == 10
    for twin in twins:
        assert twin.patient_id.startswith("real_twin_")
        assert twin.persona == "questionnaire_derived"
        assert twin.n_days == 180
        assert twin.split == "real"
        assert twin.missingness_profile is not None
        assert twin.agency_profile is not None
        assert twin.event_schedule is not None
        assert 0.0 <= twin.activity_propensity <= 1.0
        assert 240.0 <= twin.sleep_total_min_mean <= 570.0
        assert 0.55 <= twin.sleep_efficiency <= 0.97
        assert 45.0 <= twin.base_rhr <= 85.0
        assert 0.70 <= twin.isf_multiplier <= 1.35
        assert 0.70 <= twin.cr_multiplier <= 1.35
        assert 0.75 <= twin.basal_multiplier <= 1.25


def test_all_answered_questionnaire_produces_tighter_distributions_than_all_none():
    default_priors = questionnaire_to_patientconfig_priors(QuestionnaireAnswers())
    answered_priors = questionnaire_to_patientconfig_priors(
        QuestionnaireAnswers(
            bedtime_category=BedtimeCategory.NORMAL,
            sleep_consistency=SleepConsistency.VERY,
            exercise_freq=ExerciseFreq.DAILY,
            exercise_type=ExerciseType.CARDIO,
            fitness_level=FitnessLevel.VERY_FIT,
            diet_type=DietType.LOW_CARB,
            stress_level=StressLevel.RARELY,
            aggressiveness=Aggressiveness.WILLING,
            trust_level=TrustLevel.TRUSTING,
        )
    )

    keys = [
        "sleep_regularity",
        "activity_propensity",
        "fitness_level",
        "isf_multiplier",
        "stress_baseline",
    ]
    default_avg = sum(default_priors[key][1] for key in keys) / len(keys)
    answered_avg = sum(answered_priors[key][1] for key in keys) / len(keys)
    assert answered_avg < default_avg


def test_athlete_answers_raise_isf_mean():
    priors = questionnaire_to_patientconfig_priors(
        QuestionnaireAnswers(
            exercise_freq=ExerciseFreq.DAILY,
            exercise_type=ExerciseType.CARDIO,
            fitness_level=FitnessLevel.VERY_FIT,
            sleep_consistency=SleepConsistency.VERY,
        )
    )
    assert priors["isf_multiplier"][0] > 1.10


def test_night_owl_answers_raise_sleep_schedule_offset():
    priors = questionnaire_to_patientconfig_priors(
        QuestionnaireAnswers(bedtime_category=BedtimeCategory.VERY_LATE)
    )
    assert priors["sleep_schedule_offset_h"][0] > 1.0


def test_high_stress_answers_raise_stress_baseline():
    priors = questionnaire_to_patientconfig_priors(
        QuestionnaireAnswers(stress_level=StressLevel.OFTEN)
    )
    assert priors["stress_baseline"][0] > 0.40


def test_cross_interaction_high_exercise_plus_poor_sleep_elevates_stress_reactivity():
    priors = questionnaire_to_patientconfig_priors(
        QuestionnaireAnswers(
            exercise_freq=ExerciseFreq.DAILY,
            sleep_consistency=SleepConsistency.IRREGULAR,
        )
    )
    assert priors["stress_reactivity"][0] >= population_defaults()["stress_reactivity"][0] + 0.09


def test_very_variable_diet_widens_cr_distribution():
    priors = questionnaire_to_patientconfig_priors(
        QuestionnaireAnswers(diet_type=DietType.VERY_VARIABLE)
    )
    assert priors["cr_multiplier"][1] >= 0.15


def test_shift_work_reduces_sleep_regularity():
    baseline = questionnaire_to_patientconfig_priors(
        QuestionnaireAnswers(sleep_consistency=SleepConsistency.FAIRLY)
    )
    shifted = questionnaire_to_patientconfig_priors(
        QuestionnaireAnswers(
            sleep_consistency=SleepConsistency.FAIRLY,
            schedule_type=ScheduleType.SHIFT,
        )
    )
    assert shifted["sleep_regularity"][0] < baseline["sleep_regularity"][0]


def test_physical_priors_from_twins_output_is_chamelia_compatible():
    priors = questionnaire_to_patientconfig_priors(
        QuestionnaireAnswers(
            exercise_freq=ExerciseFreq.DAILY,
            exercise_type=ExerciseType.CARDIO,
        )
    )
    twins = sample_twins_from_priors(priors, n=50, seed=5)
    physical = physical_priors_from_twins(twins)

    assert "isf_multiplier" in physical
    assert "base_rhr" in physical
    assert all(isinstance(key, str) for key in physical)
    assert all(len(value) == 2 for value in physical.values())
    assert all(value[1] >= 0.02 for value in physical.values())


def test_sampled_twins_are_all_physiologically_valid():
    priors = questionnaire_to_patientconfig_priors(
        QuestionnaireAnswers(
            bedtime_category=BedtimeCategory.VERY_LATE,
            exercise_freq=ExerciseFreq.DAILY,
            exercise_type=ExerciseType.CARDIO,
            diet_type=DietType.HIGH_CARB,
            stress_level=StressLevel.OFTEN,
        )
    )
    twins = sample_twins_from_priors(priors, n=50, seed=42)

    for twin in twins:
        validated = apply_cross_parameter_interactions(twin)
        assert 0.0 <= validated.activity_propensity <= 1.0
        assert 0.0 <= validated.sleep_regularity <= 1.0
        assert 240.0 <= validated.sleep_total_min_mean <= 570.0
        assert 0.55 <= validated.sleep_efficiency <= 0.97
        assert 0.0 <= validated.stress_reactivity <= 1.0
        assert 0.0 <= validated.meal_regularity <= 1.0
        assert 0.70 <= validated.isf_multiplier <= 1.35
        assert 0.70 <= validated.cr_multiplier <= 1.35
        assert 0.75 <= validated.basal_multiplier <= 1.25


def test_questionnaire_to_agency_priors_maps_and_adjusts():
    priors = questionnaire_to_agency_priors(
        QuestionnaireAnswers(
            aggressiveness=Aggressiveness.VERY_CAUTIOUS,
            trust_level=TrustLevel.SKEPTICAL,
            stress_level=StressLevel.ALWAYS,
        )
    )

    assert priors["aggressiveness"][0] < 0.3
    assert priors["initial_trust"][0] <= 0.15
    assert "trust_growth_rate" in priors
    assert "revert_threshold" in priors

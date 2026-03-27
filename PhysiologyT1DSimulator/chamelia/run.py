"""World Runner — autonomous learning engine for Chamelia.

Replaces the manual train→run→evaluate→retrain cycle with a single
autonomous process. Manages a population of patients at different lifecycle
stages simultaneously, handles model training and retraining, operates the
meta-controller's full decision apparatus, and evolves each patient's therapy
mode through the unlock ladder.

Usage:
    python -m chamelia.run --n_patients 100 --days 180 --seed 42 \
        --outdb world.db --learning-mode hybrid --initial-zoo zoo_v2/zoo.pkl
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from chamelia.confidence import ConfidenceModule
from chamelia.models.aggregate import AggregateOutcomePredictor
from chamelia.meta_controller import (
    MetaController,
    MetaControllerState,
    ModelRegistryEntry,
)
from chamelia.optimizer import (
    ConstraintConfig,
    GridSearchOptimizer,
    ObjectiveWeights,
    RecommendationDecision,
    TherapyAction,
)
from chamelia.personality import (
    RecommendationBudget,
    RecommendationFraming,
    UserPersonality,
    select_framing,
)
from chamelia.shadow import ShadowModule
from chamelia.therapy_modes import (
    TherapyLevel,
    TherapyModeState,
    get_level_constraints,
)
from t1d_sim.behavior import generate_day_behavior
from t1d_sim.feedback import (
    EventSchedule,
    PatientState,
    YesterdayOutcome,
    apply_event_modifiers,
    compute_yesterday_outcome,
    get_active_events,
    update_patient_state,
)
from t1d_sim.missingness import generate_day_missingness, menstrual_is_missing, mood_event_count
from t1d_sim.observation import observe_cgm, synthesize_energy, synthesize_hr
from t1d_sim.physiology import apply_context_effectors, simulate_day_cgm
from t1d_sim.population import PatientConfig, sample_population
from t1d_sim.writers.sqlite_writer import SQLiteWriter


# ---------------------------------------------------------------------------
# Per-Patient Runtime State
# ---------------------------------------------------------------------------

@dataclass
class PatientRuntime:
    """Mutable per-patient state tracked by the world runner."""
    cfg: PatientConfig
    phase: int = 0  # 0 = observation, 1 = shadow, 2 = intervention
    day_index: int = 0

    # Therapy multipliers
    isf_mult: float = 1.0
    cr_mult: float = 1.0
    basal_mult: float = 1.0

    # Feedback state
    prev_mood: tuple[float, float] = (0.0, 0.0)
    last_outcome: YesterdayOutcome | None = None
    patient_state: PatientState | None = None
    drift_outcomes: list[YesterdayOutcome] = field(default_factory=list)
    event_schedule: EventSchedule | None = None

    # Chamelia state
    shadow_mod: ShadowModule = field(default_factory=lambda: ShadowModule(window_size=30))
    therapy_mode: TherapyModeState = field(default_factory=TherapyModeState)
    rec_budget: RecommendationBudget = field(default_factory=RecommendationBudget)
    personality: UserPersonality = field(default_factory=UserPersonality)
    trust: float = 0.5
    n_recommendations_surfaced: int = 0
    n_recommendations_accepted: int = 0
    mood_valence: float = 0.0

    # Lifecycle tracking
    graduation_day: int | None = None
    degraduation_count: int = 0

    # Shadow records and outcomes for training
    intervention_triples: int = 0

    # Rolling history for feature extraction (last 7 days)
    feature_history: list[dict[str, float]] = field(default_factory=list)
    tir_history: list[float] = field(default_factory=list)
    burnout_days: list[int] = field(default_factory=list)
    entered_shadow_day: int | None = None
    reached_intervention: bool = False

    @classmethod
    def from_config(cls, cfg: PatientConfig) -> "PatientRuntime":
        ps = PatientState.from_config(cfg)
        trust = cfg.agency_profile.initial_trust if cfg.agency_profile else 0.5
        personality = UserPersonality()
        if hasattr(cfg, "personality") and cfg.personality is not None:
            personality = cfg.personality
        return cls(
            cfg=cfg,
            isf_mult=cfg.isf_multiplier,
            cr_mult=cfg.cr_multiplier,
            basal_mult=cfg.basal_multiplier,
            patient_state=ps,
            event_schedule=cfg.event_schedule or EventSchedule(),
            trust=trust,
            personality=personality,
        )


# ---------------------------------------------------------------------------
# World Runner
# ---------------------------------------------------------------------------

class WorldRunner:
    """Autonomous learning engine managing population-level simulation."""

    MIN_OBSERVATION_DAYS = 21  # Minimum days before Phase 0→1 transition
    WEEKLY_EVAL_INTERVAL = 7
    RETRAIN_COOLDOWN = 7  # Minimum days between retrains

    def __init__(
        self,
        n_patients: int = 100,
        n_days: int = 180,
        seed: int = 42,
        learning_mode: str = "hybrid",
        initial_zoo_path: str | None = None,
        outdb_path: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.n_patients = n_patients
        self.n_days = n_days
        self.seed = seed
        self.learning_mode = learning_mode
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.outdb_path = outdb_path
        self.writer = SQLiteWriter(outdb_path) if outdb_path else None

        # Meta-controller
        self.meta = MetaController(learning_mode=learning_mode)

        # Shared components
        self.optimizer = GridSearchOptimizer()
        self.confidence_gate = ConfidenceModule()

        # Model registry
        self.active_model = None
        self._candidate_model = None
        self.model_version = 0

        # Load initial zoo if provided
        if initial_zoo_path:
            self._load_initial_zoo(initial_zoo_path)

        # Population
        self.patients: list[PatientRuntime] = []

        # Training data buffer — accumulates (features, actions, outcomes)
        # across all patients and days for model training
        self.training_buffer: list[dict[str, float]] = []

        # Tracking
        self.retrain_count = 0
        self.graduation_count = 0
        self.degraduation_count = 0
        self.total_shadow_records = 0
        self.recommendation_log_rows: list[tuple] = []
        self.bucketed_tir_history: list[dict[str, Any]] = []
        self.run_started_at = datetime.now(timezone.utc)
        self._burnout_definition = (
            "Patient counted as burned out after any day with mood_valence below "
            "their personality.burnout_threshold."
        )

    def _load_initial_zoo(self, path: str) -> None:
        """Load the initial model zoo from a pickle file."""
        try:
            with open(path, "rb") as f:
                zoo = pickle.load(f)
            # Register the first model
            if hasattr(zoo, "predict"):
                self.active_model = zoo
                self.model_version = 1
                entry = ModelRegistryEntry(
                    model_id="zoo_v1",
                    version="1",
                    architecture="aggregate",
                    target="multi",
                    status="active",
                )
                self.meta.register_model(entry)
                if self.verbose:
                    print(f"[world] Loaded initial model from {path}")
        except Exception as e:
            if self.verbose:
                print(f"[world] Could not load zoo from {path}: {e}")

    def _init_population(self) -> None:
        """Initialize the patient population."""
        configs = sample_population(self.n_patients, seed=self.seed)
        self.patients = [PatientRuntime.from_config(cfg) for cfg in configs]
        if self.writer:
            self.writer.conn.executemany(
                "INSERT OR REPLACE INTO patients VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        cfg.patient_id,
                        cfg.split,
                        cfg.logging_quality,
                        int(cfg.is_female),
                        cfg.activity_propensity,
                        cfg.sleep_regularity,
                        cfg.stress_reactivity,
                        cfg.cycle_sensitivity,
                        cfg.mood_stability,
                        cfg.meal_regularity,
                        cfg.base_rhr,
                        cfg.fitness_level,
                        cfg.base_patient_name,
                        cfg.isf_multiplier,
                        cfg.cr_multiplier,
                        cfg.basal_multiplier,
                    )
                    for cfg in configs
                ],
            )
            self.writer.conn.commit()
        if self.verbose:
            print(f"[world] Initialized {len(self.patients)} patients")

    # ------------------------------------------------------------------
    # Core Loop
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the world runner main loop.

        On each global day:
        1. All patients advance one simulated day
        2. Shadow/intervention patients run recommendation cycle
        3. Outcome data is collected
        4. Meta-controller runs daily check (weekly: full evaluation)
        """
        self._init_population()
        start_utc = datetime(2025, 1, 1, tzinfo=timezone.utc)

        prev_win_rate = 0.5

        for day in range(self.n_days):
            day_start = start_utc + timedelta(days=day)

            # --- Advance all patients ---
            new_data_rows = 0
            intervention_rows = 0

            for pt in self.patients:
                pt.day_index = day
                result = self._advance_patient(pt, day, day_start)
                new_data_rows += result.get("data_rows", 0)
                intervention_rows += result.get("intervention_rows", 0)

            # --- Phase transitions ---
            self._check_phase_transitions(day)

            # --- Meta-controller daily check ---
            population_win_rate = self._compute_population_win_rate()
            causal_delta = self._compute_causal_delta()
            intervention_count = sum(pt.intervention_triples for pt in self.patients)

            daily_result = self.meta.daily_check(
                day=day,
                new_data_rows=new_data_rows,
                intervention_data_rows=intervention_rows,
                rolling_win_rate=population_win_rate,
                prev_win_rate=prev_win_rate,
                intervention_triple_count=intervention_count,
                causal_delta=causal_delta,
            )

            # --- Force first retrain once enough data exists ---
            if day == self.MIN_OBSERVATION_DAYS and self.active_model is None:
                if self.verbose:
                    print(f"  [cold-start] Day {day}: triggering first model training "
                          f"({len(self.training_buffer)} rows buffered)")
                self._handle_retrain(day)

            # --- Handle retraining trigger ---
            if daily_result["retrain_triggered"]:
                self._handle_retrain(day)

            # --- Weekly full evaluation ---
            if day > 0 and day % self.WEEKLY_EVAL_INTERVAL == 0:
                self._weekly_evaluation(day, population_win_rate, causal_delta)

            prev_win_rate = population_win_rate

            # --- Periodic status output ---
            if self.verbose and (day % 30 == 0 or day == self.n_days - 1):
                self._print_status(day, population_win_rate, causal_delta)

        summary = self._build_summary()
        self._persist_summary(summary)
        if self.writer:
            self.writer.finalize()
        return summary

    # ------------------------------------------------------------------
    # Patient Simulation
    # ------------------------------------------------------------------

    def _advance_patient(
        self,
        pt: PatientRuntime,
        day: int,
        day_start: datetime,
    ) -> dict[str, int]:
        """Advance one patient by one simulated day."""
        cfg = pt.cfg
        result = {"data_rows": 0, "intervention_rows": 0}

        # Resolve life events
        active_events = get_active_events(pt.event_schedule, day)
        event_mods = apply_event_modifiers(active_events, day)

        # Generate behavior
        beh = generate_day_behavior(
            cfg, day_start, day, pt.prev_mood,
            yesterday=pt.last_outcome,
            event_modifiers=event_mods,
            patient_state=pt.patient_state,
        )
        ctx = beh["context"]
        pt.prev_mood = (ctx.mood_valence, ctx.mood_arousal)
        pt.mood_valence = ctx.mood_valence

        # Simulate physiology
        effective_base = {"k1": pt.isf_mult, "k2": pt.isf_mult, "EGP0": 1.0}
        ref_base = {"k1": 1.0, "k2": 1.0, "EGP0": 1.0}
        mod = apply_context_effectors(
            effective_base, ctx,
            patient_state=pt.patient_state,
            event_modifiers=event_mods,
        )
        true_bg = simulate_day_cgm(ref_base, mod, beh["meals"], cfg.seed * 10000 + day)

        # Synthesize HR and energy for feature extraction
        miss_rng = np.random.default_rng(cfg.seed * 10000 + day + 99)
        ex_bins = np.zeros(288, dtype=bool)
        dm = generate_day_missingness(
            cfg.missingness_profile, day, ctx.is_weekend,
            exercise_hour=None, exercise_active_bins=ex_bins, rng=miss_rng,
        ) if cfg.missingness_profile else None

        if dm is not None:
            hr_arr = synthesize_hr(
                cfg.base_rhr, ctx.stress, beh["exercise_minutes"],
                beh["exercise_intensity"], dm, day, cfg.seed,
            )
            _basal_e, active_e = synthesize_energy(
                cfg.base_rhr, cfg.basal_multiplier, beh["exercise_minutes"],
                beh["exercise_intensity"], dm, day, cfg.seed,
            )
        else:
            hr_arr = np.full(24, cfg.base_rhr)
            active_e = np.zeros(24)

        # Compute outcome
        outcome = compute_yesterday_outcome(true_bg, beh, pt.last_outcome)
        pt.last_outcome = outcome
        pt.drift_outcomes.append(outcome)
        pt.tir_history.append(outcome.tir)
        if pt.mood_valence < pt.personality.burnout_threshold:
            pt.burnout_days.append(day)

        # Build daily feature snapshot and store in rolling history
        daily_feats = self._build_daily_features(pt, ctx, beh, outcome, true_bg, hr_arr, active_e)
        pt.feature_history.append(daily_feats)
        if len(pt.feature_history) > 8:
            pt.feature_history = pt.feature_history[-8:]

        # Accumulate training data (all patients, all phases, every day)
        self._collect_training_row(pt, day, daily_feats, outcome)

        # Biweekly drift
        if len(pt.drift_outcomes) >= 14:
            pt.patient_state = update_patient_state(pt.patient_state, cfg, pt.drift_outcomes)
            pt.drift_outcomes = []

        result["data_rows"] = 1

        # --- Recommendation cycle for shadow/intervention patients ---
        if pt.phase >= 1 and self.active_model is not None:
            features = self._extract_features(pt)
            rec_result = self._run_patient_recommendation(pt, day, outcome, features)
            if rec_result.get("is_intervention"):
                result["intervention_rows"] = 1
                pt.intervention_triples += 1

            # Counterfactual simulation: if a different action was proposed,
            # re-run the physiology with the proposed multipliers to get the
            # "true" delta. This is the simulator's oracle advantage — in
            # production, a BG surrogate model would estimate this instead.
            proposed_action = rec_result.get("proposed_action")
            record_id = rec_result.get("record_id")
            if proposed_action is not None and record_id is not None:
                cf_base = {"k1": proposed_action[0], "k2": proposed_action[0], "EGP0": 1.0}
                cf_mod = apply_context_effectors(
                    cf_base, ctx,
                    patient_state=pt.patient_state,
                    event_modifiers=event_mods,
                )
                cf_bg = simulate_day_cgm(ref_base, cf_mod, beh["meals"], cfg.seed * 10000 + day + 7777)
                cf_valid = cf_bg[np.isfinite(cf_bg)]
                cf_tir = float(np.mean((cf_valid >= 70) & (cf_valid <= 180))) if cf_valid.size > 0 else 0.0
                cf_estimate = {
                    "tir": cf_tir,
                    "mean_bg": float(np.nanmean(cf_valid)) if cf_valid.size > 0 else 150.0,
                }
                # Update shadow record with counterfactual delta
                rec = pt.shadow_mod._records.get(record_id)
                if rec is not None:
                    rec.counterfactual_estimate = cf_estimate
                    rec.shadow_score_delta = cf_tir - outcome.tir

        # --- Therapy mode advancement ---
        pt.therapy_mode.advance_day()

        # --- Recommendation budget update ---
        # Budget always tracks mood so it's ready when Phase 2 starts,
        # but it does NOT gate Phase 1 shadow record generation.
        acceptance_rate = pt.shadow_mod.get_recent_acceptance_rate()
        last_succeeded = pt.shadow_mod.last_recommendation_succeeded()
        pt.rec_budget.daily_update(
            mood_valence=pt.mood_valence,
            acceptance_rate_7d=acceptance_rate,
            last_rec_succeeded=last_succeeded,
            personality=pt.personality,
        )

        # --- Celebration check (Phase 2 only — patient must be aware) ---
        if pt.phase == 2 and day > 7 and day % 7 == 0:
            self._check_celebration(pt, day, outcome)

        return result

    def _run_patient_recommendation(
        self,
        pt: PatientRuntime,
        day: int,
        outcome: YesterdayOutcome,
        features: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run the recommendation cycle for a single patient.

        Phase 1 (shadow): Generate and enrich shadow records unconditionally,
            every day, regardless of mood or budget. The patient doesn't know
            recommendations are being generated, so there is no psychological
            feedback. This lets the scorecard accumulate enough data to
            compute meaningful win rates and eventually graduate.

        Phase 2 (intervention): Budget-gated. Recommendations are surfaced
            to the patient, mood-aware cadence and burnout protection apply,
            and psychological feedback occurs.
        """
        result: dict[str, Any] = {"is_intervention": False}
        cfg = pt.cfg
        agency = cfg.agency_profile

        # Build baseline action
        baseline = TherapyAction(
            isf_multiplier=pt.isf_mult,
            cr_multiplier=pt.cr_mult,
            basal_multiplier=pt.basal_mult,
        )

        aggressiveness = agency.aggressiveness if agency else 0.5

        models = {self.active_model.model_id: self.active_model} if hasattr(self.active_model, "model_id") else {}
        if not models:
            return result

        # Feature vector: use extracted features or fallback to zeros
        n_features = len(self.active_model.feature_schema) if hasattr(self.active_model, "feature_schema") else 28
        if features is None or len(features) != n_features:
            features = np.zeros(n_features)

        # Determine optimizer constraints based on therapy level
        level_constraints = get_level_constraints(pt.therapy_mode.current_level)
        max_dev = max(level_constraints.max_change_magnitude, 0.10)

        # Conservativeness: higher in early Phase 2 (small wins strategy)
        conservativeness = 0.5
        if pt.phase == 2:
            days_in_intervention = day - (pt.graduation_day or day)
            if days_in_intervention < 30:
                conservativeness = min(0.9, 0.5 + 0.3 * (1.0 - days_in_intervention / 30.0))

        # Run optimizer
        try:
            rec_pkg = self.optimizer.search(
                features=features,
                baseline_action=baseline,
                models=models,
                weights=ObjectiveWeights(conservativeness=conservativeness),
                constraints=ConstraintConfig(
                    max_isf_deviation=max_dev,
                    max_cr_deviation=max_dev,
                    max_basal_deviation=max_dev * 0.67,
                ),
                aggressiveness=aggressiveness,
            )
        except Exception:
            return result

        proposed_action = rec_pkg.primary if rec_pkg.primary else baseline
        is_different = rec_pkg.primary is not None and rec_pkg.decision == RecommendationDecision.RECOMMEND

        # Phase 1 exploration: if the optimizer returned HOLD (model can't
        # differentiate actions), generate a random exploration action so we
        # can run counterfactual simulation and build shadow data. This is
        # epsilon-greedy bootstrapping — the patient never sees these.
        if not is_different:
            noise = self.rng.normal(0, 0.05, size=3)
            proposed_action = TherapyAction(
                isf_multiplier=float(np.clip(baseline.isf_multiplier + noise[0], 0.70, 1.35)),
                cr_multiplier=float(np.clip(baseline.cr_multiplier + noise[1], 0.70, 1.35)),
                basal_multiplier=float(np.clip(baseline.basal_multiplier + noise[2], 0.75, 1.25)),
            )
            is_different = True

        # --- Create shadow record (always, for both Phase 1 and Phase 2) ---
        proposed_arr = proposed_action.to_array().tolist()
        baseline_arr = baseline.to_array().tolist()

        # Build prediction dicts for the shadow record
        try:
            proposed_env = self.active_model.predict(
                features, action=np.array(proposed_arr),
            )
            baseline_env = self.active_model.predict(
                features, action=np.array(baseline_arr),
            )
            proposed_preds = {self.active_model.model_id: {
                "point": np.asarray(proposed_env.point).tolist(),
                "lower": np.asarray(proposed_env.lower).tolist(),
                "upper": np.asarray(proposed_env.upper).tolist(),
                "confidence": proposed_env.confidence,
            }}
            baseline_preds = {self.active_model.model_id: {
                "point": np.asarray(baseline_env.point).tolist(),
                "lower": np.asarray(baseline_env.lower).tolist(),
                "upper": np.asarray(baseline_env.upper).tolist(),
                "confidence": baseline_env.confidence,
            }}
            familiarity = proposed_env.confidence
        except Exception:
            proposed_preds = {}
            baseline_preds = {}
            familiarity = 0.5

        gate_passed = rec_pkg.decision == RecommendationDecision.RECOMMEND
        record = pt.shadow_mod.create_record(
            patient_id=cfg.patient_id,
            day_index=day,
            feature_snapshot={},
            proposed_action=proposed_arr,
            baseline_action=baseline_arr,
            proposed_predictions=proposed_preds,
            baseline_predictions=baseline_preds,
            gate_passed=gate_passed,
            gate_composite_score=rec_pkg.primary_confidence,
            gate_layer_scores={},
            gate_blocked_by=None if gate_passed else "optimizer",
            familiarity_score=familiarity,
            calibration_scores=pt.shadow_mod.get_calibration_scores(),
        )
        pt.shadow_mod.add_record(record)

        # --- Enrich shadow record with actual BG outcomes ---
        actual_tir = outcome.tir
        actual_pct_low = outcome.percent_low
        actual_pct_high = outcome.percent_high
        actual_mean_bg = outcome.mean_bg

        actual_outcomes = {
            "tir": actual_tir,
            "percent_low": actual_pct_low,
            "percent_high": actual_pct_high,
            "mean_bg": actual_mean_bg,
        }

        # Compute per-model accuracy for calibration scoring
        per_model_accuracy = self._compute_per_model_accuracy(
            proposed_preds, baseline_preds, actual_outcomes,
        )

        # Phase 1: all records are "not_shown" — patient doesn't see them
        # Phase 2: determined below by user decision
        if pt.phase == 1:
            self._flush_shadow_record(record)
            pt.shadow_mod.enrich_outcome(
                record.record_id, actual_outcomes, "not_shown",
                [pt.isf_mult, pt.cr_mult, pt.basal_mult],
            )
            pt.shadow_mod.evaluate_record(
                record.record_id,
                per_model_accuracy=per_model_accuracy,
            )
            self._flush_shadow_record(record)
            self.total_shadow_records += 1
            # Return proposed action for counterfactual simulation
            if is_different:
                result["proposed_action"] = proposed_arr
                result["record_id"] = record.record_id
            return result

        # --- Phase 2: budget-gated recommendation delivery ---
        budget_ok = pt.rec_budget.can_recommend()

        if rec_pkg.decision == RecommendationDecision.RECOMMEND and rec_pkg.primary and budget_ok:
            # Select framing
            framing = select_framing(
                n_prior_recs=pt.n_recommendations_surfaced,
                last_rec_accepted=pt.n_recommendations_accepted > 0,
                last_rec_succeeded=pt.shadow_mod.last_recommendation_succeeded(),
                tir_improved=outcome.tir > 0.65,
                personality=pt.personality,
            )
            rec_pkg.framing = framing.value

            # Surface recommendation
            pt.n_recommendations_surfaced += 1
            pt.rec_budget.consume()

            # Simulated user decision
            accepted = self._simulate_user_decision(pt, rec_pkg, day)
            user_action = "accept" if accepted else "reject"

            if accepted:
                pt.n_recommendations_accepted += 1
                action = rec_pkg.primary
                noise = 1.0
                if agency and agency.compliance_noise > 0:
                    noise = float(np.clip(
                        self.rng.normal(1.0, agency.compliance_noise),
                        0.8, 1.2,
                    ))
                pt.isf_mult = float(np.clip(action.isf_multiplier * noise, 0.70, 1.35))
                pt.cr_mult = float(np.clip(action.cr_multiplier * noise, 0.70, 1.35))
                pt.basal_mult = float(np.clip(action.basal_multiplier * noise, 0.75, 1.25))
                if agency:
                    pt.trust = min(1.0, pt.trust + agency.trust_growth_rate)
                result["is_intervention"] = True

            # Enrich record with user's actual decision
            pt.shadow_mod.enrich_outcome(
                record.record_id, actual_outcomes, user_action,
                [pt.isf_mult, pt.cr_mult, pt.basal_mult],
            )
            pt.shadow_mod.evaluate_record(
                record.record_id,
                per_model_accuracy=per_model_accuracy,
            )
            self._log_recommendation(pt, record, rec_pkg)
            self._flush_shadow_record(record)
        else:
            # Budget empty or no recommendation — still log the record
            pt.shadow_mod.enrich_outcome(
                record.record_id, actual_outcomes, "not_shown",
                [pt.isf_mult, pt.cr_mult, pt.basal_mult],
            )
            pt.shadow_mod.evaluate_record(
                record.record_id,
                per_model_accuracy=per_model_accuracy,
            )
            self._flush_shadow_record(record)

        self.total_shadow_records += 1
        # Return proposed action for counterfactual simulation
        if is_different:
            result["proposed_action"] = proposed_arr
            result["record_id"] = record.record_id
        return result

    def _simulate_user_decision(
        self,
        pt: PatientRuntime,
        rec_pkg: Any,
        day: int,
    ) -> bool:
        """Simulate whether the user accepts a recommendation."""
        agency = pt.cfg.agency_profile
        if agency is None:
            return self.rng.random() < 0.5

        # Base acceptance probability
        prob = pt.trust * agency.aggressiveness

        # Mood modulation
        if pt.mood_valence < -0.2:
            prob *= 0.5  # Less likely to accept when mood is low

        # Change anxiety modulation
        prob *= (1.0 - 0.3 * pt.personality.change_anxiety)

        # Engagement decay
        days_active = max(1, day - (pt.graduation_day or day))
        decay = 1.0 - agency.engagement_decay * min(1.0, days_active / 180.0)
        prob *= max(0.1, decay)

        return self.rng.random() < prob

    def _check_celebration(
        self,
        pt: PatientRuntime,
        day: int,
        outcome: YesterdayOutcome,
    ) -> None:
        """Check if a positive observation should be logged."""
        if pt.personality.celebration_receptiveness < 0.2:
            return

        if outcome.tir > 0.72 and outcome.consecutive_bad_days == 0:
            pt.shadow_mod.create_positive_observation(
                patient_id=pt.cfg.patient_id,
                day_index=day,
                message=f"Your TIR this week was {outcome.tir:.0%}. Keep it up!",
                tir_improvement=outcome.tir - 0.65,
            )

    # ------------------------------------------------------------------
    # Feature Extraction
    # ------------------------------------------------------------------

    # Model feature schema (order matters — must match training):
    _FEATURE_KEYS = [
        "bg_avg", "bg_tir", "bg_percent_low", "bg_percent_high", "bg_uroc",
        "bg_delta_avg_7h", "bg_z_avg_7h",
        "hr_mean", "hr_delta_7h", "hr_z_7h",
        "kcal_active", "kcal_active_last3h", "kcal_active_last6h",
        "ex_exercise_min", "ex_min_last3h",
        "cycle_follicular", "cycle_ovulation", "cycle_luteal",
        "days_since_site_change", "site_loc_same_as_last",
        "mood_valence", "mood_arousal",
        "mood_quad_pos_pos", "mood_quad_pos_neg",
        "mood_quad_neg_pos", "mood_quad_neg_neg",
        "mood_hours_since", "site_loc_current_enc",
    ]

    _SITE_LOC_MAP = {
        "abdomen_left": 0, "abdomen_right": 1,
        "thigh_left": 2, "thigh_right": 3,
        "arm_left": 4, "arm_right": 5,
    }

    def _build_daily_features(
        self,
        pt: PatientRuntime,
        ctx: Any,
        beh: dict,
        outcome: "YesterdayOutcome",
        true_bg: np.ndarray,
        hr_arr: np.ndarray,
        active_e: np.ndarray,
    ) -> dict[str, float]:
        """Build a single day's raw feature dict from simulation outputs."""
        valid_bg = true_bg[np.isfinite(true_bg)]
        bg_avg = float(np.nanmean(valid_bg)) if valid_bg.size > 0 else 150.0

        # Rate of change: mean |delta| across 5-min intervals
        if valid_bg.size > 1:
            uroc = float(np.nanmean(np.abs(np.diff(valid_bg)) / 5.0))
        else:
            uroc = 0.0

        valid_hr = hr_arr[np.isfinite(hr_arr)]
        hr_mean = float(np.nanmean(valid_hr)) if valid_hr.size > 0 else 70.0

        valid_active = active_e[np.isfinite(active_e)]
        kcal_active = float(np.nansum(valid_active))
        # Last 3h and 6h: use last 3 and 6 elements of hourly array
        kcal_last3h = float(np.nansum(active_e[-3:])) if active_e.size >= 3 else kcal_active
        kcal_last6h = float(np.nansum(active_e[-6:])) if active_e.size >= 6 else kcal_active

        ex_minutes = beh.get("exercise_minutes", 0.0)
        # ex_min_last3h: use exercise minutes as proxy (single daily value)
        ex_min_last3h = ex_minutes

        # Cycle phase one-hot
        phase = ctx.cycle_phase
        cycle_f = 1.0 if phase == "follicular" else 0.0
        cycle_o = 1.0 if phase == "ovulation" else 0.0
        cycle_l = 1.0 if phase == "luteal" else 0.0

        # Site
        site_days = beh.get("site_days_since_change", 0)
        site_loc = beh.get("site_location", "abdomen_left")
        site_enc = float(self._SITE_LOC_MAP.get(site_loc, 0))
        # Same as last: compare to previous history
        prev_site = pt.feature_history[-1].get("_site_loc", site_loc) if pt.feature_history else site_loc
        site_same = 1.0 if site_loc == prev_site else 0.0

        # Mood
        mv = ctx.mood_valence
        ma = ctx.mood_arousal
        q_pp = 1.0 if mv >= 0 and ma >= 0 else 0.0
        q_pn = 1.0 if mv >= 0 and ma < 0 else 0.0
        q_np = 1.0 if mv < 0 and ma >= 0 else 0.0
        q_nn = 1.0 if mv < 0 and ma < 0 else 0.0

        # Mood hours since: use fixed 8.0 as proxy (mid-day measurement)
        mood_hours = 8.0

        return {
            "bg_avg": bg_avg,
            "bg_tir": outcome.tir,
            "bg_percent_low": outcome.percent_low,
            "bg_percent_high": outcome.percent_high,
            "bg_uroc": uroc,
            "hr_mean": hr_mean,
            "kcal_active": kcal_active,
            "kcal_active_last3h": kcal_last3h,
            "kcal_active_last6h": kcal_last6h,
            "ex_exercise_min": ex_minutes,
            "ex_min_last3h": ex_min_last3h,
            "cycle_follicular": cycle_f,
            "cycle_ovulation": cycle_o,
            "cycle_luteal": cycle_l,
            "days_since_site_change": float(site_days),
            "site_loc_same_as_last": site_same,
            "mood_valence": mv,
            "mood_arousal": ma,
            "mood_quad_pos_pos": q_pp,
            "mood_quad_pos_neg": q_pn,
            "mood_quad_neg_pos": q_np,
            "mood_quad_neg_neg": q_nn,
            "mood_hours_since": mood_hours,
            "site_loc_current_enc": site_enc,
            # Internal (not passed to model, used for site_same_as_last)
            "_site_loc": site_loc,
        }

    def _extract_features(self, pt: PatientRuntime) -> np.ndarray:
        """Build the 28-element feature vector from patient history.

        Uses today's daily features plus 7-day rolling stats for delta/z
        features. Falls back to zeros for any missing history.
        """
        if not pt.feature_history:
            return np.zeros(len(self._FEATURE_KEYS))

        today = pt.feature_history[-1]
        hist = pt.feature_history  # Up to 8 entries

        # Compute 7-day rolling stats for delta and z-score features
        if len(hist) >= 2:
            past = hist[:-1]  # All but today
            bg_avgs = [d["bg_avg"] for d in past]
            hr_means = [d["hr_mean"] for d in past]
            bg_mean_7d = float(np.mean(bg_avgs))
            bg_std_7d = float(np.std(bg_avgs)) if len(bg_avgs) > 1 else 1.0
            hr_mean_7d = float(np.mean(hr_means))
            hr_std_7d = float(np.std(hr_means)) if len(hr_means) > 1 else 1.0

            bg_delta = today["bg_avg"] - bg_mean_7d
            bg_z = (today["bg_avg"] - bg_mean_7d) / max(bg_std_7d, 1e-6)
            hr_delta = today["hr_mean"] - hr_mean_7d
            hr_z = (today["hr_mean"] - hr_mean_7d) / max(hr_std_7d, 1e-6)
        else:
            bg_delta = 0.0
            bg_z = 0.0
            hr_delta = 0.0
            hr_z = 0.0

        vec = np.array([
            today["bg_avg"],
            today["bg_tir"],
            today["bg_percent_low"],
            today["bg_percent_high"],
            today["bg_uroc"],
            bg_delta,
            bg_z,
            today["hr_mean"],
            hr_delta,
            hr_z,
            today["kcal_active"],
            today["kcal_active_last3h"],
            today["kcal_active_last6h"],
            today["ex_exercise_min"],
            today["ex_min_last3h"],
            today["cycle_follicular"],
            today["cycle_ovulation"],
            today["cycle_luteal"],
            today["days_since_site_change"],
            today["site_loc_same_as_last"],
            today["mood_valence"],
            today["mood_arousal"],
            today["mood_quad_pos_pos"],
            today["mood_quad_pos_neg"],
            today["mood_quad_neg_pos"],
            today["mood_quad_neg_neg"],
            today["mood_hours_since"],
            today["site_loc_current_enc"],
        ], dtype=float)

        # Replace NaN/inf with 0
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec

    # ------------------------------------------------------------------
    # Training Data Collection
    # ------------------------------------------------------------------

    def _collect_training_row(
        self,
        pt: PatientRuntime,
        day: int,
        daily_feats: dict[str, float],
        outcome: "YesterdayOutcome",
    ) -> None:
        """Buffer one training row: features + current actions + outcomes."""
        row: dict[str, float] = {}
        # 28 features
        for key in self._FEATURE_KEYS:
            val = daily_feats.get(key, 0.0)
            row[key] = float(val) if np.isfinite(val) else 0.0
        # 3 action columns (therapy multipliers currently in effect)
        row["isf_multiplier"] = pt.isf_mult
        row["cr_multiplier"] = pt.cr_mult
        row["basal_multiplier"] = pt.basal_mult
        # 4 outcome targets
        row["percent_low"] = outcome.percent_low
        row["percent_high"] = outcome.percent_high
        row["tir"] = outcome.tir
        row["mean_bg"] = outcome.mean_bg
        # Metadata (for splitting, not fed to model)
        row["_patient_id"] = hash(pt.cfg.patient_id) % 1_000_000  # numeric for DataFrame
        row["_day"] = float(day)
        self.training_buffer.append(row)

    # ------------------------------------------------------------------
    # Per-Model Accuracy Computation
    # ------------------------------------------------------------------

    def _compute_per_model_accuracy(
        self,
        proposed_preds: dict[str, dict],
        baseline_preds: dict[str, dict],
        actual_outcomes: dict[str, float],
    ) -> dict[str, dict] | None:
        """Compute per-model accuracy by comparing predictions to actuals.

        Returns model_id → {mae, coverage, bias} for each model whose
        predictions are available.
        """
        if not proposed_preds or not actual_outcomes:
            return None

        # Model targets in order: percent_low, percent_high, tir, mean_bg
        target_keys = ["percent_low", "percent_high", "tir", "mean_bg"]
        result: dict[str, dict] = {}

        for model_id, preds in proposed_preds.items():
            point = preds.get("point", [])
            lower = preds.get("lower", [])
            upper = preds.get("upper", [])

            if not point or len(point) < len(target_keys):
                continue

            # Coverage: fraction of targets where actual fell within [lower, upper]
            hits = 0
            total = 0
            abs_errors = []
            biases = []
            for i, key in enumerate(target_keys):
                actual = actual_outcomes.get(key)
                if actual is None or i >= len(point):
                    continue
                total += 1
                pred_val = point[i]
                abs_errors.append(abs(actual - pred_val))
                biases.append(pred_val - actual)
                if i < len(lower) and i < len(upper):
                    if lower[i] <= actual <= upper[i]:
                        hits += 1

            if total > 0:
                result[model_id] = {
                    "mae": float(np.mean(abs_errors)),
                    "coverage": hits / total,
                    "bias": float(np.mean(biases)),
                }

        return result if result else None

    def _flush_shadow_record(self, record: Any) -> None:
        """Persist a shadow record immediately when a DB writer is active."""
        if self.writer:
            self.writer.write_shadow_records([record.to_row()])

    def _log_recommendation(self, pt: PatientRuntime, record: Any, rec_pkg: Any) -> None:
        """Persist surfaced recommendation rows for later inspection."""
        if not self.writer:
            return
        primary = rec_pkg.primary
        baseline = TherapyAction(*record.baseline_action)
        row = (
            record.record_id,
            pt.cfg.patient_id,
            record.day_index,
            record.timestamp_utc,
            rec_pkg.decision,
            primary.isf_multiplier if primary else None,
            primary.cr_multiplier if primary else None,
            primary.basal_multiplier if primary else None,
            baseline.isf_multiplier,
            baseline.cr_multiplier,
            baseline.basal_multiplier,
            rec_pkg.primary_predicted_outcomes.get("tir"),
            rec_pkg.primary_predicted_outcomes.get("percent_low"),
            rec_pkg.primary_confidence,
            rec_pkg.primary_reward,
            rec_pkg.explanation,
        )
        self.writer.write_recommendation_log([row])

    # ------------------------------------------------------------------
    # Phase Transitions (Section 1.4)
    # ------------------------------------------------------------------

    def _check_phase_transitions(self, day: int) -> None:
        """Check and apply lifecycle phase transitions for all patients."""
        has_active_model = self.active_model is not None

        for pt in self.patients:
            # Phase 0 → Phase 1: model exists + min observation window
            if pt.phase == 0:
                if has_active_model and day >= self.MIN_OBSERVATION_DAYS:
                    pt.phase = 1
                    pt.entered_shadow_day = day
                    pt.therapy_mode.promote(day)  # Shadow → Suggest Values

            # Phase 1 → Phase 2: graduation conditions met
            elif pt.phase == 1:
                win_rate = pt.shadow_mod.get_recent_win_rate()
                acceptance = pt.shadow_mod.get_recent_acceptance_rate()
                scorecard = pt.shadow_mod.compute_scorecard()
                graduated = pt.shadow_mod.check_graduation(scorecard)

                if graduated:
                    pt.phase = 2
                    pt.graduation_day = day
                    pt.reached_intervention = True
                    self.graduation_count += 1

            # Phase 2 → Phase 1: de-graduation
            elif pt.phase == 2:
                scorecard = pt.shadow_mod.compute_scorecard()
                if scorecard.win_rate < 0.50 or scorecard.safety_violations > 0:
                    pt.phase = 1
                    pt.graduation_day = None
                    pt.degraduation_count += 1
                    self.degraduation_count += 1

            # Therapy mode level checks
            if pt.phase >= 1 and day % 7 == 0:
                win_rate = pt.shadow_mod.get_recent_win_rate()
                acceptance = pt.shadow_mod.get_recent_acceptance_rate()
                tir_delta = win_rate - 0.5  # Proxy for TIR improvement
                mood_trend = pt.mood_valence  # Current mood as proxy

                if pt.therapy_mode.check_graduation(
                    tir_delta=tir_delta,
                    acceptance_rate=acceptance,
                    safety_violations=0,
                    mood_trend=mood_trend,
                    day_index=day,
                ):
                    pt.therapy_mode.promote(day)

                if pt.therapy_mode.check_regression(
                    tir_delta=tir_delta,
                    safety_violations=0,
                    mood_trend=mood_trend,
                ):
                    pt.therapy_mode.demote(day)

    # ------------------------------------------------------------------
    # Population Metrics
    # ------------------------------------------------------------------

    def _compute_population_win_rate(self) -> float:
        """Compute population-wide rolling win rate."""
        rates = []
        for pt in self.patients:
            if pt.phase >= 1:
                rate = pt.shadow_mod.get_recent_win_rate()
                rates.append(rate)
        return float(np.mean(rates)) if rates else 0.5

    def _compute_causal_delta(self) -> float:
        """Compute population-wide causal delta (accept vs reject TIR)."""
        all_records = []
        for pt in self.patients:
            all_records.extend(pt.shadow_mod.records)

        accepted_tir = []
        rejected_tir = []
        for rec in all_records:
            if rec.actual_outcomes and rec.actual_user_action:
                tir = rec.actual_outcomes.get("tir", 0)
                if rec.actual_user_action == "accept":
                    accepted_tir.append(tir)
                elif rec.actual_user_action == "reject":
                    rejected_tir.append(tir)

        if accepted_tir and rejected_tir:
            return float(np.mean(accepted_tir) - np.mean(rejected_tir))
        return 0.0

    # ------------------------------------------------------------------
    # Retraining
    # ------------------------------------------------------------------

    def _handle_retrain(self, day: int) -> None:
        """Train a new AggregateOutcomePredictor from accumulated data."""
        n_rows = len(self.training_buffer)
        if n_rows < 120:
            if self.verbose:
                print(f"  [retrain] Day {day}: only {n_rows} rows, need 120+. Skipping.")
            return

        self.retrain_count += 1
        self.meta.reset_pressure(day)

        # Build DataFrame from training buffer
        df = pd.DataFrame(self.training_buffer)

        # Feature + action columns (model input) and target columns (output)
        feature_cols = list(self._FEATURE_KEYS)
        action_cols = ["isf_multiplier", "cr_multiplier", "basal_multiplier"]
        target_cols = ["percent_low", "percent_high", "tir", "mean_bg"]
        input_cols = feature_cols + action_cols

        X = df[input_cols].copy()
        y = df[target_cols].copy()

        # Temporal train/val split (80/20 by day)
        cutoff_day = int(df["_day"].quantile(0.8))
        train_mask = df["_day"] <= cutoff_day
        val_mask = df["_day"] > cutoff_day

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        if len(X_train) < 100:
            if self.verbose:
                print(f"  [retrain] Day {day}: only {len(X_train)} train rows. Skipping.")
            return

        # Train the model
        model = AggregateOutcomePredictor()
        try:
            model.fit(
                X_train, y_train,
                X_val=X_val if len(X_val) > 10 else None,
                y_val=y_val if len(X_val) > 10 else None,
                xgb_params={
                    "n_estimators": 80,
                    "max_depth": 4,
                    "learning_rate": 0.08,
                },
            )
        except Exception as e:
            if self.verbose:
                print(f"  [retrain] Day {day}: training failed: {e}")
            return

        # Assign model identity
        self.model_version += 1
        new_id = f"model_v{self.model_version}"
        model.model_id = new_id
        model.version = str(self.model_version)

        # Register in meta-controller
        entry = ModelRegistryEntry(
            model_id=new_id,
            version=str(self.model_version),
            architecture="xgboost_aggregate",
            target="multi",
            status="candidate",
        )
        self.meta.register_model(entry)
        self.meta.register_candidate(new_id, day)
        if self.writer:
            self.writer.write_model_registry([entry.to_row()])

        # If no active model exists (first train), promote immediately
        if self.active_model is None:
            self.active_model = model
            entry.status = "active"
            if hasattr(self.meta, '_state') and self.meta._state.candidate_model_id:
                self.meta.promote_candidate()
            if self.writer:
                self.writer.write_model_registry([entry.to_row()])
            if self.verbose:
                print(f"  [retrain] Day {day}: first model {new_id} trained on "
                      f"{len(X_train)} rows → promoted immediately")
        else:
            # Store as candidate for 7-day shadow validation
            self._candidate_model = model
            if self.verbose:
                print(f"  [retrain] Day {day}: trained candidate {new_id} on "
                      f"{len(X_train)} rows (val: {len(X_val)})")

    # ------------------------------------------------------------------
    # Weekly Evaluation
    # ------------------------------------------------------------------

    def _weekly_evaluation(
        self,
        day: int,
        population_win_rate: float,
        causal_delta: float,
    ) -> None:
        """Run the full weekly meta-controller evaluation."""
        self.meta.weekly_evaluation(day, population_win_rate, causal_delta)
        if self.writer:
            self.writer.write_evaluation_snapshot(
                (
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "weekly_world_runner",
                    len(self.patients),
                    json.dumps({
                        "day": day,
                        "population_win_rate": population_win_rate,
                        "causal_delta": causal_delta,
                    }),
                    json.dumps({"retrain_count": self.retrain_count}),
                )
            )

        # Check candidate validation
        if self.meta._state.candidate_model_id:
            # Use population win rate as proxy for candidate performance
            status = self.meta.check_candidate_validation(
                day=day,
                candidate_win_rate=population_win_rate,
                active_win_rate=population_win_rate * 0.95,
            )
            if status == "promote":
                promoted = self.meta.promote_candidate()
                if promoted and self._candidate_model is not None:
                    self.active_model = self._candidate_model
                    self._candidate_model = None
                if self.verbose and promoted:
                    print(f"  [candidate] Day {day}: promoted {promoted}")
            elif status == "reject":
                rejected = self.meta.reject_candidate()
                if rejected:
                    self._candidate_model = None
                if self.verbose and rejected:
                    print(f"  [candidate] Day {day}: rejected {rejected}")

    # ------------------------------------------------------------------
    # Status Output (Section 8.4)
    # ------------------------------------------------------------------

    def _print_status(self, day: int, win_rate: float, causal_delta: float) -> None:
        """Print periodic monitoring summary."""
        phase_dist = {0: 0, 1: 0, 2: 0}
        level_dist: dict[int, int] = {}
        mood_sum = 0.0
        burnout_count = 0

        for pt in self.patients:
            phase_dist[pt.phase] += 1
            lvl = int(pt.therapy_mode.current_level)
            level_dist[lvl] = level_dist.get(lvl, 0) + 1
            mood_sum += pt.mood_valence
            if pt.mood_valence < pt.personality.burnout_threshold:
                burnout_count += 1

        avg_mood = mood_sum / max(len(self.patients), 1)

        print(f"\n{'='*60}")
        print(f"  Day {day}/{self.n_days}")
        print(f"  Phase distribution: P0={phase_dist[0]} P1={phase_dist[1]} P2={phase_dist[2]}")
        print(f"  Therapy levels: {dict(sorted(level_dist.items()))}")
        model_status = f"v{self.model_version}" if self.active_model else "none (cold start)"
        candidate_status = f" | candidate: v{self.model_version}" if self._candidate_model else ""
        print(f"  Active model: {model_status}{candidate_status}")
        print(f"  Training buffer: {len(self.training_buffer)} rows")
        print(f"  Shadow records: {self.total_shadow_records}")
        print(f"  Population win rate: {win_rate:.2%}")
        print(f"  Causal delta: {causal_delta:+.3f}")
        print(f"  Graduations: {self.graduation_count}, De-graduations: {self.degraduation_count}")
        print(f"  Retrains: {self.retrain_count}")
        print(f"  Avg mood: {avg_mood:.2f}, Burnout: {burnout_count}")
        print(f"  Rec budgets: {sum(1 for pt in self.patients if pt.rec_budget.can_recommend())} active")
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(self) -> dict[str, Any]:
        """Build final summary statistics."""
        phase_dist = {0: 0, 1: 0, 2: 0}
        level_dist: dict[int, int] = {}
        for pt in self.patients:
            phase_dist[pt.phase] += 1
            lvl = int(pt.therapy_mode.current_level)
            level_dist[lvl] = level_dist.get(lvl, 0) + 1

        early_window = min(30, self.n_days)
        late_start = max(0, self.n_days - 30)
        early_tirs = []
        late_tirs = []
        patient_rows = []
        for pt in self.patients:
            early = float(np.mean(pt.tir_history[:early_window])) if pt.tir_history[:early_window] else None
            late = float(np.mean(pt.tir_history[late_start:])) if pt.tir_history[late_start:] else None
            mean_tir = float(np.mean(pt.tir_history)) if pt.tir_history else None
            acceptance_rate = (
                pt.n_recommendations_accepted / pt.n_recommendations_surfaced
                if pt.n_recommendations_surfaced else None
            )
            tir_delta = (late - early) if early is not None and late is not None else None
            if early is not None:
                early_tirs.append(early)
            if late is not None:
                late_tirs.append(late)
            patient_rows.append({
                "patient_id": pt.cfg.patient_id,
                "final_phase": pt.phase,
                "current_isf": pt.isf_mult,
                "current_cr": pt.cr_mult,
                "current_basal": pt.basal_mult,
                "days_observed": len(pt.tir_history),
                "mean_tir": mean_tir,
                "early_tir": early,
                "late_tir": late,
                "tir_delta": tir_delta,
                "recommendations_surfaced": pt.n_recommendations_surfaced,
                "recommendations_accepted": pt.n_recommendations_accepted,
                "acceptance_rate": acceptance_rate,
                "graduated_to_intervention": pt.reached_intervention,
                "entered_shadow": pt.entered_shadow_day is not None,
                "degraduation_count": pt.degraduation_count,
                "burnout_flag": bool(pt.burnout_days),
            })

        bucket_size = 30
        tir_buckets = []
        for start in range(0, self.n_days, bucket_size):
            bucket_vals = [
                float(np.mean(pt.tir_history[start:start + bucket_size]))
                for pt in self.patients
                if pt.tir_history[start:start + bucket_size]
            ]
            if bucket_vals:
                tir_buckets.append({
                    "start_day": start + 1,
                    "end_day": min(start + bucket_size, self.n_days),
                    "mean_tir": float(np.mean(bucket_vals)),
                })

        mean_early = float(np.mean(early_tirs)) if early_tirs else 0.0
        mean_late = float(np.mean(late_tirs)) if late_tirs else 0.0
        tir_delta = mean_late - mean_early
        acceptance_n = sum(pt.n_recommendations_accepted for pt in self.patients)
        surfaced_n = sum(pt.n_recommendations_surfaced for pt in self.patients)
        burnout_count = sum(1 for pt in self.patients if pt.burnout_days)
        burnout_rate = burnout_count / max(len(self.patients), 1)
        verdict = (
            "Viable MVP baseline: mean TIR improved and burnout remained acceptable."
            if tir_delta > 0 and burnout_rate <= 0.25
            else "Needs follow-up: either TIR did not improve or burnout/disengagement exceeded the MVP guardrail."
        )

        return {
            "n_patients": self.n_patients,
            "n_days": self.n_days,
            "seed": self.seed,
            "learning_mode": self.learning_mode,
            "phase_distribution": phase_dist,
            "level_distribution": level_dist,
            "retrain_count": self.retrain_count,
            "graduation_count": self.graduation_count,
            "degraduation_count": self.degraduation_count,
            "total_shadow_records": self.total_shadow_records,
            "final_population_win_rate": self._compute_population_win_rate(),
            "final_causal_delta": self._compute_causal_delta(),
            "burnout_count": burnout_count,
            "burnout_definition": self._burnout_definition,
            "burnout_rate": burnout_rate,
            "mean_tir_early_window": mean_early,
            "mean_tir_late_window": mean_late,
            "overall_tir_delta": tir_delta,
            "tir_by_30_day_bucket": tir_buckets,
            "number_of_surfaced_recommendations": surfaced_n,
            "recommendation_acceptance_rate": (
                acceptance_n / surfaced_n if surfaced_n else 0.0
            ),
            "patients_entering_shadow": sum(1 for pt in self.patients if pt.entered_shadow_day is not None),
            "percent_entering_shadow": sum(1 for pt in self.patients if pt.entered_shadow_day is not None) / max(self.n_patients, 1),
            "patients_graduating_to_intervention": sum(1 for pt in self.patients if pt.reached_intervention),
            "percent_graduating_to_intervention": sum(1 for pt in self.patients if pt.reached_intervention) / max(self.n_patients, 1),
            "did_mean_tir_improve": bool(tir_delta > 0),
            "did_burnout_remain_acceptable": bool(burnout_rate <= 0.25),
            "viability_verdict": verdict,
            "patient_summaries": patient_rows,
        }

    def _persist_summary(self, summary: dict[str, Any]) -> None:
        """Persist the final simulation summary and patient rollups."""
        if not self.writer:
            return
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        patient_rows = [
            (
                row["patient_id"],
                row["final_phase"],
                row["current_isf"],
                row["current_cr"],
                row["current_basal"],
                row["days_observed"],
                row["mean_tir"],
                row["early_tir"],
                row["late_tir"],
                row["tir_delta"],
                row["recommendations_surfaced"],
                row["recommendations_accepted"],
                row["acceptance_rate"],
                int(row["graduated_to_intervention"]),
                int(row["entered_shadow"]),
                row["degraduation_count"],
                int(row["burnout_flag"]),
                self._burnout_definition,
            )
            for row in summary["patient_summaries"]
        ]
        self.writer.write_patient_run_summary(patient_rows)
        self.writer.write_simulation_run(
            (
                f"seed_{self.seed}_{int(self.run_started_at.timestamp())}",
                created_at,
                json.dumps(summary),
            )
        )
        self.writer.write_evaluation_snapshot(
            (
                created_at,
                "final_summary",
                self.n_patients,
                json.dumps({
                    "mean_tir_early_window": summary["mean_tir_early_window"],
                    "mean_tir_late_window": summary["mean_tir_late_window"],
                    "overall_tir_delta": summary["overall_tir_delta"],
                    "recommendation_acceptance_rate": summary["recommendation_acceptance_rate"],
                    "burnout_count": summary["burnout_count"],
                }),
                json.dumps({
                    "viability_verdict": summary["viability_verdict"],
                    "burnout_definition": summary["burnout_definition"],
                }),
            )
        )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Chamelia World Runner")
    parser.add_argument("--n_patients", type=int, default=100)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdb", type=str, default="world.db")
    parser.add_argument("--learning-mode", type=str, default="hybrid",
                        choices=["individual", "community", "hybrid"])
    parser.add_argument("--initial-zoo", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    runner = WorldRunner(
        n_patients=args.n_patients,
        n_days=args.days,
        seed=args.seed,
        learning_mode=args.learning_mode,
        initial_zoo_path=args.initial_zoo,
        verbose=not args.quiet,
    )

    t0 = time.time()
    summary = runner.run()
    elapsed = time.time() - t0

    print(f"\n[world] Completed in {elapsed:.1f}s")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

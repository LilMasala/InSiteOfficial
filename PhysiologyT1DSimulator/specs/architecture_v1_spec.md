**CHAMELIA**

Self-Learning Therapy Recommender

System Architecture Document

Version 1.0 • March 2026

InSite Project • Confidential

**Table of Contents**

**1. System Overview**

Chamelia is a self-learning recommender framework for personal health
optimization. The first application is insulin pump therapy settings
(ISF, CR, basal schedule), with the architecture designed to generalize
to behavioral interventions (exercise, sleep, nutrition) and eventually
any domain with streaming time-series data, tunable parameters, and
measurable outcomes.

The system collects streaming health data from HealthKit and Firestore,
generates parameter recommendations that optimize user-defined
objectives, proves its own robustness in shadow mode before surfacing
recommendations, adapts its model architecture in response to drift, and
exposes user-friendly control knobs for objectives, constraints, and
conservativeness.

**1.1 Design Principles**

- **Safety first:** No recommendation surfaces without passing both a
  confidence gate and an absolute safety check. The system would rather
  miss a good recommendation than surface a dangerous one.

- **Prove before you show:** Every recommendation strategy must survive
  a sustained shadow evaluation period with zero safety violations
  before graduating to user-facing mode.

- **Adapt or die:** The meta-controller continuously monitors for
  distribution drift and model degradation, escalating from gentle
  reweighting to full retraining as needed.

- **User agency:** The user controls objective weights,
  conservativeness, locked time windows, and planning horizon. The
  system optimizes within the user\'s preferences, not over them.

- **Uncertainty-aware:** Every prediction carries calibrated uncertainty
  bounds. The system optimizes risk-adjusted reward, not expected reward
  alone.

**1.2 Nine-Block Architecture**

The complete system comprises nine interlocking components that form a
cycle, not a pipeline. Data feeds models, models generate
recommendations, recommendations feed the simulator (or real users), and
outcomes become new data.

  -------- --------------------- -------------------------------------------------
  **\#**   **Block**             **Purpose**

  1        **Data Layer**        HealthKit/Firestore ingestion, CTX builders,
                                 FeatureFrameHourly schema, Firestore upload

  2        **Sim Layer**         t1d_sim synthetic population generator; extended
                                 with closed-loop intervention, user agency
                                 profiles, and forked timelines

  3        **Model Zoo**         Individual predictor models (aggregate, temporal,
                                 surrogate, anomaly), each conforming to the
                                 PredictorCard interface

  4        **Confidence Module** Four-layer gating system (GP familiarity,
                                 ensemble agreement, calibration tracker, effect
                                 size) plus hard safety gate

  5        **Shadow Module**     Logging, retrospective evaluation, scorecard,
                                 graduation/de-graduation logic

  6        **Meta-Controller**   Drift detection, model trust routing, retraining
                                 triggers, action family selection (Chameleon)

  7        **Optimization        Constrained action search (grid, Bayesian,
           Engine**              offline RL), reward function, constraint system,
                                 recommendation packaging

  8        **User Agency Layer** Acceptance/compliance/reversion modeling for
                                 simulated users; decision profiles and trust
                                 dynamics

  9        **Evaluation Layer**  Counterfactual pairs from forked timelines,
                                 off-policy evaluation, simulator
                                 cross-validation, robustness triangulation
  -------- --------------------- -------------------------------------------------

**2. Data Layer (Block 1: Built)**

The data layer is fully implemented across both the iOS app and the
Python simulation. Both pipelines produce the same \~38-feature hourly
schema, ensuring that models trained on synthetic data can be directly
applied to real user data.

**2.1 iOS Pipeline**

On-device, the pipeline flows from HealthKit through a series of context
builders (CTXBuilder.swift) that produce per-hour context structs:
BGCTX, HRCTX, EnergyCTX, SleepCTX, ExerciseCTX, MenstrualCTX, SiteCTX,
and MoodCTX. The FeatureJoiner merges these into FeatureFrameHourly
structs, which the HealthDataUploader pushes to Firestore under the
ml_feature_frames subcollection.

**2.2 Synthetic Pipeline**

The t1d_sim Python package generates synthetic patient populations with
persona-based trait sampling (solid_sleeper, athlete, high_stress,
etc.), context-driven physiology modulation (exercise, sleep, stress,
menstrual cycle, illness), realistic observation noise (CGM lag, drift,
missingness), and behavioral logging models (meal confirmation, mood
logging fidelity). The simulation writes to SQLite or Firestore and
produces the identical feature schema.

**2.3 Feature Schema**

The FeatureFrameHourly contains approximately 38 features organized into
seven signal groups: BG metrics (avg, TIR, %low, %high, uROC, 7-day
delta and z-score), HR metrics (mean, 7-day delta, z-score, resting
daily), energy metrics (active kcal, 3h/6h rolling sums, 7-day delta and
z-score), sleep metrics (previous night total, 7-day debt, minutes since
wake), exercise metrics (move/exercise minutes, 3h rolling sum, hours
since exercise), menstrual context (days since period start, cycle phase
one-hot), site context (days since change, location, repeat flag), and
mood context (valence, arousal, quadrant flags, hours since logged).

**3. Simulation Layer (Block 2: Extension Required)**

The existing t1d_sim is an open-loop simulator: patients are generated
with fixed physiology and therapy settings for 180 days. The extension
transforms it into a closed-loop simulator where the modeling system
participates in the patient\'s life, generating recommendations that the
simulated user responds to.

**3.1 Three-Phase Patient Timeline**

**Phase 1 --- Observation Only (Days 1--N):** The system collects data
and builds feature frames but makes no recommendations. Models are
training and the shadow module is accumulating baseline statistics. N
varies per patient based on model convergence.

**Phase 2 --- Shadow Active (Days N--M):** The model generates daily
recommendations silently. These are logged against actual outcomes. The
scorecard accumulates evidence of the system\'s predictive accuracy and
recommendation quality. M is when the confidence gate signals readiness
for graduation.

**Phase 3 --- Intervention (Days M--180):** Recommendations surface. The
user\'s decision profile determines acceptance, partial compliance, or
rejection. The physiology engine runs forward under the actual settings
(recommended if accepted, baseline if rejected, noisy if partially
complied). Training data now includes model-influenced outcomes.

**3.2 User Agency Profiles**

Each simulated patient receives a decision profile governing how they
interact with recommendations. These traits interact with existing
persona characteristics (e.g., a high-stress, low-logging-quality user
likely has low initial trust and high engagement decay).

  ----------------------- ----------- -------------------------------------------
  **Trait**               **Range**   **Description**

  **aggressiveness**      0.0--1.0    Willingness to accept large setting
                                      changes; maps to optimizer constraint width

  **initial_trust**       0.0--1.0    Starting confidence in the system; affects
                                      acceptance probability early in Phase 3

  **trust_growth_rate**   float       How quickly trust builds as shadow
                                      predictions prove accurate

  **compliance_noise**    0.0--1.0    How precisely the user implements
                                      recommendations (0 = perfect, 1 = heavy
                                      noise)

  **revert_threshold**    float       How bad a day must be before the user rolls
                                      back to previous settings

  **engagement_decay**    float       Rate at which the user stops checking
                                      recommendations over time
  ----------------------- ----------- -------------------------------------------

**3.3 Fork-of-Forks Timeline Branching**

At every meaningful decision point during Phase 3, the simulation
branches into two paths: one where the user accepts the recommendation
and one where they reject it. Crucially, forks themselves fork at
subsequent decision points, producing a tree of intervention
trajectories.

This tree structure provides several capabilities that no other data
source can. First, every sibling pair in the tree is a perfectly
controlled experiment --- same patient, same history, same state,
different action. This yields causal effect estimates without
statistical gymnastics. Second, the full tree captures how sequential
decisions interact: the effect of recommendation #3 depends on whether
recommendations #1 and #2 were accepted. Third, the tree provides the
diverse trajectory dataset that offline RL training requires.

**3.3.1 Managing Exponential Growth**

Unconstrained fork-of-forks produces 2\^K timelines for K decision
points. Several pruning strategies keep this tractable:

- **Depth budget:** Maximum 8--10 decision points per patient tree,
  yielding 256--1024 terminal timelines. Across 100 patients, this
  produces 25K--100K unique trajectories.

- **Fork only at meaningful decisions:** If the optimizer recommends
  \'hold\' (no change), both branches are identical and no fork is
  needed. In practice, real changes occur every 3--7 days.

- **Prune converged branches:** If accept and reject paths lead to
  similar outcomes after a few days (because the change was small), stop
  one branch.

- **Stochastic branching:** With probability p=0.3 at each decision,
  fork; otherwise let the user agency profile determine the single path.
  This reduces expected branching factor to \~1.3.

**3.3.2 Path Identification**

Each path through the tree is identified by a bitstring where each bit
represents accept (1) or reject (0) at each decision point. Path
\'11010\' means: accepted at decisions 1 and 2, rejected at decision 3,
accepted at decision 4, rejected at decision 5. This enables efficient
querying: \'give me all paths that accepted at decision 3\' or \'compare
all paths that diverged at decision 4.\'

**3.3.3 Partial Compliance**

Rather than creating a third branch type, partial compliance is modeled
within the \'accept\' branch. The user agency profile adds noise to the
recommended action (e.g., recommended ISF=45, user sets ISF=48). The
\'reject\' branch remains the clean counterfactual (baseline settings),
while the \'accept\' branch reflects realistic human compliance.

**4. Model Zoo (Block 3)**

**4.1 PredictorCard Interface**

Every model in the zoo conforms to a single abstract interface. This
makes the meta-controller\'s job trivial: it does not care whether it is
talking to an XGBoost model or a transformer. It asks: given this state
and this proposed action, what do you predict, and how sure are you?

The interface specifies: a unique model ID and version string; the
target variable being predicted (e.g., \'tir\', \'percent_low\',
\'bg_trajectory\'); the input feature schema (which subset of the \~38
features the model requires); the action schema (what action
representation it expects, or none if outcome-only); and a predict()
method that returns a PredictionEnvelope.

**4.1.1 PredictionEnvelope**

Every model returns not just a point estimate but a distribution
summary. The PredictionEnvelope contains: point (float or array for
trajectories), lower bound (e.g., 10th percentile), upper bound (e.g.,
90th percentile), a self-assessed confidence score (0--1), and metadata
(feature importances, nearest training neighbors). This envelope is what
feeds the confidence gate downstream.

**4.2 Model 1: Aggregate Outcome Predictor**

**Question answered:** Given a day\'s feature context plus a proposed
therapy setting, what will the outcome metrics be?

**Targets:** %low, %high, TIR, mean BG, BG variance.

**Training data:** From the simulator, ground truth per day per patient.
Features are daily rollups of hourly frames. The \'action\' input is the
therapy params (ISF multiplier, CR multiplier, basal multiplier).

**Architecture:** XGBoost or LightGBM with multi-output or separate
heads per target. Deliberately simple and interpretable. Trains fast,
handles missing features gracefully, provides feature importances for
explainability. For uncertainty: quantile regression variants (10th,
50th, 90th percentiles) or conformal prediction on a calibration
holdout.

**Why first:** This model directly answers the optimization question. If
you can query \'what TIR do I get if I set ISF to X and CR to Y given
today\'s context?\', you can search the action space for optimal
settings. This is the supervised recommendation path.

**4.3 Model 2: Temporal Sequence Model**

**Question answered:** Given the last N hours of feature frames, what
will the next M hours of BG look like?

**Training data:** 288 five-minute BG points per day from the sim,
windowed into hourly summaries matching the feature schema. Produces
(input_sequence, future_sequence) pairs.

**Architecture:** Lightweight transformer encoder (or LSTM). Input is
the hourly feature frame sequence; output is predicted BG summary stats
for the next 4--24 hours. Therapy param embedding is concatenated or
cross-attended into the sequence. Ensemble of 3--5 models with different
random seeds provides natural uncertainty.

**Why it matters:** Powers the \'simulated preview\' in the UI --- a
projected BG curve, not just a single TIR number. Catches temporal
patterns the aggregate model misses (e.g., \'BG always spikes 3 hours
after a late dinner when sleep-deprived\'). Provides a second opinion
for ensemble agreement.

**4.4 Model 3: BG Dynamics Surrogate**

**Question answered:** Given context and therapy params, what does the
actual BG curve look like over the next 24 hours?

**Architecture:** Conditional neural ODE or a feedforward that predicts
parameters of a parametric BG curve (decay constants, meal response
amplitudes). Deliberately small and fast --- designed to be called
thousands of times inside an optimization loop.

**Why it matters:** One leg of the robustness triangulation. Cheaper
than the full sim, can be personalized as real data arrives. Powers
counterfactual evaluation in the shadow module.

**4.5 Model 4: Anomaly / Regime Detector**

**Question answered:** Is the current state normal, or has something
shifted?

**Architecture:** Gaussian Process (sparse GP) fitted in the feature
space, or an autoencoder where reconstruction error flags anomalies. For
any new observation, reports how far the user is from the training
distribution.

**Why it matters:** Front line of drift detection. If the user enters a
state the sim never generated, the anomaly detector fires, the
confidence gate suppresses recommendations, and the shadow module logs
it as a learning opportunity.

**4.6 Model 5 (Future): Behavior Response Model**

Predicts how exercise timing, wind-down routines, and other behavioral
interventions affect next-day BG and HR/HRV. Same PredictorCard
interface, different action schema. Deferred until the therapy path is
working, but the zoo architecture accommodates it from day one.

**4.7 Build Order**

1.  **Model 1 first** (XGBoost aggregate). Fastest to train, easiest to
    validate, immediately useful for the optimization loop.

2.  **Model 4 next** (anomaly detector). Before trusting Model 1, know
    when it\'s extrapolating. Cheap to build, feeds confidence gate.

3.  **Model 2 third** (temporal sequence). Gives preview capability and
    a second ensemble opinion.

4.  **Model 3 later** (dynamics surrogate). Primarily for counterfactual
    evaluation. Can wait until shadow module is running.

**5. Confidence Module (Block 4)**

The confidence module determines whether a recommendation should be
surfaced. It protects against four distinct failure modes: model
extrapolation (the state is unfamiliar), model disagreement (the
ensemble is inconsistent), miscalibration (the model is confident but
historically wrong), and marginal effect (the predicted improvement is
within noise). A separate hard safety gate runs independently.

**5.1 Layer 1: GP Familiarity**

**Purpose:** Determine how far the current state is from the training
distribution.

The \~38-dimensional feature space is projected to 8--12 dimensions (via
PCA or a learned encoder), and a sparse Gaussian Process is fitted over
the training data in that reduced space. The GP does not predict
outcomes --- it models the density of the training data. For any new
input, the posterior standard deviation indicates familiarity.

Low variance means the system has seen many similar states and models
are likely reliable. High variance means uncharted territory ---
suppress recommendations or widen uncertainty bands. The GP updates
naturally as new data arrives (from the closed-loop sim or real users),
and the familiar region expands organically.

**5.2 Layer 2: Ensemble Agreement**

**Purpose:** Check whether the zoo models agree on their predictions.

For overlapping targets (e.g., Models 1 and 2 both predict %low), the
module checks whether their prediction intervals overlap. A concordance
score normalizes pairwise interval overlap across all models predicting
the same target. For non-overlapping targets (e.g., the surrogate
predicts a BG curve while the aggregate predicts summary stats), summary
stats are derived from the curve and compared. Concordance below
threshold closes the gate.

**5.3 Layer 3: Calibration Tracker**

**Purpose:** Verify that models have been honest in their recent
predictions.

Maintained by the shadow module and consumed by the confidence gate. For
each model over a rolling 30-day window, tracks coverage (did 80%
confidence intervals contain the true value 80% of the time?), sharpness
(how wide are the intervals?), and bias (is the model systematically
over/under-predicting?). Produces a per-model reliability score that
weights ensemble votes.

**5.4 Layer 4: Effect Size Gate**

**Purpose:** Verify that the predicted improvement is practically
significant.

Computes the predicted improvement of the recommended action vs.
baseline as a signal-to-noise ratio: improvement divided by prediction
uncertainty. If the improvement is within the noise, the recommendation
is suppressed. The threshold is adjusted by the user\'s aggressiveness
profile --- conservative users require larger effect sizes.

**5.5 Hard Safety Gate**

Runs independently of the confidence layers. For each hard safety
constraint (e.g., %low \< 4%), checks the worst-case scenario within the
prediction uncertainty --- the upper bound, not the point estimate. If
the worst-case violates any hard constraint, the recommendation is
blocked regardless of confidence.

This is non-negotiable. A highly confident recommendation with even a
small tail probability of dangerous lows gets blocked. Safety is about
consequences, not about model confidence.

**5.6 Gate Composition**

All four layers run sequentially. If any layer closes the gate, the
recommendation is suppressed. If all pass, the module produces a
composite confidence score combining familiarity, concordance,
reliability, and effect SNR. This score is what the user sees (\'87%
confident this improves your day\') and what gets logged to the shadow
record.

**6. Shadow Module (Block 5)**

The shadow module is the system\'s conscience. Nothing reaches a user
without first surviving a shadow period where predictions are logged,
compared to reality, and scored. It manages the full lifecycle of
recommendations from generation through retrospective evaluation through
graduation.

**6.1 Shadow Record Structure**

Every recommendation cycle writes a ShadowRecord with three sections
filled at different times:

**6.1.1 At Recommendation Time (Immutable)**

- Full state snapshot (feature frame window the models saw)

- Proposed action and baseline action

- All zoo predictions (PredictionEnvelope per model, for both proposed
  and baseline)

- Confidence gate decision (which layers passed/failed, overall score)

- Active ensemble metadata, GP familiarity score, calibration scores

**6.1.2 At Outcome Time (\~24h Later)**

- Actual outcome metrics (true %low, %high, TIR, mean BG)

- Actual user action (accept, reject, partial comply, not shown)

- Actual settings in effect during the outcome period

**6.1.3 At Evaluation Time**

- Counterfactual estimate from BG dynamics surrogate

- Per-model accuracy (prediction vs. actual, interval coverage)

- Shadow score delta (did recommendation outperform baseline?)

**6.2 Scorecard**

The scorecard is a rolling summary computed over a window of recent
enriched shadow records (last 30 days or 50 records). It tracks six
dimensions:

  ------------------ ----------------------------------------------------
  **Dimension**      **Description**

  **Prediction       RMSE, MAE, and bias per model per target. Trending
  Accuracy**         upward = losing touch.

  **Calibration      Empirical coverage vs. stated confidence intervals.
  Quality**          80% CI should contain truth \~80% of the time.

  **Win Rate**       Fraction of recommendations that would have improved
                     outcomes vs. baseline, via counterfactual estimates.
                     Must exceed \~65--70% for graduation.

  **Safety Record**  Count of recommendations that would have violated a
                     hard safety constraint. Must be zero.

  **Consistency**    Win rate stability across context slices
                     (time-of-day, activity level, cycle phase, stress).
                     No wide regime-dependent gaps.

  **Improvement      Is performance improving, stable, or degrading?
  Trend**            Degradation is a drift signal.
  ------------------ ----------------------------------------------------

**6.3 Graduation Logic**

Graduation is a conjunction of conditions, all of which must hold
simultaneously for a sustained period:

- **Minimum shadow duration:** At least 21--30 days of shadow data,
  ensuring the system has seen weekdays, weekends, varying activity, and
  (if applicable) a full menstrual cycle.

- **Win rate above threshold:** Recommendation win rate exceeds 60% over
  the full window.

- **Zero safety violations:** No predicted hard constraint violations in
  the shadow window. Not \'few\' --- zero.

- **Calibration within tolerance:** Empirical coverage within a band of
  the nominal level (70--90% for 80% stated intervals).

- **GP familiarity above minimum:** At least 90% of shadow records had
  familiarity scores above the threshold.

- **Cross-context consistency:** Win rate does not vary more than a
  defined amount across context slices.

- **Sustained period:** All conditions hold for 7+ consecutive daily
  evaluation cycles. Any failure resets the clock.

**6.4 Post-Graduation Monitoring**

After graduation, the shadow module transitions from gating to
monitoring. Every live recommendation still gets a shadow record. The
scorecard keeps running. If performance degrades below a de-graduation
threshold (set lower than graduation to avoid oscillation, e.g., 50% win
rate), the system automatically drops back to shadow-only mode.

**6.5 Acceptance Rate as Meta-Metric**

The shadow module tracks user acceptance rate alongside outcome metrics.
A system that is objectively good but subjectively rejected is not
working. Low acceptance feeds back into the optimizer\'s constraint
system: tighten conservativeness, reduce recommendation frequency, or
shift toward smaller changes.

**7. Meta-Controller / Chameleon (Block 6)**

The meta-controller sits above the zoo and makes structural decisions
about the modeling stack. It detects drift, routes decisions to the most
trustworthy models, and triggers adaptation when performance degrades.

**7.1 Drift Detection**

**7.1.1 Feature Drift (Covariate Shift)**

Track rolling statistics (mean, variance, quantiles) of each feature
against a reference window. Population Stability Index (PSI) or
Kolmogorov-Smirnov tests per feature, computed daily. The GP familiarity
layer contributes: downward-trending familiarity scores indicate drift
into unfamiliar state space.

**7.1.2 Outcome Drift (Concept Shift)**

The relationship between features and outcomes has changed even if
features look normal. Detected via the shadow module\'s prediction
residuals. CUSUM (cumulative sum) or Page-Hinkley tests on the residual
stream catch gradual drift. Sudden spikes in residual magnitude catch
abrupt shifts (illness, travel, medication change).

**7.1.3 Action Drift (Behavioral Shift)**

The user\'s response to recommendations has changed. Tracked via rolling
acceptance rate, compliance gap (recommended vs. actual settings), and
revert frequency. Significant shifts signal a need to update the user
model or increase conservativeness.

**7.1.4 Regime Change (Structural Break)**

Step-function changes that invalidate prior learning: pregnancy, new
medication, timezone move, pump switch. Detected via simultaneous
multi-feature shifts or manual user flags. Response is aggressive:
potentially reset shadow graduation clock and enter rapid-relearning
phase.

**7.2 Model Trust Routing**

The meta-controller maintains a trust weight per model, derived from
recent shadow accuracy, calibration quality, regime compatibility, and
staleness. Trust weights feed into the confidence module (weighting
ensemble predictions) and the optimization engine (which can restrict to
top-K trusted models).

**7.3 Adaptation Escalation Ladder**

When drift is detected or trust weights degrade, the meta-controller
follows an escalation sequence, trying the least disruptive intervention
first:

  ---------- ---------------- ---------------------------------------------------
  **Step**   **Action**       **Description**

  1          **Reweight**     Shift ensemble weights toward better-performing
                              models. No retraining. Happens automatically via
                              trust weight updates.

  2          **Fine-tune**    Continue training the best model on recent data
                              (additional trees for XGBoost, few epochs for
                              neural). Preserves existing knowledge.

  3          **Retrain**      Full retraining with a window emphasizing recent
                              data (higher weight). Expensive but necessary after
                              regime changes.

  4          **Architecture   Swap to a different model type (e.g., fall back
             Switch**         from transformer to XGBoost during data-sparse
                              recovery periods).

  5          **Add Model**    If no architecture adapts well, flag for human
                              review and potential new model design.
                              Human-in-the-loop for the POC.
  ---------- ---------------- ---------------------------------------------------

Each escalation step requires a trigger condition (e.g., \'reweighting
did not improve scorecard after 7 days → escalate to fine-tuning\').
This prevents overreaction.

**7.4 Action Family Selection**

The meta-controller decides whether the current situation calls for a
TherapyParam change, a BehaviorPlan nudge, or both. This is based on
outcome attribution: if the dominant BG signal correlates with
miscalibrated settings, route to therapy. If it correlates with
behavioral patterns (late eating, poor sleep, exercise timing), route to
behavior. Initially rule-based; can be learned over time.

**7.5 Model Registry**

A structured catalog of every model trained, with metadata: architecture
type, training date and data window, hyperparameters, validation
metrics, current trust weight, status
(active/standby/deprecated/retraining), drift sensitivity, and
applicable regime tags. The registry enables introspection and powers
explainability features in the UI.

**8. Optimization Engine (Block 7)**

The optimizer receives a state, a baseline action, a user profile, and
access to the model zoo. It returns a recommended action that maximizes
risk-adjusted reward subject to safety constraints --- or \'no
recommendation\' if nothing beats the baseline confidently.

**8.1 Action Space**

For the therapy path, an action is a TherapySchedule: a
piecewise-constant function over time-of-day intervals, each specifying
ISF, CR, and basal values. Interval boundaries are set by the user\'s
minimum interval length preference (hourly, 4-hour, day/night). For the
behavior path, an action is a BehaviorPlan with time-boxed items
(exercise, wind-down, breathing, meal windows), each with typed
parameters, burden estimates, and constraints.

**8.2 Search Strategy Tiers**

**8.2.1 Tier 1: Grid Search With Pruning**

Discretize the action space: for each interval, define a grid of
ISF/CR/basal values centered on current settings, bounded by the
aggressiveness profile\'s max deviation. Decompose by optimizing each
interval independently, then jointly evaluate the top-K combined
schedules. Easy to implement, easy to debug; the bottleneck is zoo
evaluation speed (fast for XGBoost).

**8.2.2 Tier 2: Bayesian Optimization**

Fits a GP surrogate over the (action → predicted reward) surface and
uses an acquisition function (Expected Improvement or Constrained EI) to
decide which candidates to evaluate. Sample-efficient for expensive
evaluations. Naturally handles the constraint and uncertainty objectives
via CVaR-based acquisition.

**8.2.3 Tier 3: Offline RL Policy**

A policy network that directly maps states to actions, trained via
conservative Q-learning on the trajectory dataset from the forked sim.
Advantage is speed (single forward pass); disadvantage is data
requirements and difficulty with constraints. Runs alongside
search-based optimizer as a warm start, not a replacement.

**8.3 Reward Function**

Constructed from the user\'s objective weights:

reward = w_tir × TIR - w_low × %low - w_high × %high - w_var × BG_var +
w_stab × stability

Augmented with:

- **Change penalty:** Cost proportional to the magnitude of deviation
  from current settings, scaled by the user\'s conservativeness.
  Prevents unnecessary disruption.

- **Uncertainty penalty:** Risk-adjusted reward subtracts a term
  proportional to prediction uncertainty. Prefers confident
  recommendations even at slightly lower expected reward.

**8.4 Constraint System**

**8.4.1 Pre-Filter (Before Evaluation)**

- User-locked time windows (e.g., \'do not change night basal\')
  eliminate candidates.

- Aggressiveness bounds (e.g., ISF ±10% max) prune the search space.

- Minimum step size filters (e.g., changes \< 2 units are ignored).

**8.4.2 Post-Evaluation Safety Check**

For each hard safety constraint, check the upper bound of the prediction
interval (e.g., 90th percentile %low). If the worst-case scenario
violates the constraint, the candidate is rejected. This is deliberately
conservative.

**8.5 Output: RecommendationPackage**

The optimizer produces a package containing: the primary recommended
action with predicted outcomes, confidence score, improvement vs.
baseline, explanation, and risk assessment; 0--2 diverse alternative
strategies; the baseline prediction (what happens if the user changes
nothing); and the overall decision (recommend, hold, or insufficient
confidence).

A \'hold\' decision (recommending no change) is actively communicated
rather than silent. The user seeing \'your current settings look good
for today\' builds trust and is itself a logged prediction that gets
scored.

**9. Evaluation Layer (Block 9)**

Robustness is proven by triangulation across multiple independent
evaluation methods. Consistency across all methods constitutes the
robustness proof.

**9.1 Evaluation Methods**

- **Surrogate model replay:** Replay historical days with recommended
  settings through the BG dynamics surrogate. Compare predicted outcomes
  under recommended vs. actual settings.

- **Simulator cross-validation:** Test recommended policies on virtual
  patients in the full t1d_sim. Measures generalization across patient
  profiles.

- **Off-policy evaluation:** Statistical reweighting methods (Inverse
  Propensity Weighting, doubly robust estimators) applied to
  observational data. Provides estimates without requiring intervention.

- **Shadow mode retrospective:** Long-term logs of \'what the model
  would have done\' compared against what actually happened. The shadow
  scorecard is the primary vehicle.

- **Forked timeline causal analysis:** Direct comparison of sibling
  branches in the simulation tree. The gold standard for causal effect
  estimation.

**9.2 Using the Forked Tree**

The forked timeline tree enables evaluation questions that no other
method can answer. These include the causal effect of any specific
recommendation controlling for all prior decisions; whether the
system\'s policy improves within a patient over time; the optimal
recommendation frequency and magnitude; and where and why the system
fails.

**10. System Flow: Daily Cycle**

The following describes the end-to-end daily cycle once all blocks are
operational:

5.  **Data ingestion:** HealthKit sync triggers CTXBuilder →
    FeatureJoiner → FeatureFrameHourly. Uploaded to Firestore.

6.  **Drift check:** Meta-controller checks feature statistics, residual
    trends, and GP familiarity against reference windows.

7.  **Model health:** Meta-controller updates trust weights from latest
    scorecard. Escalates adaptation if needed.

8.  **Recommendation generation:** Optimization engine receives state +
    user profile. Searches action space via current tier (grid/BO/RL).

9.  **Ensemble evaluation:** Each candidate action is scored by the
    active zoo models. PredictionEnvelopes collected.

10. **Confidence gating:** Four-layer gate runs: GP familiarity,
    ensemble agreement, calibration check, effect size. Hard safety gate
    runs independently.

11. **Shadow logging:** Full ShadowRecord written with state, action,
    predictions, gate decision.

12. **User interaction:** If graduated: recommendation surfaces with
    confidence + explanation + preview. If not: silent logging only.

13. **User response:** Accept, reject, or partial comply. Logged to
    shadow record.

14. **Outcome arrival:** \~24h later, actual outcomes enrich the shadow
    record.

15. **Retrospective evaluation:** Counterfactual estimate computed.
    Per-model accuracy scored. Scorecard updated.

16. **Graduation check:** Scorecard conditions evaluated. Graduate,
    maintain, or de-graduate as appropriate.

**11. Deployment Phases**

  ------------ ------------------ ----------------------------------------
  **Phase**    **Mode**           **Description**

  **Phase 1**  Silent / Shadow    Backend-only. Models train, shadow
                                  records accumulate. No user-facing
                                  output. Goal: prove the system can
                                  predict accurately.

  **Phase 2**  Preview            Users see recommendations but cannot
                                  apply them. Goal: validate UX, gather
                                  subjective feedback, monitor acceptance
                                  patterns.

  **Phase 3**  Controlled Rollout Apply button enabled with safety nets.
                                  Automatic revert on adverse outcomes.
                                  Tight constraint bounds. Goal: prove
                                  safety in production.

  **Phase 4**  Full Deployment    Adaptive recommendations continuously
                                  refined. Constraint bounds widen as
                                  trust builds. Behavioral action family
                                  activated.
  ------------ ------------------ ----------------------------------------

**12. Vision Beyond Insulin**

The Chamelia architecture is domain-agnostic. The nine blocks generalize
to any context with streaming time-series health data, tunable
parameters, and measurable outcomes. If proven in diabetes, the same
skeleton applies to continuous insulin delivery (closed-loop), sleep
optimization (bedtime, wake time, light exposure), exercise
prescriptions (intensity, duration, timing, rest days), and nutrition
planning (meal timing, carb distribution, macro targets).

The action surface is explicitly pluggable. TherapyParam is one plugin.
BehaviorPlan is another. Any new action family can be added by defining
its action schema, registering models in the zoo that predict outcomes
given that action type, and letting the meta-controller learn when to
route to it.

The core contribution is not an insulin optimizer --- it is a
general-purpose, self-learning recommender framework that collects
streaming data, generates parameter recommendations, proves robustness
before surfacing, adapts its own model architecture, and exposes
user-friendly control knobs. If successful, this framework becomes the
foundation for autonomous, robust, and user-guided AI health copilots
across domains.

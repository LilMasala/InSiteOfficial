**CHAMELIA**

World Runner, Meta-Controller & Human-Centered Recommendation Engine

System Specification v2.0

March 2026 • InSite Project • Confidential

**Table of Contents**

**1. World Runner: Autonomous Learning Engine**

The world runner replaces the manual train→run→evaluate→retrain cycle
with a single autonomous process. It manages a population of patients at
different lifecycle stages simultaneously, handles model training and
retraining, operates the meta-controller\'s full decision apparatus, and
evolves each patient\'s therapy mode through the unlock ladder --- all
within a single invocation.

**1.1 Design Philosophy**

The fundamental goal is not to optimize time-in-range. It is to help
each person feel that their diabetes is manageable and that things are
getting better. TIR improvement is one way that happens, but the system
must also optimize for mood, sustainability, and the person\'s sense of
agency. A system that achieves 85% TIR but causes anxiety and burnout
has failed.

Every design decision flows from this principle: the system earns trust
incrementally, adapts its behavior to the person\'s personality and
emotional state, backs off when the person is struggling, celebrates
when things go well, and never sacrifices the human relationship for a
marginal statistical improvement.

**1.2 Invocation**

python -m chamelia.run \\ \--n_patients 100 \--days 180 \--seed 42 \\
\--outdb world.db \\ \--learning-mode hybrid \\ \--initial-zoo
zoo_v2/zoo.pkl

**1.3 Core Loop**

The world runner interleaves simulation and learning day-by-day rather
than running all patients then training. On each global day: all
patients advance one simulated day (behavior, physiology, observation,
missingness), patients in shadow or intervention phases run the
recommendation cycle (optimizer, confidence gate, shadow logging),
outcome data is collected, and at configurable intervals the
meta-controller runs its full evaluation --- checking drift, updating
trust weights, evaluating retraining triggers, and managing the model
registry.

The key difference from the batch approach: the model improves while
patients are still being simulated. Patient 1 might graduate on Day 45
using model v1, while Patient 83 graduates on Day 90 using model v3 that
was retrained twice since Patient 1 started.

**1.4 Patient Lifecycle**

**Phase 0 → Phase 1:** Triggered when the model registry has at least
one active model AND the patient has accumulated a minimum observation
window (default 21 days). Patients created before any model exists stay
in Phase 0 until the first training run.

**Phase 1 → Phase 2:** Triggered when the patient\'s individual
scorecard meets graduation conditions (win rate \> 60%, zero safety
violations, calibration within tolerance, sustained 7+ days). Different
patients graduate at different times, or never.

**Phase 2 → Phase 1 (de-graduation):** Triggered when the scorecard
degrades below the de-graduation threshold. The patient drops back to
shadow-only until the system re-proves itself.

**2. Learning Modes**

Cross-patient learning is powerful but dangerous. Two patients who look
identical on paper can have completely different insulin dynamics. The
system provides three modes with an explicit user-facing toggle.

**2.1 Individual Mode**

Each patient gets their own model instance trained exclusively on their
data. No cross-patient information whatsoever. The cold-start problem is
real --- a patient needs 30--60 days before meaningful predictions.
Graduation is slow but recommendations are truly personalized.

Best for: experienced T1D patients with stable patterns who want maximum
personalization and are willing to wait.

**2.2 Community Mode**

One global model trained on all patients\' pooled data. Every patient
benefits from every other patient\'s data. Cold start is fast. The
downside: the model optimizes for average-case accuracy. Outlier
patients get worse predictions than the population mean.

Best for: initial deployment, small user bases, and patients with
typical insulin dynamics.

**2.3 Hybrid Mode (Recommended)**

A two-tier model structure. Tier 1 is a population prior trained on
pooled data, capturing general patterns (\'luteal phase increases
insulin resistance,\' \'sleep debt worsens morning BG\'). Tier 2 is an
individual residual model that predicts the difference between the
population model\'s prediction and the patient\'s actual outcome
(\'Patient 12 is 15% more cycle-sensitive than average\').

The final prediction is: population_prediction + individual_residual.

A personalization weight blends the tiers and evolves over time:

  ------------------ ------------------ ---------------------------------
  **Days**           **Weight**         **Behavior**

  0--14              0.0                Pure community. No individual
                                        data yet.

  14--60             0.0 → 0.5          Individual residuals begin
                                        contributing as data accumulates.

  60+                0.5 → 0.85         Individual dominates. Community
                                        provides fallback for rare/unseen
                                        states.
  ------------------ ------------------ ---------------------------------

The weight never reaches 1.0 --- the community prior remains available
as a fallback when the GP familiarity layer indicates the individual
model hasn\'t seen the current state (first illness, new site location,
regime change). In those moments the system automatically shifts weight
back toward the community prior for that specific prediction.

**2.4 Privacy Constraints**

No individual\'s data is ever visible to another patient. The community
model trains on aggregated, de-identified data. A patient can opt out of
contributing to the community model while still benefiting from it. In
the simulator this is a flag on PatientConfig. In production it is a
real privacy setting with Firestore security rules.

**2.5 User-Facing Control**

In the app: Settings → Learning Mode: \'Personal Only\' /
\'Community-Enhanced\' / \'Community.\' Default is Community-Enhanced
(hybrid). The explanation: \'Personal Only: recommendations based solely
on your data. Slower to start but fully personalized.
Community-Enhanced: starts with patterns learned from similar users,
then personalizes to you. Recommended for most users.\'

**3. Therapy Mode Unlock Ladder**

Each level is a small trust increment earned through demonstrated
performance. The system never leaps --- it earns the right to make
progressively more sophisticated changes to the user\'s therapy. The
initial block count and boundaries come from the user\'s actual pump
profile on day one, not from defaults.

  ----------- ---------------- ------------------------------------------------
  **Level**   **Mode**         **What the System Can Do**

  **0**       **Shadow**       Watch, learn, log. No user-facing output. The
                               system is accumulating data and proving it can
                               predict.

  **1**       **Suggest        Suggest ISF, CR, or basal value changes within
              Values**         the user\'s existing blocks. Structure
                               untouched. \'Change your 6am--noon ISF from 50
                               to 47.\'

  **2**       **Shift          Suggest moving block start/end times. Block
              Boundaries**     count stays the same. \'Shift your morning block
                               from 6am to 5:30am.\'

  **3**       **Create/Merge   Propose splitting or merging blocks. Total count
              Blocks**         changes, bounded by user\'s max-blocks
                               preference. \'Split your afternoon block at
                               3pm.\'

  **4**       **Context        Create named profiles and recommend switching
              Profiles**       based on context. \'Use your Luteal profile for
                               the next 10 days.\' Each profile is still
                               blocks.

  **5**       **Autonomous     Continuous B-spline functions conditioned on
              Curves**         context. Adapts daily. No manual blocks.
                               Requires explicit opt-in and clinical
                               endorsement.
  ----------- ---------------- ------------------------------------------------

**3.1 Unlock Graduation Criteria**

Each level transition requires all of the following to hold
simultaneously:

- **Minimum time at current level:** 21+ days operating successfully.

- **Positive causal effect:** Recommendations at the current level show
  sustained positive TIR delta.

- **Acceptance rate above threshold:** The user is actually following
  recommendations (not ignoring 80%).

- **Zero safety violations:** No predicted or actual hard constraint
  violations during the period.

- **Mood stability or improvement:** The user\'s mood trajectory has not
  degraded since entering the current level.

- **Explicit user opt-in:** Every level transition requires user
  confirmation. They can decline and stay at their current level
  indefinitely.

**3.2 Regression**

If performance degrades at any level, the system recommends dropping
back. This is de-graduation applied to therapy mode. The message might
read: \'Your recent outcomes have been less stable since enabling
profile switching. Would you like to return to single-profile mode while
the system recalibrates?\' The user decides.

**3.3 Level 5: Continuous Function Representation**

At Level 5, therapy settings are represented as B-splines with K knot
points (typically 6--8) spread across 24 hours. The value at any time t
is a weighted sum of basis functions: ISF(t) = Σ wᵢ Bᵢ(t). The weights
are conditioned on context --- sleep, exercise, cycle, stress --- so
different days produce different curves.

A slope constraint bounds the maximum rate of change: \|dISF/dt\| \<
max_slope. This prevents wild oscillations. The constraint is enforced
by bounding adjacent spline weight differences: \|wᵢ₊₁ - wᵢ\| \< Δmax.
Conservative users get tighter bounds; aggressive users allow steeper
transitions.

The optimizer at this level requires Bayesian optimization (Tier 2) or
the offline RL policy (Tier 3), as the action space is too large for
grid search.

**4. Meta-Controller: The Autonomous Brain**

The meta-controller makes five categories of decisions that together
make the world runner autonomous. It is the nervous system connecting
all other blocks into a coherent whole.

**4.1 Decision Category 1: When to Retrain**

The meta-controller maintains a retraining pressure score that
accumulates continuously and triggers a retrain when it crosses a
threshold. Multiple signals contribute pressure:

- **Data staleness pressure:** Accumulates daily. Rate proportional to
  new data volume, with intervention data contributing more than
  observation-only rows.

- **Performance degradation pressure:** Proportional to the magnitude of
  rolling win rate decline. A 2% drop adds mild pressure; a 10% drop
  adds heavy pressure.

- **Drift alarm pressure:** Burst pressure when drift detectors fire.
  Individual patient drift adds moderate pressure; correlated
  multi-patient drift adds heavy pressure.

- **Intervention data milestone pressure:** Large burst when the count
  of (state, action, outcome) triples where action differs from baseline
  crosses 100, 500, 1000, etc.

- **Negative causal delta pressure:** Strongest signal. Sustained
  negative accept-vs-reject TIR delta means recommendations are actively
  hurting. Continuous heavy pressure.

Pressure decays after each retrain (reset to zero) and during sustained
good performance (slow decay if win rate \> 65% and delta is positive).
Early in the run the retrain threshold is low (retrain eagerly). Later
it rises (retrain only with genuine reason).

**4.2 Decision Category 2: Training Data Curation**

- **Recency weighting:** Exponential decay --- today\'s data has weight
  1.0, 30 days ago has 0.7, 90 days ago has 0.4. XGBoost supports sample
  weights natively.

- **Intervention data upweighting:** Rows where the patient operated
  under changed settings get an additional weight boost. These teach
  causal effects.

- **Outlier downweighting:** Device hiatus days (mostly nulls, unrelated
  outcomes) get downweighted to prevent learning garbage patterns.

- **Per-patient balance:** Weights normalized so each patient
  contributes roughly equally regardless of data volume.

- **Learning mode influence:** In individual mode, training set is one
  patient\'s data. In community/hybrid, it is the weighted pool. The
  meta-controller manages the appropriate training runs.

**4.3 Decision Category 3: Escalation Ladder**

The meta-controller tries the cheapest intervention first, escalating
only when gentler approaches fail:

  ---------- ---------------- ------------------------------ -------------------
  **Step**   **Action**       **Description**                **Patience**

  1          **Reweight**     Shift ensemble trust weights   Wait 7 days
                              toward better-performing       
                              models. No retraining.         

  2          **Fine-tune**    Warm-start XGBoost with        Wait 7 days
                              50--100 new trees on recent    
                              data. Preserves existing       
                              knowledge.                     

  3          **Full retrain** Discard current model, train   Wait 14 days
                              from scratch on curated        
                              dataset emphasizing recent     
                              data.                          

  4          **Architecture   Activate additional zoo models Wait 21 days
             expand**         (temporal, surrogate). More    
                              ensemble opinions and better   
                              confidence.                    

  5          **Human          Flag for human review. System  N/A
             escalation**     continues at safe operating    
                              level.                         
  ---------- ---------------- ------------------------------ -------------------

**4.4 Decision Category 4: Model Registry Management**

**Active:** In the ensemble, receiving trust-weighted queries from the
optimizer.

**Candidate:** Just trained, undergoing 7-day shadow validation
alongside active models. Promoted to active only if it matches or
outperforms.

**Standby:** Previous version kept for instant rollback if a new model
proves worse.

**Deprecated:** Old models kept for audit trail, never queried.

The candidate validation period prevents bad retrains from degrading
recommendations. During validation, patients still receive
recommendations from the active model. The candidate shadows silently.

**4.5 Decision Category 5: Cross-Patient Intelligence**

In community and hybrid modes, the meta-controller manages cross-patient
transfer carefully:

- **Cohort detection:** Cluster patients by metabolic behavior (not
  demographics). Identify which dimensions of similarity predict similar
  therapeutic responses.

- **Transfer confidence:** For new patients, assess similarity to
  existing patients. High confidence → community prior is more
  informative. Low confidence → individual model weighted higher.

- **Anomaly isolation:** If a patient is a genuine outlier (doesn\'t
  cluster with anyone), shift their learning mode toward individual
  regardless of global setting.

- **Population drift monitoring:** Distinguish individual drift
  (reweight that patient) from systemic drift (retrain the population
  model).

**5. Personality-Aware Recommendation Engine**

The system adapts its behavior to the person\'s personality and
emotional state. Same optimizer, same models, same safety gates --- but
different recommendation cadence, framing, detail level, and emotional
tone.

**5.1 Personality Traits**

  ------------------------------- ----------- ----------------------------------------
  **Trait**                       **Range**   **Effect on System Behavior**

  **explanation_appetite**        0--1        0 = just tell me what to do. 1 = explain
                                              everything with confidence intervals and
                                              feature importances.

  **notification_tolerance**      0--1        0 = minimal contact, only surface
                                              high-confidence recs. 1 = keep me
                                              informed of everything.

  **autonomy_preference**         0--1        0 = I decide everything manually. 1 =
                                              handle it for me, auto-apply within
                                              safety bounds.

  **change_anxiety**              0--1        0 = changes don\'t bother me. 1 = every
                                              change is stressful. High values
                                              increase change penalty and revert
                                              sensitivity.

  **celebration_receptiveness**   0--1        0 = don\'t patronize me. 1 = I
                                              appreciate positive feedback. Controls
                                              frequency of \'things are going well\'
                                              messages.

  **education_need**              0--1        0 = I\'m an expert, skip the basics. 1 =
                                              explain what ISF means. Controls
                                              explanation depth.
  ------------------------------- ----------- ----------------------------------------

These traits can be set explicitly during onboarding, inferred from
behavior over time (frequent reverters have high change_anxiety, users
who never read explanations have low explanation_appetite), or a
combination.

**5.2 Personality Archetypes**

**5.2.1 The Anxious Optimizer**

Checks CGM 30 times a day. Stresses about every spike. Wants maximum
transparency. The system provides detailed explanations, confidence
intervals, and simulated previews. However, the system also sets their
revert threshold higher (don\'t revert after one bad day) and provides
reassurance alongside recommendations. High explanation_appetite, high
notification_tolerance, low autonomy_preference, high change_anxiety.

**5.2.2 The Hands-Off Delegator**

Tired of thinking about diabetes. Wants the system to handle it.
Minimize notifications, maximize recommendation quality (only surface
high-confidence suggestions), frame everything simply. Auto-apply
approved changes. Low explanation_appetite, low notification_tolerance,
high autonomy_preference, low change_anxiety.

**5.2.3 The Skeptic**

Doesn\'t trust algorithms. Needs to be proven wrong slowly. Extended
shadow mode, transparent track record display, comparison of system
suggestions to personal intuition. The unlock ladder moves slower. Low
autonomy_preference, high explanation_appetite, high change_anxiety.

**5.2.4 The Newly Diagnosed**

Barely understands pump settings. Needs education alongside
recommendations. Explain not just what to change but why, in plain
language. High education_need, moderate change_anxiety, moderate
celebration_receptiveness.

**6. Mood-Integrated Reward Function**

The system does not optimize for TIR alone. It optimizes for a composite
objective that includes mood, sustainability, and the person\'s sense of
progress.

**6.1 The Reward Function**

reward = w_tir × TIR - w_low × %low - w_high × %high + w_mood ×
predicted_mood_delta - w_burden × recommendation_burden - w_change ×
change_magnitude - w_anxiety × change_anxiety_cost

The mood weight means the optimizer naturally prefers gentler, less
disruptive recommendations when the patient\'s mood is fragile, and can
be more ambitious when the patient is in a good mental state and has
capacity for change.

The anxiety cost is a function of the user\'s change_anxiety personality
trait multiplied by the recommendation\'s novelty. A person\'s
first-ever recommendation carries high anxiety cost. Their 20th
recommendation (if previous ones went well) carries much less.

**6.2 Recommendation Cadence Modulation**

Recommendation frequency adapts to emotional state. The meta-controller
maintains a per-patient recommendation budget:

- **Budget fills:** When mood is stable or positive, and when the
  person\'s recent acceptance rate is healthy.

- **Budget drains:** When mood is negative, when acceptance rate drops
  (disengagement signal), or when multiple recent recommendations
  failed.

- **Budget empty:** The system goes quiet. No recommendations surfaced
  even if high-confidence options exist. The recommendation can wait.
  The person\'s mental health cannot.

- **Budget overflow:** System has been quiet for a while during good
  mood period. May surface a slightly bolder recommendation than usual.

**6.3 The Small Wins Strategy**

The system deliberately front-loads easy, high-probability-of-success
recommendations. The first recommendation a user ever sees should be
something they will almost certainly accept and that will almost
certainly help --- even if the improvement is tiny.

During the first 30 days of Phase 2, the change penalty is 2--3× the
normal level. The system proves itself on easy calls before attempting
harder ones. Each successful recommendation lowers the implicit barrier
for the next one, creating a virtuous cycle of trust building.

**6.4 Recommendation Framing**

The RecommendationPackage includes a framing field that controls how the
UI presents it:

  --------------------- ------------------------------------------------------
  **Framing**           **Context & Presentation**

  **tentative**         First time suggesting this type of change. Soft
                        language: \'Some people find that\...\' or \'You might
                        consider\...\'

  **reinforcing**       They\'ve done this before and it helped. \'Last time
                        you adjusted this, your mornings improved by 8%.\'

  **gentle_reminder**   They accepted but haven\'t followed through. No
                        judgment: \'Still interested in trying the morning
                        adjustment?\'

  **celebrating**       They did it and outcomes improved. \'Your Tuesday was
                        really solid, especially overnight. Your adjustment is
                        working.\'
  --------------------- ------------------------------------------------------

**6.5 Celebrating Improvements**

The system actively recognizes when things are going well. \'Your TIR
this week was 72%, up from 65% last month. Your morning numbers have
been especially stable.\' This is not a recommendation --- it is
positive reinforcement. It costs nothing and builds the emotional
foundation that makes future recommendations welcome.

In the shadow record schema, there is a record type for \'positive
observation\' alongside \'recommendation.\' The system logs moments
worth celebrating even when it has no suggestion to make.

**6.6 Backing Off During Difficulty**

When mood dips, the system backs off. Fewer recommendations. More
celebration of what is going well. Gentler framing. It waits for
stabilization before re-engaging.

The system never says \'your numbers have been bad this week, here\'s
what you should change.\' Instead, during a bad stretch it says nothing
about the bad numbers. When things improve, it says \'your Thursday was
really solid, especially overnight.\'

If a simulated or real patient\'s mood drops below a burnout threshold,
they disengage entirely --- acceptance rate goes to zero, logging
quality degrades, exercise drops. The system\'s job is to prevent this
from ever happening by managing its own intrusiveness.

**7. The Psychological Feedback Loop**

When the system gets this right, a virtuous cycle emerges:

1.  Small, comfortable recommendation → user accepts.

2.  BG improves slightly → user feels a sense of progress.

3.  Mood improves → better sleep (less anxiety) → better BG.

4.  System detects improvement → celebrates it.

5.  User feels supported → more open to next recommendation.

6.  Slightly bolder recommendation → user accepts → larger improvement.

7.  Genuine sense of \'my diabetes is getting easier\' → sustained mood
    improvement.

8.  System earns the right to suggest behavioral changes.

9.  User tries a morning walk → feels good physically AND proud.

10. Mood boost + BG boost → compound improvement.

The downward protection is equally important. When mood dips, the system
backs off. Fewer recommendations. More celebration. Gentler framing. It
waits for stabilization. It never kicks someone when they are down.

**7.1 Psychological Response Model (Simulation)**

Simulated patients need traits that model their emotional response to
system interactions:

  --------------------------------- --------------------------------------------
  **Trait**                         **Description**

  **mood_boost_from_success**       Positive mood delta when an accepted
                                    recommendation produces good outcomes.

  **mood_hit_from_failure**         Negative mood delta when an accepted
                                    recommendation makes things worse.

  **mood_hit_from_overload**        Negative mood delta from receiving too many
                                    recommendations in a short period.

  **mood_boost_from_celebration**   Positive mood delta from receiving positive
                                    reinforcement messages.

  **recommendation_fatigue_rate**   How quickly frequent recommendations become
                                    annoying (higher = faster fatigue).

  **burnout_threshold**             Mood level below which the patient
                                    disengages entirely. Acceptance drops to
                                    zero, logging degrades.
  --------------------------------- --------------------------------------------

These traits create a realistic simulation where the system\'s own
behavior affects patient engagement. An over-aggressive recommender
causes burnout. A well-calibrated one builds a sustained upward
trajectory.

**8. Meta-Controller State & Daily Loop**

**8.1 State**

MetaControllerState: pressure_score: float last_retrain_day: int
escalation_level: int escalation_patience_remaining: int registry:
ModelRegistry candidate_model_id: str \| None
candidate_validation_start: int \| None rolling_population_win_rate:
float rolling_causal_delta: float drift_alarm_count: int
intervention_triple_count: int patient_cohort_assignments: dict
global_learning_mode: str per_patient_recommendation_budget: dict

**8.2 Daily Check (Lightweight)**

Every global day, the meta-controller: accumulates pressure from all
signals, decays pressure if performance is strong, checks candidate
model validation period, checks retraining trigger against threshold.

**8.3 Weekly Evaluation (Full)**

Every 7 global days, the meta-controller: updates population-wide win
rate and causal delta, runs drift detection sweep across all patients,
updates patient cohort assignments (hybrid/community mode), updates
per-patient trust weights, checks de-graduation candidates, updates
therapy mode unlock eligibility, adjusts per-patient recommendation
budgets based on mood trajectories.

**8.4 Monitoring Output**

During execution the world runner prints periodic summaries showing:
patient phase distribution, active model count and version, branch count
and shadow record count, population win rate and causal delta,
graduation and de-graduation counts, and recommendation budget status.

**9. Implementation Roadmap**

**9.1 New Files**

- **chamelia/run.py:** World runner main loop with \_\_main\_\_.py entry
  point.

- **chamelia/personality.py:** UserPersonality traits, recommendation
  budget, framing selection.

- **chamelia/therapy_modes.py:** Unlock ladder state machine,
  level-specific optimizer constraints, block/spline representations.

**9.2 Modified Files**

- **chamelia/optimizer.py:** Accept therapy_mode_level constraint. Level
  1 uses current grid search. Levels 2--3 add boundary/structure search.
  Level 5 stubs spline weights.

- **chamelia/shadow.py:** Add \'positive_observation\' record type for
  celebrations. Add recommendation_budget tracking.

- **chamelia/meta_controller.py:** Full pressure-based retraining
  trigger. Candidate validation. Escalation ladder with patience.
  Cross-patient cohort detection.

- **chamelia/confidence.py:** Integrate mood-based gating (suppress
  recommendations when mood budget is empty).

- **t1d_sim/feedback.py:** Add psychological response model
  (mood_boost_from_success, mood_hit_from_overload, burnout_threshold).

- **t1d_sim/population.py:** Add UserPersonality to PatientConfig.
  Sample personality traits alongside agency traits.

- **t1d_sim/patient_threephase.py:** Wire personality into
  recommendation cycle. Respect recommendation budget. Generate
  celebration records.

**9.3 Build Order**

  ----------- ------------------------ ----------------------------------------
  **Phase**   **What**                 **Why This Order**

  1           **personality.py +       Pure dataclasses and state machines.
              therapy_modes.py**       Zero breakage. Testable in isolation.

  2           **Meta-controller        The retraining trigger is the core new
              pressure system**        logic. Must work before the world runner
                                       can be autonomous.

  3           **Meta-controller        Prevents bad retrains from degrading
              candidate validation**   production. Safety-critical.

  4           **Mood reward +          Integrates mood into the optimization
              recommendation budget**  objective and cadence modulation.

  5           **World runner           Orchestration layer that ties everything
              (chamelia/run.py)**      together. Depends on phases 1--4.

  6           **Psychological response Adds mood feedback from system
              model in sim**           interactions. Makes sim produce
                                       realistic engagement trajectories.

  7           **Therapy mode unlock    Wire level transitions into the world
              integration**            runner loop. Level 1 full, 2--4 stubbed,
                                       5 design-only.

  8           **End-to-end             100 patients, 180 days, verify:
              validation**             autonomous retraining, positive causal
                                       delta, mood-aware cadence, no burnout.
  ----------- ------------------------ ----------------------------------------

**9.4 Success Criteria**

The world runner is considered successful when a 100-patient, 180-day
run demonstrates:

- Autonomous retraining triggered 3--5 times based on pressure signals,
  not manual intervention.

- Positive causal delta (accept vs reject TIR) of +1% or greater by the
  final 30 days.

- Zero safety violations across all patients and branches.

- Mood trajectory: population average mood is stable or improving, not
  degrading.

- No simulated patient hitting the burnout threshold due to system
  overload.

- At least 60% of patients graduating from Phase 1 to Phase 2.

- At least 10% of patients unlocking Level 2 (boundary shifts) by day
  180.

- Model registry shows candidate validation catching at least one bad
  retrain attempt.

**10. Vision**

The world runner is the realization of the Chamelia vision: an
autonomous, self-learning, human-centered health copilot that collects
streaming data, generates recommendations, proves its own robustness,
adapts its model architecture, respects the person\'s emotional state
and personality, earns trust through demonstrated results, and
progressively unlocks more sophisticated capabilities as that trust
grows.

If successful in insulin therapy, the same architecture generalizes to
any domain with streaming health data, tunable parameters, and
measurable outcomes --- with the critical addition that it always treats
the human not as a compliance target but as a whole person whose mood,
autonomy, and quality of life are first-class optimization objectives.

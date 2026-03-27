Phase 1 — Validate the pre-JEPA MVP actually tells the truth
Goal
Make sure the new simulation runner is not just runnable, but decision-useful.
What should happen here
After the runner is built, you should run repeated experiments like:
100 patients / 180 days
300 patients / 180 days
1000 patients / 180 days
multiple seeds
And check:
Does late-window TIR improve vs early-window TIR?
Does intervention-phase TIR improve vs pre-intervention baseline?
Are recommendations being accepted?
Are burnout / disengagement rates low?
Do some patients graduate while some do not?
This phase is about sanity and repeatability, not adding new modeling yet.
Why this comes first
Because if the report is unstable or misleading, every later architectural improvement will be hard to interpret.
Phase 2 — Tighten the evaluation contract
Once the runner works, the next plan should be to improve how you know it’s working.
Right now the repo already contains the ingredients for stronger evaluation:

shadow records,
scorecards,
branch/fork logic,
evaluation layer ideas,
counterfactual fields. chamelia/shadow.py; t1d_sim/patient_threephase.py; specs/architecture_v1_spec.md.
What to improve next
1. Better causal measurement
Move from:
simple “early vs late TIR”
toward:
matched within-patient intervention effects,
accepted vs rejected comparisons,
eventually sibling-branch comparisons.
2. Better scorecard truth
Use actual:
TIR deltas,
safety violations,
calibration quality,
familiarity,
context consistency,
instead of runner-level proxies.
3. Better viability dashboards
Add outputs like:
TIR by patient cohort,
recommendation success by personality type,
burnout by recommendation cadence,
retrain events vs performance shifts.
Why this is Phase 2
Because after the MVP, the biggest lever is not adding smarter models yet — it is making the evidence sharper.
Phase 3 — Harden the adaptive loop
Once the system can show directional benefit and the evaluation is more trustworthy, the next plan is to make the autonomous adaptation loop real.
The spec already makes this a central feature:

retraining pressure,
drift detection,
candidate validation,
escalation ladder,
model registry management. specs/world_runner_spec.md; chamelia/meta_controller.py.
What gets built here
1. Real candidate-vs-active validation
Replace heuristic comparisons with real head-to-head scoring on shadow records.
2. Real drift signal ingestion
Feed:
feature windows,
residual streams,
acceptance changes,
regime events
into MetaController rather than relying mostly on summary counters.
3. Stronger model registry persistence
Persist:
active/candidate/standby transitions,
validation metrics,
retrain reasons,
promotion/rejection events.
Why this matters
This is the stage where Chamelia starts becoming a true autonomous learner instead of a simulation with periodic retrain logic.
Phase 4 — Expand the action/control surface carefully
Only after the basic loop and adaptation are trustworthy should you broaden the action space.
The long-term plan in the specs is:

Level 1: value changes
Level 2: boundary shifts
Level 3: create/merge blocks
Level 4: profiles
Level 5: curves. specs/world_runner_spec.md; chamelia/therapy_modes.py.
But the immediate next step should be only:
Level 2 support
Implement a real action representation for:
shifting schedule boundaries,
not just global multipliers.
Then evaluate:
does added action freedom improve TIR?
does it reduce acceptance?
does it increase burnout or safety risk?
Why not jump to all therapy levels?
Because every expansion in action space increases:
optimizer complexity,
interpretability burden,
recommendation burden,
causal ambiguity.
So this should be incremental and measured.
Phase 5 — Implement true learning modes
The specs place a lot of weight on:
individual,
community,
hybrid modes. specs/world_runner_spec.md.
But the current code mostly stores the mode rather than changing model behavior. chamelia/run.py; chamelia/therapy_modes.py.
After the pre-JEPA runner, the next plan should be:
1. Community mode
One pooled model — likely what you’ll have first anyway.
2. Hybrid-lite
Population prior + lightweight patient correction.
3. Individual mode
Only after enough patient-specific data exists.
Why this phase comes before JEPA
Because learning-mode semantics are operationally foundational.
You want to know how the system behaves under:
pooled learning,
personalized learning,
blended learning,
before you upgrade the representation engine underneath.
Phase 6 — Add JEPA / latent patient state
This is where JEPA belongs.
Not at the beginning, but after:

the runner exists,
the reporting is honest,
adaptation is real,
learning modes mean something,
the training/evaluation contracts are stable.
At that point, a JEPA-style latent world model can become the next major architecture layer.
Where JEPA should slot in
Your description is strong, and the repo already points toward this future:
feature frames already act like a structured context representation,
zoo models already act like decoder heads,
shadow records and trajectories already form the right training substrate. specs/architecture_v1_spec.md.
JEPA should likely be introduced as:
1. A latent patient-state encoder
Encode recent context into a patient embedding.
2. A predictive latent transition model
Predict next embedding from current embedding + action + context.
3. A shared representation for:
outcome decoders,
regime detection,
cohort similarity,
planning,
eventually world-model behavior.
Why later
Because then JEPA is improving an already coherent loop, not compensating for an incoherent one.
Phase 7 — World-model-driven planning and richer optimization
After JEPA lands successfully, the plan becomes:
richer counterfactual planning,
action-impact comparison in latent space,
improved optimizer search,
better multi-objective consistency.
That would be the moment to revisit:
Bayesian optimization,
offline RL warm starts,
more expressive world-model-driven policy search. chamelia/optimizer.py; specs/architecture_v1_spec.md


Notes and others 
Implement burnout attribution in /workspace/PhysiologyT1DSimulator.

Goal:
Decompose burnout into:
1. realized burnout
2. counterfactual burnout risk attributable to the recommendation policy

Estimate attribution using patient-matched forked rollouts:
- same patient
- same local decision state
- baseline/null action vs treated/recommended action
- repeated stochastic rollouts when practical

Do NOT add JEPA or latent world models yet.

Use existing code:
- `chamelia/run.py`
- `t1d_sim/patient_threephase.py`
- `t1d_sim/population.py`
- existing personality / mood / shadow machinery

Requirements:
- keep current action space (global ISF/CR/basal multipliers)
- define evaluable recommendation decision points
- estimate burnout probability under treated vs baseline over a configurable horizon
- compute delta_burnout_risk = P(burnout|treated) - P(burnout|baseline)
- report both realized burnout and attributable burnout risk in the final simulation summary
- add tests
- document assumptions and limitations

Prefer the smallest honest implementation over a broad redesign.
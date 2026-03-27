# Chamelia: Mathematical Formulation of a LeCunian Health Decision Architecture

**Version 1.1 — March 2026**

**Revision notes.** This version addresses five structural critiques of v1.0: (1) the system is now explicitly a belief-state constrained control problem; (2) intrinsic cost and epistemic uncertainty are cleanly separated into distinct formal layers; (3) the trainable critic is sharpened as a terminal-value / residual-horizon estimator; (4) the digital twin is decomposed into prior, posterior, dynamic state, and rollout noise; (5) the configurator is formalized as a meta-policy. All calculations are tightened and holes filled.

---

## SECTION 1 — SYSTEM IDENTIFICATION

### Problem Class

Chamelia is a **belief-state constrained stochastic control problem with coupled physiological–psychological dynamics, causal attribution requirements, and lexicographic safety constraints**, organized within a LeCunian energy-minimization architecture.

The critical upgrade from v1.0: the system never has access to the true patient state $x_t$. It maintains a *belief* $b_t$ — a probability distribution over possible states — and all decisions are functions of this belief. This is not a notational convenience; it is the central mathematical fact that unifies confidence gating, anomaly detection, conservative action selection, and calibration monitoring into a single framework.

More precisely, the system belongs to the intersection of:

1. **Belief-state MDP (equivalently, POMDP solved in belief space).** The true patient state $x_t \in \mathcal{X}$ is never directly observed. The system receives noisy, incomplete signals $o_t$ and maintains a belief $b_t \in \mathcal{P}(\mathcal{X})$ over the latent state. All downstream modules — World, Cost, Actor — operate on $b_t$, not on $x_t$.

2. **Constrained stochastic control with lexicographic safety barriers.** The system operates under inviolable constraints (e.g., $P(\text{hypoglycemia} > \epsilon) < \delta$) that cannot be traded off against reward at any exchange rate. These are barrier constraints enforced at the belief level: if the belief admits any plausible state violating safety, the action is blocked.

3. **Causal decision problem.** The system must distinguish outcomes caused by its recommendations from outcomes that would have occurred under a null policy. Burnout risk must be decomposed into endogenous and policy-attributable components via counterfactual reasoning.

4. **Risk-sensitive planning problem.** The system optimizes a risk measure (CVaR) of a *normative cost* function, subject to *epistemic feasibility constraints* — not a cost function that mixes the two.

5. **Multi-timescale dynamical system.** Glucose dynamics (minutes–hours), sleep/exercise effects (hours–days), trust and engagement (days–weeks), burnout (weeks–months). The system must reason across all horizons simultaneously.

### Why Belief-State Framing Matters

Elevating the belief state from an implementation detail to the central mathematical object resolves several tensions:

- **Confidence gating** becomes: "is the belief $b_t$ concentrated enough to support action?" — a property of $b_t$, not a separate heuristic.
- **Anomaly detection** becomes: "is the observation $o_t$ consistent with the current belief?" — a likelihood test against $b_t$.
- **Conservative action selection** becomes: "does the action remain safe under all plausible states in the support of $b_t$?" — a robust optimization over $b_t$.
- **Calibration** becomes: "are the belief's predictive distributions well-calibrated empirically?" — a statistical property of $b_t$.

All of these were separate mechanisms in v1.0. They are now unified under one object.

### Why This Is More Than Standard RL

Standard RL optimizes cumulative discounted reward in an MDP. Chamelia departs in five critical ways:

- **Belief-state control.** The "state" is a distribution, not a vector. The transition dynamics include a Bayesian update step.
- **Coupled human–system dynamics.** The agent's policy changes the transition kernel (trust, engagement, compliance evolve in response to recommendations).
- **Lexicographic safety.** Hard constraints cannot be encoded as reward penalties.
- **Causal attribution.** The system must report policy-attributable risk, not just outcomes.
- **Non-stationarity by design.** The patient drifts, the model improves, the trust builds or erodes.

---

## SECTION 2 — NOTATION

### Three Ontological Categories

Following the reviewer's critique, all quantities are now organized into three clean categories:

**A. What the world does** (dynamics, transitions, stochastic outcomes):

| Symbol | Domain | Description |
|--------|--------|-------------|
| $x_t$ | $\mathcal{X} \subseteq \mathbb{R}^{d_x}$ | True latent world/patient state (never directly observed) |
| $x_t^{\text{phys}}$ | $\mathbb{R}^{d_p}$ | Physiological substate (insulin sensitivity, glucose dynamics, hormonal state) |
| $x_t^{\text{psy}}$ | $\mathbb{R}^{d_q}$ | Psychological/behavioral substate (mood, engagement, trust, burden) |
| $o_t$ | $\mathcal{O} \subseteq \mathbb{R}^{d_o}$ | Observed signals (CGM, HR, activity, mood logs, etc.) |
| $p(x_{t+1} \mid x_t, \tilde{a}_t, \xi_t)$ | — | True transition dynamics |
| $p(o_t \mid x_t, \nu_t)$ | — | Observation/emission model |
| $r_t$ | $\{0, \delta, 1\}$ | User response: reject (0), partial comply ($\delta$), accept (1) |
| $\tilde{a}_t$ | $\mathcal{A}$ | Realized action after compliance noise |
| $\xi_t$ | — | Process noise (stochastic life events, meal timing, etc.) |
| $\nu_t$ | — | Observation noise (CGM lag, drift, missingness) |

**B. What is undesirable in the world** (normative cost — purely about outcomes):

| Symbol | Domain | Description |
|--------|--------|-------------|
| $C^{\text{glyc}}_t$ | $\mathbb{R}_{\geq 0}$ | Glycemic discomfort (lows, highs, variability) |
| $C^{\text{burden}}_t$ | $\mathbb{R}_{\geq 0}$ | Recommendation burden cost |
| $C^{\text{trust}}_t$ | $\mathbb{R}_{\geq 0}$ | Trust erosion cost |
| $C^{\text{burn}}_t$ | $\mathbb{R}_{\geq 0}$ | Burnout-related cost |
| $C^{\text{int}}_t$ | $\mathbb{R}_{\geq 0}$ | Total intrinsic cost (sum of above) |
| $\hat{V}_\psi$ | $\mathbb{R}$ | Critic: terminal value estimate beyond rollout horizon |

**C. What the system does not know well** (epistemic state — about model/knowledge limitations):

| Symbol | Domain | Description |
|--------|--------|-------------|
| $b_t$ | $\mathcal{P}(\mathcal{X})$ | Belief state: probability distribution over true latent state |
| $\Sigma_t^{\text{pred}}$ | $\mathbb{R}^{d_y \times d_y}$ | Predictive uncertainty of the world model |
| $\kappa_t$ | $[0, 1]$ | GP familiarity score (how well-covered is $b_t$ by training data) |
| $\rho_t$ | $[0, 1]$ | Ensemble concordance (do models agree given $b_t$?) |
| $\eta_t$ | $[0, 1]$ | Calibration quality (are prediction intervals historically honest?) |
| $\mathcal{F}_t$ | $\{0, 1\}$ | Epistemic feasibility flag: $\mathcal{F}_t = 1$ iff $\kappa_t, \rho_t, \eta_t$ all exceed thresholds |

### Twin Parameters (Four-Part Decomposition)

| Symbol | Description |
|--------|-------------|
| $\theta^{\text{prior}}_i$ | Population/cohort prior for patient $i$ (archetype-derived) |
| $\theta^{\text{post}}_i(t)$ | Patient-specific posterior estimate at time $t$ (updated from data) |
| $x_t$ | Current latent dynamic state (changes hourly–daily) |
| $\xi_t$ | Rollout noise / exogenous stochastic perturbation |

Formally, the twin for patient $i$ at time $t$ is the tuple:
$$\mathcal{T}_i(t) = \bigl(\theta^{\text{prior}}_i, \; \theta^{\text{post}}_i(t), \; x_t, \; p(\xi)\bigr)$$

### Actions

| Symbol | Domain | Description |
|--------|--------|-------------|
| $a_t$ | $\mathcal{A}$ | Proposed action (therapy change, behavioral nudge, or hold) |
| $\mathbf{a}_{t:t+K}$ | $\mathcal{A}^K$ | Action sequence over horizon $K$ |
| $a^0$ | $\mathcal{A}$ | Null/hold action |

### Memory

| Symbol | Description |
|--------|-------------|
| $\mathcal{M}_t$ | Short-term memory buffer at time $t$ |

### Configurator

| Symbol | Description |
|--------|-------------|
| $\phi_t$ | Configurator state (meta-policy output): masks, horizons, weights, constraints |

### Horizons

| Symbol | Typical Range | Description |
|--------|---------------|-------------|
| $H_{\text{perc}}$ | 6–24 h | Perception lookback |
| $H_{\text{short}}$ | 4–24 h | Short rollout (glucose preview) |
| $H_{\text{med}}$ | 3–14 d | Medium rollout (therapy evaluation) |
| $H_{\text{long}}$ | 30–90 d | Long horizon (burnout, trust dynamics) |
| $H_{\text{burn}}$ | 14–60 d | Burnout attribution horizon |
| $H_{\text{mem}}$ | 30–90 d | Memory retention |
| $H_{\text{config}}$ | 1–7 d | Configurator update cadence |

### Safety, Burnout, Trust, Burden

| Symbol | Domain | Description |
|--------|--------|-------------|
| $g_j(b_t, a_t)$ | $\mathbb{R}$ | $j$-th safety constraint evaluated under belief; feasibility requires $g_j \leq 0$ |
| $B_t$ | $[0, 1]$ | Burnout state (continuous disengagement level) |
| $\lambda^B_t$ | $\mathbb{R}_{\geq 0}$ | Burnout hazard rate under current policy |
| $\Delta^B_H(\pi)$ | $\mathbb{R}$ | Policy-attributable burnout risk over horizon $H$ |
| $\tau_t$ | $[0, 1]$ | Trust level |
| $\beta_t$ | $\mathbb{R}_{\geq 0}$ | Cumulative recommendation burden |
| $\omega_t$ | $[0, 1]$ | Engagement level |

---

## SECTION 3 — LECUNIAN ROLE FORMALIZATION

### 3.1 Configurator (Meta-Policy)

**Mathematical Role.** The Configurator is a **meta-policy** $\mu$ that maps a summary of the current belief, memory, and user preferences to a configuration vector that parameterizes all other modules:

$$\phi_t = \mu\bigl(\text{summary}(b_t), \; \text{stats}(\mathcal{M}_t), \; \text{prefs}(u), \; \text{drift\_flags}_t\bigr)$$

The output $\phi_t = (\phi^{\text{perc}}_t, \phi^{\text{world}}_t, \phi^{\text{cost}}_t, \phi^{\text{act}}_t)$ specifies:

- $\phi^{\text{perc}}_t = (M_{\text{signal}}, H_{\text{perc}}, \delta_{\text{anom}})$: active signal mask $M_{\text{signal}} \in \{0,1\}^{38}$, perception horizon, anomaly sensitivity
- $\phi^{\text{world}}_t = (\mathcal{I}_{\text{model}}, H_{\text{short}}, H_{\text{med}}, N_{\text{roll}})$: model index set, rollout horizons, stochastic rollout count
- $\phi^{\text{cost}}_t = (\mathbf{w}, \{\epsilon_j\}, H_{\text{burn}})$: cost weight vector, safety thresholds, burnout attribution horizon
- $\phi^{\text{act}}_t = (\Delta_{\max}, \mathcal{F}_{\text{families}}, N_{\text{search}})$: action deviation bounds, allowed action families, search budget

**The meta-policy decides adaptively:**
- When belief entropy $\mathbb{H}(b_t)$ is high → shrink $\Delta_{\max}$, increase $N_{\text{roll}}$, tighten $\epsilon_j$
- When the scorecard win rate is strong → expand $\Delta_{\max}$, allow richer action families
- When drift flags fire → increase $H_{\text{perc}}$, reduce $H_{\text{med}}$ (don't trust long rollouts during drift), escalate adaptation
- When the patient has been stable for weeks → relax rollout count, lengthen $H_{\text{config}}$

**Learned vs. Explicit.** In the POC, $\mu$ is a rule-based function (lookup tables and threshold logic). The mathematical formulation as a meta-policy makes it clear what "learning the configurator" means later: learning $\mu$ from the (meta-state, configuration, downstream-performance) triples stored in memory.

**Interfaces:** Reads from Memory (scorecard statistics), Perception (belief summary), and user preferences. Writes configuration to all other modules.

### 3.2 Perception Module (Belief Estimator)

**Mathematical Role.** The Perception module is a **belief estimator**: it maintains and updates a probability distribution $b_t$ over the true latent state $x_t$ given all observations.

Formally, the belief update follows a Bayesian filtering recursion:

**(a) Prediction step** (time update, using the world model):
$$\bar{b}_t(x) = \int p(x \mid x', \tilde{a}_{t-1}, \theta^{\text{post}}_i(t)) \; b_{t-1}(x') \; dx'$$

**(b) Update step** (measurement update, incorporating new observation):
$$b_t(x) = \frac{p(o_t \mid x) \; \bar{b}_t(x)}{\int p(o_t \mid x') \; \bar{b}_t(x') \; dx'} \propto p(o_t \mid x) \; \bar{b}_t(x)$$

In practice, the full Bayesian recursion is intractable for the high-dimensional state. The POC implements an **approximate belief** using:

- **Point estimate + uncertainty:** $b_t \approx \mathcal{N}(\hat{x}_t, \Sigma_t)$ where $\hat{x}_t$ is the MAP state estimate and $\Sigma_t$ captures estimation uncertainty
- **Factored structure:** $b_t = b_t^{\text{phys}} \otimes b_t^{\text{psy}}$, exploiting conditional independence between physiological and psychological substates given observations
- **Sufficient statistics for each factor:** For physiology, the CTXBuilder z-scores and rolling statistics serve as a compressed belief representation. For psychology, the mood trajectory, engagement trend, and burden accumulation serve as belief summaries.

**Outputs:**
- Belief state $b_t$ (represented as $(\hat{x}_t, \Sigma_t)$ or equivalent sufficient statistics)
- Belief entropy $\mathbb{H}(b_t)$ (summary of overall state uncertainty)
- Observation likelihood $p(o_t \mid b_{t-1})$ (for anomaly detection: low likelihood $\Rightarrow$ anomaly)
- Epistemic quality indicators: GP familiarity $\kappa_t$, calibration $\eta_t$

**Key distinction from v1.0.** The Perception module does not output "the state." It outputs a *distribution over states*. Downstream modules must operate on this distribution, not on a point estimate. When the belief is diffuse (high $\mathbb{H}(b_t)$), the system acts conservatively — not because uncertainty is added to the cost, but because the epistemic feasibility constraints tighten.

**Learned vs. Explicit.** In the POC, the belief representation is explicit: the CTXBuilder pipeline produces structured features, rolling z-scores provide implicit uncertainty (a z-score of 3 means "this observation is unusual relative to recent history"), and the GP familiarity layer provides the anomaly score. A JEPA-style encoder would later learn a richer belief representation from raw signals.

**Interfaces:** Reads observations from Data Layer. Reads configuration from Configurator. Writes $b_t$ to World Module, Cost Module, Actor, and Memory.

### 3.3 World Module (Twin-Based Forward Model)

**Mathematical Role.** The World Module is a **conditional generative model** that predicts future state trajectories given the current belief and a candidate action. It implements two operations:

**(a) Belief-conditioned rollout.** Sample a state from the belief, then simulate forward:

For rollout $i = 1, \ldots, N_{\text{roll}}$:
1. Draw initial state: $x^{(i)}_t \sim b_t$
2. Simulate forward: for $k = 0, \ldots, H-1$:
$$x^{(i)}_{t+k+1} = f_{\text{dyn}}(x^{(i)}_{t+k}, \tilde{a}_{t+k}, \theta^{\text{post}}_i(t); \xi^{(i)}_{t+k})$$
where $\xi^{(i)}_{t+k} \sim p(\xi)$ is rollout noise and $\tilde{a}_{t+k}$ incorporates the user agency model's predicted response.

3. Record trajectory: $\mathbf{x}^{(i)}_{t:t+H}$

**(b) Outcome distribution.** Collect summary statistics across rollouts:
$$\hat{y}(a) = \frac{1}{N_{\text{roll}}}\sum_{i=1}^{N_{\text{roll}}} y({\mathbf{x}}^{(i)}_{t:t+H})$$
$$\Sigma^{\text{pred}}(a) = \text{Var}_{i}[y(\mathbf{x}^{(i)}_{t:t+H})]$$

where $y(\cdot)$ extracts outcome metrics (TIR, %low, %high, mean BG, BG variance).

The variance $\Sigma^{\text{pred}}$ has two sources:
- **Aleatoric uncertainty** (irreducible): stochastic variation in meals, sleep, stress, etc. Present even with perfect knowledge of $x_t$ and $\theta$.
- **Epistemic uncertainty** (reducible): uncertainty in $x_t$ (from $b_t$) and uncertainty in $\theta^{\text{post}}_i(t)$. Reduced by more data.

By sampling initial states from $b_t$ (not from a point estimate), the rollouts automatically propagate both sources.

**The twin at multiple fidelities:**
- *Full simulator* (t1d_sim): $f_{\text{dyn}}$ uses the ODE-based physiology engine + behavioral generator. High fidelity, ~0.1s per day-rollout. Used for $N_{\text{roll}} \leq 100$ evaluations.
- *Surrogate model* (Model 3): neural ODE or parametric BG curve. Medium fidelity, ~1ms per evaluation. Used inside tight optimization loops.
- *Aggregate predictor* (Model 1, XGBoost): directly predicts $y(a)$ without simulating trajectories. Fastest. Primary workhorse for grid search.

**Learned vs. Explicit.** The physiological dynamics $f_{\text{dyn}}$ begin as the explicit ODE engine (your $apply_context_effectors$ + $simulate_day_cgm$). The behavioral/mood dynamics begin as the explicit stochastic generator ($generate_day_behavior$). Both can later be replaced by learned models, with the critical constraint that learned models must produce calibrated uncertainty (not just point predictions).

**Interfaces:** Receives $b_t$ from Perception, action proposals from Actor, configuration from Configurator. Writes predicted trajectories and outcome distributions to Cost Module. Writes user response predictions to Actor.

### 3.4 Intrinsic Cost Module (Normative Cost — Category B Only)

**Mathematical Role.** The Intrinsic Cost computes a measure of *how bad the world-state is*, evaluated purely in terms of patient outcomes and wellbeing. It contains **no epistemic terms** — no model uncertainty, no confidence, no familiarity.

This is the clean separation demanded by the reviewer: the intrinsic cost answers only "how undesirable is this state of affairs?" — not "how well do we know it?"

Formally:
$$C^{\text{int}}_t = C^{\text{int}}(x_t, \mathbf{x}_{t+1:t+H})$$

In practice, evaluated over the World Module's rollout distribution:
$$\bar{C}^{\text{int}}_t(a) = \frac{1}{N_{\text{roll}}}\sum_{i=1}^{N_{\text{roll}}} C^{\text{int}}(x^{(i)}_t, \mathbf{x}^{(i)}_{t+1:t+H})$$

**Component decomposition:**

**(i) Glycemic cost:**
$$C^{\text{glyc}}_t = w_{\text{low}} \cdot \widehat{\%\text{low}} + w_{\text{high}} \cdot \widehat{\%\text{high}} + w_{\text{var}} \cdot \widehat{\text{BG\_var}} - w_{\text{tir}} \cdot \widehat{\text{TIR}}$$

where the quantities are computed from the predicted BG trajectory within each rollout. The asymmetry matters: physiologically, %low is ~5× more dangerous per unit than %high, and the weights should reflect this.

**(ii) Burden cost:**
$$C^{\text{burden}}_t(a) = c_{\text{freq}} \cdot \mathbb{1}[a \neq a^0] + c_{\text{mag}} \cdot \|a - a^{\text{current}}\|_1 + c_{\text{accum}} \cdot \beta_t$$

where the accumulated burden is:
$$\beta_t = \sum_{\tau=1}^{t} \gamma_\beta^{t-\tau} \cdot \mathbb{1}[a_\tau \neq a^0], \quad \gamma_\beta \in (0.9, 0.98)$$

This is a discounted count of recent recommendations. The decay rate $\gamma_\beta$ determines how quickly past recommendation load is "forgotten" — a recommendation 30 days ago contributes less burden than one yesterday.

**(iii) Trust erosion cost:**
$$C^{\text{trust}}_t(a) = c_\tau \cdot \max\bigl(0, \; \mathbb{E}_{b_t}[\tau_t] - \hat{\tau}_{t+1}(a)\bigr)$$

where $\hat{\tau}_{t+1}(a)$ is the predicted trust level after proposing action $a$, computed from the user agency model. This penalizes actions predicted to *decrease* trust (not low absolute trust — the system cannot control the baseline).

The trust dynamics follow:
$$\tau_{t+1} = \text{clip}\Bigl[\tau_t + \kappa_\tau \cdot \text{outcome\_quality}_t - \lambda_\tau \cdot |\Delta a_t| - \mu_\tau \cdot \mathbb{1}[\text{bad outcome after accept}], \; 0, \; 1\Bigr]$$

where $\kappa_\tau, \lambda_\tau, \mu_\tau$ are agency parameters and $\text{outcome\_quality}_t$ is the realized glycemic improvement.

**(iv) Burnout cost:**
$$C^{\text{burn}}_t = c_B \cdot B_t + c_{\lambda} \cdot \lambda^B_t$$

where $B_t$ is the current burnout state (penalizing realized burnout) and $\lambda^B_t$ is the current burnout hazard rate (penalizing states that are approaching burnout). This dual penalty means the system cares about both being burned out and being at risk of becoming burned out.

**Total intrinsic cost:**
$$C^{\text{int}}_t(a) = C^{\text{glyc}}_t(a) + C^{\text{burden}}_t(a) + C^{\text{trust}}_t(a) + C^{\text{burn}}_t(a)$$

**What is NOT in the intrinsic cost:**
- No uncertainty penalty
- No confidence score
- No familiarity term
- No model disagreement term

These belong to the epistemic constraint layer (Section 3.8).

**Learned vs. Explicit.** Entirely explicit and hardwired. The cost weights $\mathbf{w}$ are set by user preferences (via the Configurator) and can be reviewed by clinicians. The functional forms are fixed by clinical knowledge. This is deliberately non-learnable: the system's definition of "bad" must be auditable.

### 3.5 Trainable Critic (Terminal Value Estimator)

**Mathematical Role.** The Critic is a **terminal value function** that estimates the residual cumulative intrinsic cost *beyond the explicit rollout horizon*. It is not a generic value function. It has a precise, limited job: the World Module rolls out $H$ steps explicitly; the Critic estimates what happens from step $H$ onward.

Formally, define the **residual cost** beyond horizon $H$ as:
$$R_H(x_{t+H}) = \sum_{k=1}^{H_{\text{long}} - H} \gamma^k \; C^{\text{int}}_{t+H+k}$$

This is the discounted sum of intrinsic costs from the end of the explicit rollout to the long planning horizon. The Critic approximates:

$$\hat{V}_\psi(s_{t+H}) \approx \mathbb{E}[R_H(x_{t+H}) \mid s_{t+H}]$$

where $s_{t+H}$ is the terminal state of the explicit rollout and $\psi$ are the learned parameters.

**Why this formulation is sharp.** The Critic captures exactly the phenomena that short-horizon rollouts miss:

- **Slow trust erosion:** A recommendation that slightly degrades trust will show minimal cost within a 7-day rollout but compounds over 60 days. The Critic's job is to learn: "terminal states with declining trust trajectories have high residual cost."
- **Gradual burnout accumulation:** Burnout builds over weeks. The 7-day rollout shows $B_{t+7}$, but the Critic estimates the probability that $B_{t+7}$'s level will cascade into clinical burnout by $t + 90$.
- **Physiological adaptation:** A therapy change may have a transient benefit (captured by rollout) but lead to a compensatory physiological shift (captured by Critic learning from past experience).

**Training objective.** The Critic is trained by temporal difference learning on realized outcomes stored in Memory:

$$\mathcal{L}(\psi) = \sum_{(s_\tau, R_\tau) \in \mathcal{M}} \bigl(\hat{V}_\psi(s_\tau) - R_\tau\bigr)^2$$

where $R_\tau$ is the realized residual cost, computed retrospectively:
$$R_\tau = \sum_{k=1}^{\min(H_{\text{long}} - H, \; T - \tau)} \gamma^k \; C^{\text{int}}_{\tau+k}$$

with $T$ being the current time (we use whatever realized future is available, up to $H_{\text{long}} - H$ steps).

**Critical constraint on training.** The Critic is trained on *realized intrinsic costs* from Memory — not on the World Module's predictions. This prevents a self-reinforcing loop where the Critic trusts the World Module, the World Module trusts the Critic, and neither is grounded in reality.

**Total energy for action evaluation.** The Actor evaluates a candidate action $a$ by combining the explicit rollout cost with the Critic's terminal value:

$$E_t(a) = \underbrace{\bar{C}^{\text{int}}_{t:t+H}(a)}_{\text{explicit rollout cost}} + \underbrace{\gamma^H \cdot \hat{V}_\psi(\hat{s}_{t+H}(a))}_{\text{terminal value estimate}}$$

where:
$$\bar{C}^{\text{int}}_{t:t+H}(a) = \frac{1}{N_{\text{roll}}}\sum_{i=1}^{N_{\text{roll}}} \sum_{k=0}^{H-1} \gamma^k \; C^{\text{int}}(x^{(i)}_{t+k}, x^{(i)}_{t+k+1})$$

is the average discounted intrinsic cost across stochastic rollouts, and $\hat{s}_{t+H}(a)$ is the terminal state (averaged or sampled across rollouts).

**Learned vs. Explicit.** Entirely learned. In the POC, the Critic is initially absent ($\hat{V}_\psi \equiv 0$), reducing the architecture to pure finite-horizon MPC. After 30+ days of memory accumulation, a lightweight Critic (linear model or small MLP on handcrafted terminal state features: trust level, burnout state, engagement trend, glycemic trend, burden level) is bootstrapped.

### 3.6 Short-Term Memory

**Mathematical Role.** Memory is a structured experience buffer supporting three distinct functions: (1) Critic training, (2) Twin calibration, (3) Policy evaluation.

Each record in $\mathcal{M}_t$ stores:

**At recommendation time $\tau$ (immutable):**
- Belief snapshot: $(\hat{x}_\tau, \Sigma_\tau)$ and summary statistics
- Proposed action $a_\tau$ and baseline $a^0_\tau$
- All model predictions: per-model $(\hat{y}_\tau, \text{CI}_\tau, \text{conf}_\tau)$
- Epistemic state: $(\kappa_\tau, \rho_\tau, \eta_\tau, \mathcal{F}_\tau)$
- Configurator settings $\phi_\tau$

**At outcome time $\tau + \Delta$ (mutable once):**
- Realized observations $o_{\tau+1:\tau+\Delta}$
- Realized outcome metrics $y^{\text{real}}_\tau$
- User response $r_\tau$
- Realized action $\tilde{a}_\tau$
- Realized intrinsic cost $C^{\text{int}}_\tau$

**At evaluation time (mutable, progressive):**
- Counterfactual outcome estimate $y^{\text{cf}}_\tau$ (from baseline rollout)
- Per-model accuracy $(y^{\text{real}}_\tau - \hat{y}_\tau)$ and interval coverage
- Shadow score delta
- Retrospective critic target: $R_\tau = \sum_{k=1}^{K} \gamma^k C^{\text{int}}_{\tau+k}$ (extended as more future data arrives)

**Trust/burden/engagement trajectory:**
- $(\tau_\tau, \beta_\tau, \omega_\tau, B_\tau)$ at each timestep

**How memory supports each learning process:**

*Critic training:* Memory provides $(s_\tau, R_\tau)$ pairs where $R_\tau$ is the realized residual cost computed from downstream intrinsic costs. Older records provide longer retrospective windows (better targets). Newer records reflect the current regime.

*Twin calibration:* Memory provides $(b_\tau, a_\tau, y^{\text{real}}_\tau)$ tuples revealing when twin predictions are systematically biased. The posterior $\theta^{\text{post}}_i(t)$ is updated by minimizing calibration loss over recent memory.

*Policy evaluation:* Memory provides the shadow scorecard inputs: win rate, safety record, consistency across contexts, acceptance rate trajectory.

*Stabilizing learning:* Old records from stable regimes regularize the Critic against overfitting to transient drift periods.

### 3.7 Actor (Constrained Action Search Under Belief)

**Mathematical Role.** The Actor solves a constrained optimization over actions, minimizing total energy subject to safety constraints, all evaluated under the belief distribution:

$$a^*_t = \arg\min_{a \in \mathcal{A}(\phi^{\text{act}}_t)} \; \text{CVaR}_\alpha\Bigl[\frac{1}{N_{\text{roll}}}\sum_i E^{(i)}_t(a)\Bigr]$$

$$\text{subject to:} \quad g_j(b_t, a) \leq 0 \quad \forall j \in \mathcal{J}$$

$$\text{and:} \quad \mathcal{F}_t = 1 \quad \text{(epistemic feasibility)}$$

where $E^{(i)}_t(a) = \sum_{k=0}^{H-1} \gamma^k C^{\text{int}}(x^{(i)}_{t+k}) + \gamma^H \hat{V}_\psi(s^{(i)}_{t+H})$ is the total energy from rollout $i$.

**The CVaR computation.** Given $N_{\text{roll}}$ rollout energies $\{E^{(1)}, \ldots, E^{(N)}\}$ for action $a$:

1. Sort: $E^{(\sigma(1))} \leq E^{(\sigma(2))} \leq \ldots \leq E^{(\sigma(N))}$
2. Take the worst $\lceil (1-\alpha) \cdot N \rceil$ rollouts
3. Average them: $\text{CVaR}_\alpha = \frac{1}{\lceil (1-\alpha)N \rceil} \sum_{j=\lfloor \alpha N \rfloor + 1}^{N} E^{(\sigma(j))}$

With $\alpha = 0.8$ and $N = 50$, this averages over the worst 10 rollouts. This focuses the Actor on actions that are robust to bad-case realizations.

**Note:** CVaR is applied to the *normative cost* (intrinsic cost + critic), not to an uncertainty-augmented cost. Epistemic uncertainty is handled by the feasibility constraint $\mathcal{F}_t$ and by the belief-conditioned rollouts (which naturally produce wider energy distributions when the belief is uncertain). This eliminates the double-counting concern from v1.0.

**Safety constraints under belief.** Each hard safety constraint is evaluated in worst-case over the belief:

$$g_j(b_t, a) = \sup_{x \in \text{supp}_\delta(b_t)} \; g_j(x, a) \leq 0$$

where $\text{supp}_\delta(b_t)$ is the $\delta$-credible region of the belief (e.g., 95% highest posterior density). In practice, this is implemented by checking safety on the worst-case rollout among those sampled from $b_t$:

$$g_j(b_t, a) \approx \max_{i=1,\ldots,N_{\text{roll}}} \; g_j(x^{(i)}_t, a)$$

If *any* rollout (sampled from the belief) violates a hard safety constraint, the action is rejected.

### 3.8 Epistemic Constraint Layer (New — Separated from Cost)

**Mathematical Role.** This layer determines whether the system's knowledge state is adequate to support action. It is **not a cost term** — it is a constraint. The system does not trade off "knowing less" against "better outcomes." If epistemic feasibility fails, no action is taken regardless of predicted benefit.

$$\mathcal{F}_t = \mathbb{1}\bigl[\kappa_t \geq \kappa_{\min}\bigr] \cdot \mathbb{1}\bigl[\rho_t \geq \rho_{\min}\bigr] \cdot \mathbb{1}\bigl[\eta_t \geq \eta_{\min}\bigr]$$

where:

**GP Familiarity** $\kappa_t$:
$$\kappa_t = 1 - \frac{\sigma_{\text{GP}}(\hat{x}_t)}{\sigma_{\max}}$$

where $\sigma_{\text{GP}}(\hat{x}_t)$ is the GP posterior standard deviation at the belief mean in the ~12-dimensional projected feature space. High $\sigma_{\text{GP}}$ means the system has not seen similar states → low $\kappa_t$ → gate closes.

**Ensemble Concordance** $\rho_t$:
$$\rho_t = \frac{1}{|\mathcal{P}|}\sum_{(m_1, m_2) \in \mathcal{P}} \text{IoU}\bigl(\text{CI}_{m_1}(a), \; \text{CI}_{m_2}(a)\bigr)$$

where $\mathcal{P}$ is the set of model pairs predicting the same target, and IoU is the interval overlap divided by interval union. Low $\rho_t$ means models disagree → gate closes.

**Calibration Quality** $\eta_t$:
$$\eta_t = 1 - \max_{m \in \mathcal{I}} \bigl|\hat{p}_m(0.8) - 0.80\bigr|$$

where $\hat{p}_m(0.8)$ is the empirical coverage rate of model $m$'s 80% prediction intervals over the last 30 memory records. If any model's intervals are badly miscalibrated, $\eta_t$ drops → gate closes.

**Effect size gate** (separate from epistemic feasibility):
$$\delta_{\text{eff}}(a) = \frac{\bar{C}^{\text{int}}_t(a^0) - \bar{C}^{\text{int}}_t(a)}{\sqrt{\text{Var}_{\text{roll}}[C^{\text{int}}_t(a)]}} > \delta_{\min}$$

Recommend action $a$ only if the predicted improvement exceeds the noise by factor $\delta_{\min}$ (scaled by user aggressiveness).

**Composing the two constraint systems.** The Actor's search is gated by two independent constraint families:

1. **Epistemic feasibility** ($\mathcal{F}_t$): Is the system's knowledge adequate? If no → hold, regardless.
2. **Safety constraints** ($g_j \leq 0$): Is the action safe under the belief? If no → reject action, try others.
3. **Effect size** ($\delta_{\text{eff}} > \delta_{\min}$): Is the improvement meaningful? If no → hold.

The hierarchy is lexicographic: epistemic feasibility is checked first (can we act at all?), then safety (is this particular action safe?), then effect size (is this worth doing?), then CVaR optimization (which safe, meaningful action is best?).

---

## SECTION 4 — DIGITAL TWIN: FOUR-PART DECOMPOSITION

### The Twin Tuple

For patient $i$ at time $t$, the digital twin is:

$$\mathcal{T}_i(t) = \bigl(\underbrace{\theta^{\text{prior}}_i}_{\text{cohort prior}}, \;\; \underbrace{\theta^{\text{post}}_i(t)}_{\text{personalized posterior}}, \;\; \underbrace{x_t}_{\text{dynamic state}}, \;\; \underbrace{p(\xi)}_{\text{rollout noise}}\bigr)$$

Each component has a distinct role, update mechanism, and timescale:

### 4.1 Cohort Prior $\theta^{\text{prior}}_i$

**What it is.** Population-level parameter estimates derived from the patient's archetype/persona. This is the system's starting knowledge before any patient-specific data.

**Components:**
- Physiological prior: $\theta^{\text{phys,prior}}_i$ — drawn from persona archetype (e.g., "athlete" → ISF multiplier $\sim \mathcal{N}(1.2, 0.1)$, base RHR $\sim \mathcal{N}(50, 5)$)
- Behavioral prior: $\theta^{\text{beh,prior}}_i$ — persona-derived (e.g., "high\_stress" → stress\_reactivity $\sim \mathcal{N}(0.85, 0.08)$, mood\_stability $\sim \mathcal{N}(0.25, 0.10)$)
- Agency prior: $\theta^{\text{agency,prior}}_i$ — initial estimates of trust growth, compliance noise, engagement decay

These are the `PatientConfig` fields in your codebase, drawn from the persona distributions in `constants.py`.

**Update mechanism:** Fixed at patient creation. Never updated.

**Role in the system:** Provides the starting point for posterior estimation. Regularizes early estimates when data is sparse. Defines the "population-typical" behavior for counterfactual comparison.

### 4.2 Patient-Specific Posterior $\theta^{\text{post}}_i(t)$

**What it is.** The system's best current estimate of the patient's *time-stable* characteristics, updated from observed data.

**Components:**
- $\theta^{\text{phys,post}}_i(t)$: estimated ISF multiplier, EGP0 scaling, meal response parameters, exercise sensitivity — calibrated from realized BG outcomes
- $\theta^{\text{beh,post}}_i(t)$: estimated sleep regularity, meal patterns, activity propensity — calibrated from observed behavior
- $\theta^{\text{agency,post}}_i(t)$: estimated trust dynamics, compliance pattern, engagement trajectory — calibrated from acceptance/rejection history

**Update mechanism:** Bayesian posterior update (or MAP estimation) given data:

$$\theta^{\text{post}}_i(t) = \arg\max_\theta \; p(\theta \mid o_{1:t}, a_{1:t}) \propto p(o_{1:t} \mid \theta, a_{1:t}) \; p(\theta \mid \theta^{\text{prior}}_i)$$

In the POC, this is implemented as:
- Physiological params: sliding-window MLE on recent (state, action, BG outcome) tuples. E.g., the ISF multiplier is estimated by finding the value that best explains observed BG responses to meals and insulin over the last 14–30 days.
- Behavioral params: rolling statistics of observed behavior (mean sleep duration, meal regularity score, exercise frequency).
- Agency params: logistic regression on (recommendation features, acceptance decision) pairs from Memory.

**Update cadence:** Daily for behavioral stats. Weekly for physiological parameters (need stable windows). After every 10+ recommendations for agency parameters.

**The prior → posterior pathway:**
$$\theta^{\text{post}}_i(t) = \theta^{\text{prior}}_i + \Delta\theta_i(t)$$

where $\Delta\theta_i(t)$ is the data-driven correction. Early in the system's life ($t$ small, few observations), $\Delta\theta_i \approx 0$ and the twin relies heavily on the prior. As data accumulates, the posterior moves away from the prior.

**This decomposition clarifies identifiability.** A parameter is identifiable if $\Delta\theta$ converges to a stable estimate as $t \to \infty$ under typical data. Parameters that remain uncertain (high posterior variance) after 60+ days are effectively non-identifiable from the available signals and should remain at their prior values.

### 4.3 Dynamic State $x_t$

**What it is.** The current state of rapidly-varying quantities that change on the timescale of hours to days.

**Components:**
- Current effective insulin sensitivity (within-day variation around $\theta^{\text{phys,post}}$ baseline)
- Current blood glucose trajectory (recent CGM readings)
- Current mood (valence, arousal — today's value, not the stable tendency)
- Current stress level (today's acute stress, not baseline reactivity)
- Current sleep status (hours since wake, last night's quality)
- Current exercise state (hours since last session, accumulated fatigue)
- Current engagement/trust state (today's willingness to interact)
- Current burnout state $B_t$
- Current burden accumulation $\beta_t$
- Menstrual cycle day (if applicable)
- Infusion site age and location

**Update mechanism:** Bayesian filtering (the Perception module's belief update at each observation).

**Relationship to parameters:** The dynamic state fluctuates *around* the stable parameters. For example:
- The patient's true insulin sensitivity at time $t$ is $\text{ISF}_t = \theta^{\text{phys,post}}_{\text{ISF}} \cdot f_{\text{circadian}}(t) \cdot f_{\text{exercise}}(t) \cdot f_{\text{sleep}}(t) \cdot f_{\text{stress}}(t) \cdot f_{\text{cycle}}(t)$
- The modulation factors $f_{\cdot}(t)$ depend on the dynamic state $x_t$, while the baseline $\theta^{\text{phys,post}}_{\text{ISF}}$ is a stable parameter.

### 4.4 Rollout Noise $p(\xi)$

**What it is.** The stochastic perturbation model used during forward simulation. Captures irreducible randomness in the patient's life.

**Components:**
- Meal timing jitter: $\xi^{\text{meal}}_k \sim \text{distribution parameterized by } \theta^{\text{beh,post}}_{\text{meal\_reg}}$
- Meal size variation: $\xi^{\text{carb}}_k \sim \text{LogNormal}(\mu_{\text{carb}}, \sigma_{\text{carb}})$
- Exercise occurrence: $\xi^{\text{ex}}_k \sim \text{Bernoulli}(\theta^{\text{beh,post}}_{\text{activity}})$
- Sleep quality variation: $\xi^{\text{sleep}}_k \sim \text{Normal}(0, \sigma_{\text{sleep}})$
- Stress perturbation: $\xi^{\text{stress}}_k \sim \text{Normal}(0, \sigma_{\text{stress}})$
- Mood perturbation: $\xi^{\text{mood}}_k$ (AR(1) noise around mood trajectory)
- Rare events: illness ($\xi^{\text{ill}}_k \sim \text{Bernoulli}(0.01)$), site degradation, sensor failure

**Note on noise vs. uncertainty.** Rollout noise is *aleatoric* — it represents genuine randomness in the patient's life. It is distinct from *epistemic* uncertainty about the parameters or state, which enters through sampling $x_t \sim b_t$ and through the posterior variance on $\theta^{\text{post}}$.

### 4.5 Twin Adequacy Criterion (Tightened)

The twin $\mathcal{T}_i(t)$ is adequate for proof-of-concept rollouts if all of the following hold over a 30-day held-out evaluation window:

1. **Marginal coverage:** For each target $y \in \{\text{TIR}, \%\text{low}, \%\text{high}\}$, the twin's 80% prediction interval (from stochastic rollouts) covers the realized value in $\geq 72\%$ of held-out days. (The 72% threshold accounts for modest miscalibration while requiring approximate honesty.)

2. **Directional accuracy:** When comparing action $a$ vs. $a^0$, the twin correctly predicts which produces lower %low in $\geq 65\%$ of cases.

3. **Safety conservatism:** $\max_i \widehat{\%\text{low}}^{(i)} \geq \%\text{low}_{\text{real}}$ in $\geq 90\%$ of days. The twin's worst-case rollout must be pessimistic about hypoglycemia.

4. **Behavioral plausibility:** Simulated acceptance rates match observed rates within $\pm 15\%$ absolute.

---

## SECTION 5 — HORIZONS

### Why Multiple Horizons Are Necessary

The system makes decisions at timescale $\Delta t$ (daily recommendation cycle) about processes operating at timescales spanning three orders of magnitude:

| Process | Timescale | Relevant Horizon |
|---------|-----------|------------------|
| Glucose trajectory | 5 min – 6 h | $H_{\text{short}}$ |
| Meal/exercise/sleep effects | 4 – 48 h | $H_{\text{short}}$ |
| Therapy setting effect | 3 – 14 d | $H_{\text{med}}$ |
| Trust building | 7 – 30 d | $H_{\text{long}}$ via Critic |
| Engagement dynamics | 7 – 60 d | $H_{\text{long}}$ via Critic |
| Burnout accumulation | 14 – 90 d | $H_{\text{burn}}$, $H_{\text{long}}$ via Critic |
| Physiological drift | 30 – 180 d | Twin recalibration |

**The fundamental tension:** Longer rollouts capture more phenomena but are less reliable. The architecture resolves this by using explicit rollouts for short/medium horizons (where the twin is reliable) and the learned Critic for long horizons (where the twin is unreliable but memory-based learning can extrapolate).

### Perception Horizon $H_{\text{perc}}$: 6–24 hours

Lookback window for the belief update. Must capture at least one full sleep–wake cycle. The 7-day same-hour z-scores in the CTXBuilder effectively provide implicit context beyond $H_{\text{perc}}$ without requiring the belief estimator to process 7 days of raw data.

**What it supports:** Belief estimation, anomaly detection (via observation likelihood under the prior predictive), recent trend identification.

**Configurator adaptation:** When drift is detected, $H_{\text{perc}}$ shrinks (recent data is more relevant). When the patient is stable, $H_{\text{perc}}$ can extend for smoother estimates.

### Short Rollout $H_{\text{short}}$: 4–24 hours

Explicit forward simulation for BG trajectory preview. This is the horizon over which the World Module produces trajectories that the user can inspect ("projected glucose for the next 12 hours").

**What it supports:** Immediate safety checking ("will this cause a low tonight?"), glucose trajectory preview for user interface, short-term intrinsic cost computation.

**Why bounded:** BG prediction degrades rapidly beyond ~12h due to unpredictable meals, exercise, and stress. The World Module's stochastic rollouts capture this: the prediction interval widens with horizon, and by 24h it is typically too wide to be actionable for trajectory-level predictions (though summary statistics remain useful).

### Medium Rollout $H_{\text{med}}$: 3–14 days

The horizon over which therapy changes are evaluated. The Actor evaluates candidate actions by running $N_{\text{roll}}$ stochastic rollouts of $H_{\text{med}}$ days and computing the average intrinsic cost + terminal value.

**What it supports:** Therapy evaluation (average TIR improvement over 7 days), recommendation win rate estimation, trust/burden dynamics over the recommendation cycle.

**Configurator adaptation:** During stable periods, $H_{\text{med}} = 7$–$14$ days (longer horizon → better estimate of sustained effect). During drift, $H_{\text{med}} = 3$–$5$ days (don't trust the twin beyond the drift timescale).

### Long Planning Horizon $H_{\text{long}}$: 30–90 days

This is **not** a rollout horizon. It is the discounting horizon for the Critic's training targets. The Critic learns to estimate:

$$\hat{V}_\psi(s_{t+H_{\text{med}}}) \approx \mathbb{E}\left[\sum_{k=1}^{H_{\text{long}} - H_{\text{med}}} \gamma^k C^{\text{int}}_{t + H_{\text{med}} + k}\right]$$

**What it supports:** Long-term consequences of current actions: trust trajectories, burnout risk, engagement decay, physiological adaptation.

**Why the Critic and not the twin:** Running the twin for 90 days produces unreliable trajectories (behavioral noise compounds, regime shifts are likely, the twin may be stale). The Critic learns from *realized* costs stored in Memory, so it captures long-horizon effects without requiring long-horizon simulation.

### Burnout Attribution Horizon $H_{\text{burn}}$: 14–60 days

The horizon for counterfactual burnout risk estimation ($\Delta^B_H(\pi)$).

**Why it needs its own horizon:** Burnout attribution requires matched pair rollouts (treated vs. baseline) from the same initial state. These rollouts must be long enough for burnout to potentially manifest (at least 14 days, typically 30) but short enough that the counterfactual remains meaningful (beyond 60 days, the treated and baseline paths diverge so much that comparison is meaningless).

**Sensitivity analysis requirement:** $\hat{\Delta}^B_H(\pi)$ should be computed at multiple horizons $H \in \{14, 21, 30, 45, 60\}$ and checked for consistency. If the sign of $\hat{\Delta}^B$ changes with $H$, the estimate is unstable and should not be trusted.

### Memory Horizon $H_{\text{mem}}$: 30–90 days

Retention window for experience records. Drives:
- Critic training target computation (need realized costs extending $H_{\text{long}} - H_{\text{med}}$ steps beyond each record)
- Calibration statistics (rolling 30-day window)
- Twin posterior updates (recent data weighted more heavily)
- Scorecard evaluation (sustained performance over 30+ days)

### Configurator Update Horizon $H_{\text{config}}$: 1–7 days

Cadence at which the meta-policy $\mu$ re-evaluates its configuration. Daily for cost weights, action bounds, and search parameters. Weekly for model trust routing and adaptation escalation decisions.

---

## SECTION 6 — COST / ENERGY FORMULATION (REVISED)

### Architectural Separation: Three Layers

The v1.0 formulation mixed normative cost and epistemic uncertainty. The v1.1 architecture cleanly separates three layers:

**Layer 1: Normative Cost (what matters to the patient)**
$$C^{\text{int}}_t(a) = C^{\text{glyc}}_t(a) + C^{\text{burden}}_t(a) + C^{\text{trust}}_t(a) + C^{\text{burn}}_t(a)$$

This is purely about outcomes. It answers: "if the world unfolds this way, how bad is it?"

**Layer 2: Terminal Value (what the Critic predicts about the long run)**
$$\hat{V}_\psi(s_{t+H}) \approx \mathbb{E}[R_H(x_{t+H})]$$

This extends the normative cost beyond the rollout horizon. It is also purely about outcomes — just further into the future.

**Layer 3: Epistemic Constraints (what the system doesn't know)**
$$\mathcal{F}_t = 1 \quad \text{and} \quad g_j(b_t, a) \leq 0$$

These are *constraints*, not costs. They gate whether action is taken, but they do not enter the objective function.

### The Optimization Problem (Complete Statement)

The Actor's problem is:

$$a^*_t = \arg\min_{a \in \mathcal{A}(\phi^{\text{act}}_t)} \; \text{CVaR}_\alpha\left[E_t(a)\right]$$

where:

$$E_t(a) = \frac{1}{N_{\text{roll}}}\sum_{i=1}^{N_{\text{roll}}}\left[\sum_{k=0}^{H-1} \gamma^k C^{\text{int}}(x^{(i)}_{t+k}, a) + \gamma^H \hat{V}_\psi(s^{(i)}_{t+H})\right]$$

subject to:

$$\text{(Epistemic feasibility)} \quad \mathcal{F}_t = 1$$
$$\text{(Hard safety)} \quad \max_{i} \; g_j(x^{(i)}_t, a) \leq 0 \quad \forall j$$
$$\text{(Effect size)} \quad \frac{\bar{E}_t(a^0) - \bar{E}_t(a)}{\sqrt{\text{Var}_{\text{roll}}[E_t(a)]}} > \delta_{\min}$$

**No double-counting.** Uncertainty enters through exactly two channels:
1. The stochastic rollouts sample initial states from $b_t$ (propagating belief uncertainty) and draw process noise $\xi$ (propagating aleatoric uncertainty). This makes the CVaR computation naturally sensitive to both — actions with high outcome variance under $b_t$ will have worse CVaR.
2. The epistemic feasibility gate $\mathcal{F}_t$ prevents action when the belief is too diffuse or the models too poorly calibrated.

Neither channel adds an uncertainty *cost term*. The cost function is clean.

### Detailed Cost Term Specifications

#### Glycemic Cost (within each rollout)

For rollout $i$ producing BG trajectory $\text{BG}^{(i)}_{1:T}$ at 5-minute resolution:

$$C^{\text{glyc},(i)} = w_{\text{low}} \cdot \frac{|\{t : \text{BG}^{(i)}_t < 70\}|}{T} + w_{\text{high}} \cdot \frac{|\{t : \text{BG}^{(i)}_t > 180\}|}{T} + w_{\text{var}} \cdot \text{CV}(\text{BG}^{(i)}) - w_{\text{tir}} \cdot \frac{|\{t : 70 \leq \text{BG}^{(i)}_t \leq 180\}|}{T}$$

Default weight guidance (from clinical practice, user-adjustable):
- $w_{\text{low}} = 5.0$ (hypoglycemia is ~5× more dangerous per unit than hyperglycemia)
- $w_{\text{high}} = 1.0$
- $w_{\text{tir}} = 1.0$
- $w_{\text{var}} = 0.5$

#### Burden Cost

$$C^{\text{burden}}_t(a) = c_{\text{freq}} \cdot \mathbb{1}[a \neq a^0] + c_{\text{mag}} \cdot \sum_{d \in \text{dims}} \left|\frac{a_d - a^{\text{curr}}_d}{a^{\text{curr}}_d}\right| + c_{\text{accum}} \cdot \beta_t$$

The magnitude term uses relative change (not absolute), so that a 5% ISF change is penalized equally whether the baseline ISF is 30 or 60.

$$\beta_t = \sum_{\tau < t} \gamma_\beta^{t - \tau} \cdot \mathbb{1}[a_\tau \neq a^0], \quad \gamma_\beta = 0.95$$

With $\gamma_\beta = 0.95$ and daily cadence, the effective half-life is $\frac{\ln 2}{\ln(1/0.95)} \approx 13.5$ days. Recommendations more than ~4 weeks old contribute negligibly to burden.

#### Trust Dynamics and Cost

Trust evolves as:
$$\tau_{t+1} = \text{clip}\Bigl[\tau_t + \underbrace{\kappa^+_\tau \cdot q_t^+ \cdot \mathbb{1}[r_t = 1]}_{\text{trust gain from good outcome}} - \underbrace{\kappa^-_\tau \cdot q_t^- \cdot \mathbb{1}[r_t = 1]}_{\text{trust loss from bad outcome}} - \underbrace{\lambda_\tau \cdot d(a_t, a^0)}_{\text{trust cost of large change}}, \; 0, \; 1\Bigr]$$

where:
- $q_t^+ = \max(0, \text{TIR}_{t+1} - \text{TIR}_t)$ is glycemic improvement
- $q_t^- = \max(0, \%\text{low}_{t+1} - \%\text{low}_t)$ is glycemic deterioration (especially hypoglycemia increase)
- $d(a_t, a^0) = \|a_t - a^0\|_1 / \|a^0\|_1$ is relative action magnitude
- $\kappa^+_\tau, \kappa^-_\tau$ are trust growth/decay rates (agency parameters)
- $\lambda_\tau$ scales the trust cost of surprising changes

Trust erosion cost:
$$C^{\text{trust}}_t(a) = c_\tau \cdot \mathbb{E}_{b_t}\bigl[\max(0, \tau_t - \hat{\tau}_{t+1}(a))\bigr]$$

Asymmetry is crucial: trust builds slowly ($\kappa^+_\tau$ small) but can be damaged quickly ($\kappa^-_\tau$ larger). A single bad recommendation can undo weeks of trust building.

#### Burnout Cost

$$C^{\text{burn}}_t = c_B \cdot B_t + c_\lambda \cdot \lambda^B_t$$

where the burnout hazard is (detailed in Section 7):
$$\lambda^B_t = \lambda_0 \exp\bigl(\beta_1 B_t + \beta_2 \beta_t + \beta_3 (1 - \tau_t) + \beta_4 (1 - \omega_t) + \beta_5 \cdot \text{glyc\_frustration}_t\bigr)$$

and $\text{glyc\_frustration}_t = \text{clip}[\bar{\%\text{low}}_{7d} + 0.5 \cdot (1 - \bar{\text{TIR}}_{7d}), 0, 1]$ captures glycemic frustration as a burnout driver.

---

## SECTION 7 — BURNOUT ATTRIBUTION (TIGHTENED)

### Burnout State Dynamics

The burnout state $B_t \in [0, 1]$ evolves according to:

$$B_{t+1} = \sigma\Bigl(\sigma^{-1}(B_t) + \underbrace{\Delta B^{\text{endo}}_t}_{\text{endogenous}} + \underbrace{\Delta B^{\text{policy}}_t}_{\text{policy-induced}} + \underbrace{\eta^B_t}_{\text{noise}}\Bigr)$$

where $\sigma$ is the logistic sigmoid and $\sigma^{-1}$ is the logit, so that the dynamics are additive in log-odds space (ensuring $B_t$ stays in $[0,1]$ without artificial clipping).

**Endogenous burnout drivers** (would occur even without the system):
$$\Delta B^{\text{endo}}_t = \alpha_1 \cdot \text{glyc\_frustration}_t + \alpha_2 \cdot \text{sleep\_debt}_t + \alpha_3 \cdot \text{stress}_t + \alpha_4 \cdot (1 - \text{mood\_stability}) + \alpha_5 \cdot \mathbb{1}[\text{ill}_t]$$

**Policy-induced burnout drivers** (attributable to the system's actions):
$$\Delta B^{\text{policy}}_t = \alpha_6 \cdot \beta_t + \alpha_7 \cdot \mathbb{1}[\text{bad outcome after accept}]_t + \alpha_8 \cdot \text{cognitive\_load}_t$$

where $\text{cognitive\_load}_t$ is a function of recommendation complexity and information volume.

**Noise:** $\eta^B_t \sim \mathcal{N}(0, \sigma^2_B)$, reflecting unmeasured life events.

### Burnout Hazard

The hazard of *clinical* burnout (crossing threshold $B_{\text{thresh}} = 0.7$, meaning the patient stops engaging meaningfully) is:

$$\lambda^B_t = \lambda_0 \exp\bigl(\beta_1 B_t + \beta_2 \beta_t + \beta_3 (1 - \tau_t) + \beta_4 (1 - \omega_t) + \beta_5 \cdot \text{glyc\_frustration}_t\bigr)$$

The survival probability over horizon $H$:
$$S_H(t) = P(B_\tau < B_{\text{thresh}} \;\forall \tau \in [t, t+H]) \approx \exp\left(-\sum_{k=0}^{H-1} \lambda^B_{t+k} \cdot \Delta t\right)$$

Burnout probability:
$$P^B_H(t) = 1 - S_H(t)$$

### Counterfactual Burnout Attribution (Full Algorithm)

**Goal:** Compute $\hat{\Delta}^B_H(\pi) = \hat{P}^B_H(\pi) - \hat{P}^B_H(\pi^0)$ with confidence intervals.

**Algorithm:**

**Input:** Current belief $b_t$, active policy $\pi$, null policy $\pi^0$, twin $\mathcal{T}_i(t)$, horizon $H = H_{\text{burn}}$, number of rollout pairs $N = 100$.

**For $i = 1, \ldots, N$:**

1. Draw shared initial state: $x^{(i)}_0 \sim b_t$
2. Draw shared noise sequence: $\xi^{(i)}_{0:H-1} \sim p(\xi)$ (same exogenous randomness for both paths)

3. **Treated path ($\pi$):**
   For $k = 0, \ldots, H-1$:
   - Actor proposes: $a^{(i)}_k = \pi(b^{\pi,(i)}_k)$
   - User responds: $r^{(i)}_k \sim p(r \mid x^{\pi,(i)}_k, a^{(i)}_k, \theta^{\text{agency,post}})$
   - Realized action: $\tilde{a}^{(i)}_k = r^{(i)}_k a^{(i)}_k + (1 - r^{(i)}_k) a^0$
   - State transition: $x^{\pi,(i)}_{k+1} = f_{\text{dyn}}(x^{\pi,(i)}_k, \tilde{a}^{(i)}_k, \theta^{\text{post}}; \xi^{(i)}_k)$
   - Update burnout: $B^{\pi,(i)}_{k+1}$ from dynamics equation

4. **Baseline path ($\pi^0$):**
   Same initial state and exogenous noise, but $a_k = a^0$ always:
   - $x^{0,(i)}_{k+1} = f_{\text{dyn}}(x^{0,(i)}_k, a^0, \theta^{\text{post}}; \xi^{(i)}_k)$
   - $B^{0,(i)}_{k+1}$ from dynamics equation (with $\Delta B^{\text{policy}} = 0$ since no recommendations)

5. Record burnout events:
   - $Z^{\pi}_i = \mathbb{1}[\exists k : B^{\pi,(i)}_k \geq B_{\text{thresh}}]$
   - $Z^0_i = \mathbb{1}[\exists k : B^{0,(i)}_k \geq B_{\text{thresh}}]$

**Compute:**

Treated burnout probability:
$$\hat{P}^{\pi} = \frac{1}{N}\sum_{i=1}^N Z^{\pi}_i$$

Baseline burnout probability:
$$\hat{P}^{0} = \frac{1}{N}\sum_{i=1}^N Z^{0}_i$$

Attributable risk:
$$\hat{\Delta}^B_H(\pi) = \hat{P}^{\pi} - \hat{P}^{0}$$

**Confidence interval (paired):**

Because the pairs share the same initial state and exogenous noise, the individual differences $D_i = Z^{\pi}_i - Z^0_i$ are correlated. The paired standard error is:

$$\text{SE}_{\text{paired}} = \frac{s_D}{\sqrt{N}}, \quad s_D = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (D_i - \bar{D})^2}$$

95% CI: $\hat{\Delta}^B_H \pm t_{0.025, N-1} \cdot \text{SE}_{\text{paired}}$

This is tighter than the unpaired SE from v1.0 because the shared noise cancels out common variance.

**Sensitivity analysis:**
- Compute $\hat{\Delta}^B_H$ at $H \in \{14, 21, 30, 45, 60\}$ days
- Flag inconsistency: if $\text{sign}(\hat{\Delta}^B_H)$ varies across horizons, the estimate is unreliable
- Twin robustness: repeat with perturbed $\theta^{\text{post}} \pm \epsilon$ and check sign stability

### Decision Rule

A policy $\pi$ is acceptable if:
$$\hat{\Delta}^B_H(\pi) + t_{0.025, N-1} \cdot \text{SE}_{\text{paired}} < \epsilon_{\text{burn}}$$

where $\epsilon_{\text{burn}}$ is the maximum tolerable policy-attributable burnout risk (e.g., 5%). This uses the upper confidence bound — the system must be confident that the policy doesn't increase burnout risk, not just estimate that it doesn't.

### Comparing "Better TIR but Higher Burnout Risk"

When the Actor finds an action that improves glycemic outcomes but has positive $\hat{\Delta}^B_H$, the system reports a structured tradeoff:

- Glycemic improvement: $\Delta \text{TIR} = \bar{\text{TIR}}(a^*) - \bar{\text{TIR}}(a^0)$
- Burnout risk increase: $\hat{\Delta}^B_H \pm \text{CI}$
- Recommendation: if $\hat{\Delta}^B_H + \text{CI} > \epsilon_{\text{burn}}$, the action is flagged even if glycemically superior
- Alternative: search for an action with comparable glycemic improvement but lower burnout risk (e.g., smaller change, lower frequency)

---

## SECTION 8 — MEMORY AND LEARNING (REVISED)

### Memory Schema (Tightened)

Each record $m_\tau \in \mathcal{M}_t$ is a tuple:

$$m_\tau = (b_\tau, \; a_\tau, \; a^0_\tau, \; r_\tau, \; \tilde{a}_\tau, \; \hat{y}_\tau, \; y^{\text{real}}_\tau, \; y^{\text{cf}}_\tau, \; C^{\text{int}}_\tau, \; \hat{V}_\tau, \; \tau_\tau, \; \beta_\tau, \; \omega_\tau, \; B_\tau, \; \mathcal{F}_\tau, \; \phi_\tau)$$

**Size estimate:** Each record is ~500 floats. At daily cadence with 90-day retention: ~45K floats $\approx$ 180KB. Negligible storage.

### Critic Training Pipeline (Explicit)

**Step 1: Compute realized residual costs.** For each record $m_\tau$ with $\tau \leq t - H_{\text{med}}$, compute:

$$R_\tau = \sum_{k=1}^{\min(H_{\text{long}} - H_{\text{med}}, \; t - \tau - H_{\text{med}})} \gamma^k \; C^{\text{int}}_{\tau + H_{\text{med}} + k}$$

This uses realized (not predicted) intrinsic costs from subsequent memory records.

**Step 2: Construct training set.** Collect $(s_{\tau + H_{\text{med}}}, R_\tau)$ pairs from memory, where $s_{\tau + H_{\text{med}}}$ is the terminal state summary from the record at time $\tau + H_{\text{med}}$.

Terminal state features for the Critic:
- Trust level $\tau_{\tau + H}$
- Trust trend (slope of $\tau$ over the preceding 7 days)
- Burnout state $B_{\tau + H}$
- Burnout trend
- Engagement $\omega_{\tau + H}$
- Engagement trend
- Burden $\beta_{\tau + H}$
- Average glycemic cost over the preceding 7 days
- Glycemic trend (improving/stable/degrading)

**Step 3: Train.** Minimize MSE:
$$\hat{\psi} = \arg\min_\psi \sum_{(s, R) \in \mathcal{D}_{\text{critic}}} (\hat{V}_\psi(s) - R)^2$$

For the POC, $\hat{V}_\psi$ is a ridge regression on the ~10 terminal state features. This is deliberately simple — overfitting a complex Critic to limited data is worse than a slightly biased linear estimator.

**Step 4: Validate.** Hold out the most recent 20% of records. Compute Critic prediction error on holdout. If Critic is worse than the constant predictor $\hat{V}_\psi \equiv \bar{R}$, do not use the Critic (revert to $\hat{V}_\psi \equiv 0$).

### Twin Posterior Update Pipeline

**Physiological parameter update:**
Using the last 14–30 days of $(b_\tau, a_\tau, y^{\text{real}}_\tau)$ records, solve:

$$\theta^{\text{phys,post}}_i(t) = \arg\min_\theta \sum_{\tau \in \mathcal{W}} \|y^{\text{real}}_\tau - \hat{y}_\tau(\theta)\|^2 + \lambda_{\text{reg}} \|\theta - \theta^{\text{prior}}_i\|^2$$

The regularization term $\lambda_{\text{reg}} \|\theta - \theta^{\text{prior}}\|^2$ implements the Bayesian prior — with limited data, the posterior stays near the prior.

**Agency parameter update:**
Using acceptance/rejection records from Memory, fit:

$$P(r = 1 \mid s_\tau, a_\tau; \theta^{\text{agency}}) = \sigma(\theta^{\text{agency}}_0 + \theta^{\text{agency}}_1 \cdot d(a_\tau, a^0) + \theta^{\text{agency}}_2 \cdot \tau_\tau + \theta^{\text{agency}}_3 \cdot \omega_\tau)$$

via logistic regression on the (features, acceptance) pairs in Memory.

---

## SECTION 9 — ACTOR AND CONSTRAINED ACTION SEARCH (REVISED)

### Complete Actor Algorithm for POC

```
ALGORITHM: ChameliaActor(b_t, φ_t, WorldModule, CostModule, Critic, Memory)

Input:
  b_t: current belief state
  φ_t: configurator settings (Δ_max, δ_min, N_roll, H_med, α)
  
Step 0: Epistemic Feasibility Check
  Compute κ_t, ρ_t, η_t
  If F_t = 0: RETURN (hold, reason="insufficient epistemic confidence")

Step 1: Generate Candidate Actions
  A_cand = GridSearch(a_current, Δ_max, step_sizes)
  A_cand = A_cand ∪ {a^0}  // always include hold
  A_cand = FilterLockedWindows(A_cand, user_prefs)
  // Typical |A_cand| ≈ 100–200

Step 2: Evaluate Each Candidate
  For each a ∈ A_cand:
    For i = 1, ..., N_roll:
      x_0^(i) ~ b_t                          // sample from belief
      ξ^(i) ~ p(ξ)                            // draw noise
      Simulate H_med days: x^(i)_{0:H} = WorldModule.rollout(x_0^(i), a, θ^post, ξ^(i))
      Compute per-rollout intrinsic cost: C^(i) = Σ_k γ^k C_int(x^(i)_k, a)
      Compute terminal value: V^(i) = γ^H · Critic(s^(i)_{t+H})  // 0 if Critic not yet trained
      E^(i) = C^(i) + V^(i)
    
    // Safety gate (hard, on worst-case rollout)
    If max_i g_j(x^(i), a) > 0 for any j: REJECT a
    
    // CVaR computation
    Sort E^(1), ..., E^(N_roll)
    CVaR_α(a) = mean of top ⌈(1-α)·N_roll⌉ values

Step 3: Effect Size Gate
  For each surviving a:
    δ_eff(a) = (CVaR(a^0) - CVaR(a)) / std_roll(E(a))
    If δ_eff(a) < δ_min: REJECT a

Step 4: Select Best
  If no candidate survives: RETURN (hold, reason="no action improves on baseline")
  a* = argmin_{surviving a} CVaR_α(a)

Step 5: Package
  RETURN RecommendationPackage(
    primary = a*,
    predicted_improvement = CVaR(a^0) - CVaR(a*),
    confidence = composite of κ_t, ρ_t, η_t, δ_eff,
    alternatives = top-2 diverse surviving candidates,
    baseline_prediction = outcomes under a^0,
    burnout_attribution = Δ^B_H(π) if computed this cycle
  )
```

### Hold as Active Decision

When the Actor returns "hold," it is a deliberate, logged prediction: "your current settings are appropriate given the current state and our uncertainty about it." This prediction gets a shadow record, is scored against future outcomes, and contributes to the scorecard. A system that frequently holds and is usually right (stable TIR, no safety events during holds) builds trust.

### Conservative Action Bounds

The maximum allowable deviation $\Delta_{\max}$ is a function of:

$$\Delta_{\max}(t) = \Delta_{\text{base}} \cdot f_{\text{aggr}}(\theta^{\text{agency}}) \cdot f_{\text{trust}}(\tau_t) \cdot f_{\text{score}}(\text{scorecard}_t) \cdot f_{\text{drift}}(\text{drift}_t)$$

where:
- $\Delta_{\text{base}}$ = 10% (hard ceiling on any single parameter change for the POC)
- $f_{\text{aggr}} \in [0.3, 1.0]$ scales by user's aggressiveness preference
- $f_{\text{trust}} = \min(1, \tau_t / \tau_{\text{req}})$ scales by current trust level
- $f_{\text{score}} = \min(1, \text{win\_rate} / 0.65)$ scales by shadow scorecard performance
- $f_{\text{drift}} = 1.0$ if no drift detected, $0.5$ if drift detected

A newly initialized system with low trust and no shadow history will have $\Delta_{\max} \approx 3\%$ — only tiny adjustments are permitted.

---

## SECTION 10 — WHAT IS EXPLICIT VS. WHAT SHOULD EVENTUALLY BE LEARNED

### Disciplined Decomposition

| Component | POC (Explicit) | V2 (Learnable) | Always Explicit |
|-----------|---------------|-----------------|-----------------|
| **Perception (belief estimator)** | CTXBuilder + z-scores + GP familiarity | JEPA encoder → latent belief $b_t$ | — |
| **World Module: physiology** | t1d_sim ODE engine | Neural ODE from real data | — |
| **World Module: behavior** | Stochastic generator (behavior.py) | Latent state-space model | — |
| **World Module: agency** | Parameterized acceptance model | Learned from acceptance data | — |
| **Actor** | Grid search + safety gate | Offline RL (CQL on fork-of-forks) | — |
| **Critic** | Linear ridge regression on terminal features | Deep value network | — |
| **Configurator** | Rule-based meta-policy | Learned meta-policy from (config, outcome) data | — |
| **Intrinsic cost function** | — | — | Hardwired, clinician-reviewed |
| **Safety constraints** | — | — | Hardwired, non-negotiable |
| **Safety gate logic** | — | — | Worst-case checking over belief |
| **Memory structure** | — | — | Structured buffer, explicit schema |
| **Burnout attribution** | — | — | Counterfactual rollout algorithm |
| **Epistemic feasibility** | — | — | Constraint, not cost |

### What JEPA Would Replace (Precision Statement)

A JEPA implementation would:

1. **Replace** the CTXBuilder pipeline with a learned encoder $e_\phi : o_{t-H:t} \to z_t$ producing latent state representations. The belief would become a distribution in latent space: $b_t \in \mathcal{P}(\mathcal{Z})$.

2. **Replace** the explicit forward dynamics with a latent predictor $p_\theta : (z_t, a_t) \to z_{t+1}$ trained via self-supervised prediction loss $\|z_{t+1} - p_\theta(z_t, a_t)\|^2$ in representation space. This is the core JEPA insight: predict in latent space, not in observation space.

3. **Augment** the behavioral/mood dynamics by learning latent psychological state transitions from observed mood, engagement, and acceptance trajectories.

4. **NOT replace:** intrinsic cost (hardwired), safety constraints (hardwired), memory schema (explicit), burnout attribution (algorithmic), epistemic constraints (explicit). The cost function would need a decoder $d : \mathcal{Z} \to \mathcal{Y}$ to map latent states to outcome metrics that the cost function evaluates, but the cost function itself remains explicit.

---

## SECTION 11 — SOLUTION FAMILIES (REVISED)

### Family 1: Explicit Twin + Constrained MPC + Separated Epistemic Layer

**Description.** World Module = t1d_sim with four-part twin decomposition. Actor = constrained grid search (MPC-style). Cost = pure normative (no uncertainty terms). Epistemic layer = GP familiarity + ensemble concordance + calibration, as constraints. CVaR over stochastic rollouts for risk sensitivity.

**Strengths:** Fully interpretable. No learned dynamics. Clean cost/uncertainty separation. Directly leverages existing codebase. Validates the architectural decomposition before adding learned components.

**Weaknesses:** Simulator fidelity limits quality. Grid search doesn't scale to high-dimensional action spaces. No long-horizon planning without Critic.

**Data needs:** Per-patient twin calibration from 14–30 days of observation.

**POC suitability:** **Excellent.** This is the recommended path.

### Family 2: Explicit Physiology + Learned Latent Psychology (Hybrid Twin)

**Description.** Physiological dynamics remain ODE-based. Psychological/behavioral dynamics (mood, engagement, trust, burnout) are modeled by a learned latent state-space model: $z^{\text{psy}}_{t+1} = f_\theta(z^{\text{psy}}_t, a_t, x^{\text{phys}}_t; \xi_t)$. Trained from longitudinal mood/engagement/acceptance data.

**Strengths:** Leverages physiology domain knowledge while learning the poorly-specified psychological dynamics. Modular — can be developed independently. The physiological–psychological interface is clean (glycemic outcomes → mood inputs, recommendations → burden inputs).

**Weaknesses:** Requires sufficient longitudinal psychological data. Interface design between the two components is non-trivial.

**Data needs:** Simulator behavioral data (immediate), real longitudinal mood/engagement data (later).

**POC suitability:** **Good as Phase 2.** Build Family 1 first, then swap the behavioral model.

### Family 3: Conservative Batch RL on Forked Timelines

**Description.** Train a CQL-style policy on the fork-of-forks trajectory dataset. The policy maps belief summaries to actions. The explicit safety gate runs independently.

**Strengths:** Fast inference (single forward pass). Naturally handles sequential decision-making. CQL's conservatism prevents overestimation (important for safety).

**Weaknesses:** Requires fork-of-forks dataset to be built and validated. Policy is opaque. Coverage of the state-action space by the training data limits generalization.

**Data needs:** 25K–100K terminal timelines from the forked tree.

**POC suitability:** **Moderate.** Can augment Family 1 as a warm-start once the fork tree exists.

### Family 4: Energy-Based World Model (Full LeCunian)

**Description.** An EBM $F_\theta(x, a)$ scores state-action compatibility. Low energy = good pairing. The World Module is the EBM itself. The Actor performs gradient-based energy minimization to find actions.

**Strengths:** Most faithful to LeCun's theoretical vision. Handles multi-modal futures naturally. Elegant mathematical formulation.

**Weaknesses:** EBM training is difficult (contrastive methods, careful negative sampling). Safety guarantees are hard to formalize. No existing implementation infrastructure.

**POC suitability:** **Poor.** Theoretically motivated but practically premature. Worth revisiting after the explicit twin is validated.

### Family 5: Bayesian Twin + Thompson Sampling Actor

**Description.** Maintain a full Bayesian posterior over twin parameters $p(\theta \mid o_{1:t})$. At each decision point, sample $\theta \sim p(\theta \mid o_{1:t})$ and act optimally for that twin (Thompson sampling). This naturally balances exploration (acting on uncertain parameter samples) and exploitation (acting on likely parameters).

**Strengths:** Principled uncertainty quantification. Natural exploration. No need for separate GP familiarity layer (the posterior width plays that role).

**Weaknesses:** Full Bayesian inference over the twin parameter space is computationally expensive. Approximate methods (variational inference, particle filters) introduce their own biases.

**Data needs:** Same as Family 1, but requires more computation per decision.

**POC suitability:** **Moderate.** The four-part twin decomposition already enables a lightweight version of this (sample $\theta^{\text{post}}$ from its posterior uncertainty during rollouts).

---

## SECTION 12 — RECOMMENDED MINIMUM TRUE PROOF OF CONCEPT (REVISED)

### Architecture: Family 1 with Full Decomposition

| Component | Implementation |
|-----------|---------------|
| **Belief state** | $b_t \approx (\hat{x}_t, \Sigma_t)$ from CTXBuilder + rolling uncertainty estimates |
| **Twin** | t1d_sim with four-part decomposition: persona prior $\theta^{\text{prior}}$, posterior $\theta^{\text{post}}(t)$ updated from 14–30 day windows, dynamic state $x_t$ from Perception, rollout noise $p(\xi)$ from behavioral generator |
| **World Module** | t1d_sim rollouts ($N = 50$, $H_{\text{med}} = 7$ days) sampling $x_0 \sim b_t$ |
| **Intrinsic cost** | $C^{\text{glyc}} + C^{\text{burden}} + C^{\text{trust}} + C^{\text{burn}}$ (pure normative, no uncertainty terms) |
| **Epistemic constraints** | GP familiarity $\kappa_t \geq 0.6$, ensemble concordance $\rho_t \geq 0.5$, calibration $\eta_t \geq 0.7$ |
| **Safety gate** | Worst-case %low across rollouts $< 4\%$ |
| **Actor** | Grid search: ISF $\pm\{0, 5\%, 10\%\}$, CR $\pm\{0, 5\%, 10\%\}$, basal $\pm\{0, 5\%\}$. CVaR at $\alpha = 0.8$. Effect size gate $\delta_{\min} = 0.5$. |
| **Critic** | Initially $\hat{V} \equiv 0$ (pure finite-horizon MPC). After 30+ days: linear ridge on terminal state features. |
| **Burnout attribution** | 100 paired rollouts × 30 days, treated vs. null, paired SE confidence interval |
| **Memory** | 60-day rolling buffer, ~45K floats total |
| **Configurator** | Rule-based: user prefs → cost weights, trust/drift → $\Delta_{\max}$, daily update |
| **Shadow graduation** | 21-day minimum, 60% win rate, zero safety violations, sustained 7 consecutive days |

### What Must Remain Explicit Restriction Gates

1. **Epistemic feasibility** ($\mathcal{F}_t$): Cannot act if $\kappa_t, \rho_t, \eta_t$ below thresholds
2. **Hard safety** ($g_j \leq 0$): Cannot recommend if worst-case rollout violates %low threshold
3. **Effect size** ($\delta_{\text{eff}} > \delta_{\min}$): Cannot recommend if improvement is within noise
4. **Shadow graduation**: Cannot surface recommendations before 21+ days of validated shadow logging
5. **Burnout risk ceiling**: Flag if $\hat{\Delta}^B_H + \text{CI} > 5\%$

---

## SECTION 13 — OPEN MATHEMATICAL QUESTIONS (UPDATED)

1. **Belief representation adequacy.** The approximate belief $b_t \approx (\hat{x}_t, \Sigma_t)$ assumes Gaussian uncertainty. The true posterior over patient state (especially the psychological components) may be multi-modal or heavy-tailed. When does the Gaussian approximation break down, and what are the consequences for safety guarantees?

2. **Identifiability in the four-part decomposition.** Which parameters are identifiable in $\theta^{\text{post}}$ vs. confounded with the dynamic state $x_t$? For example: is current low mood ($x_t^{\text{psy}}$) distinguishable from low mood stability ($\theta^{\text{beh}}_{\text{mood\_stability}}$) given typical observation density (1 mood log per day)?

3. **Prior sensitivity.** How sensitive is the system's behavior (especially early recommendations) to the choice of persona prior $\theta^{\text{prior}}$? If the prior is wrong, how quickly does the posterior correct, and what damage can be done during the correction period?

4. **Critic bootstrapping stability.** The Critic is trained on realized residual costs, which depend on the policy, which depends on the Critic. Even though the Critic uses realized (not predicted) costs, the policy-dependence creates a mild feedback loop. Under what conditions does this converge?

5. **Burnout dynamics calibration.** The burnout state dynamics use a logistic-additive model with coefficients $\{\alpha_i\}$. These coefficients are currently set heuristically. What data would be needed to calibrate them empirically? How sensitive is $\hat{\Delta}^B_H(\pi)$ to misspecification of the burnout dynamics?

6. **CVaR estimation under finite rollouts.** With $N = 50$ rollouts and $\alpha = 0.8$, the CVaR is estimated from only 10 samples. This estimate has high variance. What is the resulting decision error rate (probability of choosing a suboptimal action due to CVaR estimation noise)? Should $N$ be increased?

7. **Cost weight elicitation.** The weights $\{w_{\text{low}}, w_{\text{tir}}, w_{\text{burden}}\}$ encode value judgments. Can these be elicited from patients via preference queries ("would you prefer 5% better TIR or 30% fewer recommendations?")? What is the minimum number of queries for stable elicitation?

8. **Safety under belief misspecification.** The safety gate checks worst-case rollouts sampled from $b_t$. If $b_t$ is a poor approximation to the true posterior (e.g., the true state is outside the 95% credible region because of an undetected regime change), the safety gate may clear an unsafe action. How often does this happen, and what are the consequences? The GP familiarity layer provides partial protection, but its coverage is not formally guaranteed.

9. **Multi-objective Pareto structure.** The weighted scalar cost collapses a multi-objective problem. Patients may benefit from seeing the Pareto frontier. What is the computational cost of tracing the frontier, and how should tradeoffs be presented?

10. **Trust as a multi-dimensional construct.** Trust may have components (trust in safety, trust in efficacy, trust in the system's understanding of the patient). A scalar $\tau_t$ may be insufficient. What dimensions of trust are identifiable from acceptance/rejection patterns?

11. **Horizon sensitivity of burnout attribution.** Formal analysis: for the burnout dynamics model specified in Section 7, derive the sensitivity $\partial \hat{\Delta}^B_H / \partial H$ and characterize when this sensitivity is large (indicating the estimate is unstable with respect to horizon choice).

12. **Convergence rate of the posterior.** Given the observation model and typical missingness patterns, how many days of data are needed for $\|\theta^{\text{post}} - \theta^{\text{true}}\| < \epsilon$ with probability $\geq 1 - \delta$? This determines the minimum Phase 1 (observation) duration.

---

## CLOSING SUMMARIES

### One-Paragraph Formal Problem Statement

Chamelia is a belief-state constrained stochastic control system operating over coupled physiological–psychological patient dynamics, where a LeCunian Configurator (meta-policy over horizons, masks, and constraints) parameterizes a Perception module (Bayesian belief estimator maintaining $b_t \in \mathcal{P}(\mathcal{X})$ over the true latent patient state from noisy multi-modal signals), a World Module (patient-specific digital twin with four-part decomposition — cohort prior, personalized posterior, dynamic state, rollout noise — producing stochastic forward trajectories under candidate actions), a two-part Cost Module (hardwired intrinsic cost over glycemic outcomes, recommendation burden, trust dynamics, and burnout state; plus a terminal-value Critic estimating residual cumulative cost beyond the explicit rollout horizon, trained from realized outcomes in memory), and an Actor (constrained CVaR-minimizing search over a conservative action set). Intrinsic cost and epistemic uncertainty are architecturally separated: the cost function measures only normative badness, while epistemic constraints (familiarity, concordance, calibration) gate whether action is permissible — eliminating the risk of double-counting uncertainty. The system computes policy-attributable burnout risk $\Delta^B_H(\pi)$ via paired counterfactual rollouts with shared initial state and exogenous noise, reporting calibrated confidence intervals, and blocks any policy whose upper confidence bound on burnout risk exceeds a hard threshold, regardless of glycemic benefit.

### Minimum Viable Mathematical Architecture

**Belief:** $b_t \approx (\hat{x}_t, \Sigma_t)$ from CTXBuilder features + rolling uncertainty. **Twin:** four-part: persona prior, posterior (MLE on 14–30 day windows), dynamic state (from Perception), rollout noise (from behavioral generator). **World Module:** t1d_sim, 50 rollouts × 7 days, sampling $x_0 \sim b_t$. **Cost:** pure normative: weighted glycemic + burden + trust + burnout, no uncertainty terms. **Epistemic layer:** GP familiarity + ensemble concordance + calibration, as hard constraints, not costs. **Safety:** worst-case rollout %low < 4%. **Actor:** grid search, CVaR at $\alpha$=0.8, effect-size gate at $\delta$=0.5. **Critic:** initially zero, bootstrapped after 30 days as linear ridge on terminal features. **Burnout:** 100 paired rollouts × 30 days, paired confidence intervals. **Memory:** 60-day buffer. **Configurator:** rule-based meta-policy.

### What a Future JEPA-Style Implementation Would Replace or Augment

JEPA replaces the handcrafted CTXBuilder with a learned encoder $e_\phi : o_{t-H:t} \to z_t$ producing latent belief representations; replaces the explicit ODE forward model with a latent predictor $z_{t+1} = p_\theta(z_t, a_t)$ trained by self-supervised prediction in latent space (the core JEPA insight: predict representations, not observations); and augments psychological dynamics with learned latent models of engagement, trust, and burnout from observed trajectories. It does NOT replace: the intrinsic cost function (hardwired, auditable), the safety constraints (hardwired, non-negotiable), the epistemic constraint layer (explicit, separate from cost), the memory schema (structured, explicit), the burnout attribution algorithm (counterfactual, algorithmic), or the architectural separation between normative cost and epistemic state — which is a structural property of the architecture that persists regardless of whether the components are explicit or learned.

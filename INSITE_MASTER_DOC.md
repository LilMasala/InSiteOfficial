# InSite: System Architecture & Goals

*Last updated: 2026-04-06*

---

## What InSite Is

InSite is a closed-loop AI system for Type 1 Diabetes management. The goal is not to replace endocrinologists or CGM controllers — it is to sit *above* the loop, learning each patient's physiology, lifestyle, and behavioral patterns, and nudging insulin therapy in ways that improve Time-in-Range (TIR) without burning the patient out.

The system has three main pieces:

1. **PhysiologyT1DSimulator** — synthetic patient population and simulation engine for pre-clinical development
2. **ChameliaV2** — the AI agent: trained via curriculum, deployed to reason over patient state and recommend therapy adjustments
3. **InSite (iOS app)** — the patient-facing product: integrates with Loop/LoopKit, HealthKit, CGM data, and surfaces Chamelia's recommendations

---

## The Problem Being Solved

T1D management is a continuous, high-stakes decision problem under uncertainty:
- Blood glucose is affected by food, insulin, exercise, sleep, stress, illness, and interaction effects between all of these
- CGM + closed-loop pumps (Loop) handle minute-to-minute corrections, but basal rates, carb ratios, and correction factors need periodic tuning
- Patients experience **burnout** from constant decisions and alerts — a system that maximizes TIR by demanding too much engagement will be abandoned
- Doctors see patients every 3 months; the data between visits is largely unexamined

The clinical goal is: **improve TIR without increasing patient burden**, and do it in a way that generalizes across the enormous patient-to-patient variability in T1D.

---

## PhysiologyT1DSimulator

**Directory:** `PhysiologyT1DSimulator/`

A synthetic cohort simulator used to develop and pre-validate Chamelia before real-patient deployment.

### What it models

- **Physiology** (`t1d_sim/physiology.py`): glucose dynamics using a compartment model (similar to Hovorka/Dalla Man). Takes insulin doses and carb inputs → outputs BG trajectory.
- **Patient population** (`t1d_sim/population.py`): generates synthetic patients with varying ISF (insulin sensitivity factor), CR (carb ratio), basal rates, dawn phenomenon severity, exercise response, etc.
- **Behavior** (`t1d_sim/behavior.py`): models whether patients accept or reject recommendations, log meals, engage with the app. This is critical — a recommendation that is ignored is worse than no recommendation.
- **Three-phase lifecycle** (`t1d_sim/patient_threephase.py`): each patient goes through:
  1. **Observation** — Chamelia watches, doesn't intervene
  2. **Shadow** — Chamelia makes recommendations that are shown to the patient but not acted on (builds trust)
  3. **Intervention** — Chamelia's recommendations are surfaced and acted on
- **Therapy** (`t1d_sim/therapy.py`): represents Loop settings (basal profile, ISF, CR) that Chamelia can adjust

### Key modeling details

- **Meal absorption** (`physiology.py`): dual-exponential kernel with FAST/MEDIUM/SLOW profiles. Returns probability distribution over 5-min bins.
- **Context effectors**: stress, menstrual cycle, exercise, sleep modulate physiology parameters at each step.
- **Behavior model** (`behavior.py`): models meal timing variability, incomplete bolusing (under/over), exercise patterns, sleep, and recommendation acceptance/rejection. A system that patients don't accept is useless.
- **Long-term drift** (`feedback.py`): ISF changes with fitness, CIR adapts to diet, burnout reduces adherence. Parameters update biweekly.
- **Questionnaire** (`questionnaire.py`): maps patient answers → physiological priors → sample synthetic "twin" patients for population diversity.

### What the sim produces

Running `chamelia.run_simulation` produces a SQLite database and JSON report that answers:
- Did mean TIR improve?
- Did burnout remain acceptable?
- How many patients progressed through observation → shadow → intervention?
- What was the acceptance rate of recommendations?

### Why this matters

ChameliaV2 is trained on a curriculum, not patient data (we don't have enough real data yet). The simulator lets us validate that Chamelia's trained reasoning capabilities actually translate to correct T1D management decisions before any real-patient exposure.

---

## ChameliaV2

**Directory:** `ChameliaV2/`

The AI agent. Not a language model, not a rule system — a planning agent that builds latent representations of patient state, reasons over them, and outputs therapy adjustment recommendations.

### Core Architecture

```
Patient State Tokens
        ↓
  ┌─────────────┐
  │  HJEPA      │  ← Hierarchical Joint-Embedding Predictive Architecture
  │  (encoder)  │    Context encoder + Target encoder (EMA)
  └─────────────┘    SequenceRoPE1D — no fixed token limit
        ↓
  ┌─────────────┐
  │Configurator │  ← Cross-attention over latent memory
  └─────────────┘    Produces task-conditioned context tokens
        ↓
  ┌─────────────┐
  │   Actor     │  ← 6-candidate action head (mode1=fast/mode2=deliberate)
  └─────────────┘    Outputs therapy adjustment recommendation
        ↓
  ┌─────────────┐
  │ WorldModel  │  ← Predicts future patient state given action
  └─────────────┘    Used for planning/evaluation
        ↓
  ┌─────────────┐
  │ CostModule  │  ← Trainable critic + intrinsic cost
  └─────────────┘    Scores action quality; learns from outcomes
        ↓
  ┌─────────────┐
  │LatentMemory │  ← Episodic + semantic memory
  └─────────────┘    Stores patient history, retrieves relevant context
```

### HJEPA — The Backbone

HJEPA is the encoder. It is *not* a generative model — it learns representations by predicting masked token representations in latent space (not pixel/token values). This is the key idea from JEPA (Yann LeCun, 2022):

**The thesis:** Intelligence requires building world models in representation space, not predicting raw observations. Predicting pixels or tokens forces the model to waste capacity on irrelevant low-level details. Predicting representations forces it to capture structure and semantics.

**How HJEPA works:**
1. Input tokens (patient state, game moves, reasoning chains — whatever the curriculum domain provides) are split into context (visible) and target (masked) sets
2. **Context encoder** processes the visible tokens → context representations
3. **Target encoder** (an EMA copy of the context encoder, no gradient) processes the full input → target representations
4. **Predictor** takes context representations + mask position information → predicts target representations
5. **Loss** is smooth L1 between predicted and actual target representations

The target encoder uses **Exponential Moving Average (EMA)** rather than gradient descent — its weights slowly track the context encoder. This prevents collapse (the trivial solution where both encoders output constants).

**Why EMA prevents collapse:** If both encoders were trained with gradients, the loss could be minimized by making all representations identical. EMA breaks this by making one encoder a slowly-moving "teacher" that the other must track. The predictor is forced to actually predict.

**HJEPA adds hierarchy** over vanilla I-JEPA: the FPN (Feature Pyramid Network) extracts representations at multiple scales, giving the predictor multi-resolution targets. Fine-grained (local) and coarse (global) structure are learned simultaneously.

**SequenceRoPE1D** (just added): replaces ViT's absolute positional embedding (hardcoded at 197 positions) with 1D Rotary Position Embeddings. Positions are encoded by rotating Q and K vectors in attention by angle `θ = position × frequency`. The rotation is relative — attention between position 40 and position 43 encodes "distance 3" regardless of absolute position. No fixed length limit.

### The Configurator

A cross-attention transformer that reads from `LatentMemory` to produce task-conditioned context tokens. This is how Chamelia's "persona" adapts to each patient — the same HJEPA backbone, but the Configurator shapes its context based on what it has learned about this specific patient's physiology and behavior.

### The Actor

Produces therapy adjustment recommendations. Has two modes:
- **Mode 1 (fast):** Direct action from context representations — low-latency, used when rapid response is needed
- **Mode 2 (deliberate):** Full pipeline including WorldModel rollouts — used for planned adjustments

The Actor outputs 6 candidate actions with associated confidence scores. The CostModule scores them; the highest-scoring non-blunder candidate is selected.

### The WorldModel

Predicts future patient state given a proposed therapy adjustment. Allows the Actor to evaluate "if I increase basal rate by 10% here, what happens to BG over the next 4 hours?" without waiting for reality.

### The CostModule

Has two components:
- **IntrinsicCost**: immediate quality signal (e.g., is the recommended action within safe range? does it align with patient preferences?)
- **TrainableCritic**: a learned value function that estimates long-term outcomes. Trained from delayed feedback — when a recommendation is accepted and the patient's TIR improves over the next week, that outcome is used to update the critic's parameters.

### LatentMemory

Episodic + semantic memory. Stores patient interaction history as compressed latent vectors. The Configurator reads from it to personalize context. The Critic reads from it to estimate outcome quality based on similar past situations.

---

## Curriculum Training

Chamelia is not trained on T1D data directly — it first learns general cognitive capabilities through a structured curriculum, then those capabilities are applied to T1D reasoning. This is both a data constraint (we don't have enough T1D data) and a principled design choice (generalization requires broad pre-training).

### The 6-Stage Curriculum

| Stage | Domain | What's being learned |
|---|---|---|
| 0 | Language (xnli, cc100) | Token prediction, basic language understanding |
| 1 | Reasoning (LSAT, GSM8K, ProofWriter, FOLIO, LSAT) | Logical deduction, mathematical reasoning, MCQ |
| 2 | Patterns (OEIS, arithmetic, HMM sequences) | Rule detection, regime identification, counterfactual prediction |
| 3 | Games (chess, go, poker) | Strategic planning, action evaluation, long-horizon reasoning |
| 4 | Collaborative (TBD) | Multi-agent interaction, negotiation |
| 5 | Health (T1D-specific) | Glucose dynamics, therapy optimization |

Each stage has 4-6 levels. A domain advances when a set of probe metrics exceed thresholds. The runner can extend a stage (more steps), retune learning rate, or eventually fail a domain and move on.

### Key Training Concepts

**Cost functions over actions:** Each curriculum domain defines cost functions that evaluate the model's output action. The model doesn't predict tokens autoregressively — it predicts a single action (e.g., next move, MCQ answer, therapy adjustment) from a compressed representation. Loss is computed over this action distribution.

**Masking strategy:** Inputs are partially masked before encoding. The model must predict answer tokens from incomplete context — this forces it to build robust representations rather than copying input.

**Advancement probes:** At each evaluation step, the curriculum runner calls `run_advancement_probe()` on the domain to get metrics (e.g., `game_score`, `accuracy`, `blunder_rate`). If thresholds are met, the domain level advances.

**EMA of target encoder:** Continues through all stages. The representations Chamelia learns in stage 0 become the foundation for stage 1, etc. — curriculum learning works because of this continuity.

### Data on Cluster (Wesleyan HPC)

| Dataset | Stage | Size |
|---|---|---|
| xnli, cc100 | 0 | ~GB-scale |
| agieval (lsat-ar/lr/rc, sat-en, logiqa-en) | 1 | 230-651 records/split |
| gsm8k | 1 | ~8.5K records |
| hendrycks_math | 1 | 7.5K train / 5K test |
| proofwriter | 1 | ~845K rows total |
| folio | 1 | 203 val lines |
| open_platypus | 1 | 24.9K rows |
| OEIS sequences | 2 | synthetic fallback |
| chess (lichess + engine annotations) | 3 | 4K train records |

### Checkpoint Strategy

- **Status checkpoints** (small, ~KB): stage/level events → NFS (`checkpoints/{run_tag}/`)
- **Bridge artifacts** (large, ~1.5GB): full model state → local scratch (`$TMPDIR/bridge_artifacts/`)
- Only the **best-scored bridge artifact** is copied to NFS automatically (`best_model.pth`)
- At job end, manually copy scratch artifacts if needed: `cp -r $TMPDIR/bridge_artifacts/ checkpoints/{run_tag}/bridge_artifacts_all/`

---

## What Chamelia Actually Controls (InSiteBridgeDomain)

The domain plugin (`src/chamelia/plugins/insite_t1d.py`) defines what signals Chamelia sees and what actions it can output.

**Input signals** — 8 features, daily aggregates:
```
bg_avg              Blood glucose average (normalized by 250)
tir_7d              Time in range 70-180 mg/dL, 7-day rolling
pct_low_7d          % time <70 mg/dL
pct_high_7d         % time >180 mg/dL
bg_var              Glucose variance
exercise_mins       Exercise duration (normalized by 120)
cycle_phase_menstrual  Menstrual cycle phase
cycle_phase_luteal     Luteal phase
```

**Output action** — 8-dimensional vector decoded into therapy adjustments:
```
hold_bias           Confidence adjustment (how assertive to be)
basal_adjustment    Base insulin rate change
correction_bias     Aggressiveness of high-BG correction
meal_bias           Carb-ratio aggressiveness
support_intensity   Social/contextual support level surfaced to user
stability_bias      Conservative vs. experimental posture
probe_bias          Information-seeking (e.g., meal timing exploration)
trust_preservation  Continuity of user trust / recommendation style
```

**Intrinsic cost functions** (weighted sum → scalar cost):
- Hypoglycemia cost (weight 0.5): `ReLU((95 - bg) / 55) + pct_low + aggressiveness_penalty`
- Hyperglycemia cost (weight 0.3): `ReLU((bg - 160) / 120) + pct_high + support_penalty`
- Volatility cost (weight 0.2): `bg_variance + intervention_size_penalty`

The CostModule's TrainableCritic learns *realized* long-term costs on top of these intrinsic signals.

---

## Serving & Bridge Runtime

**Directory:** `ChameliaV2/src/serving/`

The bridge runtime manages per-patient sessions. Each session has its own Chamelia model instance + memory.

**Per-decision flow:**
```
observe(signals)
  ↓ tokenize → [B, N, D]
  ↓ HJEPA encode → hierarchical reps
  ↓ memory retrieve (kNN → relevance reranker)
  ↓ Configurator → context tokens [B, 16, D]
  ↓ Actor → 6 candidates
  ↓ WorldModel rollout (3 steps each)
  ↓ Critic → 30-step value estimate
  ↓ select best candidate
  ↓ return recommendation to app

user_accepts/rejects
  ↓ next day: record_outcome(signals, realized_cost)
  ↓ ingest_replay_examples() → offline critic/retrieval training
```

**Deployment modes:**
- **local**: StubSequenceHJEPA (64-dim), CPU, for dev/testing
- **remote**: Full HJEPA on Cloud Run GPU, production
- **hybrid**: Local inference, remote training

---

## InSite iOS App

**Directory:** `InSite/`

The patient-facing application. Integrates:
- **LoopKit / Loop** — reads current CGM data, active insulin, pump state
- **HealthKit** — glucose readings, activity, sleep
- **tconnectsync** — Tandem pump integration
- **Chamelia (via bridge runtime)** — sends patient state, receives recommendations

The app surfaces recommendations gently: it does not override Loop's closed-loop control. It suggests adjustments to basal rate, ISF, or CR that a patient or their endocrinologist reviews.

---

## The Goal, Restated

```
PhysiologyT1DSimulator
    → validates Chamelia decisions are physiologically sound
    
ChameliaV2 curriculum training
    → builds a planning agent capable of reasoning over
      complex, uncertain, long-horizon sequential decisions
    
ChameliaV2 stage 5 (health)
    → fine-tunes that capability on T1D-specific problems:
      predicting glucose response, evaluating therapy changes,
      personalizing to individual patient physiology
      
InSite iOS app
    → deploys to real patients, respects their autonomy,
      learns from outcomes, improves TIR without burnout
```

The north star metric is: **patients using InSite should have better TIR and lower diabetes distress scores than patients using Loop alone**, as measured in a controlled study.

---

## Current Status (2026-04-06)

- Stage 0 (language) training: **working**, graduating in ~200 steps
- Stage 1 (reasoning): recently fixed (GSM8K MCQ conversion, ProofWriter binary MCQ, logiqa2 remapped, seq_len crash fixed via SequenceRoPE1D)
- Stages 2-3: data pipeline fixed (dataset sizes, chess `best_move` extraction)
- Stages 4-5: data pipeline not yet validated
- Bridge artifacts: NFS crash fixed (scratch-first write strategy)
- Benchmark eval (`--eval-only`): implemented, not yet run on a trained checkpoint

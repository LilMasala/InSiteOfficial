# Cognitive Architecture Specification
### A General Learning System Inspired by Human Memory and Skill Acquisition
**Status:** Draft v0.7 — Living Document  
**Author:** [You]  
**Last Updated:** April 2026

> **Literature foundation (2024–2025):** LOVE · Stitch · LILO · RSD · JEPA-Reasoner · HWM · CSR · IOB · BODE-GEN · Dream Decompiling · Choreographer · EB-JEPA. Named algorithms are used throughout rather than re-inventing solutions the field has already solved.

> **Core design principles:**
> 1. Replace binary gates with continuous weighting by signal quality — growth is a gradient, not a switch
> 2. Inference never touches disk — hot tier fully loaded at startup
> 3. Nothing is replaced — everything is extended upward and outward
> 4. Skills stay in latent space — no token generation ever enters the reasoning loop

---

## 1. Motivation

Current LLMs re-derive everything from shared weights on every forward pass. When a model sees `+`, it doesn't *call* an addition function — it statistically reconstructs what addition looks like from patterns in its weights. This is fundamentally unlike how learning systems (humans) operate.

Humans:
- Learn procedures once and *call* them by recognition
- Store specific experiences (episodic) separately from reusable skills (procedural)
- Consolidate memory during sleep — compressing frequently-used paths into faster routes
- Develop domain expertise without losing competence in other domains
- Transfer skills across domains by building new trigger networks into existing procedures
- Reason in a continuous internal space and only produce output when needed

This document specifies an architecture that moves toward all of these properties. The 2024–2025 literature validates the design direction and provides specific named algorithms for each component. This spec uses those algorithms rather than inventing new ones.

---

## 2. Core Thesis

**Learning = encoding new experience + building associations through a memory network + consolidating frequent paths into callable procedures.**

When you encounter a problem, your brain doesn't recompute everything. It:
1. Recognises the situation (routes to a domain)
2. Retrieves relevant past experiences (episodic memory)
3. Invokes known procedures that apply (procedural memory)
4. Uses the base reasoning system only for the novel connective tissue between known pieces
5. During sleep, compresses any new paths that proved useful into more direct routes

This is **transductive learning through a memory network** — reasoning across stored examples and procedures rather than re-deriving from raw weights.

---

## 3. Memory Architecture

### 3.1 Two Distinct Memory Types

The system maintains two fundamentally different memory stores. This separation is critical.

#### Episodic Memory
*"What happened here before"*

- Stores specific past experiences as `EpisodeRecord`s keyed by latent state vectors
- Retrieved by similarity to current situation + outcome quality weighting
- Domain-local — each domain cluster has its own episodic pool
- Decays and is pruned over time (circular buffer)
- **Already implemented in `LatentMemory` / `memory.py`**

#### Procedural Memory
*"What always works when I see this pattern"*

- Stores reusable skills: `differentiate`, `plan_forward`, `consider_circumstances`, `add`, etc.
- **Domain-agnostic** — skills float in their own space, not owned by any cluster
- Each skill is a **compressed latent action** — a fixed-dimension encoding of an action subsequence. This keeps everything in latent space. No tokens, no symbolic programs.
- Retrieved via latent transition query: `Δz = P(z_current, z_goal)` → FAISS → matching skill embeddings
- Indexed using CSR (Contrastive Sparse Representation) or IOB (Information-Ordered Bottleneck) embeddings for efficient retrieval (see section 11)
- Never decays — skills are stable once consolidated
- **Not yet implemented — primary gap in current architecture**

### 3.2 What a Skill Actually Is

A skill is a **compressed latent macro-action** — a learned fixed-dimension vector encoding an action subsequence that reliably produces a specific type of latent state transition.

Skills live in the Hierarchical World Model's (HWM) high-level action space. The high-level planner operates on latent macro-actions of fixed dimension (e.g. 4–16 dims). These macro-actions are encoded from low-level action sequences by a Latent Action Encoder (small transformer). A skill is one of these latent macro-actions that has been consolidated, named, and indexed.

This means:
- Skills are naturally composable — the high-level planner can sequence them
- Skills stay entirely in latent space — no token generation ever enters the loop
- The skill representation is directly executable by the existing WorldModel
- FAISS can index them natively since they're fixed-dimension vectors

### 3.3 Skill Granularity and Composition

Skills are maximally granular — as atomic as possible. `differentiate`, `substitute`, `simplify`, `recognise_pattern` are separate primitives. `integrate_by_parts` is not a new primitive — it's a **path** through those primitives via the high-level planner.

Paths get promoted to compiled skills only when traversal frequency makes the reconstruction cost a bottleneck. Decision criterion: *is this path traversed often enough that MCTS search time matters?* If yes, compile to a latent macro-action and index it. If no, leave it as a composable sequence. The primitive library grows slowly and deliberately — no explosion of redundant near-duplicate skills.

### 3.4 Interaction Between Memory Types

```
Problem arrives
    ↓
Latent transition query: Δz = P(z_t, z_goal)
    ↓
FAISS skill retrieval: top-k matching skill embeddings        [< 1ms]
    ↓
    ├── Match found + confident:
    │   → Execute skill (latent macro-action → WorldModel)    [System 1]
    │   → If outcome OK: store episode, return
    │   → If outcome poor: narrow skill scope, fall through
    │
    └── No match / low confidence:
        → MCTS + HWM deliberate reasoning                     [System 2]
        → Store episode WITH full MCTS trace
        → Async: sleep phase mines trace for skill candidates
```

Stored episodes contain full MCTS traces — which primitive sequences were explored, which succeeded, what costs were found. This is what the sleep phase mines to form new skills.

---

## 4. Domain Clustering (Dirichlet Process)

### 4.1 The Problem It Solves

Catastrophic interference: shared weights that get good at math get overwritten when learning biology. The model averages out. The fix is structural, not regularization-based. **Stop fighting catastrophic interference — route around it.**

### 4.2 How It Works

Domains are discovered nonparametrically via a **Dirichlet Process** — the system doesn't pre-specify how many domains exist. They emerge from experience.

Each domain cluster contains:
- A **centroid** in latent space (the domain's address)
- Its own **episodic memory pool** (HDF5 archive)
- Its own **trigger weights** over the shared procedural memory library
- Its own **posture seeds** — how to frame problems in this domain
- A **LoRA-style weight adapter** — the diff from base weights representing peak competence

### 4.3 Cluster Formation

New cluster spawns when incoming experience lands far from all existing centroids AND performance on existing clusters would degrade by absorbing it. Sleep phase confirms it's genuinely new.

### 4.4 Routing

```
Input → HJEPA encodes → latent z
    ↓
Match z against domain index (cosine similarity to cluster centroids)
    ↓
    ├── Close to existing cluster:
    │   → Load that cluster's LoRA adapter
    │   → Retrieve from that cluster's episodic memory
    │   → Use that cluster's trigger weights for skill retrieval
    │
    └── Novel / uncertain:
        → Use base model (no adapter)
        → Flag for sleep-phase cluster evaluation
```

The base model only needs to be good enough to recognise which domain it's in. The specialist handles competence within the domain.

### 4.5 Skill Transfer Across Domains

Skills are domain-agnostic. The math cluster has strong trigger weights toward `differentiate`. Biology initially doesn't — but the first time biology reasoning traverses a path that invokes `differentiate`, that cross-cluster edge is stored. Sleep strengthens it. Eventually biology has its own direct trigger.

**Expertise transfer = building new trigger networks into existing skills.** The physicist learning biology doesn't re-learn calculus — they learn to recognise where calculus applies in the new domain.

---

## 5. The Thinker / Talker Architecture

### 5.1 The Core Decoupling

This is the answer to "action-decision-only mode that isn't just an autoregressive decoder on the final latent state."

JEPA-Reasoner (2025) validates and formalises this: **decouple reasoning from output generation entirely.**

```
┌─────────────────────────────────────────────┐
│  THINKER                                    │
│  Reasons exclusively in continuous          │
│  normalised latent space.                   │
│  Never generates tokens.                    │
│  Runs on every inference.                   │
└────────────────┬────────────────────────────┘
                 │ frozen latent reasoning chain
                 │ (one-way — never feeds back)
                 ▼
┌─────────────────────────────────────────────┐
│  TALKER                                     │
│  Reconstructs natural language from the    │
│  latent reasoning chain.                   │
│  Only invoked when output is needed.        │
│  Token sampling errors cannot propagate    │
│  back into reasoning.                      │
└─────────────────────────────────────────────┘
```

In action-decision mode (InSite hot path): **Thinker only**. Encode, retrieve, reason, select action — entirely in latent space. Talker never runs.

In explanation mode: Thinker runs and completes reasoning. Talker then reconstructs from the frozen latent chain. Token sampling failures cannot corrupt the already-completed plan.

### 5.2 Why This Matters Empirically

JEPA-Reasoner (0.9B parameters) achieved **149.5% improvement** over a coupled transformer baseline on mathematical reasoning (GSM8K) purely by implementing this decoupling. The gains come from:

- **Error containment**: token-level generation failures don't propagate into the reasoning chain
- **Continuous guidance**: Talker has access to the entire lossless reasoning trajectory
- **Multi-hypothesis superposition**: Thinker can maintain parallel hypotheses in latent space before committing — something autoregressive generation cannot do

### 5.3 Relationship to Existing Architecture

The **Thinker is your existing Chamelia pipeline**: HJEPA → Configurator → Actor → WorldModel → CostModule. It already reasons in latent space. It already doesn't generate tokens. It's already 90% of the Thinker.

The **Talker is a new module** — lightweight decoder taking the latent reasoning chain → natural language. Downstream of everything, never feeds back, invoked on demand only.

---

## 6. The Sleep Phase — LOVE + Stitch + RSD

### 6.1 Purpose

Sleep is a dedicated non-inference phase. It:
1. Decomposes MCTS episode traces into skill-shaped segments (**LOVE**)
2. Compresses those segments into reusable λ-abstractions (**Stitch**)
3. Encodes abstractions into latent skill vectors and indexes them (Latent Action Encoder + FAISS)
4. Optionally documents new skills in natural language (**LILO AutoDoc**)
5. Discovers new frontier skills via adversarial challenge (**RSD**)
6. Refines domain cluster boundaries
7. Strengthens cross-cluster skill triggers
8. Prunes low-quality episodes

Sleep never runs inference or makes recommendations. It reorganises.

### 6.2 Triggers

- **Opportunistic**: low activity windows (natural for InSite's health monitoring context)
- **Periodic**: every N inference steps (configurable)
- **Buffer pressure**: episodic memory approaching capacity

Sleep runs as a background thread. Pauses and resumes if inference arrives mid-sleep.

### 6.3 Stage 1 — LOVE Decomposition

**LOVE (Learning Options Via comprEssion)** partitions MCTS episode traces into skill-shaped segments by optimising:

```
J = E[log p(experience | skills)] - β · DescriptionLength(skills)
```

The compression penalty forces the model to find statistical regularities — action subsequences that recur across many episodes in the same structural form. These are the natural skill boundaries. β controls granularity: higher β = coarser fewer skills; lower β = finer more primitives.

LOVE is scalable to high-dimensional observations and avoids degeneracy problems from pure maximum-likelihood objectives, which often find trivial non-generalising skills.

Output: a partition of each episode's action trace into named segments with shared structure across episodes.

### 6.4 Stage 2 — Stitch Compression

**Stitch** (POPL 2023) takes LOVE segments and finds optimal λ-abstractions — reusable sub-programs that minimally re-describe the full corpus. Stitch achieves 1,000x–10,000x speedup over DreamCoder's compression algorithm.

`integrate_by_parts` emerges as a λ-abstraction over `recognise_pattern`, `differentiate`, `substitute`, `simplify` — not because it was hand-specified, but because those four primitives co-occur in that order across enough episodes that Stitch finds it worth naming.

**Stitch needs discrete symbolic handles.** This is a key reason for the VQ layer — quantised codes give Stitch the discrete representations it needs to find structural recurrences. With continuous vectors you'd cluster first; with VQ codes Stitch runs directly on the symbolic representation.

### 6.5 Stage 3 — Latent Action Encoding

The Stitch abstractions are symbolic descriptions of action subsequences. A **Latent Action Encoder** (small transformer) converts each abstraction into a fixed-dimension latent macro-action vector:

```
Stitch abstraction (symbolic) → Latent Action Encoder → latent macro-action [d_skill]
```

This latent macro-action IS the skill. It's executable by the WorldModel's high-level planner. It's indexable by FAISS. From this point forward everything stays in latent space.

The Latent Action Encoder is trained during sleep on (symbolic_abstraction → execution_outcome_in_latent_space) pairs.

### 6.5b Stage 3b — Dream Decompiling (Amortised Knowledge Extraction)

**Dream Decompiling (ICML 2024)** provides a complementary mechanism to Stitch. Rather than waiting for a task to be *solved* before extracting skill components, Dream Decompiling mines the neural search policy's amortised knowledge directly — identifying functionality worth chunking into the library even from *unsolved* tasks.

Concretely: the MCTS Actor builds up internal representations of promising directions even when no final solution is found. Dream Decompiling treats the Actor's learned search behaviour as a signal — "what subproblems is the Actor implicitly decomposing this into?" — and extracts those subproblems as skill candidates.

This is critical for hard problems where Stitch alone might find nothing (no solved traces to compress). Dream Decompiling gives the sleep phase signal even from failed episodes.

Integration: runs in parallel with LOVE/Stitch on the same episode corpus. Candidates from Dream Decompiling feed the same Latent Action Encoder → FAISS pipeline. The two approaches are complementary: Stitch compresses successful paths (reduces depth), Dream Decompiling extracts partial knowledge from failures (reduces breadth on next attempt).

### 6.5c Stage 3c — BODE-GEN for Novel Skill Search

**BODE-GEN (Bayesian Optimisation for Code Generation)** demonstrates that search for complex structured skills can stay in a continuous embedding space — avoiding discrete token-by-token generation while maintaining expressiveness.

Applied here: when RSD's adversarial generator proposes a new frontier skill target that doesn't yet exist in the library, BODE-GEN-style search finds a latent prompt (a conditioning vector) that, when fed through the Latent Action Encoder, produces a skill embedding that achieves the target behaviour. The search is Bayesian optimisation over the continuous latent space — efficient, non-autoregressive, and fully within the latent substrate.

This gives the system a principled way to generate genuinely new skills, not just compress existing ones. The output is still a latent macro-action vector — no tokens, no symbolic programs, no external model dependencies.

### 6.6 Stage 4 — LILO AutoDoc (Optional)

**LILO** augments Stitch with an LLM-powered documentation step that infers natural language names and docstrings for discovered abstractions. Fills the `description` field in `skill_constraints` and provides human-readable skill names. Useful for debugging and interpretability. Runs offline during sleep and never touches the hot path.

### 6.7 Stage 5 — RSD: Frontier Skill Discovery

**RSD (Regret-aware Skill Discovery)** expands the skill library adversarially.

A skill generator policy `π_θ2` proposes new challenging skills that maximise the agent's regret — potential for value improvement. The agent policy `π_θ1` attempts to master them. This min-max game ensures:
- Skill space doesn't fill with redundant easy skills (already mastered → zero regret)
- Skill space doesn't stall on impossible skills (too hard → regret saturates, generator moves on)
- New skills are discovered along the actual frontier of current competence

RSD runs during sleep after LOVE + Stitch have consolidated existing skills. It expands the frontier rather than re-processing what's already known.

Empirical result: RSD achieves **15% improvement** in zero-shot generalisation in high-dimensional environments over uniform skill discovery.

### 6.8 Stage 6 — Indexing and Pruning

```
1. CSR/IOB encode new skill latent vectors
2. Add to FAISS skill index
3. Update SQLite (confidence, constraints, source_episodes, AutoDoc names)
4. Evaluate domain clusters (split / merge / flag as needed)
5. Prune episodic memory:
   - Remove episodes fully compiled into skills
   - Remove low-quality episodes below retention threshold
   - Keep outliers and surprising episodes (most informative)
6. Serialize updated FAISS index to disk
7. Commit sleep workspace HDF5 to domain archive
```

### 6.9 Relationship to Existing Training

Sleep is separate from online training. Online training updates base model weights. Sleep reorganises the memory graph and writes new procedural memory entries. LoRA adapters can be updated during sleep using accumulated episodes — domain-specific fine-tuning without touching base weights.

---

## 7. The Memory Graph

### 7.1 Node and Edge Types

```
Node types:
  - EpisodeNode     (specific past experience, keyed by latent state)
  - SkillNode       (compiled latent macro-action, FAISS-indexed)
  - DomainNode      (cluster centroid + LoRA adapter + episodic pool)
  - ConceptNode     (abstract concept recurring across episodes — VQ code cluster)

Edge types:
  - SIMILAR_TO      (cosine similarity, weighted — existing retrieval)
  - TRIGGERED_BY    (skill ← latent transition pattern that activates it)
  - USED_IN         (skill → episodes where it was invoked)
  - LEADS_TO        (episode → outcome quality)
  - BELONGS_TO      (episode → domain cluster)
  - COMPOSES        (skill → sub-skills / primitives it's built from)
  - CROSS_CLUSTER   (path from one domain cluster through another's skill)
```

### 7.2 Pathfinding = Problem Solving

- **Short path** (direct skill match): System 1, sub-5ms
- **Medium path** (MCTS through existing skills): System 2, 50–200ms
- **Long path** (MCTS through novel primitive combinations): 200ms+
- **No path**: genuinely novel — base model, trace recorded for sleep

The world model rollout IS this pathfinding made explicit. Successful paths become new edges. Frequently-used paths become new skills.

### 7.3 Transductive Learning

The core learning mechanism: when the system sees a new problem it asks "what path through my memory network reaches a solution?" rather than deriving a rule from scratch or pattern-matching to training data. Associations are traversed, not recomputed. The graph is the knowledge. Expertise is having short direct paths to correct procedures.

---

## 8. Hierarchical World Model (HWM)

### 8.1 Two-Level Planning

The existing flat WorldModel is upgraded to a Hierarchical World Model:

```
HIGH-LEVEL PLANNER
  Input:  latent macro-actions (skills) [d_skill dim]
  Horizon: long (hours for T1D)
  Output: sequence of subgoal latent states z_subgoal

      ↓ first subgoal passed to low-level planner

LOW-LEVEL PLANNER  (= existing WorldModel, unchanged internally)
  Input:  primitive actions [action_dim] + subgoal z_subgoal
  Horizon: short (minutes)
  Output: actual action recommendation
```

Empirical result from literature: HWM achieves **70% zero-shot success** on complex robotic tasks where single-level world models achieve 0%. Planning cost reduction up to 4x.

### 8.2 Relationship to Existing WorldModel

The existing `world_model.py` becomes the low-level planner with no internal changes. A new `HighLevelPlanner` module operates above it, issuing macro-actions (skills) that the low-level planner executes. The interface between them is the subgoal latent state `z_subgoal`.

This is purely additive.

### 8.3 Choreographer: Imagination-First Skill Learning

**Choreographer** (parallel literature) provides a blueprint for how the HWM interacts with skill discovery during sleep. Choreographer decouples exploration from skill learning — skills are discovered and refined entirely within the world model's imagination, without requiring real-world interactions for every refinement.

Applied here: during sleep, the HWM acts as a simulation environment. RSD's adversarial skill generator proposes frontier skills. A meta-controller evaluates and adapts candidate skills in parallel in imagination. Only skills that survive imagination testing get written to the FAISS skill library. This dramatically improves data efficiency — the system can evaluate thousands of skill candidates per sleep cycle without any real patient interactions.

This is especially important for InSite: real patient interactions carry safety constraints and delay. Imagination-based skill learning means the system gets smarter between patient sessions, not just during them.

---

### 9.1 Role in the Two-Speed Model

Skill invocation is System 1 — fast, pattern-matched. MCTS is System 2 — deep sequential search for when System 1 fails or the problem is genuinely novel.

The current `reasoning_steps` loop in `chamelia.py` is flat parallel search: K candidates evaluated side by side. This misses problems that only manifest multiple steps ahead (the pizza crash — glucose looks fine at step 1, crashes at step 12 because of slow digestion). MCTS finds those because it keeps expanding promising branches deep enough to discover the eventual cost.

### 9.2 The Latent Node

```
N = { z, ctx, ψ }

z:    latent physical/world state [D]         — from WorldModel
ctx:  situational awareness tokens [C, D]     — from Configurator
ψ:    psychological/domain state              — trust, burnout, burden
```

### 9.3 The Four Stages

**Stage I — Selection (UCT, cost minimization form)**

```
UCT(s, a) = -V(s, a)  +  C × sqrt( ln N(s) / N(s, a) )

V(s, a):   average cost of branch (s, a) from prior simulations
N(s):      total visit count of node s
N(s, a):   visit count of action a from node s
C:         exploration constant
```

**Stage II — Expansion**

Actor acts as policy head at each leaf, proposing K next-steps. Each becomes a child node. Posture seeds still apply — different postures seed different search directions.

**Stage III — Simulation with safety gates**

Roll WorldModel forward H steps accumulating cost. Safety gates abort dangerous branches mid-playout — not as post-hoc filters. This saves compute and lets the safety signal backpropagate as a real cost, not a binary reject. Branches near danger accumulate higher V values; the tree learns to steer away.

**Stage IV — Backpropagation**

Total simulation cost flows back up the tree. Every node on the selection path updates V(s,a) and N(s,a).

### 9.4 Full Inference Flow

```
chamelia.forward():

1. Encode input → z, ctx                                [HJEPA]
2. Domain routing → load LoRA adapter                   [DomainIndex]
3. Latent transition query → FAISS skill retrieval      [ProceduralMemory, <1ms]
4. If skill found + confident:
   → Execute latent macro-action via HWM                [System 1, <10ms total]
5. If no skill / low confidence:
   → Init MCTS tree at (z, ctx, ψ)
   → Run N simulations: UCT + Actor + HWM + safety gates
   → Select: argmin_a V(root, a)
   → Store episode with full MCTS trace                 [System 2, 50–200ms]
6. Async: update skill confidence scores
7. Async: sleep phase mines trace for skill candidates
```

### 9.5 Compute Budget

| Parameter | Value | Rationale |
|---|---|---|
| N simulations | 32–128 | Sweet spot for T1D decision horizon |
| H rollout depth | 6–24 steps | Hours-scale horizon |
| K expansion width | 4–8 | Matches current candidate count |
| Tree reuse | Yes | Patient state evolves slowly — warm start across calls |

### 9.6 The Oedipus Effect: Causal Prediction in Latent Space

Because MCTS reasoning happens in the Thinker's continuous latent space before any action is taken, the system maintains a **superposition of possible futures** rather than committing to a single predicted trajectory. This is the "Oedipus effect" from the literature — the prediction system can evaluate the causal impact of a skill *before* the prediction intervenes in the causal chain it's reasoning about.

Practically: the Thinker holds multiple MCTS branches alive simultaneously in latent space. It doesn't collapse to a single action until the search budget is exhausted. This prevents the "reflexivity trap" where a premature commitment to one branch causes the model to misinterpret subsequent evidence as confirming that branch. The Talker only ever sees the completed, committed plan — never the superposition.

This is why the Thinker/Talker decoupling matters beyond just speed. It's epistemically safer. The reasoning doesn't collapse until it has to.

---

## 10. Embedding Compression: CSR and IOB

### 10.1 Why Standard Dense Embeddings Are Suboptimal

Plain cosine similarity over dense vectors treats all dimensions equally and has no ordering. Fine for small libraries, becomes a bottleneck as the skill library grows to tens of thousands of entries.

### 10.2 Information-Ordered Bottleneck (IOB)

IOB produces embeddings where dimensions are ordered by importance — most discriminative information in the first few dimensions. Enables coarse-to-fine retrieval: FAISS searches truncated embeddings for fast approximate matching, then refines with full embeddings for the top candidates. Can be truncated at any width without retraining. Single-stage training with low overhead.

Use for: **episodic memory retrieval** — ordered dimensions aid coarse-to-fine search over large episode archives.

### 10.3 Contrastive Sparse Representation (CSR)

CSR produces high-dimensional but selectively activated embeddings — most dimensions near-zero, only a small active subset fires per skill. Outperforms Matryoshka (nested) embeddings in both accuracy and retrieval speed with significantly less training overhead. Adjustable active set size at inference time — the system can trade retrieval speed for precision dynamically based on available compute.

Use for: **skill trigger matching** — sparse activations make sub-millisecond FAISS search practical at scale.

### 10.4 Isotropic Super-Resolution for Compact Skill Storage

Recent work on extreme compression shows that skills (as latent action sequences) can be stored as as few as 32 discrete tokens without meaningful loss of behavioural fidelity. The full-resolution skill is only reconstructed when it needs to be *executed* — during hot-path invocation. At rest in the FAISS index, skills live in their compressed form.

This means the skill library can be far more compact than intuition suggests. A library of 100k skills might occupy only a few hundred MB in the compressed CSR representation — easily hot in RAM.

The pipeline: skill → VQ compression → 32-token discrete representation → stored in FAISS. At execution: 32 tokens → Latent Action Encoder → full latent macro-action → WorldModel input.

---

## 11. Data Structure and Storage

### 11.1 Hot / Cold Separation

**Inference never touches disk.** Hot tier is fully loaded at startup. Disk I/O only during sleep, startup, or domain switch.

**Hot tier** (RAM during inference):

| Component | Technology | Size | Notes |
|---|---|---|---|
| VQ codebook | Model checkpoint | ~MB | Always loaded |
| CSR skill FAISS index | GPU FAISS | ~10s MB | Loaded at startup |
| IOB episode FAISS index | FAISS | ~100s MB | Hot buffer |
| Domain cluster centroids | NumPy | ~KB | Always hot |
| Active LoRA adapter | PyTorch tensors | ~10–50MB | Swaps on domain change |
| Skill metadata cache | SQLite in-memory | ~10s MB | Loaded at startup |

**Cold tier** (disk, accessed deliberately):

| Component | Technology | Notes |
|---|---|---|
| Episodic archive | HDF5 per domain | Tensor-native, fast bulk I/O |
| Skill metadata + constraints | SQLite | Queryable relational store |
| LoRA adapters | PyTorch `.pt` per cluster | Loaded on domain routing |
| FAISS snapshots | Serialized FAISS index | Written after each sleep |
| Sleep workspace | Temp HDF5 | Committed on sleep completion; safe to discard if interrupted |

### 11.2 Why Not PyTorch Native Graphs

PyTorch is optimised for dense matrix operations, not sparse graph traversal, sub-millisecond lookups, mixed-type node operations, or pointer chasing. A PyTorch-native graph would be 10–100ms per operation. The hot path needs sub-millisecond skill matching.

### 11.3 Recommended Stack

```
Inference hot path:      GPU FAISS (CSR + IOB) + in-memory skill cache
Skill metadata:          SQLite — lightweight, queryable, zero-dependency
Episodic archive:        HDF5 per domain — tensor-native, fast bulk I/O
LoRA adapters:           PyTorch .pt per cluster — load on domain switch
VQ codebook:             Embedded in model checkpoint
Sleep workspace:         Temp HDF5 → committed on completion
Future (production):     LanceDB — unified persistent vector DB
                         replacing FAISS + HDF5 when system scales
```

### 11.4 HDF5 Domain Archive Structure

```
domain_math.h5
├── metadata/
│   ├── cluster_centroid          [D]
│   ├── created_step              scalar
│   └── last_updated_step         scalar
├── episodes/
│   ├── keys                      [N, D]            latent state vectors
│   ├── actions                   [N, A]            action vectors
│   ├── outcomes                  [N]               realized costs
│   ├── skill_traces              [N, max_skills]   skills invoked
│   ├── mcts_traces               [N, ...]          full MCTS tree snapshots
│   └── timestamps                [N]
└── archived_skills/
    └── (skills promoted from this domain's episodes)
```

### 11.5 SQLite Schema

```sql
CREATE TABLE skills (
    skill_id        TEXT PRIMARY KEY,
    name            TEXT,        -- populated by LILO AutoDoc
    output_type     TEXT,
    confidence      REAL,
    use_count       INTEGER,
    formation_step  INTEGER,
    deprecated_by   TEXT REFERENCES skills(skill_id),
    created_at      TIMESTAMP,
    updated_at      TIMESTAMP
);

CREATE TABLE skill_constraints (
    constraint_id   INTEGER PRIMARY KEY,
    skill_id        TEXT REFERENCES skills(skill_id),
    constraint_type TEXT,        -- 'valid_when', 'invalid_when'
    description     TEXT,        -- human-readable (LILO AutoDoc)
    latent_region   BLOB         -- encoded region where constraint applies
);

CREATE TABLE skill_invocations (
    invocation_id   INTEGER PRIMARY KEY,
    skill_id        TEXT REFERENCES skills(skill_id),
    episode_id      TEXT,
    outcome_quality REAL,
    succeeded       BOOLEAN,
    step            INTEGER
);

CREATE TABLE domain_clusters (
    cluster_id      TEXT PRIMARY KEY,
    name            TEXT,
    centroid_file   TEXT,
    lora_file       TEXT,
    episode_file    TEXT,
    formation_step  INTEGER,
    last_sleep_step INTEGER
);
```

### 11.6 The VQ Codebook as Semantic Dictionary

The VQ codebook is a learned semantic dictionary. "four plus five", "4+5", "add four to five" all compress to the same discrete code during training — not hand-engineered, emergent.

Load-bearing for sleep: Stitch needs discrete symbolic handles to find structural recurrences. VQ codes provide exactly this. The codebook is also the shared vocabulary across all domain clusters, making cross-cluster skill transfer automatic.

### 11.7 Startup Sequence

```
1. Load model checkpoint (base weights + VQ codebook)
2. Load domain cluster index (centroids → RAM)
3. Load FAISS skill index (CSR) from last snapshot
4. Load skill metadata from SQLite → RAM cache
5. Identify active domain → load that cluster's LoRA adapter
6. Fill hot episodic buffer from domain HDF5 (recent N episodes)
7. Ready — inference never touches disk until sleep or domain switch
```

Target: under 10 seconds for ~20 domain clusters, ~10k skills.

---

## 12. Architecture Backbone: Transformer vs. Mamba

### 12.1 Mathematical Comparison

```
Transformer:  output_t = Attention(Q_t, K_{1..t}, V_{1..t})    O(t²) per step
Mamba:        h_t = A(x_t) × h_{t-1} + B(x_t) × x_t           O(1) per step
              output_t = C(x_t) × h_t
```

A, B, C are input-dependent (selective) — what separates Mamba from earlier SSMs. This is not a decision to make lightly mid-project. The table below reflects where the inductive bias genuinely differs.

### 12.2 Component-by-Component Decision

| Component | Current | Recommendation | Rationale |
|---|---|---|---|
| HJEPA encoder | Transformer | **Keep** | Long-range patch dependencies need attention |
| WorldModel (low-level) | Transformer | **Migrate to Mamba** | Pure state-space rollout — Mamba's home turf |
| HWM high-level planner | (new) | **Mamba** | Long-horizon macro-action sequences |
| Configurator | Transformer | **Evaluate Mamba** | Large episodic context windows benefit from O(n) |
| Actor | Transformer | **Keep** | Candidate proposal needs global context visibility |
| MemoryRelevanceScorer | Transformer | **Keep** | Retrieval is attention-native |
| Talker | (new) | **Transformer** | Natural language generation is attention-native |

The WorldModel migration is highest-value, lowest-risk: self-contained module, clear interface (`z, actions → trajectory`), direct measurable benefit on long rollouts, nothing else in the pipeline changes.

### 12.3 Migration Approach

Implement Mamba WorldModel in parallel with existing Transformer WorldModel. Train both on same data. Compare rollout accuracy, training stability, and latency before committing. Use Mamba-2 (addresses Mamba-1 training instabilities). Existing Transformer WorldModel stays as reference and fallback.

---

## 13. Three-Level Capability Hierarchy

The literature identifies a natural scaling structure for how capabilities build on each other. This maps directly onto the implementation phases:

| Level | Capability | Scaling Property | Implementation |
|---|---|---|---|
| I — Perceptual | Visual/signal information aggregation | Linear with sensor density | HJEPA encoder (exists) |
| II — Dynamic | Temporal modelling via latent actions | Quadratic with history length | HWM + Mamba WorldModel |
| III — Reasoning | Decoupled multimodal logic | **Logarithmic with library size** (FAISS) | Thinker + MCTS + Skill library |

The logarithmic scaling at Level III is the key insight from the literature. Once you have a well-populated skill library, adding more skills costs log(n) in retrieval time. The system gets smarter without getting slower. This is why the skill library architecture matters so much — it's the mechanism that keeps reasoning cost bounded as the system learns more.

---

## 14. Design Decisions — Resolved

> **Core design principle:** Replace binary gates with continuous weighting by signal quality. Every place you're tempted to put a threshold, ask whether a weight works instead. Growth is a gradient, not a switch.

---

**Q4: LoRA adapter size** ✅ RESOLVED  
**Rank 8, attention projection matrices only (Q, K, V, O).**

Rank 8 is the right starting point — aggressive enough to be efficient, permissive enough to work before episodic memory is populated. Rank 4 is too aggressive during early training when the episodic buffer is sparse. Rank 16 is likely redundant once memory is rich. The adapter applies to attention projections only — this is where domain-specific routing happens. FFN layers encode general knowledge and must not be overwritten by domain adaptation. As memory richness grows, adapter rank can be empirically pruned downward. Testing approach: train to peak domain performance, snapshot, progressively reduce rank while adding memory, measure degradation. Expected result: effective rank drops toward 4 as memory fills.

---

**Q7: LILO AutoDoc integration** ✅ RESOLVED  
**Lazy, post-sleep, low-priority background thread. Local Gemma 2B.**

AutoDoc has zero effect on runtime behaviour — it names skills for human interpretability only. Implementation: maintain a `skills_pending_documentation: List[skill_id]` queue. After each sleep cycle, newly promoted skills are added to the queue. A low-priority background thread processes the queue whenever the system is idle. Results write to `skills.name` and `skill_constraints.description` in SQLite. Never blocks inference or sleep. Model: Gemma 2B running locally — no API dependency, no per-call cost, works offline. One-shot prompt per skill: execution trace in, name + docstring out.

---

**Q8: RSD adversarial stability** ✅ RESOLVED  
**Weighted effective regret — no hard boundary on the proposal space.**

Hard thresholds on proposal distance are brittle and suppress growth. The generator proposes freely. The regret signal is weighted by estimated probability of mastery:

```
effective_regret(s, a) = regret(s, a) × P(mastery | current_competence, skill_distance)

P(mastery) estimated from:
  - distance to nearest mastered skill in FAISS index
  - historical mastery rate on skills at similar distances
  - rate of improvement on current frontier skills
```

Skills far beyond the frontier still get proposed, but their effective regret contribution approaches zero because P(mastery) is near zero — the generator naturally steers toward the learnable frontier without being told where it is. Skills near the frontier produce strong effective regret and drive learning. The generator maximises effective regret, not raw regret. It discovers the frontier from the signal, not from a geometric constraint. As competence grows, the effective frontier expands automatically.

---

**Q9: z_goal estimation** ✅ RESOLVED  
**InSite: trivially known (target glucose range). General case: MCTS bootstrap.**

For InSite the goal state is always pre-specified — target physiological range is known before inference. `z_goal` is the HJEPA encoding of that target state.

For general open-ended reasoning: use the MCTS tree as the goal estimator. First MCTS iteration runs without skill retrieval — pure Actor proposal, standard WorldModel rollout. Terminal latent states of the most promising branches become candidate `z_goal` estimates. Second iteration onwards uses those terminal states for `Δz` skill retrieval. The tree search and skill retrieval are mutually reinforcing — the search produces goal estimates, goal estimates enable skill shortcuts, shortcuts make the search faster. No separate goal-encoding module needed.

---

**Q10: Dream Decompiling signal quality** ✅ RESOLVED  
**Weighted by Actor consistency — always running, influence proportional to signal quality.**

Dream Decompiling runs from the first sleep cycle. It is never gated. Its candidates are weighted by Actor consistency before contributing to skill formation:

```
skill_candidate_weight = f(Actor consistency at similar states)

Actor consistency measured by:
  entropy of Actor's child proposals across similar parent latent states
  → high entropy (inconsistent search) = low weight
  → low entropy (consistent, structured search) = high weight
```

Early in training, Actor entropy is high — Dream Decompiling candidates get low weight and don't dominate skill formation. As the Actor develops coherent search representations, entropy drops, weights rise, and Dream Decompiling's contribution increases smoothly. No wasted episodes, no arbitrary gate. The path from noisy to informative is continuous and self-regulating. LOVE + Stitch dominate early; Dream Decompiling grows in influence as the Actor matures.

---

**Q11: BODE-GEN search stability** ✅ RESOLVED  
**Lipschitz regularisation on the Latent Action Encoder.**

Gaussian process acquisition landscape stability is inherited from encoder smoothness. If the Latent Action Encoder is Lipschitz-smooth, similar action sequences produce similar latent macro-actions, and nearby points in skill space produce similar behaviours. Discontinuities in the acquisition landscape would indicate encoder failure, not a BODE-GEN limitation.

Implementation: add a Lipschitz regularisation term to the Latent Action Encoder training objective. Spectral normalisation on the encoder's weight matrices is the standard efficient approach — constrains the Lipschitz constant without explicit penalty computation. Validate on simple toy skill generation (can BODE-GEN find a skill that moves from state A to state B?) before deploying on frontier skill search.

---



## 15. Relationship to Existing Chamelia Architecture

### 15.1 What Already Exists (ChameliaV2)

| Concept | Implementation |
|---|---|
| Latent state encoding | `HJEPA` — hierarchical joint-embedding |
| Episodic memory (flat) | `LatentMemory` + `EpisodeRecord` |
| Similarity retrieval | `retrieve_scored()` — cosine + quality reranking |
| Postures (soft skill analog) | `Actor.posture_seeds` |
| Memory-biased reasoning | `_memory_posture_bias()`, `_memory_refinement_bias()` |
| Low-level planner | `WorldModel` rollout |
| Iterative refinement | `Actor.refine()` — seed for MCTS |
| Retrieval reranking | `MemoryRelevanceScorer` |
| Thinker | Entire existing forward pipeline |

### 15.2 What's Missing

| Concept | Status |
|---|---|
| Procedural memory store + CSR FAISS index | ❌ |
| Latent Action Encoder (rank 8, Q/K/V/O projections) | ❌ New module |
| HWM high-level planner | ❌ Existing WorldModel = low-level |
| LOVE decomposition | ❌ Use existing library |
| Stitch compression | ❌ Use existing library |
| Dream Decompiling (Actor-consistency-weighted) | ❌ |
| BODE-GEN latent skill search (Lipschitz encoder) | ❌ |
| RSD adversarial discovery (effective regret weighting) | ❌ |
| Choreographer imagination-based evaluation | ❌ |
| Sleep phase coordinator | ❌ |
| MCTS replacing flat reasoning loop | ❌ |
| Talker module | ❌ Thinker exists; Talker missing |
| Domain cluster index + LoRA adapters (rank 8) | ❌ |
| VQ layer on encoder | ❌ Architectural addition |
| IOB episodic embeddings | ❌ Upgrade to existing retrieval |
| CSR skill embeddings | ❌ New embedding type |
| AutoDoc queue + Gemma 2B worker | ❌ |

### 15.3 Migration Path

Nothing is replaced — everything is extended upward and outward. Existing `WorldModel` becomes HWM low-level planner with no internal changes. Existing flat reasoning loop becomes the seed for MCTS. Existing `LatentMemory` becomes the episodic layer. New `ProceduralMemory` sits alongside it. Talker is a new downstream module that doesn't touch anything existing.

---

## 16. Next Steps

**Phase 1 — Foundation**
1. Storage scaffold — SQLite schema, HDF5 layout per domain, startup loader
2. CSR/IOB FAISS skill index alongside existing episodic search; serialize/deserialize after sleep
3. `ProceduralMemory` class — skill store backed by SQLite + FAISS; confidence tracking; constraint annotations; `deprecated_by` pointers
4. Latent Action Encoder — small transformer encoding primitive action sequences → fixed-dim latent macro-action vector
5. Add MCTS traces to `EpisodeRecord` — full tree snapshots stored in HDF5 for sleep mining

**Phase 2 — Thinker / Talker**
6. Formalise Thinker boundary — document exact interface; ensure nothing downstream feeds back
7. Talker module — lightweight decoder: frozen latent reasoning chain → natural language; invoked on demand only

**Phase 3 — MCTS**
8. `MCTSSearch` class — UCT selection, Actor as policy head, WorldModel as simulation engine, safety gate mid-playout
9. `CostModule` tree averaging — extend `TrainableCritic` to track V(s,a) and N(s,a) per tree node
10. Tree reuse — warm-start from previous call when patient state is stable

**Phase 4 — Sleep pipeline**
11. Sleep phase coordinator — background thread, opportunistic + periodic triggers, temp HDF5 workspace
12. LOVE integration — partition MCTS episode traces into skill-shaped segments
13. Stitch integration — compress LOVE segments into λ-abstractions using existing Stitch library
14. Dream Decompiling — mine Actor's search policy for skill candidates from unsolved episodes; runs in parallel with LOVE/Stitch
15. Latent Action Encoder training — train on (Stitch abstraction + Dream Decompiling candidate → latent macro-action) pairs
16. BODE-GEN latent search — Bayesian optimisation over skill embedding space for novel frontier skill generation
17. Choreographer evaluation loop — imagination-based skill evaluation; only index skills that survive HWM simulation testing
18. RSD adversarial discovery — min-max skill generator vs agent policy; uses Choreographer's imagination loop as evaluation oracle
19. LILO AutoDoc — optional natural language naming of discovered skills; offline, never on hot path

**Phase 5 — HWM and domain clustering**
17. HWM high-level planner — new module above existing WorldModel; operates on latent macro-actions
18. Domain cluster index — Dirichlet process discovery; persist in SQLite
19. LoRA adapter per cluster — snapshot weight diffs at peak domain performance
20. Empirical adapter size study — target rank 4–16

**Phase 6 — Representation upgrades**
22. VQ layer evaluation — VQ-VAE or product quantization on HJEPA encoder; prerequisite for Stitch running directly on symbolic codes; codebook as shared semantic dictionary
23. IOB episodic embeddings — replace current flat cosine retrieval with IOB-ordered embeddings for coarse-to-fine episode search
24. CSR skill embeddings — fine-tune skill embeddings for sparse high-dimensional retrieval; enables sub-ms FAISS matching at 100k+ skill scale
25. Isotropic super-resolution — 32-token compressed skill storage; full reconstruction only at execution time
26. Mamba WorldModel prototype — parallel implementation; evaluate against Transformer baseline on rollout accuracy, stability, latency
27. LanceDB assessment — production replacement for FAISS + HDF5 at scale

---

*This document is a living spec. Update it as the architecture evolves. The goal is to think clearly before building, not to be right on the first draft.*

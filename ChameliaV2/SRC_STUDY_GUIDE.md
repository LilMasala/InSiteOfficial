# ChameliaV2 `src` Study Guide

This guide is for reading the codebase as a system, not as isolated files.

The short version:

- `src/models` learns the latent representation (`HJEPA`)
- `src/chamelia` turns that latent representation into planning, memory, and action
- `src/chamelia/cognitive` adds the higher-level architecture: MCTS, procedural memory, sleep, VQ/IOB/CSR, Mamba, LanceDB, clustering
- `src/losses` and `src/trainers` are the training side for the representation model

If you keep one mental model in your head, use this:

1. Observe the world into latent space
2. Retrieve relevant past episodes and skills
3. Build context
4. Propose candidate action paths
5. Imagine futures
6. Score futures
7. Select an action
8. Store the episode
9. Later, during sleep, compress episodes into reusable skills

## 1. Best Reading Order

Read these in order if you want the fastest path to understanding:

1. `src/chamelia/chamelia.py`
2. `src/chamelia/hjepa_adapter.py`
3. `src/models/hjepa.py`
4. `src/chamelia/configurator.py`
5. `src/chamelia/actor.py`
6. `src/chamelia/world_model.py`
7. `src/chamelia/cost.py`
8. `src/chamelia/memory.py`
9. `src/chamelia/cognitive/planning.py`
10. `src/chamelia/cognitive/procedural.py`
11. `src/chamelia/cognitive/sleep.py`
12. `src/chamelia/cognitive/representation.py`
13. `src/chamelia/cognitive/storage.py`
14. `src/trainers/trainer.py`
15. `src/losses/hjepa_loss.py` and `src/losses/combined.py`

If you only want the smallest useful set, read:

1. `src/chamelia/chamelia.py`
2. `src/chamelia/actor.py`
3. `src/chamelia/world_model.py`
4. `src/chamelia/cost.py`
5. `src/chamelia/memory.py`
6. `src/chamelia/cognitive/planning.py`
7. `src/chamelia/cognitive/procedural.py`
8. `src/chamelia/cognitive/sleep.py`

## 2. The Main Runtime Story

The top-level system is assembled in `src/chamelia/chamelia.py`.

`Chamelia.forward(...)` is the best single place to understand the runtime.

At a high level, it does this:

1. Run HJEPA to get latent features
2. Extract the scene summary latent `z`
3. Retrieve episodic memory from `LatentMemory`
4. Rerank that memory with the learned retrieval scorer
5. Build context tokens with the `Configurator`
6. Either:
   - run the flat actor/world-model/cost loop, or
   - run the MCTS planner
7. Decode the chosen action through the domain plugin
8. Store an `EpisodeRecord`
9. Later, `fill_outcome(...)` patches in the realized outcome

Important methods in `src/chamelia/chamelia.py`:

- `forward(...)`: full inference pipeline
- `_run_system1_skill(...)`: quick procedural-memory path
- `_run_planner_sample(...)`: one MCTS-backed planning sample
- `fill_outcome(...)`: attaches delayed outcomes to stored episodes
- `train_critic_from_memory(...)`: critic supervision from completed episodes
- `train_world_model_from_memory(...)`: rollout supervision from memory

## 3. Representation Layer

### `src/models/hjepa.py`

This is the latent representation backbone.

What it does:

- encodes context tokens
- encodes target tokens
- predicts masked target features from context
- builds hierarchical outputs
- optionally applies a shared VQ bottleneck to encoder features

Why it matters:

- Chamelia assumes everything downstream works in latent space
- the quality of `z`, context features, and target features determines the whole stack

Key ideas:

- `HJEPA` is the core model
- `_TokenVectorQuantizer` is the token-level shared VQ bottleneck
- the model now emits `context_codes`, `target_codes`, and `vq_commitment_loss`

### `src/chamelia/hjepa_adapter.py`

This file is the bridge between Chamelia and HJEPA.

It lets Chamelia run HJEPA on:

- images
- already embedded token sequences

This file is important because Chamelia domains often do not look like ordinary image classification pipelines.

## 4. Context Builder

### `src/chamelia/configurator.py`

The `Configurator` turns:

- hierarchical latent features from HJEPA
- retrieved episodic memory summaries

into:

- context tokens `ctx_tokens`

Those context tokens are what the actor, world model, and critic condition on.

Mental model:

- HJEPA says what is present
- memory says what has happened before
- Configurator turns both into the working context for planning

## 5. Action Proposal

### `src/chamelia/actor.py`

The `Actor` proposes candidate action paths.

Two key phases:

- `propose(...)`: initial candidate generation
- `refine(...)`: iterative improvement after imagined rollouts are scored

Important details:

- candidate 0 is kept as a simple baseline
- postures are soft skill-like latent steering variables
- memory can bias both posture initialization and refinement
- `forward(...)` is a convenience wrapper; the richer interface is `propose(...)`

When studying this file, focus on:

- `_memory_posture_bias(...)`
- `_memory_refinement_bias(...)`
- `propose(...)`
- `refine(...)`

## 6. Imagination and Scoring

### `src/chamelia/world_model.py`

The `ActionConditionedWorldModel` rolls imagined latent futures forward from:

- current latent state `z`
- candidate action paths
- context tokens
- optional postures / reasoning states

Outputs:

- latent trajectory
- terminal latent
- summary token

### `src/chamelia/cost.py`

The `CostModule` scores imagined futures.

It combines:

- `IntrinsicCost`: domain-defined immediate cost
- `TrainableCritic`: learned future cost

The important interface is `score_candidates(...)`, which turns imagined futures into:

- `ic`
- `tc`
- `total`

This is the point where candidate paths become comparable.

## 7. Episodic Memory

### `src/chamelia/memory.py`

This is the episodic memory store.

The central record is `EpisodeRecord`.

It now stores much more than a flat `(state, action)` pair:

- selected path
- candidate paths
- candidate costs
- selected posture
- retrieval trace
- MCTS trace
- skill trace
- goal latent
- domain cluster id

The main class is `LatentMemory`.

Important behavior:

- circular-buffer storage
- delayed outcome filling via `fill_outcome(...)`
- retrieval through `retrieve(...)` / `retrieve_scored(...)`
- optional IOB-based shortlist generation

Important clarification:

- the retrieval path is not “just flat cosine” anymore when IOB is configured
- the seam is still the same: `retrieve_scored(...)`
- that is where coarse-to-fine episode search lives

## 8. Retrieval Reranking

### `src/chamelia/retrieval.py`

This file adds a learned reranker on top of the raw episodic retrieval shortlist.

`MemoryRelevanceScorer` uses:

- key similarity
- summary similarity
- quality
- posture similarity
- learned interactions

This module matters because memory selection is not just nearest-neighbor lookup anymore.

## 9. Thinker, Talker, and MCTS

### `src/chamelia/cognitive/planning.py`

This file contains the deliberate-planning layer.

Core pieces:

- `FrozenReasoningChain`: read-only latent reasoning trace
- `ThinkerOutput`: frozen planner output for downstream use
- `Talker`: decodes frozen latent reasoning into language tokens
- `HighLevelPlanner`: maps retrieved skills into macro subgoals
- `MCTSSearch`: tree search over latent futures

How to think about it:

- Thinker stays in latent space
- Talker only reads Thinker output
- MCTS replaces the old flat reasoning loop when enabled

Important detail:

- `MCTSNode` stores the tree state, including `visit_count` and total/mean cost
- this is where the tree-level `V/N` bookkeeping lives

## 10. Procedural Memory

### `src/chamelia/cognitive/procedural.py`

This is the skill library.

The key abstraction is `SkillRecord`.

A skill contains:

- latent embedding
- retrieval vector
- action path
- confidence
- source episodes
- constraints
- optional symbolic program
- optional compressed codes

`ProceduralMemory` handles:

- adding skills
- loading skills
- retrieval
- storage/index backends

Important details:

- CSR can be used for sparse retrieval vectors
- Isotropic skill compression is now the runtime storage format when a codec is configured
- LanceDB is now a live backend, not just an assessment stub

## 11. Sleep Pipeline

### `src/chamelia/cognitive/sleep.py`

This is the biggest “architecture” file in the repo.

It is where episodes get reorganized into reusable skills.

Important pieces, in rough order:

- `LOVEDecomposer`: partitions traces using a compression-style objective
- `StitchCompressor`: compresses discrete segments into abstractions
- `BODEGenOptimizer`: Bayesian optimization over latent prompts
- `DreamDecompiler`: extracts chunks from unresolved search frontiers
- `ChoreographerEvaluator`: imagination-based candidate evaluation
- `RSDAdversary`: adversarial frontier skill discovery
- `GemmaAutoDocWorker` / `LILOAutoDoc`: optional naming and documentation
- `SleepCoordinator`: orchestrates the whole process

The sleep cycle is roughly:

1. gather episodes
2. decompose solved traces with LOVE
3. compress with Stitch
4. mine unresolved frontiers with DreamDecompiler
5. generate hard frontier skills with RSD
6. refine frontier skills with BODE-GEN in prompt space
7. evaluate candidates with the Choreographer
8. promote surviving skills into procedural memory
9. archive episodes

This file is worth reading twice:

- first for the high-level pipeline
- second for the exact data structures that move between the stages

## 12. Representation Upgrades

### `src/chamelia/cognitive/representation.py`

This file contains the phase-6 representation utilities:

- `VectorQuantizer`
- `InformationOrderedBottleneck`
- `ContrastiveSparseRepresentation`
- `IsotropicSkillCodec`

How they fit:

- VQ gives discrete symbolic handles
- IOB gives coarse-to-fine episodic search
- CSR gives sparse skill retrieval vectors
- Isotropic codec gives compressed skill storage

## 13. Storage and Backends

### `src/chamelia/cognitive/storage.py`

This file is the persistence layer.

It handles:

- SQLite metadata
- LanceDB archive and vector storage glue
- serialization of tensors and codes

If you want to understand what is persisted versus what stays purely in-memory, read this right after `procedural.py` and `sleep.py`.

### `src/chamelia/cognitive/mamba_world_model.py`

This is the alternative world-model backend.

It mirrors the transformer world model at the interface level so the rest of Chamelia can swap between:

- transformer rollout model
- Mamba rollout model

### `src/chamelia/cognitive/lancedb_assessment.py`

This benchmarks or compares vector-storage backends.

It is useful for systems understanding, but not the first file to read.

## 14. Domain Layer

### `src/chamelia/plugins`

This is how Chamelia becomes domain-specific.

`base.py` defines the domain interface.

`insite_t1d.py` is the current concrete domain plugin.

The domain decides things like:

- tokenization
- action decoding
- intrinsic cost pieces
- domain state shape

## 15. Tokenizers

### `src/chamelia/tokenizers`

These files convert domain observations into token sequences.

The important point is not which tokenizer exists, but that Chamelia expects domain observations to become a sequence that HJEPA can process.

## 16. Training Stack

### `src/trainers/trainer.py`

This is the HJEPA training loop.

The core training step does:

1. build masks
2. run the model
3. extract predictions and targets
4. pass `context_features` into the loss for VICReg-style regularization
5. pass `vq_commitment_loss` into the loss when VQ is enabled
6. update the target encoder with EMA

### `src/losses/hjepa_loss.py`

This is the base JEPA loss.

It now also supports the optional VQ commitment term through `vq_commitment_loss`.

### `src/losses/combined.py`

This composes:

- JEPA prediction loss
- VICReg regularization
- optional VQ contribution

If you care about “how does the representation actually learn?”, read `trainer.py`, then `hjepa_loss.py`, then `combined.py`.

## 17. How One Episode Moves Through the System

Here is the end-to-end mental trace for one inference:

1. Domain observation is tokenized
2. `forward_hjepa(...)` runs HJEPA
3. `Chamelia` extracts the scene summary latent `z`
4. `LatentMemory` retrieves episodic neighbors
5. retrieval is reranked by `MemoryRelevanceScorer`
6. `Configurator` builds `ctx_tokens`
7. planning happens:
   - flat actor/refine loop, or
   - MCTS over actor proposals and world-model rollouts
8. `CostModule` scores imagined futures
9. Chosen action is decoded by the domain
10. an `EpisodeRecord` is stored
11. when the real outcome arrives, `fill_outcome(...)` patches in the realized future
12. later, sleep mines those stored episodes into procedural skills

## 18. What To Ignore On First Pass

If you are getting overwhelmed, ignore these until the second pass:

- `src/data`
- `src/evaluation`
- `src/visualization`
- `src/serving`
- `src/inference`

They matter, but they are not the core architecture.

## 19. Best Tests To Read As Executable Documentation

These are the highest-signal tests for understanding how things fit:

- `tests/test_chamelia.py`
- `tests/test_cognitive_architecture.py`
- `tests/test_phase123_optimizations.py`
- `tests/test_losses.py`
- `tests/test_combined_losses.py`
- `tests/test_trainers.py`

Suggested test reading order:

1. `tests/test_chamelia.py`
2. `tests/test_cognitive_architecture.py`
3. `tests/test_losses.py`
4. `tests/test_combined_losses.py`
5. `tests/test_trainers.py`

## 20. A Good 3-Pass Study Strategy

### Pass 1: Runtime Skeleton

Goal: understand the inference path.

Read:

1. `src/chamelia/chamelia.py`
2. `src/chamelia/configurator.py`
3. `src/chamelia/actor.py`
4. `src/chamelia/world_model.py`
5. `src/chamelia/cost.py`
6. `src/chamelia/memory.py`

Question to answer:

- How does a latent observation become a chosen action?

### Pass 2: Deliberation and Memory Growth

Goal: understand how the system becomes more capable over time.

Read:

1. `src/chamelia/cognitive/planning.py`
2. `src/chamelia/cognitive/procedural.py`
3. `src/chamelia/cognitive/sleep.py`
4. `src/chamelia/cognitive/storage.py`
5. `src/chamelia/cognitive/representation.py`

Question to answer:

- How do episodes turn into skills?

### Pass 3: Learning the Representation

Goal: understand how the latent space is trained.

Read:

1. `src/models/hjepa.py`
2. `src/chamelia/hjepa_adapter.py`
3. `src/trainers/trainer.py`
4. `src/losses/hjepa_loss.py`
5. `src/losses/combined.py`

Question to answer:

- Why should the latent space be good enough for planning and memory?

## 21. Fast Glossary

- `z`: scene summary latent
- `ctx_tokens`: working context tokens from the configurator
- `candidate_paths`: proposed action sequences
- `candidate_postures`: latent steering variables for candidates
- `reasoning_states`: actor-side internal candidate reasoning states
- `ic`: intrinsic cost
- `tc`: trainable future cost
- `EpisodeRecord`: one episodic memory item
- `SkillRecord`: one procedural memory item
- `ThinkerOutput`: frozen planner output for downstream reading
- `SleepCoordinator`: offline consolidation pipeline

## 22. If You Want One File To Annotate By Hand

Use `src/chamelia/chamelia.py`.

If you can annotate:

- where `z` is formed
- where memory is read
- where context is built
- where planning happens
- where the action is selected
- where memory is written

then the whole repository starts to make sense much faster.

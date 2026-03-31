# Chamelia Architecture Scorecard

## Premise

This scorecard is written against the intended architecture, not the current drifted implementation.

Intended role of Chamelia:

- `Perception` infers belief from available evidence.
- `WorldModule` rolls belief forward under candidate actions.
- `Cost` defines badness in domain terms and estimates downstream badness.
- `Actor` searches for the least-bad / best action under those futures.
- `Memory` stores what happened and what kinds of reasoning worked.
- `Configurator` is not a knob tuner. It is the metacognitive layer that changes how the rest of the system reasons.

Important clarifications:

- JEPA is supposed to be the main world-model substrate, not just one estimator among many.
- Chamelia should adapt and reshape how JEPA reasons, not default to living as a menu of estimator swaps.
- Domain-specific semantics, especially what counts as "bad", should be defined by the plugin/domain layer, not hardcoded into Chamelia core.
- Chamelia core is supposed to be the reasoning engine. InSite is supposed to supply domain meaning, action semantics, and control affordances.
- The configurator is supposed to reason about reasoning. It should change how other modules think, not merely tune scalar settings.

## Executive Summary

Current grades:

- `Perception`: `B`
- `WorldModule`: `B`
- `Cost`: `C`
- `Actor`: `B+`
- `Memory`: `B-`
- `Configurator`: `D`
- `Core/Plugin Boundary`: `D`

Top-line assessment:

- The system already has a real reasoning skeleton.
- The strongest modules are `Actor`, `Perception`, and `WorldModule`.
- The weakest module is `Configurator`, because it does not yet function as a metacognitive controller.
- The second major problem is architectural leakage: too much InSite/domain-specific logic still lives in Chamelia core.
- The main path forward is not to replace JEPA or throw away the rollout architecture. It is to restore the intended role of the `Configurator`, and to push domain semantics back out of core.

## Non-Negotiable Core Principles

These are the architectural guardrails for future work. If a change violates one of these, it should be treated as architectural regression unless there is a very strong reason otherwise.

### 1. Chamelia is a reasoning engine, not a domain app

- Chamelia core should not become an InSite-specific execution engine with abstract wrappers.
- Core owns reasoning structure.
- Plugins own domain meaning.

### 2. JEPA is central

- The architecture should not drift toward "pick from a menu of estimators" as the main idea.
- JEPA should remain the primary adaptive latent substrate.
- The configurator should mainly control how JEPA is used, shaped, interpreted, and adapted.

### 3. The configurator must control cognition, not just knobs

- Changing scalar configuration values is not sufficient.
- The configurator must be able to influence how each major module reasons.
- If the configurator cannot materially alter the reasoning mode of other modules, it is too weak.

### 4. Domain semantics of badness must live outside core

- Chamelia core should not define domain-shaped harms like burnout cost, trust cost, or other product-specific badness formulas as universal truth.
- Core may define generic mechanisms for explicit cost, critic value, and constraints.
- Plugins define what those costs mean.

### 5. Actions are domain-defined, search is core-defined

- Chamelia core should know how to search over candidate actions.
- Plugins should define what actions exist and what they mean.
- Core should not be rewritten every time a domain changes its intervention vocabulary.

### 6. Memory must store ways of thinking, not just outcomes

- It is not enough to store action and realized outcome.
- Memory should preserve the internal reasoning context:
  - what belief framing was active
  - what controller mode was active
  - what prediction horizon mattered
  - what reasoning strategy succeeded or failed

### 7. Hard holds are not a substitute for intelligence

- If uncertainty rises, the correct response is not always silence.
- Chamelia should be able to shift into degraded, stabilizing, or probe modes rather than simply refusing to act.
- The system should route around reasoning failure, not just stop.

### 8. Train general cognition before domain specialization

- Pretraining should focus on latent inference, future simulation, memory-guided reasoning, and metacognitive control.
- Domain-specific semantics should come later through plugins and finetuning.
- The project should not collapse into task-specific policy learning too early.

### 9. Preserve the short-term / long-term tradeoff architecture

- A locally worse move can still be globally right.
- The explicit cost plus downstream critic architecture is important and should be preserved.
- The system must be allowed to take actions that incur short-term cost if they improve future constrained outcomes.

### 10. Refactors should increase generality, not reduce it

- If a convenience change makes Chamelia more InSite-shaped, it should be viewed skeptically.
- If a refactor increases plugin expressivity and core generality, it is probably directionally correct.

## Module Scorecard

### Perception

Grade: `B`

What it gets right:

- `Perception` really does maintain an explicit belief abstraction in [Perception.jl](/Users/anandparikh/Desktop/InSiteOfficial/Chamelia/src/Perception/Perception.jl).
- There is a consistent predict/update interface in [belief_update.jl](/Users/anandparikh/Desktop/InSiteOfficial/Chamelia/src/Perception/belief_update.jl).
- The JEPA path is real, not decorative.
- Anomaly detection and belief widening exist and are conceptually aligned with reasoning under uncertainty.

What is off:

- The current implementation still treats belief estimators too much like interchangeable implementations rather than configurable expressions of one reasoning substrate.
- Epistemic quality is still dominated by entropy-relative heuristics in [epistemic.jl](/Users/anandparikh/Desktop/InSiteOfficial/Chamelia/src/Perception/epistemic.jl), which is too weak for the architecture.
- The configurator has almost no deep authority over how perception reasons.

What needs to change:

- Treat JEPA as the primary adaptive latent model.
- Let the configurator reshape JEPA behavior: observation windows, latent objectives, uncertainty interpretation, anomaly response, signal weighting, memory-conditioned inference, and online adaptation policy.
- Stop centering the architecture around estimator choice as the main control surface.
- If alternate estimators exist at all, they should be secondary tools, not the center of the architecture.

### WorldModule

Grade: `B`

What it gets right:

- The rollout architecture is fundamentally correct in [rollout.jl](/Users/anandparikh/Desktop/InSiteOfficial/Chamelia/src/WorldModule/rollout.jl).
- It really does simulate futures from current belief plus action.
- It carries forward both physiology and psychology.
- It already incorporates contextual signals like stress and sleep into future evolution.

What is off:

- The module is still closer to a rollout executor than a fully adaptive world-model ecology.
- Too much of the practical behavior depends on fixed assumptions from outside rather than on a controller that can reshape the simulation policy.
- Some domain semantics are still too close to core.

What needs to change:

- Keep the rollout-based structure.
- Expose more control surfaces to the configurator: horizon policy, branch policy, regime-conditioned rollout policy, objective-conditioned prediction policy, and online model adjustment policy.
- Preserve JEPA as the main learned substrate, with the configurator changing how it is used rather than bypassing it.

### Cost

Grade: `C`

What it gets right:

- The decomposition into explicit rollout cost plus critic value is directionally correct in [energy.jl](/Users/anandparikh/Desktop/InSiteOfficial/Chamelia/src/Cost/energy.jl).
- The actor can already justify a short-term intrinsic worsening if long-run critic value improves.
- The critic is trained on realized outcomes, which is the right anti-self-delusion design.

What is off:

- Too much normative semantics are still hardcoded into Chamelia core.
- `compute_burden_cost`, `compute_trust_cost`, `compute_burnout_cost`, and `compute_intrinsic_cost` live in [types.jl](/Users/anandparikh/Desktop/InSiteOfficial/Chamelia/src/types.jl), which is architecturally wrong for a general reasoning engine.
- Chamelia should not define "burnout cost" or equivalent domain-shaped harms as native truth.

What needs to change:

- Move "what is bad" out of core and into plugin/domain definitions.
- Chamelia core should support abstract cost channels, constraint channels, and critic composition.
- The plugin should specify domain semantics for harms, priorities, and outcome shaping.
- Core should retain the generic mechanism for balancing explicit cost and downstream critic value, but not domain-specific badness definitions.
- The key generic idea to preserve is not specific formulas, but the architecture:
  - explicit near-horizon badness
  - downstream critic-estimated future badness
  - constrained tradeoff between them

### Actor

Grade: `B+`

What it gets right:

- `Actor` is more faithful to the intended idea than most of the rest of the system.
- It compares candidate actions against a null baseline in [Actor.jl](/Users/anandparikh/Desktop/InSiteOfficial/Chamelia/src/Actor/Actor.jl).
- It does constrained search rather than naive ranking.
- It chooses via CVaR over future rollout energies, not just one-step reward.
- It already supports the critical architecture idea that a locally unpleasant move can be globally correct if it improves the future.

What is off:

- The actor is still limited by the action ontology exposed to it.
- It is too vulnerable to upstream hard gating.
- Some action structures remain too InSite-shaped in core.

What needs to change:

- Preserve the actor structure.
- Make action families more plugin-owned.
- Give the configurator richer authority over actor mode: stabilize, recover, maintain, probe, exploit, robust-search, etc.
- Ensure the actor can optimize over plugin-defined actions without core rewrite.

### Memory

Grade: `B-`

What it gets right:

- `Memory` is structurally solid in [buffer.jl](/Users/anandparikh/Desktop/InSiteOfficial/Chamelia/src/Memory/buffer.jl).
- It stores beliefs, actions, outcomes, predicted CVaR, and latent snapshots.
- It feeds critic learning and twin posterior updates from realized experience.

What is off:

- Memory mostly stores action/outcome traces, not enough about internal reasoning configuration.
- It only weakly stores "how the system was thinking" when a decision was made.
- It does not yet richly support retrieval of successful reasoning strategies by regime or failure mode.

What needs to change:

- Store controller mode, perception mode, world-model mode, and objective mode per record.
- Add regime-conditioned memory summaries.
- Track which internal reasoning strategy worked, not just whether the final action worked.
- Let memory serve metacognition, not just retrospective scoring.
- The long-term goal is not just "experience replay", but a memory system that can help the configurator decide how to think.

### Configurator

Grade: `D`

This is the main architectural miss.

What it currently does:

- Adjusts scalar parameters like `Δ_max`, `N_roll`, `H_med`, `α_cvar`, some anomaly sensitivity, and some weight emphasis in [Configurator.jl](/Users/anandparikh/Desktop/InSiteOfficial/Chamelia/src/Configurator/Configurator.jl), [rules.jl](/Users/anandparikh/Desktop/InSiteOfficial/Chamelia/src/Configurator/rules.jl), [learned.jl](/Users/anandparikh/Desktop/InSiteOfficial/Chamelia/src/Configurator/learned.jl), and [meta_policy.jl](/Users/anandparikh/Desktop/InSiteOfficial/Chamelia/src/Configurator/meta_policy.jl).

Why that is insufficient:

- That behavior is closer to hyperparameter adaptation than metacognition.
- It does not deeply control how perception reasons.
- It does not reconfigure the world model in a meaningful way.
- It does not choose reasoning modes.
- It does not orchestrate memory retrieval policy.
- It does not reshape the actor's decision mode beyond scalar constraints.

What the configurator is supposed to be:

- The reasoning-about-reasoning layer.
- The module that says:
  - perception is not working for the task; change how you infer belief
  - world model is thinking too shallowly; project further or differently
  - actor is optimizing the wrong thing for this regime; switch mode
  - cost emphasis is wrong for the task; reweight the evaluation logic
  - memory should rely more on longer-horizon or regime-similar traces
  - JEPA's current latent framing is inadequate; adapt how the latent is formed or used

What needs to change:

- Rebuild the configurator as an executive controller.
- Give it real authority over module reasoning strategy, not just scalar settings.
- Let it control:
  - perception policy
  - world-model policy
  - actor policy
  - cost composition policy
  - memory retrieval and weighting policy
  - online adaptation policy for JEPA
  - reasoning depth, abstraction level, and what "success" means in the current context

### Core / Plugin Boundary

Grade: `D`

What is wrong:

- Too much InSite semantics are in core.
- Too many action and cost assumptions are native to Chamelia.
- Some abstractions exist, but the boundary is still not enforced strongly enough.

Examples:

- Domain-shaped costs live in core.
- InSite-specific action semantics still materially shape the actor/core types.
- Core is still too aware of schedule/profile semantics.
- The architecture has started to move toward plugins, but the core still knows too much about the current product.

What needs to change:

- Plugins should define:
  - state semantics
  - observation semantics
  - action semantics
  - regime vocabulary
  - cost semantics
  - safety semantics
  - intervention primitives
- Chamelia core should define:
  - belief formation
  - future simulation framework
  - constrained planning
  - critic integration
  - metacognitive control
  - memory and learning interfaces

## The Most Important Correction

The current codebase risks thinking about the problem as:

- "Which estimator do we choose?"

That is too small.

The intended question is:

- "How should Chamelia reason right now, using its JEPA-based world model and all available evidence, to best solve the task under constraints?"

That means the configurator should control:

- how JEPA is used
- how JEPA uncertainty is interpreted
- how the observation window is formed
- what latent features matter right now
- which objectives are active
- how far ahead to think
- how robust or exploratory the actor should be
- which memories to rely on

## Training Strategy

The likely training path is not to specialize Chamelia on insulin or health from day one. The better path is:

- first teach it how to think
- then teach it what a domain means

### What should be pretrained generally

`Configurator`

- This is the clearest general pretraining target.
- It should learn meta-reasoning:
  - when to think deeper
  - when to rely on short-horizon vs long-horizon futures
  - when to pull in memory
  - when current reasoning is failing
  - when to stabilize, probe, exploit, or recover

`Perception`

- Perception should be pretrained generally on latent formation.
- It should learn how to build useful latent state:
  - predictive
  - controllable
  - uncertainty-aware
  - useful for counterfactual planning

`WorldModule`

- The world-model substrate should be pretrained generally on:
  - latent transition prediction
  - regime-shift handling
  - counterfactual forecasting
  - short-term vs long-term consequence structure

`Memory`

- The memory mechanism should be pretrained generally.
- The goal is not just to store episodes, but to learn:
  - what reasoning traces are worth storing
  - how to summarize them
  - how to retrieve analogs by latent structure
  - when old memories matter more than recent ones

`Critic / value machinery`

- The critic should learn general long-horizon value estimation mechanics.
- Domain semantics of "bad" should still come later from the plugin.

`Actor priors`

- The actor can likely benefit from generic search priors and robust planning habits.
- Action semantics should still remain domain-owned.

### What should remain domain-specific

- action semantics
- cost semantics
- safety semantics
- intervention vocabulary
- explanation language
- domain-specific regime meanings

### Why chess is a good benchmark

Chess is useful not because it is the end goal, but because it cleanly tests the intended architecture:

- `Perception`: infer board state and salient structure
- `WorldModule`: project future states under candidate moves
- `Cost`: define good and bad futures in domain terms
- `Actor`: search over constrained futures
- `Memory`: retrieve motifs, traps, plans, and analogs
- `Configurator`: decide how to think, how deep to search, and what matters now

Chess is a strong benchmark for:

- short-term sacrifice vs long-term value
- tactical vs strategic reasoning
- memory-guided reasoning
- switching reasoning modes
- constrained future search

### Other strong pretraining domains

- `Go`
- `Poker`
- `RTS resource-control environments`
- `Hidden-regime gridworlds`
- `Robot navigation with sensor failures`
- `Supply-chain and logistics simulators`
- `Power-grid / traffic / queuing control`
- `Ecology or epidemic simulators`
- `General POMDP benchmark suites`

These are useful because they stress:

- partial observability
- hidden causes
- delayed consequences
- multiple valid reasoning modes
- hard constraints

## Training Practicality

A valid concern is storage and compute cost.

This should not require LLM-scale training if done correctly.

What keeps it tractable:

- keep the core model compact
- keep memory external and structured
- pretrain on synthetic generators rather than giant static corpora
- pretrain modules in stages instead of end-to-end monolithically
- keep plugins thin and domain-specific

What likely dominates storage:

- trajectory logs
- latent transition datasets
- memory/retrieval traces

What likely does not need to be huge:

- configurator/controller
- critic
- retrieval policy
- plugin adapters

The practical strategy is:

1. train on synthetic worlds
2. pretrain latent perception and world modeling
3. pretrain metacognitive control over those modules
4. pretrain memory retrieval over reasoning traces
5. domain-specialize later

That is much more realistic than trying to train one giant end-to-end domain-specialized model from scratch.

## Refactor Priorities

Priority 1: Redefine the configurator

- Replace "parameter tuner" with "executive controller".
- Add explicit module control surfaces.

Priority 2: Move domain cost semantics out of core

- Remove domain-shaped cost logic from `types.jl` and other core files.
- Replace with plugin-defined cost channel interfaces.

Priority 3: Clean the action boundary

- Push InSite-specific action semantics further out of core.
- Make actor operate over plugin-owned action families.

Priority 4: Upgrade memory into metacognitive memory

- Record how the system reasoned, not just what it did.

Priority 5: Keep JEPA central

- Improve configurator control over JEPA rather than building the architecture around fallback estimators.

## Final Assessment

Chamelia is not far from its intended identity structurally, but it is still far from it architecturally.

The good news:

- The reasoning skeleton is real.
- The rollout-and-critic framing is real.
- JEPA is already in the architecture.
- The actor already reasons in a future-sensitive, constraint-sensitive way.

The bad news:

- The configurator is not yet doing the job the architecture requires.
- Domain semantics still leak too deeply into core.
- Chamelia is still too much a specialized system with generic wrappers, rather than a genuinely general reasoning engine with domain plugins.

The correct direction is:

- keep JEPA central
- keep rollouts central
- keep actor search central
- rebuild the configurator as true metacognition
- push domain-specific semantics out of core

That is the shortest path to making Chamelia actually smart in the way it was intended to be.

# AGENTS.md

## Repo priorities
- This is a Julia project. Always use `julia --project=.`
- Before running code, instantiate the environment:
  `julia --project=. -e "using Pkg; Pkg.instantiate()"`
- Prefer small, surgical edits over broad rewrites.
- Do not modify `Manifest.toml` unless explicitly required.
- Do not invent mathematical assumptions unless clearly marked as provisional.

## Chamelia-specific behavior
- Treat `Docs/Chamelia_Mathematical_Formulation_v1.1.md` as the conceptual source of truth.
- When code and math disagree, first explain the mismatch.
- Preserve the intended LeCun-style module roles: Configurator, Perception, World, Cost, Memory, Actor.
- Prefer code that makes module ownership, dimensions, and dependencies explicit.
- Watch for:
  - global constants referenced before definition
  - circular include/load order problems
  - module namespace mistakes
  - structs whose fields do not match the intended math objects
  - functions that mix estimation, control, and objective logic incoherently

## Debugging workflow
- For compile/load errors, diagnose before editing.
- Check include order, module scope, const ownership, and top-level side effects first.
- Explain the exact cause of the failure in plain English.
- Then propose the minimum fix.

## Rewrite workflow
- For rewrites based on the math doc:
  1. identify the intended object in the math
  2. locate the code representation
  3. state the mismatch
  4. propose the smallest coherent correction
- Do not try to make the entire system perfect in one pass.

## Testing
- After edits, run the narrowest relevant test or load path first.
- If no test exists, add a minimal smoke test for the changed behavior.
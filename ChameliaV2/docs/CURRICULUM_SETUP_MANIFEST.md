# Curriculum Setup Manifest

This file explains how to use [curriculum_setup_manifest.yaml](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/configs/curriculum_setup_manifest.yaml) as the source of truth for `setup.sh`.

## Purpose

The manifest exists so `setup.sh` and later `train_chamelia.py` can answer four questions consistently:

1. What assets belong to each curriculum stage?
2. Which assets should be fetched automatically?
3. Which assets should be treated as optional, manual, or gated?
4. Which module paths consume each asset?

## Mode Semantics

- `default`: fetch automatically. If the relevant stage is enabled, missing assets are a setup error.
- `optional`: fetch only when explicitly requested. Missing assets should not block the run.
- `manual`: never auto-fetch. `setup.sh` should create the destination path and ingest only if the file already exists.
- `gated`: requires credentials, terms acceptance, or a DUA. `setup.sh` should skip unless the required secret or flag is present.
- `generated`: no download. The data is created locally by generators or self-play.

## Setup Policy Semantics

- `hard_fail_if_stage_enabled`: fail setup only when the stage/domain is active and the asset is required.
- `soft_skip`: skip cleanly and report that the asset is unavailable.

## Included Sources

The manifest currently covers the concrete sources we have actually selected for the curriculum:

- Stage 0: `roberta-base`, Wikimedia dumps, Project Gutenberg, XNLI, PMC Open Access, optional `The Stack`, manual `Cecil`
- Stage 1: generated arithmetic, `AGIEval`, `GSM8K`, `hendrycks_math`, `LogiQA2`, `ProofWriter`, `FOLIO`, optional `Open-Platypus`
- Stage 2: `OEIS`, `ARC-AGI-2`, generated regime-shift and pattern data
- Stage 3: `Stockfish`, `Lichess` databases, optional `KataGo`, `OpenSpiel`
- Stage 4: generated collaborative self-play traces
- Stage 5: `MEDS`, `MIMIC-IV demo MEDS`, optional credentialed `MIMIC-IV`, optional `SciBERT`, optional `BiomedBERT`, generated synthetic patients

## Why AGIEval And Open-Platypus Are Split

- `AGIEval` is the primary open LSAT source. It includes `lsat-ar`, `lsat-lr`, and `lsat-rc`, and it is a clean default candidate for Stage 1 LSAT-style work.
- `Open-Platypus` is a supplemental reasoning mixture, not the core LSAT dataset. Its upstream license mix is more complicated, so it is marked optional.

## Why Cecil Is Manual-Only

The Cecil PDF link is useful as a research seed corpus, but it should not be auto-fetched by `setup.sh`. The manifest intentionally marks it `manual` so we can support:

- local manual placement into `data/manual/medical_texts/`
- preprocessing if present
- clean skipping if absent

## Unresolved Domains

The manifest does not pretend every curriculum domain is solved. These remain explicit gaps:

- `gre`
- `mcat_cars`
- `go` corpora beyond engine-backed self-play
- open patient-doctor dialogue corpora
- dedicated care-ethics benchmarks

Each unresolved item has a fallback recorded in the YAML so `setup.sh` and `train_chamelia.py` can report the gap instead of silently improvising.

## Implemented `setup.sh` Behavior

`setup.sh` should:

1. Read the manifest.
2. Install all `default` assets for the requested stages.
3. Install `optional` assets only when a flag such as `--include-optional` is provided.
4. Create destination directories for `manual` assets and print exactly what file should be placed there.
5. Skip `gated` assets unless credentials or acceptance flags are present.
6. Record a final setup report listing fetched, skipped, manual, and unresolved assets.

This is now implemented through:

- [setup.sh](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/setup.sh)
- [setup_curriculum.py](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/scripts/setup_curriculum.py)
- [setup_manifest.py](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/training/curriculum/data/setup_manifest.py)

Example:

```bash
./setup.sh --dry-run --stage 1 --domain basic_arithmetic
./setup.sh --stage 0 --stage 1 --include-optional
./setup.sh --stage 5 --include-gated --strict
```

## Integration Spots

The main consumers already exist in the V2 tree:

- [stage0_language.py](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/training/curriculum/domains/stage0_language.py)
- [stage1_reasoning.py](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/training/curriculum/domains/stage1_reasoning.py)
- [stage2_patterns.py](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/training/curriculum/domains/stage2_patterns.py)
- [stage3_games.py](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/training/curriculum/domains/stage3_games.py)
- [stage4_collaborative.py](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/training/curriculum/domains/stage4_collaborative.py)
- [stage5_health.py](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/training/curriculum/domains/stage5_health.py)
- [preprocessors.py](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/training/curriculum/data/preprocessors.py)
- [stage_runner.py](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/training/curriculum/stage_runner.py)
- [graduation.py](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/training/curriculum/graduation.py)

## Current Limitation

`setup.sh` currently executes what it can directly:

- local/generated assets
- git-backed repositories
- manual/gated/optional bookkeeping
- machine-readable report generation

For many large datasets and Hugging Face assets, the current implementation records explicit follow-up commands instead of force-downloading immediately. That is intentional: a number of sources in the manifest are landing pages, gated assets, or term-acceptance flows rather than direct file URLs.

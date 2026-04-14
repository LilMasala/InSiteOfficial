# Protein DTI Runbook

This directory contains the acquisition, split-building, QA, and smoke-eval entrypoints for the `protein_dti` core domain plugin.

## Paths

Local defaults are repo-relative:

- data dir: `data/protein_dti`
- scratch dir: `artifacts/protein_dti_tmp`
- metadata DB: `data/protein_dti/db/protein_dti.sqlite3`
- acquisition log DB: `data/protein_dti/db/acquisition_log.sqlite3`

Swallowtail runs should use the canonical cluster paths by exporting:

- `CHAMELIA_SWALLOWTAIL_ROOT`
- `CHAMELIA_SWALLOWTAIL_SCRATCH_DIR`
- `CHAMELIA_PROTEIN_DTI_DATA_DIR`
- `CHAMELIA_PROTEIN_DTI_DB_PATH`

`run_all_swallowtail.sh` already does that wiring for the standard cluster layout.

## Local Setup

1. `bash scripts/protein_data/00_setup.sh`
2. `.venv311/bin/python scripts/protein_data/init_schema.py`
3. `.venv311/bin/python scripts/protein_data/01_tdc_pull.py`
4. `.venv311/bin/python scripts/protein_data/02_enrich_proteins.py`
5. `.venv311/bin/python scripts/protein_data/03_enrich_drugs.py`
6. `.venv311/bin/python scripts/protein_data/04_build_protein_graphs.py`
7. `.venv311/bin/python scripts/protein_data/05_build_drug_graphs.py`
8. `.venv311/bin/python scripts/protein_data/06_build_splits.py --strategy all --write-hdf5`

## QA

Quick dataset coverage report:

```bash
.venv311/bin/python scripts/protein_data/report_dataset.py
```

JSON output for automation:

```bash
.venv311/bin/python scripts/protein_data/report_dataset.py --json
```

API/integrity health check:

```bash
.venv311/bin/python scripts/protein_data/health_check.py --once
```

## Smoke Eval

Run a small held-out ranking evaluation through the same bridge/runtime model path used elsewhere in the repo:

```bash
.venv311/bin/python scripts/protein_data/evaluate_domain.py \
  --split test \
  --split-strategy protein_family \
  --affinity-type Kd \
  --num-proteins 32 \
  --backbone-mode stub
```

With a checkpoint:

```bash
.venv311/bin/python scripts/protein_data/evaluate_domain.py \
  --split test \
  --split-strategy protein_family \
  --affinity-type Kd \
  --backbone-mode hjepa \
  --checkpoint-path /absolute/path/to/checkpoint.pt
```

Reported metrics:

- primary: mean and median per-protein Spearman correlation
- secondary: per-protein MSE mean, pooled MSE, pooled AUROC

Notes:

- The domain is ranking-first. MSE is secondary and raw actor scores may be uncalibrated for early checkpoints.
- AUROC uses the supplied `--binder-threshold` against the normalized affinity scale. The default `7.0` is sensible for `Kd`/`Ki`-style p-affinity values, not necessarily for `KIBA`.
- Temporal splits are still intentionally left as TODO.

## Runtime Env Knobs

The bridge/runtime path for `protein_dti` reads these env vars when a session is created:

- `CHAMELIA_PROTEIN_DTI_DB_PATH`
- `CHAMELIA_PROTEIN_DTI_DATA_DIR`
- `CHAMELIA_PROTEIN_DTI_SPLIT`
- `CHAMELIA_PROTEIN_DTI_SPLIT_STRATEGY`
- `CHAMELIA_PROTEIN_DTI_AFFINITY_TYPE`
- `CHAMELIA_PROTEIN_DTI_MAX_CANDIDATES`
- `CHAMELIA_PROTEIN_DTI_ACTION_DIM`
- `CHAMELIA_PROTEIN_DTI_SEED`

That keeps local runs repo-relative by default while still letting cluster jobs point at the swallowtail dataset explicitly.

# Cluster Chess Training

This is the cluster runbook for the interactive chess orchestrator path.

## Official path

The official chess runtime path is the orchestrator, not the Stage 3 curriculum token trainer.

- entrypoint: [scripts/train_orchestrator.py](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/scripts/train_orchestrator.py)
- main config: [configs/orchestrator_chess.yaml](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/configs/orchestrator_chess.yaml)
- smoke config: [configs/orchestrator_chess_smoke.yaml](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/configs/orchestrator_chess_smoke.yaml)
- generic launcher: [scripts/slurm/train_orchestrator_exx512.sbatch](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/scripts/slurm/train_orchestrator_exx512.sbatch)
- smoke launcher: [scripts/slurm/train_orchestrator_chess_smoke_exx512.sbatch](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/scripts/slurm/train_orchestrator_chess_smoke_exx512.sbatch)

## Environment

All Slurm launchers now source:

1. `~/.profile`
2. `~/.bashrc`
3. the requested Conda or Micromamba environment

That keeps the Gemma and Hugging Face cache exports from `~/.profile` available in batch jobs before environment activation.

## Smoke run

```bash
cd ~/InSite/InSiteOfficial/ChameliaV2
export CONDA_ENV_NAME=chameliav2
export RUN_TAG=chess-smoke-$(date +%Y%m%dT%H%M%S)
sbatch scripts/slurm/train_orchestrator_chess_smoke_exx512.sbatch
```

This uses the smaller smoke preset and restricts the run to `chess`.

## Full chess run

```bash
cd ~/InSite/InSiteOfficial/ChameliaV2
export CONDA_ENV_NAME=chameliav2
export ORCHESTRATOR_CONFIG=$PWD/configs/orchestrator_chess.yaml
export DOMAINS=chess
export RUN_TAG=chess-$(date +%Y%m%dT%H%M%S)
sbatch scripts/slurm/train_orchestrator_exx512.sbatch
```

## LanceDB note

The orchestrator path uses the `procedural.use_lancedb` setting from the orchestrator config.

The curriculum trainer in [scripts/train_chamelia.py](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/scripts/train_chamelia.py) currently builds `LatentMemory` only, so LanceDB is not part of that training path yet.

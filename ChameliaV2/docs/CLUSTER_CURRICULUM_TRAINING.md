# Cluster Curriculum Training

This is the current cluster runbook for the repaired `ChameliaV2 -> V3` training path.

The intended sequence is:

1. foundation pretraining over stages `0-4`
2. stage-5 health/T1D specialization
3. export bridge-loadable artifacts from each run
4. point the Julia/Python bridge at the resulting `.pth` artifact

The current `train_chamelia.py` entrypoint already exports bridge-loadable artifacts under:

- `checkpoints/<run-tag>/bridge_artifacts/*.pth`

Those artifacts include:

- `model_state_dict`
- `config`
- `bridge_backbone_mode`
- `model_version`

## Cluster assumptions

These launchers are written for the Wesleyan `exx512` Slurm partition.

- partition: `exx512`
- default GPU request: `1`
- current training code path: single-process, single-GPU

If you later add distributed training, update the launchers before requesting more than one GPU.

## Data assumptions

Foundation training can run immediately even if Stage 3 and Stage 4 public datasets are incomplete.

- Stage 3 games falls back to synthetic move-history samples.
- Stage 4 collaboration falls back to synthetic coordination samples.

That makes the run reproducible without blocking on dataset curation, but the run is stronger if real data exists under:

- `data/curriculum/stage3/...`
- `data/curriculum/stage4/generated`

## Environment assumptions

Each `sbatch` script will try the following in order:

1. load `CUDA_MODULE` if provided
2. source `~/.bashrc`
3. activate `MAMBA_ENV_NAME` if provided
4. activate `CONDA_ENV_NAME` if provided
5. use `.venv311/bin/python`, then `.venv/bin/python`, then `python3`

So the easiest operator flow is:

```bash
cd ~/InSite/InSiteOfficial/ChameliaV2
export MAMBA_ENV_NAME=<your-env>
sbatch scripts/slurm/train_curriculum_foundation_exx512.sbatch
```

## Phase A: Foundation run

This phase should teach the planner the general reasoning/search substrate before health specialization.

- stages: `0,1,2,3,4`
- config: [curriculum_hjepa_foundation_single_gpu.yaml](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/configs/curriculum_hjepa_foundation_single_gpu.yaml)
- launcher: [train_curriculum_foundation_exx512.sbatch](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/scripts/slurm/train_curriculum_foundation_exx512.sbatch)

Example:

```bash
cd ~/InSite/InSiteOfficial/ChameliaV2
export MAMBA_ENV_NAME=chamelia
export RUN_TAG=foundation-20260405
export MODEL_VERSION=foundation-20260405
export DATA_ROOT=$PWD/data/curriculum
sbatch scripts/slurm/train_curriculum_foundation_exx512.sbatch
```
- `CHECKPOINT_DIR=/path/to/checkpoints/foundation-20260405`
- `CUDA_MODULE=<cluster-cuda-module>`

## Phase B: Stage-5 finetune

This phase specializes into health/T1D after the foundation run exists.

- stage: `5`
- default domains: `synthetic_patients,t1d_specific`
- config: [curriculum_hjepa_stage5_single_gpu.yaml](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/configs/curriculum_hjepa_stage5_single_gpu.yaml)
- launcher: [train_curriculum_stage5_exx512.sbatch](/Users/anandparikh/Desktop/InSiteOfficial/ChameliaV2/scripts/slurm/train_curriculum_stage5_exx512.sbatch)

Example:

```bash
cd ~/InSite/InSiteOfficial/ChameliaV2
export MAMBA_ENV_NAME=chamelia
export RUN_TAG=stage5-20260405
export MODEL_VERSION=stage5-20260405
export STAGE5_DOMAINS=synthetic_patients,t1d_specific
export INIT_CHECKPOINT=$PWD/checkpoints/foundation-20260405/bridge_artifacts/<artifact>.pth
sbatch scripts/slurm/train_curriculum_stage5_exx512.sbatch
```

If `INIT_CHECKPOINT` is set, the stage-5 run will strict-load the `model_state_dict`
from that artifact before training starts.

## Monitoring

Outputs go to:

- `logs/slurm/*.out`
- `logs/slurm/*.err`
- `checkpoints/<run-tag>/`
- `checkpoints/<run-tag>/bridge_artifacts/`

Useful commands:

```bash
squeue -u "$USER"
tail -f logs/slurm/chamelia-foundation-<jobid>.out
tail -f logs/slurm/chamelia-stage5-<jobid>.out
```

## Bridge deployment after training

Once a run finishes, point the Python bridge at one of the exported artifacts:

```bash
export CHAMELIA_BRIDGE_BACKBONE_MODE=hjepa
export CHAMELIA_BRIDGE_CHECKPOINT=/abs/path/to/checkpoints/<run-tag>/bridge_artifacts/<artifact>.pth
export CHAMELIA_BRIDGE_MODEL_VERSION=<model-version>
```

Then rerun the local or cluster-backed comparison harness to compare:

- `v1.1`
- `v1.5`
- `v3`

using a real learned HJEPA checkpoint instead of the stub bridge.

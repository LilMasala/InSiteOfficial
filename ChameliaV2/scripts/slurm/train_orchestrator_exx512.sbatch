#!/usr/bin/env bash
#SBATCH --job-name=chamelia-orchestrator
#SBATCH --partition=exx512
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/logs/slurm"
mkdir -p "${LOG_DIR}"

if [[ -n "${CUDA_MODULE:-}" ]] && command -v module >/dev/null 2>&1; then
  module load "$CUDA_MODULE"
fi

if [[ -f "${HOME}/.bashrc" ]]; then
  # shellcheck disable=SC1090
  source "${HOME}/.bashrc"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-chameliav2}"
if [[ -n "${MAMBA_ENV_NAME:-}" ]] && command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate "${MAMBA_ENV_NAME}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

if [[ -x "${ROOT_DIR}/.venv311/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv311/bin/python"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "python3 not found" >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

RUN_TAG="${RUN_TAG:-orchestrator-$(date +%Y%m%dT%H%M%S)}"
ORCHESTRATOR_CONFIG="${ORCHESTRATOR_CONFIG:-${ROOT_DIR}/configs/orchestrator_foundation.yaml}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/checkpoints/${RUN_TAG}}"
DEFAULT_SCRATCH_BASE="/sanscratch/${USER}/chamelia"
if [[ ! -d "/sanscratch" ]]; then
  DEFAULT_SCRATCH_BASE="${TMPDIR:-/tmp/chamelia_${SLURM_JOB_ID:-$$}}"
fi
SCRATCH_DIR="${SCRATCH_DIR:-${DEFAULT_SCRATCH_BASE}/${RUN_TAG}}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
DOMAINS="${DOMAINS:-}"

mkdir -p "${SCRATCH_DIR}"

echo "Starting orchestrator curriculum run"
echo "  root:          ${ROOT_DIR}"
echo "  python:        ${PYTHON_BIN}"
echo "  config:        ${ORCHESTRATOR_CONFIG}"
echo "  run tag:       ${RUN_TAG}"
echo "  device:        ${DEVICE}"
echo "  seed:          ${SEED}"
echo "  scratch dir:   ${SCRATCH_DIR}"
echo "  checkpoint dir:${CHECKPOINT_DIR}"
echo "  domains:       ${DOMAINS:-<config default>}"
echo "  node:          $(hostname)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

TRAIN_CMD=(
  "${PYTHON_BIN}" scripts/train_orchestrator.py
  --config "${ORCHESTRATOR_CONFIG}"
  --device "${DEVICE}"
  --run-dir "${SCRATCH_DIR}"
  --seed "${SEED}"
)

if [[ -n "${DOMAINS}" ]]; then
  IFS=',' read -r -a DOMAIN_LIST <<< "${DOMAINS}"
  for DOMAIN in "${DOMAIN_LIST[@]}"; do
    TRAIN_CMD+=(--domain "${DOMAIN}")
  done
fi

printf '  train cmd:     '
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n'

"${TRAIN_CMD[@]}"

mkdir -p "${CHECKPOINT_DIR}"
if command -v rsync >/dev/null 2>&1; then
  rsync -a "${SCRATCH_DIR}/" "${CHECKPOINT_DIR}/"
else
  cp -r "${SCRATCH_DIR}/." "${CHECKPOINT_DIR}/"
fi

echo ""
echo "Training complete."
echo "  Scratch run dir: ${SCRATCH_DIR}"
echo "  Copied results:  ${CHECKPOINT_DIR}"
echo "  Summary file:    ${CHECKPOINT_DIR}/orchestrator_results.yaml"

#!/usr/bin/env bash
#SBATCH --job-name=chamelia-foundation
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

RUN_TAG="${RUN_TAG:-foundation-$(date +%Y%m%dT%H%M%S)}"
MODEL_VERSION="${MODEL_VERSION:-${RUN_TAG}}"
CURRICULUM_CONFIG="${CURRICULUM_CONFIG:-${ROOT_DIR}/configs/curriculum_hjepa_foundation_single_gpu.yaml}"
CHAMELIA_CONFIG="${CHAMELIA_CONFIG:-${ROOT_DIR}/configs/chamelia.yaml}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data/curriculum}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/checkpoints/${RUN_TAG}}"
STAGES="${STAGES:-}"
START_STAGE="${START_STAGE:-}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
INITIAL_STAGE_STEPS="${INITIAL_STAGE_STEPS:-2000}"
EVAL_EVERY="${EVAL_EVERY:-200}"
MAX_TOTAL_STAGE_STEPS="${MAX_TOTAL_STAGE_STEPS:-20000}"
MAX_EXTENSIONS="${MAX_EXTENSIONS:-4}"
MAX_RETUNES="${MAX_RETUNES:-2}"
EXTENSION_FACTOR="${EXTENSION_FACTOR:-1.5}"
RETUNE_LR_FACTORS="${RETUNE_LR_FACTORS:-0.5,1.5}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
LR="${LR:-}"

echo "Starting foundation curriculum run"
echo "  root:            ${ROOT_DIR}"
echo "  python:          ${PYTHON_BIN}"
echo "  curriculum:      ${CURRICULUM_CONFIG}"
echo "  chamelia config: ${CHAMELIA_CONFIG}"
echo "  stages:          ${STAGES:-<config default>}"
echo "  device:          ${DEVICE}"
echo "  data root:       ${DATA_ROOT}"
echo "  checkpoint dir:  ${CHECKPOINT_DIR}"
echo "  log dir:         ${LOG_DIR}"
echo "  model version:   ${MODEL_VERSION}"
echo "  init checkpoint: ${INIT_CHECKPOINT:-<none>}"
echo "  start stage:     ${START_STAGE:-<config default>}"
echo "  node:            $(hostname)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

TRAIN_CMD=(
  "${PYTHON_BIN}" scripts/train_chamelia.py
  --curriculum-config "${CURRICULUM_CONFIG}"
  --chamelia-config "${CHAMELIA_CONFIG}"
  --device "${DEVICE}"
  --data-root "${DATA_ROOT}"
  --seed "${SEED}"
  --backbone-mode hjepa
  --initial-stage-steps "${INITIAL_STAGE_STEPS}"
  --eval-every "${EVAL_EVERY}"
  --max-total-stage-steps "${MAX_TOTAL_STAGE_STEPS}"
  --max-extensions "${MAX_EXTENSIONS}"
  --max-retunes "${MAX_RETUNES}"
  --extension-factor "${EXTENSION_FACTOR}"
  --retune-lr-factors "${RETUNE_LR_FACTORS}"
  --clip-grad "${CLIP_GRAD}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --model-version "${MODEL_VERSION}"
)

if [[ -n "${LR}" ]]; then
  TRAIN_CMD+=(--lr "${LR}")
fi

if [[ -n "${INIT_CHECKPOINT}" ]]; then
  TRAIN_CMD+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

if [[ -n "${START_STAGE}" ]]; then
  TRAIN_CMD+=(--start-stage "${START_STAGE}")
fi

if [[ -n "${STAGES}" ]]; then
  IFS=',' read -r -a STAGE_LIST <<< "${STAGES}"
  for STAGE in "${STAGE_LIST[@]}"; do
    TRAIN_CMD+=(--stage "${STAGE}")
  done
fi

printf '  train cmd:       '
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n'

"${TRAIN_CMD[@]}"

#!/usr/bin/env bash
#SBATCH --job-name=chamelia-stage5
#SBATCH --partition=exx512
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

if [[ -n "${MAMBA_ENV_NAME:-}" ]] && command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook --shell bash)"
  micromamba activate "${MAMBA_ENV_NAME}"
elif [[ -n "${CONDA_ENV_NAME:-}" ]] && command -v conda >/dev/null 2>&1; then
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

RUN_TAG="${RUN_TAG:-stage5-$(date +%Y%m%dT%H%M%S)}"
MODEL_VERSION="${MODEL_VERSION:-${RUN_TAG}}"
CURRICULUM_CONFIG="${CURRICULUM_CONFIG:-${ROOT_DIR}/configs/curriculum_hjepa_stage5_single_gpu.yaml}"
CHAMELIA_CONFIG="${CHAMELIA_CONFIG:-${ROOT_DIR}/configs/chamelia.yaml}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/data/curriculum}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${ROOT_DIR}/checkpoints/${RUN_TAG}}"
STAGE5_DOMAINS="${STAGE5_DOMAINS:-synthetic_patients,t1d_specific}"
SEED="${SEED:-42}"
INITIAL_STAGE_STEPS="${INITIAL_STAGE_STEPS:-2000}"
EVAL_EVERY="${EVAL_EVERY:-200}"
MAX_TOTAL_STAGE_STEPS="${MAX_TOTAL_STAGE_STEPS:-16000}"
MAX_EXTENSIONS="${MAX_EXTENSIONS:-4}"
MAX_RETUNES="${MAX_RETUNES:-2}"
EXTENSION_FACTOR="${EXTENSION_FACTOR:-1.5}"
RETUNE_LR_FACTORS="${RETUNE_LR_FACTORS:-0.5,1.5}"
CLIP_GRAD="${CLIP_GRAD:-1.0}"
LR="${LR:-1.0e-4}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

echo "Starting stage-5 curriculum run"
echo "  root:            ${ROOT_DIR}"
echo "  python:          ${PYTHON_BIN}"
echo "  curriculum:      ${CURRICULUM_CONFIG}"
echo "  chamelia config: ${CHAMELIA_CONFIG}"
echo "  stage-5 domains: ${STAGE5_DOMAINS}"
echo "  data root:       ${DATA_ROOT}"
echo "  checkpoint dir:  ${CHECKPOINT_DIR}"
echo "  log dir:         ${LOG_DIR}"
echo "  model version:   ${MODEL_VERSION}"
echo "  init checkpoint: ${INIT_CHECKPOINT:-<none>}"
echo "  node:            $(hostname)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

TRAIN_CMD=(
  "${PYTHON_BIN}" scripts/train_chamelia.py
  --curriculum-config "${CURRICULUM_CONFIG}"
  --chamelia-config "${CHAMELIA_CONFIG}"
  --stage 5
  --domain "${STAGE5_DOMAINS}"
  --device cuda
  --data-root "${DATA_ROOT}"
  --seed "${SEED}"
  --backbone-mode hjepa
  --lr "${LR}"
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

if [[ -n "${INIT_CHECKPOINT}" ]]; then
  TRAIN_CMD+=(--init-checkpoint "${INIT_CHECKPOINT}")
fi

"${TRAIN_CMD[@]}"

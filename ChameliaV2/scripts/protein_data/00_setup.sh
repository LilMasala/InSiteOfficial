#!/usr/bin/env bash
set -euo pipefail

ROOT="${CHAMELIA_PROTEIN_DTI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_DIR="${CHAMELIA_PROTEIN_DTI_DATA_DIR:-${ROOT}/data/protein_dti}"
SCRATCH_DIR="${CHAMELIA_PROTEIN_DTI_SCRATCH_DIR:-${ROOT}/artifacts/protein_dti_tmp}"
LOG_DIR="${ROOT}/logs/protein_data"

mkdir -p "${DATA_DIR}/db"
mkdir -p "${DATA_DIR}/structures/pdb"
mkdir -p "${DATA_DIR}/structures/alphafold"
mkdir -p "${DATA_DIR}/graphs/proteins"
mkdir -p "${DATA_DIR}/graphs/drugs"
mkdir -p "${DATA_DIR}/hdf5"
mkdir -p "${SCRATCH_DIR}"
mkdir -p "${LOG_DIR}"

echo "Protein DTI directories created."
echo "  Root:    ${ROOT}"
echo "  Data:    ${DATA_DIR}"
echo "  Scratch: ${SCRATCH_DIR}"
echo "  Logs:    ${LOG_DIR}"

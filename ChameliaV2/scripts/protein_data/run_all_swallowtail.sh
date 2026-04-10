#!/usr/bin/env bash
set -euo pipefail

ROOT="${CHAMELIA_SWALLOWTAIL_ROOT:-/zfshomes/aparikh02/InSite/InSiteOfficial/ChameliaV2}"
DATA="${ROOT}/data/protein_dti"
DB="${DATA}/db/protein_dti.sqlite3"
LOG_DB="${DATA}/db/acquisition_log.sqlite3"
SCRATCH="${CHAMELIA_SWALLOWTAIL_SCRATCH_DIR:-/sanscratch/aparikh02/protein_dti_tmp}"
PYTHON="${ROOT}/.venv311/bin/python"
SESSION="protein_acquisition"

cd "${ROOT}"

tmux new-session -d -s "${SESSION}" -n "setup"
tmux send-keys -t "${SESSION}:setup" \
  "export CHAMELIA_PROTEIN_DTI_ROOT='${ROOT}'; export CHAMELIA_PROTEIN_DTI_DATA_DIR='${DATA}'; export CHAMELIA_PROTEIN_DTI_SCRATCH_DIR='${SCRATCH}'; bash scripts/protein_data/00_setup.sh && ${PYTHON} scripts/protein_data/init_schema.py --db '${DB}' --log-db '${LOG_DB}'" Enter

sleep 2

tmux new-window -t "${SESSION}" -n "tdc_pull"
tmux send-keys -t "${SESSION}:tdc_pull" \
  "${PYTHON} scripts/protein_data/01_tdc_pull.py --db '${DB}' --scratch '${SCRATCH}'" Enter

tmux new-window -t "${SESSION}" -n "enrich_proteins"
tmux send-keys -t "${SESSION}:enrich_proteins" \
  "${PYTHON} scripts/protein_data/02_enrich_proteins.py --db '${DB}' --data-dir '${DATA}' --log-db '${LOG_DB}' --requests-per-second 2" Enter

tmux new-window -t "${SESSION}" -n "enrich_drugs"
tmux send-keys -t "${SESSION}:enrich_drugs" \
  "${PYTHON} scripts/protein_data/03_enrich_drugs.py --db '${DB}' --log-db '${LOG_DB}' --requests-per-second 2" Enter

tmux new-window -t "${SESSION}" -n "protein_graphs"
tmux send-keys -t "${SESSION}:protein_graphs" \
  "${PYTHON} scripts/protein_data/04_build_protein_graphs.py --db '${DB}' --data-dir '${DATA}'" Enter

tmux new-window -t "${SESSION}" -n "drug_graphs"
tmux send-keys -t "${SESSION}:drug_graphs" \
  "${PYTHON} scripts/protein_data/05_build_drug_graphs.py --db '${DB}' --data-dir '${DATA}'" Enter

tmux new-window -t "${SESSION}" -n "splits"
tmux send-keys -t "${SESSION}:splits" \
  "${PYTHON} scripts/protein_data/06_build_splits.py --db '${DB}' --data-dir '${DATA}' --strategy all --write-hdf5" Enter

tmux new-window -t "${SESSION}" -n "health"
tmux send-keys -t "${SESSION}:health" \
  "${PYTHON} scripts/protein_data/health_check.py --db '${DB}' --log-db '${LOG_DB}'" Enter

echo "Started tmux session '${SESSION}'."
echo "Attach with: tmux attach -t ${SESSION}"

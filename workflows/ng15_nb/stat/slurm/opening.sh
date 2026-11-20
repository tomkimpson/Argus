#!/bin/bash
# Detection-statistic spec Phase 1 (tasks 1.1, 1.2, 1.4) on one A100, from a pinned worktree.
#   MODE=check            timing, peak memory, batch sizing and V2 (minutes)
#   MODE=full BATCH=B     G for all 400 v3-probe draws, V1 and the 5-draw rerun
#   sbatch --export=ALL,REPO=<worktree>,MODE=check workflows/ng15_nb/stat/slurm/opening.sh
# Tracked results land in $REPO/workflows/ng15_nb/results/stat/opening/; the full G array
# lands beside the ingest caches in the main checkout's outputs/stat/opening/.

#SBATCH --job-name=ng15nb-stat-opening
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/outputs/logfiles/stat_opening_%j.out

set -euo pipefail
REPO="${REPO:?set REPO to the pinned worktree}"
MODE="${MODE:-check}"
source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate Argus
export PYTHONPATH="${REPO}/python:${REPO}"  # the editable install points at the main checkout
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NG15_INGEST_DIR=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/data/ingest  # gitignored
echo "repo ${REPO} @ $(git -C "${REPO}" rev-parse HEAD); mode ${MODE}"
nvidia-smi -L
python -c "import argus; print('argus from:', argus.__file__)"
cd "${REPO}"
if [[ "${MODE}" == check ]]; then
    time python -u workflows/ng15_nb/stat/opening_job.py check
else
    time python -u workflows/ng15_nb/stat/opening_job.py full --batch "${BATCH:?set BATCH}"
fi

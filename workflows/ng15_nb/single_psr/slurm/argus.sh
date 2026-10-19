#!/bin/bash
# Single-pulsar Argus noise runs (spec task 3.1): one pulsar per array index (manifest
# order), one A100 each. MODE=check instead runs the dense-oracle check (`--check`) and
# needs only the pulsars it is given, e.g. --array=<indices of the C1 pulsars>.
#
# Code runs from a pinned worktree so edits to the main checkout cannot reach a queued job:
#   git -C /fred/oz022/tkimpson/Argus worktree add --detach /fred/oz022/tkimpson/Argus-ng15nb-p3 <commit>
#   mkdir -p workflows/ng15_nb/outputs/logfiles
#   sbatch --export=ALL,REPO=/fred/oz022/tkimpson/Argus-ng15nb-p3 workflows/ng15_nb/single_psr/slurm/argus.sh
# Results land in the pinned worktree's workflows/ng15_nb/results/single_psr/argus/ and
# outputs/single_psr/argus/; copy the JSONs back to the main checkout to commit them.

#SBATCH --job-name=ng15nb-sp-argus
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-66
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/outputs/logfiles/sp_argus_%A_%a.out

set -euo pipefail
REPO="${REPO:?set REPO to the pinned worktree}"
source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate Argus
export PYTHONPATH="${REPO}/python:${REPO}"  # the editable install points at the main checkout
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NG15_INGEST_DIR=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/data/ingest  # gitignored
echo "repo ${REPO} @ $(git -C "${REPO}" rev-parse HEAD); index ${SLURM_ARRAY_TASK_ID}; $(nvidia-smi --query-gpu=name --format=csv,noheader)"
EXTRA=()
if [[ "${MODE:-run}" == check ]]; then EXTRA=(--check); fi
python "${REPO}/workflows/ng15_nb/single_psr/run_argus.py" --index "${SLURM_ARRAY_TASK_ID}" "${EXTRA[@]}"

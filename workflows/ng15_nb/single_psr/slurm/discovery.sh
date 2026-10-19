#!/bin/bash
# Single-pulsar Discovery-OU noise runs (spec task 3.2): one pulsar per array index
# (manifest order), one A100 each. Same pinned-worktree convention as argus.sh:
#   sbatch --export=ALL,REPO=/fred/oz022/tkimpson/Argus-ng15nb-p3 workflows/ng15_nb/single_psr/slurm/discovery.sh

#SBATCH --job-name=ng15nb-sp-disc
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-66
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/outputs/logfiles/sp_disc_%A_%a.out

set -euo pipefail
REPO="${REPO:?set REPO to the pinned worktree}"
source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate discovery
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "repo ${REPO} @ $(git -C "${REPO}" rev-parse HEAD); index ${SLURM_ARRAY_TASK_ID}; $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python "${REPO}/workflows/ng15_nb/single_psr/run_discovery.py" --index "${SLURM_ARRAY_TASK_ID}"

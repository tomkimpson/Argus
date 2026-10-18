#!/bin/bash
# D2's epoch-cell rule (task 2.5, amended 2026-10-18): two-scalar DMX-window vs
# epoch-cell filter likelihood for all 67 pulsars, in 4 shards on one A100 each.
# Writes workflows/ng15_nb/results/c1/epoch_rule/<pulsar>.json; then run
#   python workflows/ng15_nb/scripts/epoch_cell_rule.py --summarise
#
#   mkdir -p workflows/ng15_nb/outputs/logfiles
#   sbatch workflows/ng15_nb/compression/slurm/epoch_rule.sh

#SBATCH --job-name=ng15nb-erule
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3
#SBATCH --chdir=/fred/oz022/tkimpson/Argus
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/outputs/logfiles/erule_%A_%a.out

set -euo pipefail
source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate Argus
export PYTHONPATH=$PWD/python:$PWD  # the editable install points at the main checkout
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "shard ${SLURM_ARRAY_TASK_ID}  commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python workflows/ng15_nb/scripts/epoch_cell_rule.py --shard "${SLURM_ARRAY_TASK_ID}" --n-shards 4 "$@"

#!/bin/bash
# Task 2.6 / K3: the 67-psr union-grid value-plus-gradient cost on one A100.
# Writes workflows/ng15_nb/results/k3_benchmark.json.
#
#   mkdir -p workflows/ng15_nb/outputs/logfiles
#   sbatch workflows/ng15_nb/compression/slurm/k3_bench.sh

#SBATCH --job-name=ng15nb-k3
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/fred/oz022/tkimpson/Argus
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/outputs/logfiles/k3_%j.out

set -euo pipefail
source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate Argus
export PYTHONPATH=$PWD/python:$PWD  # the editable install points at the main checkout
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python workflows/ng15_nb/scripts/bench_union_grid.py "$@"

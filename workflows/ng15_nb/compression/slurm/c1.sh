#!/bin/bash
# C1 (spec task 2.5): compressed filter vs dense continuous-time oracle on the three C1
# pulsars plus the B1855+09 control, one A100 each. Writes workflows/ng15_nb/results/c1/<pulsar>.json.
#
# The oracle holds ~8 dense n×n float64 arrays (n = TOA count): ~34 GB for B1937+21
# (23,023 TOAs), ~43 GB for J1012+5307 (25,837) and ~78 GB for J1909-3744 (35,037).
# B1937+21 is the strong-red-noise pulsar rather than J1713+0747 (59,389 TOAs, ~225 GB).
#
#   mkdir -p workflows/ng15_nb/outputs/logfiles
#   sbatch workflows/ng15_nb/compression/slurm/c1.sh

#SBATCH --job-name=ng15nb-c1
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-3
#SBATCH --chdir=/fred/oz022/tkimpson/Argus
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/outputs/logfiles/c1_%A_%a.out

set -euo pipefail
source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate Argus
export PYTHONPATH=$PWD/python:$PWD  # the editable install points at the main checkout
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PULSARS=(B1937+21 J1012+5307 J1909-3744 B1855+09)
P=${PULSARS[$SLURM_ARRAY_TASK_ID]}
echo "pulsar ${P}  commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python workflows/ng15_nb/scripts/c1_compression.py --pulsar "${P}" --deltas none 7 14 30

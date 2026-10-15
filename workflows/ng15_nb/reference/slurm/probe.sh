#!/bin/bash
# Cost probe for the Discovery reference ladder (spec task 1.3/1.4) on one A100: times lnL and
# its gradient at 67 pulsars for both kernels, then runs a short NUTS probe (100 + 100, 4 chains)
# so the full rungs can be sized against the ~50 A100-h budget in spec §5.
#
#   sbatch workflows/ng15_nb/reference/slurm/probe.sh            # power law
#   sbatch --export=ALL,KERNEL=ou workflows/ng15_nb/reference/slurm/probe.sh

#SBATCH --job-name=ng15nb-probe
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/outputs/logfiles/probe_%j.out

set -euo pipefail
KERNEL="${KERNEL:-pl}"
REPO="${REPO:-/fred/oz022/tkimpson/Argus}"
CONFIG="${REPO}/workflows/ng15_nb/reference/configs/${KERNEL}.ini"

source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate discovery
export XLA_PYTHON_CLIENT_PREALLOCATE=false
nvidia-smi --query-gpu=name,memory.total --format=csv
git -C "${REPO}" rev-parse HEAD

python "${REPO}/workflows/ng15_nb/reference/run_rung.py" --config "${CONFIG}" --eps 0.5 --benchmark
python "${REPO}/workflows/ng15_nb/reference/run_rung.py" --config "${CONFIG}" --eps 0.0 \
    --num-warmup 100 --num-samples 100 --output-id "probe_${KERNEL}" \
    --out-root "${REPO}/workflows/ng15_nb/outputs/probes"

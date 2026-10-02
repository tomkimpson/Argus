#!/bin/bash
# Per-epoch trace of the marginalized filter at NUTS-frozen positions (see
# scripts/diag_filter_trace.py). 2b eps=0: chain 1 frozen, chain 0 healthy control.
# OU self-gen eps=0: all four frozen.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/diag_filter_trace.sh

#SBATCH --job-name=diag_trace
#SBATCH --account=oz022
#SBATCH --partition=milan
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/diag_trace_%j.out

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="/fred/oz022/tkimpson/Argus/python:${PYTHONPATH}"
export JAX_PLATFORMS=cpu
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD

python -u scripts/diag_filter_trace.py configs/mdc2_flat_eps000.ini \
    outputs/mdc2_flat_eps000/mdc2_flat_eps000_checkpoint.pkl 1 2>&1 | grep -v -E "^PSR:|Warning"
python -u scripts/diag_filter_trace.py configs/mdc2_ou_selfgen_eps000.ini \
    outputs/mdc2_ou_selfgen_eps000/mdc2_ou_selfgen_eps000_checkpoint.pkl 0 3 2>&1 | grep -v -E "^PSR:|Warning"

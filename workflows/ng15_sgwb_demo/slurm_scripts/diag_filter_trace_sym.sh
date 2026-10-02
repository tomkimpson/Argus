#!/bin/bash
# Hypothesis test: does symmetrising P after each Joseph update remove the -inf holes?
# Frozen points (2b ch1, self-gen ch0/ch3) plus healthy 2b ch0 to size the logL shift.
#SBATCH --job-name=diag_trace_sym
#SBATCH --account=oz022
#SBATCH --partition=milan
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/diag_trace_sym_%j.out

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="/fred/oz022/tkimpson/Argus/python:${PYTHONPATH}"
export JAX_PLATFORMS=cpu
python -u scripts/diag_filter_trace.py --symmetrise configs/mdc2_flat_eps000.ini \
    outputs/mdc2_flat_eps000/mdc2_flat_eps000_checkpoint.pkl 1 0 2>&1 | grep -v -E "^PSR:|Warning"
python -u scripts/diag_filter_trace.py --symmetrise configs/mdc2_ou_selfgen_eps000.ini \
    outputs/mdc2_ou_selfgen_eps000/mdc2_ou_selfgen_eps000_checkpoint.pkl 0 3 1 2 2>&1 | grep -v -E "^PSR:|Warning"

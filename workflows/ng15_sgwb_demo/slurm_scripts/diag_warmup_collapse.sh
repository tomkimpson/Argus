#!/bin/bash
# Diagnose the NUTS warmup step-size collapse: 1b (healthy control), 2b eps=0 (chain 1
# collapsed), OU self-gen eps=0 (all four collapsed). See scripts/diag_warmup_collapse.py.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/diag_warmup_collapse.sh

#SBATCH --job-name=diag_collapse
#SBATCH --account=oz022
#SBATCH --partition=milan
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/diag_collapse_%j.out

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="/fred/oz022/tkimpson/Argus/python:${PYTHONPATH}"
export JAX_PLATFORMS=cpu
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD

for RUN in mdc2_d1_flat_eps000 mdc2_flat_eps000 mdc2_ou_selfgen_eps000; do
    python -u scripts/diag_warmup_collapse.py configs/${RUN}.ini \
        outputs/${RUN}/${RUN}_checkpoint.pkl 2>&1 | grep -v -E "^PSR:|Warning"
done

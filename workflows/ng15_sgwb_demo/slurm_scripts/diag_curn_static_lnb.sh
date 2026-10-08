#!/bin/bash
# CPU: static CURN lnB(GW / noise) for GW-model x red-noise-prior pairs on MDC2 1b
# (scripts/diag_curn_static_lnb.py). One worker per pulsar.
#
#   sbatch slurm_scripts/diag_curn_static_lnb.sh
#   sbatch --export=ALL,TAG=mdc2_d1_fine,EXTRA="--gw-step-amp 0.1 --gw-step-shape 0.25" ...

#SBATCH --job-name=diag_curn
#SBATCH --account=oz022
#SBATCH --partition=milan
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=33
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/diag_curn_%j.out

TAG="${TAG:-mdc2_d1}"
source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD

time python -u scripts/diag_curn_static_lnb.py --data data/mdc2_d1_all \
    --truth-noise ../data/IPTA_MockDataChallenge2/group1_psr_noise.json \
    --log10-a -15.18045606445813 --workers 33 ${EXTRA} \
    --out "outputs/diag_curn_static_lnb_${TAG}.json"

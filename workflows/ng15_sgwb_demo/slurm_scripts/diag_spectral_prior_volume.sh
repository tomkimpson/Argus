#!/bin/bash
# CPU diagnostic: per-pulsar lnB(GW/noise) under the OU vs enterprise power-law red-noise
# prior (scripts/diag_spectral_prior_volume.py). Minutes, not hours; a batch job only so it
# doesn't contend on the login node.
#
#   sbatch slurm_scripts/diag_spectral_prior_volume.sh                       # all 33 pulsars
#   sbatch --export=ALL,PULSARS=J1909-3744:J1939+2134,TAG=x slurm_scripts/diag_spectral_prior_volume.sh
#   EXTRA="--step-amp 0.025 --step-ou 0.05 --step-pl 0.05" for the grid-convergence check.

#SBATCH --job-name=diag_spv
#SBATCH --account=oz022
#SBATCH --partition=milan
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/diag_spv_%j.out

# Colon-separated: sbatch --export splits on commas.
PULSARS="${PULSARS:-all}"; PULSARS="${PULSARS//:/,}"
TAG="${TAG:-mdc2_d1}"
source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD

time python -u scripts/diag_spectral_prior_volume.py --data data/mdc2_d1_all \
    --truth-noise ../data/IPTA_MockDataChallenge2/group1_psr_noise.json \
    --log10-a -15.18045606445813 --pulsars "${PULSARS}" ${EXTRA} \
    --out "outputs/diag_spectral_prior_volume_${TAG}.json" \
    --plot "outputs/diag_spectral_prior_volume_${TAG}.png"

#!/bin/bash
# Profile the likelihood along the GW pivot log-PSD at stored 1b posterior draws
# (scripts/diag_gw_amplitude_profile.py). Index 0 = CURN (eps=0), 1 = HD (eps=1).
# Asks whether the low-amplitude posterior excess that makes ln B(CURN/noise) < 0 is a
# non-monotone likelihood at fixed noise, or red noise trading off against the GW.
#
# Submit:
#   sbatch /fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/slurm_scripts/diag_gw_amplitude_profile.sh

#SBATCH --job-name=gwprof
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-1
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/gwprof_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
RUNS=(mdc2_d1_flat_uprior_eps000 mdc2_d1_flat_uprior_eps100)
RUN="${RUNS[${SLURM_ARRAY_TASK_ID:-0}]}"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
python -c "import argus.parameter_sampling as p; print('argus from:', p.__file__)"
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD
nvidia-smi -L

time python -u "${ROOT}/scripts/diag_gw_amplitude_profile.py" \
    --run "${ROOT}/outputs/${RUN}" \
    --out "${ROOT}/outputs/diag_gw_amplitude_profile_${RUN}"

#!/bin/bash
# Kalman vs static-Gaussian dlogL(s) at the same posterior draws
# (scripts/diag_kalman_vs_static.py). The Kalman side needs the GPU; the static side is
# numpy on the host.
#
#   sbatch slurm_scripts/diag_kalman_vs_static.sh
#   sbatch --export=ALL,TP=diffuse slurm_scripts/diag_kalman_vs_static.sh

#SBATCH --job-name=kal_vs_static
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/kal_vs_static_%j.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
RUN="${RUN:-mdc2_d1_flat_uprior_eps000}"
TP="${TP:-}"  # Kalman timing prior override: informative | diffuse
SUFFIX="${TP:+_tp${TP}}"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
python -c "import argus.parameter_sampling as p; print('argus from:', p.__file__)"
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD
nvidia-smi -L

time python -u "${ROOT}/scripts/diag_kalman_vs_static.py" \
    --run "${ROOT}/outputs/${RUN}" \
    ${TP:+--timing-prior ${TP}} \
    --out "${ROOT}/outputs/diag_kalman_vs_static_${RUN}${SUFFIX}"

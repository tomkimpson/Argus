#!/bin/bash
# Sky-scramble ensemble for the eps = 0 path-sampling integrand (CPU only, no new sampling).
# Generates N scrambles to a NEW archive (the single-scramble mdc2_d1_scrambles.npz that
# lnb_scrambled_readout.sh hardcodes is left alone), then scores them all against the
# existing eps000 draws of a signal ladder via scripts/scramble_ensemble_eps0.py.
#
# Usage:
#   sbatch slurm_scripts/scramble_ensemble_eps0.sh
#   LADDER   optional, default mdc2_d1_flat_diffuse (the +0.693 ladder)
#   N        optional, default 1000 scrambles (scoring is a dot product; N is free)
#   REPO_PY  optional, default the Argus-diffuse worktree (fa65942) that read out LADDER

#SBATCH --job-name=scr_ens_eps0
#SBATCH --account=oz022
#SBATCH --partition=milan
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/scr_ens_eps0_%j.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
LADDER="${LADDER:-mdc2_d1_flat_diffuse}"
N="${N:-1000}"
REPO_PY="${REPO_PY:-/fred/oz022/tkimpson/Argus-diffuse/python}"
ENSEMBLE="${ROOT}/data/scrambles/mdc2_d1_scrambles_ens${N}.npz"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"
export JAX_PLATFORMS=cpu

echo "=== scramble ensemble (eps0 proxy): ${LADDER}, N=${N} ==="
echo "library: ${REPO_PY}"; git -C "${REPO_PY}/.." rev-parse --short HEAD

set -e
if [ ! -f "${ENSEMBLE}" ]; then
    python -u "${ROOT}/scripts/sky_scrambles.py" \
        --data-path "${ROOT}/data/mdc2_d1_all" --n-scrambles "${N}" --seed 0 \
        --out "${ENSEMBLE}"
fi

time python -u "${ROOT}/scripts/scramble_ensemble_eps0.py" \
    --rung "${ROOT}/outputs/${LADDER}_eps000/${LADDER}_eps000_results.nc" \
    --config "${ROOT}/configs/${LADDER}_eps000.ini" \
    --scrambles "${ENSEMBLE}" \
    --reference-scramble "${ROOT}/data/scrambles/mdc2_d1_scrambles.npz" \
    --out "${ROOT}/outputs/scramble_ensemble_eps0_${LADDER}"

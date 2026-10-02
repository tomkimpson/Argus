#!/bin/bash
# CPU readout of a SCRAMBLED path-sampling ladder via scripts/lnb_scrambled.py, which re-applies
# the scramble before evaluating the integrand. lnb_readout.sh must never be used for these: its
# --evaluate rebuilds the data with the TRUE ORF and returns a meaningless number.
#
# Usage:
#   sbatch --export=ALL,LADDER=mdc2_d1_null_psym,REPO_PY=/fred/oz022/tkimpson/Argus-fix-psym/python \
#       slurm_scripts/lnb_scrambled_readout.sh
#   LADDER   required, the run prefix -> outputs/<LADDER>_eps*/
#   OUT      optional, default lnb_path_sampling_<LADDER>.json
#   REPO_PY  optional, the library to evaluate with (default: the main checkout)

#SBATCH --job-name=lnb_scr_readout
#SBATCH --account=oz022
#SBATCH --partition=milan
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/lnb_scr_readout_%j.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO_PY="${REPO_PY:-/fred/oz022/tkimpson/Argus/python}"
SCRAMBLE_NPZ="${ROOT}/data/scrambles/mdc2_d1_scrambles.npz"
SCRAMBLE_INDEX="${SCRAMBLE_INDEX:-0}"

if [ -z "${LADDER}" ]; then
    echo "ERROR: set LADDER, e.g. sbatch --export=ALL,LADDER=mdc2_d1_null_psym $0" >&2; exit 1
fi
OUT="${OUT:-lnb_path_sampling_${LADDER}.json}"
CONFIG="${ROOT}/configs/${LADDER}_eps000.ini"
for TAG in 000 025 050 075 100; do
    NC="${ROOT}/outputs/${LADDER}_eps${TAG}/${LADDER}_eps${TAG}_results.nc"
    if [ ! -f "${NC}" ]; then
        echo "ERROR: ${NC} missing -- rung eps${TAG} has not finished." >&2; exit 1
    fi
done

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== scrambled readout: ${LADDER} (scramble ${SCRAMBLE_INDEX}) -> outputs/${OUT} ==="
echo "library: ${REPO_PY}"; git -C "${REPO_PY}/.." rev-parse --short HEAD

time JAX_PLATFORMS=cpu python -u "${ROOT}/scripts/lnb_scrambled.py" \
    --scramble-npz "${SCRAMBLE_NPZ}" --scramble-index "${SCRAMBLE_INDEX}" \
    --from-runs "${ROOT}/outputs/${LADDER}_eps*/${LADDER}_eps*_results.nc" \
    --evaluate "${CONFIG}" \
    --batch-size 100 \
    --out "${ROOT}/outputs/${OUT}"

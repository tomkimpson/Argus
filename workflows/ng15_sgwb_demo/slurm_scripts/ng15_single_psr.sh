#!/bin/bash
# OU-adequacy step 3: single-pulsar Argus OU red-noise fit on real NG15 wideband data,
# one array task per pulsar staged under data/ng15_singles/ (slurm_scripts/ng15_single_prep.sh).
#
# Same design as M1 Stage A (configs/ng15_single_psr.ini): GW fixed negligible, flat
# per-pulsar OU priors, EFAC/EQUAD fixed. Read out with scripts/compare_single_psr_ou.py.
#
# Submit (set --array to 0..N-1 for the N staged pulsars):
#   sbatch --array=0-<N-1> workflows/ng15_sgwb_demo/slurm_scripts/ng15_single_psr.sh

#SBATCH --job-name=ng15_single_psr
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:1
# NG15 single pulsars have up to ~190 binned epochs vs MDC2's 185; MDC2 took 31-42 min.
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ng15_single_psr_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO_PY=/fred/oz022/tkimpson/Argus/python

mapfile -t PSRS < <(ls -d "${ROOT}"/data/ng15_singles/*/ | xargs -n1 basename | sort)
if [ "${#PSRS[@]}" -eq 0 ]; then
    echo "ERROR: no staged pulsar directories under ${ROOT}/data/ng15_singles/" >&2
    echo "Run: sbatch slurm_scripts/ng15_single_prep.sh" >&2
    exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#PSRS[@]}" ]; then
    echo "ERROR: array task ${SLURM_ARRAY_TASK_ID} >= ${#PSRS[@]} staged pulsars" >&2
    exit 1
fi
PSR="${PSRS[$SLURM_ARRAY_TASK_ID]}"

PSR_DIR="${ROOT}/data/ng15_singles/${PSR}/"
PSR_NOISE="${ROOT}/data/ng15_singles/${PSR}/psr_noise.json"
# Derived config must live OUTSIDE the run's own output dir (run_inference copies the
# config into it) but inside the workflow tree, with ABSOLUTE paths.
RUN="${ROOT}/outputs/derived_configs/ng15_single_${PSR}.ini"

mkdir -p "${ROOT}/outputs/derived_configs" "${ROOT}/outputs/logfiles"

sed -e "s|^data_path = .*|data_path = ${PSR_DIR}|" \
    -e "s|^noise_params_path = .*|noise_params_path = ${PSR_NOISE}|" \
    -e "s|^output_id = .*|output_id = ng15_single_${PSR}|" \
    "${ROOT}/configs/ng15_single_psr.ini" > "${RUN}"

echo "=== NG15 single-pulsar OU fit, task ${SLURM_ARRAY_TASK_ID}: ${PSR} ==="
grep -E "^data_path|^noise_params_path|^output_id|^log10_sigma_p|^log10_gamma_p" "${RUN}"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import argus.prior_models as pm; print('argus from:', pm.__file__)"
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD
nvidia-smi -L

echo "=== running: ${PSR} ==="
time python -u "${ROOT}/run_analysis.py" "${RUN}"

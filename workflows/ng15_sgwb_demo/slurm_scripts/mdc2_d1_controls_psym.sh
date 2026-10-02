#!/bin/bash
# Re-validate the two 1b controls on the symmetrised Joseph update (branch
# fix/kalman-covariance-symmetry). Both were computed on the unsymmetrised filter:
#   KIND=null   sky-scramble null      (old lnB -0.766, outputs/lnb_path_sampling_d1_null.json)
#   KIND=nogwb  no-injection control   (old lnB -0.013, outputs/lnb_path_sampling_mdc2_d1_nogwb.json)
# The 1b flat ladder itself held on the fixed filter (3.0432 vs 3.0431,
# outputs/lnb_path_sampling_mdc2_d1_flat_psym.json). Templates, scramble and data are the
# originals; only the library and the output ids (mdc2_d1_<KIND>_psym_eps<TAG>) change.
#
# Submit:
#   sbatch --export=ALL,KIND=null  .../slurm_scripts/mdc2_d1_controls_psym.sh
#   sbatch --export=ALL,KIND=nogwb .../slurm_scripts/mdc2_d1_controls_psym.sh
# Read out:
#   nogwb: sbatch --export=ALL,LADDER=mdc2_d1_nogwb_psym,REPO_PY=<worktree>/python lnb_readout.sh
#   null:  sbatch --export=ALL,LADDER=mdc2_d1_null_psym,REPO_PY=<worktree>/python lnb_scrambled_readout.sh
#   (never lnb_readout.sh for the null: it would integrate against the TRUE ORF.)

#SBATCH --job-name=d1_ctrl_psym
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/d1_ctrl_psym_%x_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO_PY=/fred/oz022/tkimpson/Argus-fix-psym/python
FROZEN="${ROOT}/configs/frozen/evidence_procedure_v1.json"
SCRAMBLE_NPZ="${ROOT}/data/scrambles/mdc2_d1_scrambles.npz"
SCRAMBLE_INDEX=0

case "${KIND}" in
    null)  DATA=mdc2_d1_all ;;
    nogwb) DATA=mdc2_d1_nogwb ;;
    *) echo "ERROR: set KIND=null or KIND=nogwb (got '${KIND}')." >&2; exit 1 ;;
esac

EPS_VALUES=(0.0 0.25 0.5 0.75 1.0)
EPS_TAGS=(000 025 050 075 100)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
EPS="${EPS_VALUES[$IDX]}"
TAG="${EPS_TAGS[$IDX]}"
if [ -z "${EPS}" ]; then
    echo "ERROR: no eps value for array index ${IDX}" >&2; exit 1
fi

OUT_ID="mdc2_d1_${KIND}_psym_eps${TAG}"
CONFIG="${ROOT}/configs/${OUT_ID}.ini"
sed -e "s/__EPS_VALUE__/${EPS}/" -e "s/__EPS_TAG__/${TAG}/" \
    -e "s/^output_id = .*/output_id = ${OUT_ID}/" \
    "${ROOT}/configs/mdc2_d1_${KIND}_rung.ini.template" > "${CONFIG}"

if grep -qE "^\s*empirical_priors_path\s*=\s*\S" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} sets empirical_priors_path." >&2; exit 1
fi
if ! grep -qE "^\s*red_noise_prior\s*=\s*flat\s*$" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} does not set red_noise_prior = flat." >&2; exit 1
fi
if ! grep -qE "^\s*data_path\s*=\s*\.\./data/${DATA}\s*$" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} does not point at ../data/${DATA}." >&2; exit 1
fi
if [ "${KIND}" = null ] && [ ! -f "${SCRAMBLE_NPZ}" ]; then
    echo "ERROR: ${SCRAMBLE_NPZ} not found." >&2; exit 1
fi

mkdir -p "${ROOT}/outputs/logfiles"
source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== env check (${OUT_ID}, eps=${EPS}) ==="
python -c "import argus.jax_kalman_filter as k, inspect; print('argus from:', k.__file__); \
print('symmetrised:', 'P = 0.5 * (P + P.T)' in inspect.getsource(k._update_marginal))"
git -C /fred/oz022/tkimpson/Argus-fix-psym rev-parse --short HEAD
nvidia-smi -L
grep -E "^(data_path|orf_path|orf_epsilon_value|red_noise_prior|output_id)" "${CONFIG}"

if [ "${KIND}" = null ]; then
    time python -u "${ROOT}/run_scrambled.py" "${CONFIG}" \
        --scramble-npz "${SCRAMBLE_NPZ}" --scramble-index "${SCRAMBLE_INDEX}"
else
    time python -u "${ROOT}/run_analysis.py" "${CONFIG}"
fi
STATUS=$?
if [ ${STATUS} -eq 0 ]; then
    python "${ROOT}/scripts/stamp_provenance.py" --output-dir "${ROOT}/outputs/${OUT_ID}" \
        --config "${CONFIG}" --frozen "${FROZEN}"
fi
exit ${STATUS}

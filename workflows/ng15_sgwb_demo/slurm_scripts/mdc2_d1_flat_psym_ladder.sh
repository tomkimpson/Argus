#!/bin/bash
# Re-validate the MDC2 1b flat-prior ladder on the symmetrised Joseph update (branch
# fix/kalman-covariance-symmetry). lnB = +3.04 (outputs/mdc2_d1_flat_eps*) was computed on the
# unsymmetrised filter, whose covariance drift corrupted finite likelihoods by up to 9 nats and
# opened -inf holes on 2b. This reruns all five rungs from the SAME template; only the library
# and the output ids change.
#
# Output ids are mdc2_d1_flat_psym_eps<TAG> (not ..._eps<TAG>_psym) so lnb_readout.sh's
# ${LADDER}_eps* glob picks up exactly these five rungs and none of the old ones.
#
# The library comes from the fix worktree (PYTHONPATH is prepended, which beats the editable
# install of the main checkout). Data, configs and outputs live in the main checkout.
#
# Submit:
#   sbatch /fred/oz022/tkimpson/Argus-fix-psym/workflows/ng15_sgwb_demo/slurm_scripts/mdc2_d1_flat_psym_ladder.sh
# Read out once all five exist (the readout must use the fixed library too):
#   sbatch --export=ALL,LADDER=mdc2_d1_flat_psym,REPO_PY=/fred/oz022/tkimpson/Argus-fix-psym/python \
#       /fred/oz022/tkimpson/Argus-fix-psym/workflows/ng15_sgwb_demo/slurm_scripts/lnb_readout.sh

#SBATCH --job-name=d1_flat_psym
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/d1_flat_psym_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO_PY=/fred/oz022/tkimpson/Argus-fix-psym/python
FROZEN="${ROOT}/configs/frozen/evidence_procedure_v1.json"

EPS_VALUES=(0.0 0.25 0.5 0.75 1.0)
EPS_TAGS=(000 025 050 075 100)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
EPS="${EPS_VALUES[$IDX]}"
TAG="${EPS_TAGS[$IDX]}"
if [ -z "${EPS}" ]; then
    echo "ERROR: no eps value for array index ${IDX}" >&2; exit 1
fi

OUT_ID="mdc2_d1_flat_psym_eps${TAG}"
CONFIG="${ROOT}/configs/${OUT_ID}.ini"
sed -e "s/__EPS_VALUE__/${EPS}/" -e "s/__EPS_TAG__/${TAG}/" \
    -e "s/^output_id = .*/output_id = ${OUT_ID}/" \
    "${ROOT}/configs/mdc2_d1_flat_rung.ini.template" > "${CONFIG}"

# empirical_priors_path overrides red_noise_prior, so its absence must be enforced.
if grep -qE "^\s*empirical_priors_path\s*=\s*\S" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} sets empirical_priors_path." >&2; exit 1
fi
if ! grep -qE "^\s*red_noise_prior\s*=\s*flat\s*$" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} does not set red_noise_prior = flat." >&2; exit 1
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

time python -u "${ROOT}/run_analysis.py" "${CONFIG}"
STATUS=$?
if [ ${STATUS} -eq 0 ]; then
    python "${ROOT}/scripts/stamp_provenance.py" --output-dir "${ROOT}/outputs/${OUT_ID}" \
        --config "${CONFIG}" --frozen "${FROZEN}"
fi
exit ${STATUS}

#!/bin/bash
# Rerun the eps = 0 rung on MDC2 2b and on the OU self-generated spike dataset with the
# symmetrised Joseph update (branch fix/kalman-covariance-symmetry). Both earlier runs froze
# chains in -inf likelihood holes. This rerun asks whether all four chains now move, and only
# then applies the spike's pre-registered decision rule.
#
# The library comes from the fix worktree (PYTHONPATH is prepended, which beats both the
# editable install of the main checkout and run_analysis.py's sys.path.append). Data, configs
# and outputs live in the main checkout, because they are gitignored. The output ids take a
# _psym suffix, so the frozen runs are kept for comparison.
#
# Submit (array index 0 = 2b, 1 = OU self-gen):
#   sbatch /fred/oz022/tkimpson/Argus-fix-psym/workflows/ng15_sgwb_demo/slurm_scripts/rerun_eps0_psym.sh

#SBATCH --job-name=eps0_psym
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-1
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/eps0_psym_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO_PY=/fred/oz022/tkimpson/Argus-fix-psym/python
FROZEN="${ROOT}/configs/frozen/evidence_procedure_v1.json"

TEMPLATES=(mdc2_flat_rung mdc2_ou_selfgen_rung)
OUT_IDS=(mdc2_flat_eps000_psym mdc2_ou_selfgen_eps000_psym)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
TEMPLATE="${ROOT}/configs/${TEMPLATES[$IDX]}.ini.template"
OUT_ID="${OUT_IDS[$IDX]}"
CONFIG="${ROOT}/configs/${OUT_ID}.ini"

sed -e "s/__EPS_VALUE__/0.0/" -e "s/__EPS_TAG__/000/" \
    -e "s/^output_id = .*/output_id = ${OUT_ID}/" "${TEMPLATE}" > "${CONFIG}"

# The same guards as the ladders: empirical_priors_path overrides red_noise_prior.
if grep -qE "^\s*empirical_priors_path\s*=\s*\S" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} sets empirical_priors_path." >&2; exit 1
fi
if ! grep -qE "^\s*red_noise_prior\s*=\s*flat\s*$" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} does not set red_noise_prior = flat." >&2; exit 1
fi

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== env check (${OUT_ID}) ==="
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

#!/bin/bash
# Rerun the MDC2 flat-prior results with red_noise_prior = flat as a TRUE bounded Uniform
# (branch fix/flat-red-noise-prior). Every earlier "flat" NUTS run sampled red noise from an
# unbounded N(mid, (hi-lo)/(6 sqrt N)); at 33 psr that is log10γp ~ N(-9, 0.17) and
# log10σp ~ N(-16, 0.23), which nearly pins the red noise. Whether the 1b detection
# (lnB +3.04) and the 2b pivot excess (+1.84 dex, one-sided) survive a genuinely flat prior is
# what this rerun decides.
#
# Array indices 0-4 = the 1b ladder (eps 0, 0.25, 0.5, 0.75, 1.0) from the same template as
# mdc2_d1_flat_psym_*. Index 5 = the 2b eps = 0 rung from the same template as
# mdc2_flat_eps000_psym. Only the library and the output ids change.
#
# The library comes from a dedicated worktree of the fix branch (PYTHONPATH is prepended, which
# beats the editable install of the main checkout), so checking out another branch here cannot
# change the code under a running job. Data, configs and outputs live in the main checkout.
#
# Submit:
#   sbatch /fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/slurm_scripts/mdc2_uprior_rerun.sh
# Read out the 1b ladder once all five rungs exist (the readout must use the same library):
#   sbatch --export=ALL,LADDER=mdc2_d1_flat_uprior,REPO_PY=/fred/oz022/tkimpson/Argus-uprior/python \
#       /fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/slurm_scripts/lnb_readout.sh

#SBATCH --job-name=uprior
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/uprior_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
LIB=/fred/oz022/tkimpson/Argus-uprior
REPO_PY="${LIB}/python"
FROZEN="${ROOT}/configs/frozen/evidence_procedure_v1.json"

EPS_VALUES=(0.0 0.25 0.5 0.75 1.0 0.0)
EPS_TAGS=(000 025 050 075 100 000)
TEMPLATES=(mdc2_d1_flat_rung mdc2_d1_flat_rung mdc2_d1_flat_rung mdc2_d1_flat_rung \
    mdc2_d1_flat_rung mdc2_flat_rung)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
EPS="${EPS_VALUES[$IDX]}"
TAG="${EPS_TAGS[$IDX]}"
if [ -z "${EPS}" ]; then
    echo "ERROR: no eps value for array index ${IDX}" >&2; exit 1
fi
if [ "${IDX}" -lt 5 ]; then
    OUT_ID="mdc2_d1_flat_uprior_eps${TAG}"
else
    OUT_ID="mdc2_flat_uprior_eps${TAG}"
fi
CONFIG="${ROOT}/configs/${OUT_ID}.ini"
sed -e "s/__EPS_VALUE__/${EPS}/" -e "s/__EPS_TAG__/${TAG}/" \
    -e "s/^output_id = .*/output_id = ${OUT_ID}/" \
    "${ROOT}/configs/${TEMPLATES[$IDX]}.ini.template" > "${CONFIG}"

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
python -c "import argus.parameter_sampling as p; print('argus from:', p.__file__); \
print('uniform red-noise prior:', hasattr(p, 'sample_uniform_parameters'))"
git -C "${LIB}" rev-parse --short HEAD
nvidia-smi -L
grep -E "^(data_path|orf_path|orf_epsilon_value|red_noise_prior|output_id)" "${CONFIG}"

time python -u "${ROOT}/run_analysis.py" "${CONFIG}"
STATUS=$?
if [ ${STATUS} -eq 0 ]; then
    python "${ROOT}/scripts/stamp_provenance.py" --output-dir "${ROOT}/outputs/${OUT_ID}" \
        --config "${CONFIG}" --frozen "${FROZEN}"
fi
exit ${STATUS}

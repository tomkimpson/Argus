#!/bin/bash
# Run the MDC2 2b path-sampling ladder under the DIFFUSE timing prior. Array indices 0-1 are
# eps = 0 (CURN) and eps = 1 (HD); 2-4 are the interior rungs 0.25, 0.5, 0.75.
# The configs (configs/mdc2_flat_diffuse_eps*.ini) are mdc2_flat_uprior_eps000.ini with only
# orf_epsilon_value, timing_prior = diffuse and output_id changed. Apart from data_path and
# output_id they are identical to the 1b ladder mdc2_d1_flat_diffuse_eps*.ini.
#
# Same pinned library as the 1b diffuse ladder (worktree Argus-diffuse at fa65942). Between
# fa65942 and main the library differs only in the timing_prior DEFAULT and the checkpoint
# fingerprint, so with timing_prior = diffuse set explicitly the likelihood is the same as main.
# The worktree already exists; to recreate it:
#   git -C /fred/oz022/tkimpson/Argus worktree add --detach /fred/oz022/tkimpson/Argus-diffuse fa65942
# Do not remove that worktree while any rung is queued or running.
#
# Submit eps 0 first to check that 2b samples cleanly before spending the other four rungs:
#   sbatch --array=0 slurm_scripts/mdc2_2b_diffuse.sh
#   sbatch --array=1-4 slurm_scripts/mdc2_2b_diffuse.sh
# lnB(HD / CURN) once all five exist (same library):
#   sbatch --export=ALL,LADDER=mdc2_flat_diffuse,REPO_PY=/fred/oz022/tkimpson/Argus-diffuse/python \
#       slurm_scripts/lnb_readout.sh

#SBATCH --job-name=2b-diffuse
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/2b_diffuse_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
LIB=/fred/oz022/tkimpson/Argus-diffuse
REPO_PY="${LIB}/python"
FROZEN="${ROOT}/configs/frozen/evidence_procedure_v1.json"

TAGS=(000 100 025 050 075)
TAG="${TAGS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUT_ID="mdc2_flat_diffuse_eps${TAG}"
CONFIG="${ROOT}/configs/${OUT_ID}.ini"
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: ${CONFIG} not found." >&2; exit 1
fi
if ! sed 's/#.*//' "${CONFIG}" | grep -qE "^\s*timing_prior\s*=\s*diffuse\s*$"; then
    echo "ERROR: ${CONFIG} does not set timing_prior = diffuse." >&2; exit 1
fi
if grep -qE "^\s*empirical_priors_path\s*=\s*\S" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} sets empirical_priors_path." >&2; exit 1
fi

mkdir -p "${ROOT}/outputs/logfiles"
source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
if [ ! -d "${REPO_PY}/argus" ]; then
    echo "ERROR: ${REPO_PY} missing; create the pinned worktree (see header)." >&2; exit 1
fi
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== env check (${OUT_ID}) ==="
python -c "import argus.parameter_sampling as p; print('argus from:', p.__file__)"
git -C "${LIB}" rev-parse --short HEAD
nvidia-smi -L
grep -E "^(data_path|orf_path|orf_epsilon_value|red_noise_prior|timing_prior|output_id)" "${CONFIG}"

time python -u "${ROOT}/run_analysis.py" "${CONFIG}"
STATUS=$?
if [ ${STATUS} -eq 0 ]; then
    python "${ROOT}/scripts/stamp_provenance.py" --output-dir "${ROOT}/outputs/${OUT_ID}" \
        --config "${CONFIG}" --frozen "${FROZEN}" --repo "${LIB}"
fi
exit ${STATUS}

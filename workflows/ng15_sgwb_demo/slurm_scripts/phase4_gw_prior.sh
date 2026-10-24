#!/bin/bash
# Spec Phase 4, task 4.3 (issue #125): the GW ridge prior, Gaussian vs bounded Uniform, on the
# same likelihood. Six rungs, each run on its own pinned library:
#   0 mdc2_d1_flat_uniformgw_eps000   1b CURN     uniform GW prior   (LIB_NEW)
#   1 mdc2_d1_flat_uniformgw_eps100   1b HD       uniform GW prior   (LIB_NEW)
#   2 mdc2_flat_uniformgw_eps000      2b CURN     uniform GW prior   (LIB_NEW)
#   3 mdc2_d1_flat_gaussgw_eps000     1b CURN     old Gaussian prior (LIB_OLD)
#   4 mdc2_d1_flat_gaussgw_eps100     1b HD       old Gaussian prior (LIB_OLD)
#   5 mdc2_flat_gaussgw_eps000        2b CURN     old Gaussian prior (LIB_OLD)
# The configs are the PR #123 / #124 diffuse rungs with only output_id (and, for the uniform
# runs, the dense-block site names) changed. The committed diffuse outputs predate PR #129's
# Lambda-jitter fix, so the Gaussian twins are rerun here rather than reused.
#
# Libraries (worktrees; do not remove while a rung is queued or running):
#   LIB_OLD = main at 7616aa7 (after PR #129, before the #125 change)
#     git -C /fred/oz022/tkimpson/Argus worktree add --detach /fred/oz022/tkimpson/Argus-phase4-old 7616aa7
#   LIB_NEW = the task 4.1 commit on branch phase-4-gw-uniform-priors
#     git -C /fred/oz022/tkimpson/Argus worktree add --detach /fred/oz022/tkimpson/Argus-phase4-new a3484db
# The two differ only in the GW prior sites (git diff 7616aa7 a3484db -- python/).
#
# Data (gitignored): ../data/mdc2_d1_all (1b) and ../data/mdc2_all (2b); see
# mdc2_d1_diffuse_rerun.sh and mdc2_2b_diffuse.sh for how to build them.
#
#   sbatch slurm_scripts/phase4_gw_prior.sh
# As run: rungs 0-2 as above (job 18961537). Rungs 3-5 waited days for a 4-GPU node, so
# they were cancelled and resubmitted on one GPU (job 19055225), chains then sequential:
#   sbatch --array=3-5 --gres=gpu:a100:1 --time=36:00:00 slurm_scripts/phase4_gw_prior.sh
# Readout: scripts/phase4_compare_gw_prior.py.

#SBATCH --job-name=p4-gwprior
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-5
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/p4_gwprior_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
LIB_OLD=/fred/oz022/tkimpson/Argus-phase4-old
LIB_NEW=/fred/oz022/tkimpson/Argus-phase4-new
FROZEN="${ROOT}/configs/frozen/evidence_procedure_v1.json"

IDS=(mdc2_d1_flat_uniformgw_eps000 mdc2_d1_flat_uniformgw_eps100 mdc2_flat_uniformgw_eps000
     mdc2_d1_flat_gaussgw_eps000 mdc2_d1_flat_gaussgw_eps100 mdc2_flat_gaussgw_eps000)
OUT_ID="${IDS[${SLURM_ARRAY_TASK_ID:-0}]}"
case "${OUT_ID}" in
    *uniformgw*) LIB="${LIB_NEW}" ;;
    *gaussgw*) LIB="${LIB_OLD}" ;;
esac
REPO_PY="${LIB}/python"
CONFIG="${ROOT}/configs/${OUT_ID}.ini"
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: ${CONFIG} not found." >&2; exit 1
fi
if ! sed 's/#.*//' "${CONFIG}" | grep -qE "^\s*timing_prior\s*=\s*diffuse\s*$"; then
    echo "ERROR: ${CONFIG} does not set timing_prior = diffuse." >&2; exit 1
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
grep -E "^(data_path|dense_mass_blocks|orf_epsilon_value|red_noise_prior|timing_prior|output_id)" "${CONFIG}"

time python -u "${ROOT}/run_analysis.py" "${CONFIG}"
STATUS=$?
if [ ${STATUS} -eq 0 ]; then
    python "${ROOT}/scripts/stamp_provenance.py" --output-dir "${ROOT}/outputs/${OUT_ID}" \
        --config "${CONFIG}" --frozen "${FROZEN}" --repo "${LIB}"
fi
exit ${STATUS}

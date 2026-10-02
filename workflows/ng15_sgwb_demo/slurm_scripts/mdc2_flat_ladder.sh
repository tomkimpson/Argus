#!/bin/bash
# GENERALISATION TEST: the full 5-rung flat-prior ladder on MDC2 dataset 2b -- the dataset
# that actually contains per-pulsar red noise.
#
# The flat-prior result everything now rests on was obtained on 1b, and the IPTA MDC2 dataset
# table lists g1.d1a(b) as WN only against g1.d2a(b) as WN+RN. So on 1b the flat model was the
# TRUE model: there was no red noise for it to get wrong. 2b is the same 33 pulsars, epochs and
# white noise with red noise injected, and is the only test of whether the procedure about to
# be frozen for real NG15 data generalises.
#
# READ THE AMPLITUDE, NOT THE BAYES FACTOR. 2b is a published non-detection for every method
# the IPTA tried (Hazboun et al., arXiv:1912.12939), so lnB ~ 0 is expected and is not a
# failure; our empirical-prior ladder already gave +0.175 +/- 0.017 and ratioing their HD/CRN
# rows gives ~ -0.13. What decides the question is whether the pivot log-PSD posterior is an
# informative upper limit near their < 1.4e-15 (posterior sd well below prior sd, as on 1b
# where it went 0.99 -> 0.21 of prior sd) or simply the prior handed back.
#
# configs/mdc2_flat_rung.ini.template is configs/mdc2_d1_flat_rung.ini.template with exactly
# two values changed (data_path, output_id), verified by diff, so 1b and 2b differ by the
# dataset and nothing else.
#
# NOTE ON THE 2b FEATHERS. data/mdc2_all was for a long time a SYMLINK into a treehouse
# worktree outside /fred. This script refuses to run against a symlink: five 2-hour A100 jobs
# should not depend on a path that can vanish, and the existing 2b results would be
# unreproducible if it did. Re-ingest into a real directory first:
#   JAX_PLATFORMS=cpu python scripts/ingest_par_tim.py \
#       workflows/data/IPTA_MockDataChallenge2/dataset_2b \
#       workflows/ng15_sgwb_demo/data/mdc2_all          # from the repo root
#
# Read the Bayes factor afterwards as a SLURM CPU job, NOT on the login node:
#   JAX_PLATFORMS=cpu python scripts/lnb_path_sampling.py \
#       --from-runs 'outputs/mdc2_flat_eps*/mdc2_flat_eps*_results.nc' \
#       --evaluate configs/mdc2_flat_eps000.ini \
#       --batch-size 100 \
#       --out outputs/lnb_path_sampling_flat.json
# --batch-size 100 is mandatory or the integrand evaluation is OOM-killed. For the truth
# check, scripts/check_mdc2_truth.py already defaults to 2b's log10_A = -14.886056647693163,
# so do NOT pass the --log10-a override the 1b scripts carry.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_flat_ladder.sh

#SBATCH --job-name=mdc2_flat
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:4
# Sized from the 1b flat and null ladders' measured usage (9.4 GB peak of 32 GB; 1h30m-2h23m
# of 16 h). gpu:4 is fixed by num_chains = 4. Checkpointing is on, so a timeout costs one
# segment rather than the run. 2b's red noise makes the posterior harder than 1b's, so the
# wall-clock allowance is kept at the 1b ladder's original 16 h rather than the null ladder's
# trimmed 8 h.
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/mdc2_flat_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO_PY=/fred/oz022/tkimpson/Argus/python
FROZEN="${ROOT}/configs/frozen/evidence_procedure_v1.json"

# The frozen uniform 5-point grid. Uniform spacing is what lets lnb_path_sampling.py
# coarsen twice (5 -> 3 -> 2) and estimate its own discretisation error by Romberg
# extrapolation; it enforces MIN_RUNGS = 5.
EPS_VALUES=(0.0 0.25 0.5 0.75 1.0)
EPS_TAGS=(000 025 050 075 100)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
EPS="${EPS_VALUES[$IDX]}"
TAG="${EPS_TAGS[$IDX]}"

if [ -z "${EPS}" ]; then
    echo "ERROR: no eps value for array index ${IDX}" >&2
    exit 1
fi

if [ ! -d "${ROOT}/data/mdc2_all" ]; then
    echo "ERROR: ${ROOT}/data/mdc2_all not found." >&2
    exit 1
fi
if [ -L "${ROOT}/data/mdc2_all" ]; then
    echo "ERROR: ${ROOT}/data/mdc2_all is a symlink. Re-ingest dataset_2b into a real" >&2
    echo "       directory under /fred before spending A100 hours on it (see header)." >&2
    exit 1
fi

mkdir -p "${ROOT}/outputs/logfiles"

# Derive this rung's config from the template.
CONFIG="${ROOT}/configs/mdc2_flat_eps${TAG}.ini"
sed -e "s/__EPS_VALUE__/${EPS}/" -e "s/__EPS_TAG__/${TAG}/" \
    "${ROOT}/configs/mdc2_flat_rung.ini.template" > "${CONFIG}"

# These guards matter MORE here than on 1b: configs/mdc2_ladder_rung.ini.template is the 2b
# EMPIRICAL ladder, it sits in the same directory, and it differs from this file only in the
# noise-prior block. empirical_priors_path takes precedence over red_noise_prior in
# get_pulsar_noise_priors, so a stray copy of that line would silently turn this run back into
# the empirical ladder it is meant to be compared against.
if grep -qE "^\s*empirical_priors_path\s*=\s*\S" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} still sets empirical_priors_path; this is not the flat ladder." >&2
    exit 1
fi
if ! grep -qE "^\s*red_noise_prior\s*=\s*flat\s*$" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} does not set red_noise_prior = flat." >&2
    exit 1
fi
if ! grep -qE "^\s*data_path\s*=\s*\.\./data/mdc2_all\s*$" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} does not point at ../data/mdc2_all." >&2
    exit 1
fi

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== env check (2b flat rung ${IDX}: eps=${EPS}) ==="
which python
python -c "import argus.gravitational_waves as gw; print('argus from:', gw.__file__); print('correlation_path available:', hasattr(gw, 'correlation_path'))"
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD
nvidia-smi -L
grep -E "data_path|orf_path|orf_epsilon_value|red_noise_prior|empirical|output_id" "${CONFIG}"

echo "=== running 2b flat rung eps=${EPS} ==="
time python -u "${ROOT}/run_analysis.py" "${CONFIG}"
STATUS=$?

# Task 1.10: every run records which frozen evidence procedure it was produced under.
if [ ${STATUS} -eq 0 ]; then
    python "${ROOT}/scripts/stamp_provenance.py" \
        --output-dir "${ROOT}/outputs/mdc2_flat_eps${TAG}" \
        --config "${CONFIG}" \
        --frozen "${FROZEN}"
fi

exit ${STATUS}

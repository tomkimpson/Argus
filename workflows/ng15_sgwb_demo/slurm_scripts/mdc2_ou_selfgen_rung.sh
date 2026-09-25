#!/bin/bash
# SPIKE step 2 (notes/SPIKE_noise_model_misspecification.md): ONE eps = 0 rung of the 2b
# flat-prior configuration, run on data drawn entirely from Argus's own generative model
# (data/mdc2_ou_selfgen, built by slurm_scripts/mdc2_ou_selfgen_generate.sh).
#
# This is slurm_scripts/mdc2_flat_ladder.sh re-pointed at the self-generated dataset and cut
# to --array=0. The empirical_priors_path / red_noise_prior = flat / data_path guards are kept.
#
# Judge on r_hat / ess_bulk / divergences / mode-splitting, NOT lnB (one rung has no Bayes
# factor). A healthy rung is 1.5-2.5 h; the failing 2b rungs took 7-13.5 h, so a long wall
# time is itself part of the signal.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_ou_selfgen_rung.sh

#SBATCH --job-name=ou_selfgen
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
#SBATCH --array=0
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/mdc2_ou_selfgen_%A_%a.out

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

if [ ! -d "${ROOT}/data/mdc2_ou_selfgen" ]; then
    echo "ERROR: ${ROOT}/data/mdc2_ou_selfgen not found." >&2
    exit 1
fi
if [ -L "${ROOT}/data/mdc2_ou_selfgen" ]; then
    echo "ERROR: ${ROOT}/data/mdc2_ou_selfgen is a symlink. Regenerate it into a real" >&2
    echo "       directory with mdc2_ou_selfgen_generate.sh first." >&2
    exit 1
fi

mkdir -p "${ROOT}/outputs/logfiles"

# Derive this rung's config from the template.
CONFIG="${ROOT}/configs/mdc2_ou_selfgen_eps${TAG}.ini"
sed -e "s/__EPS_VALUE__/${EPS}/" -e "s/__EPS_TAG__/${TAG}/" \
    "${ROOT}/configs/mdc2_ou_selfgen_rung.ini.template" > "${CONFIG}"

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
if ! grep -qE "^\s*data_path\s*=\s*\.\./data/mdc2_ou_selfgen\s*$" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} does not point at ../data/mdc2_ou_selfgen." >&2
    exit 1
fi

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== env check (OU self-gen rung ${IDX}: eps=${EPS}) ==="
which python
python -c "import argus.gravitational_waves as gw; print('argus from:', gw.__file__); print('correlation_path available:', hasattr(gw, 'correlation_path'))"
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD
nvidia-smi -L
grep -E "data_path|orf_path|orf_epsilon_value|red_noise_prior|empirical|output_id" "${CONFIG}"

echo "=== running OU self-gen rung eps=${EPS} ==="
time python -u "${ROOT}/run_analysis.py" "${CONFIG}"
STATUS=$?

# Task 1.10: every run records which frozen evidence procedure it was produced under.
if [ ${STATUS} -eq 0 ]; then
    python "${ROOT}/scripts/stamp_provenance.py" \
        --output-dir "${ROOT}/outputs/mdc2_ou_selfgen_eps${TAG}" \
        --config "${CONFIG}" \
        --frozen "${FROZEN}"
fi

exit ${STATUS}

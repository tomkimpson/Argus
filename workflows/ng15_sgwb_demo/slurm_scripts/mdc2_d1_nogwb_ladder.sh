#!/bin/bash
# NO-INJECTION CONTROL (openspec sgwb-detection-route task 1.9): the full 5-rung flat-prior
# ladder on a SYNTHESISED SIGNAL-FREE dataset at the MDC2 1b geometry.
#
# The evidence estimator has now passed two of the three scenarios in sgwb/model-selection on
# real 1b data:
#
#   injected signal   lnB = +3.043 +/- 0.015   (outputs/lnb_path_sampling_d1_flat.json)
#   sky scramble      lnB = -0.766 +/- 0.010   (outputs/lnb_path_sampling_d1_null.json)
#
# This ladder is the third. The scramble does NOT close it: a scramble mis-describes a
# correlation that is genuinely there, so it must come back below zero, and it did.
# "Consistent with zero" is a statement about data with no correlation to describe at all.
#
#   |lnB| < 1 and reliable: true  => the estimator does not manufacture HD evidence on noise.
#                                    Group 1's validation half is complete.
#   lnB near +3                   => it does. STOP; the 3.043 does not survive.
#   lnB near -0.77                => the estimator penalises ANY correlation model on
#                                    noise-only data. Not a failure, but a different finding,
#                                    and it must be reported rather than filed as a pass.
#
# The dataset is built by scripts/inject_powerlaw_gwb.py at log10_A_gw = -30 (pivot PSD
# 2.8e-37 s^3 against 1b's injected 1.2e-7), white noise on from group1_psr_noise.json, no
# per-pulsar red noise -- which matches g1.d1b exactly, since the IPTA MDC2 dataset table
# lists g1.d1a(b) as WN only and g1.d2a(b) as WN+RN. Every other feather field (TOAs, errors,
# design matrix, sky positions) is real 1b, so the geometry, epochs and cadence are unchanged.
#
# configs/mdc2_d1_nogwb_rung.ini.template is configs/mdc2_d1_flat_rung.ini.template with
# exactly two values changed (data_path, output_id), verified by diff, so this ladder is
# comparable to the +3.043 one line for line.
#
# Read the Bayes factor afterwards as a SLURM CPU job, NOT on the login node:
#   JAX_PLATFORMS=cpu python scripts/lnb_path_sampling.py \
#       --from-runs 'outputs/mdc2_d1_nogwb_eps*/mdc2_d1_nogwb_eps*_results.nc' \
#       --evaluate configs/mdc2_d1_nogwb_eps000.ini \
#       --batch-size 100 \
#       --out outputs/lnb_path_sampling_d1_nogwb.json
# --batch-size 100 is mandatory or the integrand evaluation is OOM-killed (~30 min, ~19 GB
# for five rungs). This ladder uses the TRUE ORF, so lnb_path_sampling.py is the right entry
# point -- scripts/lnb_scrambled.py is only for scrambled ladders.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_d1_nogwb_ladder.sh

#SBATCH --job-name=mdc2_d1_nogwb
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:4
# Sized from the flat and null ladders' measured usage (9.4 GB peak of 32 GB; 1h30m-2h23m of
# 16 h). gpu:4 is fixed by num_chains = 4. Checkpointing is on, so a timeout costs one segment
# rather than the run.
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/mdc2_d1_nogwb_%A_%a.out

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

if [ ! -e "${ROOT}/data/mdc2_d1_nogwb" ]; then
    echo "ERROR: ${ROOT}/data/mdc2_d1_nogwb not found; build it with" >&2
    echo "  scripts/inject_powerlaw_gwb.py --aligned-dir data/mdc2_d1_all --log10-A-gw -30" >&2
    exit 1
fi

# The control is defined by its injection record; without it the run is unattributable.
if [ ! -f "${ROOT}/data/mdc2_d1_nogwb/injection_truth.json" ]; then
    echo "ERROR: ${ROOT}/data/mdc2_d1_nogwb/injection_truth.json missing." >&2
    exit 1
fi

mkdir -p "${ROOT}/outputs/logfiles"

# Derive this rung's config from the template.
CONFIG="${ROOT}/configs/mdc2_d1_nogwb_eps${TAG}.ini"
sed -e "s/__EPS_VALUE__/${EPS}/" -e "s/__EPS_TAG__/${TAG}/" \
    "${ROOT}/configs/mdc2_d1_nogwb_rung.ini.template" > "${CONFIG}"

# Inherited from the flat ladder scripts and kept deliberately: empirical_priors_path takes
# precedence over red_noise_prior in get_pulsar_noise_priors, so a stray copy of that line
# would silently change the noise model out from under the control and make it incomparable
# to the flat ladder it is the control for.
if grep -qE "^\s*empirical_priors_path\s*=\s*\S" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} still sets empirical_priors_path; the control is not comparable." >&2
    exit 1
fi
if ! grep -qE "^\s*red_noise_prior\s*=\s*flat\s*$" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} does not set red_noise_prior = flat." >&2
    exit 1
fi
# The whole point of this ladder is that it runs on the signal-free data, not on 1b.
if ! grep -qE "^\s*data_path\s*=\s*\.\./data/mdc2_d1_nogwb\s*$" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} does not point at ../data/mdc2_d1_nogwb." >&2
    exit 1
fi

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== env check (no-injection control rung ${IDX}: eps=${EPS}) ==="
which python
python -c "import argus.gravitational_waves as gw; print('argus from:', gw.__file__); print('correlation_path available:', hasattr(gw, 'correlation_path'))"
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD
nvidia-smi -L
grep -E "data_path|orf_path|orf_epsilon_value|red_noise_prior|empirical|output_id" "${CONFIG}"

echo "=== running no-injection control rung eps=${EPS} ==="
time python -u "${ROOT}/run_analysis.py" "${CONFIG}"
STATUS=$?

# Task 1.10: every run records which frozen evidence procedure it was produced under.
if [ ${STATUS} -eq 0 ]; then
    python "${ROOT}/scripts/stamp_provenance.py" \
        --output-dir "${ROOT}/outputs/mdc2_d1_nogwb_eps${TAG}" \
        --config "${CONFIG}" \
        --frozen "${FROZEN}"
fi

exit ${STATUS}

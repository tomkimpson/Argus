#!/bin/bash
# Multi-seed calibration of the eps = 0 flat-prior GW posterior on model-correct data.
#
# data/mdc2_inject_ou (seed 0) recovered the pivot log-PSD +0.26 dex high (truth at the 2.7%
# quantile), with log10_gamma_a tight and ~0.65 dex high (truth at quantile ~0). One
# realisation cannot separate scatter from bias. Each array task draws a fresh OU GW-only
# injection (same ha, gamma_a, white noise and geometry as mdc2_inject_ou, different seed),
# then runs the same eps = 0 flat rung on the symmetrised filter. Seed 0 is
# outputs/mdc2_inject_ou_eps000_psym.
#
# Read off per seed: the truth quantile of the pivot, log10_gamma_a and log10_ha. Under a
# calibrated posterior these quantiles are ~Uniform(0, 1).
#
# Submit:  sbatch /fred/oz022/tkimpson/Argus-fix-psym/workflows/ng15_sgwb_demo/slurm_scripts/mdc2_ou_calib.sh

#SBATCH --job-name=ou_calib
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-5
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ou_calib_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO_PY=/fred/oz022/tkimpson/Argus-fix-psym/python
SEED="${SLURM_ARRAY_TASK_ID:?run as an array task}"
DATA="mdc2_inject_ou_s${SEED}"
OUT_ID="${DATA}_eps000_psym"
CONFIG="${ROOT}/configs/${OUT_ID}.ini"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"
python -c "import argus.jax_kalman_filter as k, inspect; print('argus from:', k.__file__); \
print('symmetrised:', 'P = 0.5 * (P + P.T)' in inspect.getsource(k._update_marginal))"
git -C /fred/oz022/tkimpson/Argus-fix-psym rev-parse --short HEAD

# The recipe of data/mdc2_inject_ou (notes/kernel_systematic_injection_pair.md), new seed.
JAX_PLATFORMS=cpu python -u /fred/oz022/tkimpson/Argus-fix-psym/workflows/ng15_sgwb_demo/scripts/inject_powerlaw_gwb.py \
    --mode ou --aligned-dir data/mdc2_all \
    --noise-json ../data/IPTA_MockDataChallenge2/group1_psr_noise.json \
    --out-dir "data/${DATA}" \
    --log10-ha -12.919767 --log10-gamma-a -9.0 --seed "${SEED}" --overwrite || exit 1

sed -e "s/__EPS_VALUE__/0.0/" -e "s/__EPS_TAG__/000/" \
    -e "s/^output_id = .*/output_id = ${OUT_ID}/" \
    -e "s|^data_path = .*|data_path = ../data/${DATA}|" \
    "${ROOT}/configs/mdc2_ou_selfgen_rung.ini.template" > "${CONFIG}"
if grep -qE "^\s*empirical_priors_path\s*=\s*\S" "${CONFIG}" || \
   ! grep -qE "^\s*red_noise_prior\s*=\s*flat\s*$" "${CONFIG}"; then
    echo "ERROR: ${CONFIG} is not the flat rung." >&2; exit 1
fi
grep -E "^(data_path|orf_epsilon_value|red_noise_prior|output_id)" "${CONFIG}"
nvidia-smi -L

time python -u "${ROOT}/run_analysis.py" "${CONFIG}"
STATUS=$?
# stamp_provenance.py records the MAIN checkout's HEAD as git_sha; the library actually used is
# the fix worktree's HEAD printed above.
if [ ${STATUS} -eq 0 ]; then
    python "${ROOT}/scripts/stamp_provenance.py" --output-dir "${ROOT}/outputs/${OUT_ID}" \
        --config "${CONFIG}" --frozen "${ROOT}/configs/frozen/evidence_procedure_v1.json"
fi
exit ${STATUS}

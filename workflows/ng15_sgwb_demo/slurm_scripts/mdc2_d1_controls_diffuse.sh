#!/bin/bash
# Rerun the two MDC2 1b controls under the DIFFUSE timing prior. Both were last computed under
# the informative prior, which PR #121 replaced as the default after it was shown to manufacture
# the 1b GW-vs-noise deficit:
#   KIND=null   sky-scramble null      (informative: lnB -0.766, outputs/lnb_path_sampling_mdc2_d1_null_psym.json)
#   KIND=nogwb  no-injection control   (informative: lnB -0.013, outputs/lnb_path_sampling_mdc2_d1_nogwb_psym.json)
# The signal ladder they calibrate is mdc2_d1_flat_diffuse: lnB(HD/CURN) = 0.693 +/- 0.003.
#
# Each config is the rung template with ONE change, timing_prior = diffuse (inserted ahead of the
# red-noise block, where mdc2_d1_flat_diffuse_* carries it), plus output_id
# mdc2_d1_<KIND>_diffuse_eps<TAG>. Null eps000 is CURN for any geometry, so its POSTERIOR
# matches mdc2_d1_flat_diffuse_eps000; its integrand does not (it depends on the ORF through
# dC/d eps = Gamma - I). Confirm the scramble landed via "[SCRAMBLE] Overrode hd_correlation"
# in the logs. (The config comments that say otherwise are kept byte-identical for provenance.)
#
# The library comes from a worktree pinned at 66e060a (main after PR #122). Its likelihood is
# identical to the fa65942 one the flat_diffuse ladder used when timing_prior is explicit; the
# difference is only the #121 guards. Create it first (once):
#   git -C /fred/oz022/tkimpson/Argus worktree add --detach /fred/oz022/tkimpson/Argus-controls-diffuse 66e060a
#
# Submit:
#   sbatch --export=ALL,KIND=null  slurm_scripts/mdc2_d1_controls_diffuse.sh
#   sbatch --export=ALL,KIND=nogwb slurm_scripts/mdc2_d1_controls_diffuse.sh
# Read out:
#   nogwb: sbatch --export=ALL,LADDER=mdc2_d1_nogwb_diffuse,REPO_PY=/fred/oz022/tkimpson/Argus-controls-diffuse/python slurm_scripts/lnb_readout.sh
#   null:  sbatch --export=ALL,LADDER=mdc2_d1_null_diffuse,REPO_PY=/fred/oz022/tkimpson/Argus-controls-diffuse/python slurm_scripts/lnb_scrambled_readout.sh
#   (never lnb_readout.sh for the null: it would integrate against the TRUE ORF.)

#SBATCH --job-name=d1_ctrl_diffuse
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=16:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/d1_ctrl_diffuse_%A_%a.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
LIB=/fred/oz022/tkimpson/Argus-controls-diffuse
REPO_PY="${LIB}/python"
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

OUT_ID="mdc2_d1_${KIND}_diffuse_eps${TAG}"
CONFIG="${ROOT}/configs/${OUT_ID}.ini"
sed -e "s/__EPS_VALUE__/${EPS}/" -e "s/__EPS_TAG__/${TAG}/" \
    -e "s/^output_id = .*/output_id = ${OUT_ID}/" \
    -e "s/^# 2\. Pulsar red noise/# Timing-model prior: THE ONE CHANGE from the informative-prior control.\ntiming_prior = diffuse\n\n&/" \
    "${ROOT}/configs/mdc2_d1_${KIND}_rung.ini.template" > "${CONFIG}"

if ! sed 's/#.*//' "${CONFIG}" | grep -qE "^\s*timing_prior\s*=\s*diffuse\s*$"; then
    echo "ERROR: ${CONFIG} does not set timing_prior = diffuse." >&2; exit 1
fi
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
if [ ! -d "${REPO_PY}/argus" ]; then
    echo "ERROR: ${REPO_PY} missing; create the pinned worktree (see header)." >&2; exit 1
fi
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== env check (${OUT_ID}, eps=${EPS}) ==="
python -c "import argus.parameter_sampling as p; print('argus from:', p.__file__)"
git -C "${LIB}" rev-parse --short HEAD
nvidia-smi -L
grep -E "^(data_path|orf_path|orf_epsilon_value|red_noise_prior|timing_prior|output_id)" "${CONFIG}"

if [ "${KIND}" = null ]; then
    time python -u "${ROOT}/run_scrambled.py" "${CONFIG}" \
        --scramble-npz "${SCRAMBLE_NPZ}" --scramble-index "${SCRAMBLE_INDEX}"
else
    time python -u "${ROOT}/run_analysis.py" "${CONFIG}"
fi
STATUS=$?
if [ ${STATUS} -eq 0 ]; then
    python "${ROOT}/scripts/stamp_provenance.py" --output-dir "${ROOT}/outputs/${OUT_ID}" \
        --config "${CONFIG}" --frozen "${FROZEN}" --repo "${LIB}"
fi
exit ${STATUS}

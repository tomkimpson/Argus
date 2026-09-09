#!/bin/bash
# CPU readout of a path-sampling ladder: integrate the recorded integrand and report
# ln B(HD/CURN) with its reliability diagnostics.
#
# WHY THIS IS A BATCH JOB. The readout is CPU-only but not small: ~30 min and ~19 GB resident
# for a five-rung ladder when it gets a core to itself. Run on the login node it has taken
# 6h33m at 36 GB, purely from cgroup contention -- and a multi-hour 36 GB login-node process
# can be reaped at any moment. There is no reason to risk it.
#
# --batch-size 100 is MANDATORY. Without it the integrand evaluation is OOM-killed.
#
# Usage:
#   sbatch --export=ALL,LADDER=mdc2_d1_nogwb slurm_scripts/lnb_readout.sh
#   sbatch --export=ALL,LADDER=mdc2_flat,OUT=lnb_path_sampling_flat.json slurm_scripts/lnb_readout.sh
#
#   LADDER  required. The run prefix, e.g. mdc2_d1_nogwb -> outputs/mdc2_d1_nogwb_eps*/
#   OUT     optional. Output JSON name under outputs/ (default: lnb_path_sampling_<LADDER>.json)
#
# SCRAMBLED LADDERS DO NOT GO THROUGH THIS SCRIPT. lnb_path_sampling.py --evaluate rebuilds the
# data from the config, so it picks up the TRUE ORF regardless of what the rungs were sampled
# under, and returns a well-formed, reliable-looking, meaningless number. Use
# scripts/lnb_scrambled.py for those.
#
# The check below is a NAMING TRIPWIRE, and it is worth being honest about what it cannot do.
# The ORF does not reach the filter through any config key -- it arrives through the data dict,
# which is why run_scrambled.py monkeypatches the loader instead of setting an option. Nothing
# on the run side records which ORF was used either: the checkpoint stores only a composite
# fingerprint hash (bayesian_inference._data_fingerprint does hash the ORF, but only into that
# one opaque value), and the .nc attrs carry no ORF provenance at all. At eps = 0 the two are
# genuinely identical anyway, since correlation_path returns the identity for any geometry.
# So there is no cheap way to *prove* a ladder was unscrambled from its artefacts. The tripwire
# catches the realistic mistake -- pointing this script at the mdc2_d1_null_* ladder -- and
# nothing more. The real protection is using lnb_scrambled.py deliberately.

#SBATCH --job-name=lnb_readout
#SBATCH --account=oz022
#SBATCH --partition=milan
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/lnb_readout_%j.out

ROOT=/fred/oz022/tkimpson/Argus/workflows/ng15_sgwb_demo
REPO_PY=/fred/oz022/tkimpson/Argus/python

if [ -z "${LADDER}" ]; then
    echo "ERROR: set LADDER, e.g. sbatch --export=ALL,LADDER=mdc2_d1_nogwb $0" >&2
    exit 1
fi
OUT="${OUT:-lnb_path_sampling_${LADDER}.json}"
CONFIG="${ROOT}/configs/${LADDER}_eps000.ini"

N_RUNGS=$(ls -d ${ROOT}/outputs/${LADDER}_eps*/ 2>/dev/null | wc -l)
if [ "${N_RUNGS}" -ne 5 ]; then
    echo "ERROR: found ${N_RUNGS} rung directories for ${LADDER}, expected 5." >&2
    echo "       The frozen procedure enforces MIN_RUNGS = 5; a short ladder cannot" >&2
    echo "       estimate its own discretisation error." >&2
    exit 1
fi
for TAG in 000 025 050 075 100; do
    NC="${ROOT}/outputs/${LADDER}_eps${TAG}/${LADDER}_eps${TAG}_results.nc"
    if [ ! -f "${NC}" ]; then
        echo "ERROR: ${NC} missing -- rung eps${TAG} has not finished." >&2
        exit 1
    fi
done
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: ${CONFIG} not found." >&2
    exit 1
fi
# See the header: --evaluate would silently use the true ORF. Naming tripwire, not proof.
if echo "${LADDER}" | grep -qiE "null|scramble"; then
    echo "ERROR: ladder '${LADDER}' is named as a scrambled/null ladder." >&2
    echo "       --evaluate would integrate it against the TRUE ORF and return a" >&2
    echo "       well-formed, meaningless number. Use scripts/lnb_scrambled.py." >&2
    exit 1
fi
# Comments are stripped first: the config headers discuss scrambles at length, and an earlier
# version of this guard matched its own explanatory prose and refused a clean ladder.
if sed 's/#.*//' "${CONFIG}" | grep -qiE "scramble"; then
    echo "ERROR: ${CONFIG} sets a scramble option; use scripts/lnb_scrambled.py instead." >&2
    exit 1
fi

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${REPO_PY}:${PYTHONPATH}"

echo "=== readout: ${LADDER} -> outputs/${OUT} ==="
git -C /fred/oz022/tkimpson/Argus rev-parse --short HEAD
grep -h "data_path" "${CONFIG}"

time JAX_PLATFORMS=cpu python -u "${ROOT}/scripts/lnb_path_sampling.py" \
    --from-runs "${ROOT}/outputs/${LADDER}_eps*/${LADDER}_eps*_results.nc" \
    --evaluate "${CONFIG}" \
    --batch-size 100 \
    --out "${ROOT}/outputs/${OUT}"

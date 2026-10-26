#!/bin/bash
# Spec Phase 5, task 5.2: the 67-psr ε = 0 rung of configs/frozen/ng15_nb_v1.ini on 4 A100s,
# one NUTS chain per GPU, through array/run_array.py.
#   MODE=check  one timed value-plus-gradient call and the sampled-site count, no sampling
#   MODE=smoke  1 chain, 2 + 8 iterations at tree depth 1 in two checkpointed segments (numpyro's
#               summary needs >= 4 draws a segment): the whole harness path (fingerprint,
#               checkpoint, resume, save) in minutes on one GPU
#               (sbatch --gres=gpu:a100:1 --time=1:00:00 ...)
#   MODE=probe  200 warmup + 200 draws (checkpoint every 100), to measure step size and
#               s/iter before the full pilot is committed (§5's budget is open)
#   MODE=pilot  the frozen 1000 + 1000. RESUME=1 continues from its checkpoint, for the
#               7-day segments the milan-gpu limit forces
# The probe's config is the frozen file with only num_warmup, num_samples, the checkpoint
# interval and output_id changed; the job prints the diff.
#
# Code runs from a pinned worktree so edits to the main checkout cannot reach a queued job:
#   git -C /fred/oz022/tkimpson/Argus worktree add --detach /fred/oz022/tkimpson/Argus-ng15nb-p5 <commit>
#   mkdir -p /fred/oz022/tkimpson/Argus/workflows/ng15_nb/outputs/logfiles
#   sbatch --export=ALL,REPO=/fred/oz022/tkimpson/Argus-ng15nb-p5,MODE=probe workflows/ng15_nb/array/slurm/pilot.sh
# Draws, checkpoints and provenance land in the pinned worktree's
# workflows/ng15_nb/outputs/<output_id>/; array/pilot_readout.py reads them from there.

#SBATCH --job-name=ng15nb-pilot
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:a100:4
#SBATCH --time=7-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --chdir=/fred/oz022/tkimpson/Argus
#SBATCH --output=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/outputs/logfiles/pilot_%j.out

set -euo pipefail
REPO="${REPO:?set REPO to the pinned worktree}"
MODE="${MODE:-check}"
source /fred/oz022/tkimpson/miniconda3/etc/profile.d/conda.sh
conda activate Argus
export PYTHONPATH="${REPO}/python:${REPO}"  # the editable install points at the main checkout
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export NG15_INGEST_DIR=/fred/oz022/tkimpson/Argus/workflows/ng15_nb/data/ingest  # gitignored
WF="${REPO}/workflows/ng15_nb"
FROZEN="${WF}/configs/frozen/ng15_nb_v1.ini"
echo "repo ${REPO} @ $(git -C "${REPO}" rev-parse HEAD); mode ${MODE}"
nvidia-smi -L
python -c "import argus.parameter_sampling as p; print('argus from:', p.__file__)"

case "${MODE}" in
    check | pilot)
        CONFIG="${FROZEN}"
        ;;
    probe)
        CONFIG="${WF}/configs/generated/ng15_nb_v1_probe.ini"
        mkdir -p "$(dirname "${CONFIG}")"
        sed -e 's/^num_warmup = .*/num_warmup = 200/' \
            -e 's/^num_samples = .*/num_samples = 200/' \
            -e 's/^interval = .*/interval = 100/' \
            -e 's/^output_id = .*/output_id = ng15_nb_v1_probe_eps000/' "${FROZEN}" > "${CONFIG}"
        diff "${FROZEN}" "${CONFIG}" || true
        ;;
    smoke)
        CONFIG="${WF}/configs/generated/ng15_nb_v1_smoke.ini"
        mkdir -p "$(dirname "${CONFIG}")"
        sed -e 's/^num_warmup = .*/num_warmup = 2/' \
            -e 's/^num_samples = .*/num_samples = 8/' \
            -e 's/^num_chains = .*/num_chains = 1/' \
            -e 's/^max_tree_depth = .*/max_tree_depth = 1/' \
            -e 's/^interval = .*/interval = 4/' \
            -e 's/^output_id = .*/output_id = ng15_nb_v1_smoke_eps000/' "${FROZEN}" > "${CONFIG}"
        diff "${FROZEN}" "${CONFIG}" || true
        ;;
    *)
        echo "ERROR: MODE must be check, smoke, probe or pilot" >&2; exit 1
        ;;
esac

if [[ "${MODE}" == check ]]; then
    python -u "${WF}/array/run_array.py" "${CONFIG}" --check
    exit 0
fi
if [[ "${RESUME:-0}" == 1 ]]; then
    RESUMED="${WF}/configs/generated/$(basename "${CONFIG}" .ini)_resume.ini"
    mkdir -p "$(dirname "${RESUMED}")"
    sed -e 's/^resume = .*/resume = true/' "${CONFIG}" > "${RESUMED}"
    diff "${CONFIG}" "${RESUMED}" || true
    CONFIG="${RESUMED}"
fi

OUT_ID=$(sed -n 's/^output_id = //p' "${CONFIG}")
time python -u "${WF}/array/run_array.py" "${CONFIG}"
python "${REPO}/workflows/ng15_sgwb_demo/scripts/stamp_provenance.py" \
    --output-dir "${WF}/outputs/${OUT_ID}" --config "${CONFIG}" --repo "${REPO}" \
    --frozen "${REPO}/workflows/ng15_sgwb_demo/configs/frozen/evidence_procedure_v1.json"

#!/bin/bash
# M1 Stage B: MDC2 33-pulsar array run with red noise FIXED at the Stage A
# medians (issue #111). MODE selects the ORF:
#   hd   (default) -> run_analysis.py with configs/mdc2_stage_b_hd.ini (Hellings-Downs)
#   curn           -> run_curn.py     with configs/mdc2_stage_b_curn.ini (identity ORF)
#
# Only 2 GW parameters are sampled (noise fixed), so this is cheap; it tests the
# array-run geometry, speed and wiring. Requires data/stage_a_medians.pkl from
# scripts/extract_stage_a.py.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_stage_b.sh          # hd
#          sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_stage_b.sh curn

#SBATCH --job-name=mdc2_stage_b
#SBATCH --account=oz022
# The OzSTAR job_submit plugin canonicalizes any GPU request to milan-gpu; if
# milan-gpu is down, GPU jobs queue until it returns (milan-c has no GPU route).
#SBATCH --partition=milan-gpu
# 4 GPUs matched to num_chains=4 so chains run in parallel (sequential chains
# on 1 GPU timed out the first Stage A array — see mdc2_stage_a.sh).
#SBATCH --gres=gpu:4
# 12h, not 4h: job 14593114 timed out on ONE stuck chain at max tree depth
# (healthy chains finished in 48 min). With depth capped at 8 the worst-case
# stuck chain is ~7.5 s/it x 3000 iters ~ 6.3h.
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/mdc2_stage_b_%j.out

# NB: no `set -e` -- sourcing ~/.bashrc / conda init returns non-zero in a
# non-interactive shell, which under `set -e` aborts the job before any output.

ROOT=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
WORKTREE_PY=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python

MODE="${1:-hd}"   # hd | curn
case "${MODE}" in
    hd)   DRIVER="${ROOT}/run_analysis.py" ;;
    curn) DRIVER="${ROOT}/run_curn.py" ;;
    *)    echo "ERROR: MODE must be hd or curn (got '${MODE}')" >&2; exit 1 ;;
esac
CONFIG="${ROOT}/configs/mdc2_stage_b_${MODE}.ini"

if [ ! -f "${ROOT}/data/stage_a_medians.pkl" ]; then
    echo "ERROR: ${ROOT}/data/stage_a_medians.pkl not found." >&2
    echo "Run scripts/extract_stage_a.py after the Stage A array job." >&2
    exit 1
fi

mkdir -p "${ROOT}/outputs/logfiles"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus

# CRITICAL: argus is pip-installed EDITABLE pointing at /fred/oz022/tkimpson/Argus
# (the main checkout). Prepend THIS worktree's python/ via PYTHONPATH so
# `import argus` resolves to the branch code.
export PYTHONPATH="${WORKTREE_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import jax, flax; print('jax', jax.__version__, 'flax', flax.__version__)"
python -c "import argus.prior_models as pm; print('argus.prior_models from:', pm.__file__)"
nvidia-smi -L

echo "=== running Stage B (MODE=${MODE}) ==="
time python -u "${DRIVER}" "${CONFIG}"

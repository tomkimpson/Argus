#!/bin/bash
# M1 Stage C: MDC2 33-pulsar array run with EMPIRICAL per-pulsar red noise
# priors from the Stage A posteriors (issue #111). MODE selects the ORF:
#   hd   (default) -> run_analysis.py with configs/mdc2_stage_c_hd.ini (Hellings-Downs)
#   curn           -> run_curn.py     with configs/mdc2_stage_c_curn.ini (identity ORF)
#
# 68 sampled dims (2 GW + 2x33 noise, no hyperpriors). Requires
# data/stage_a_empirical_priors.json from scripts/extract_stage_a.py.
# If mixing is poor, escalate per the config header: target_accept_prob 0.95 ->
# max_tree_depth 10 -> empirical_prior_inflation 3.0.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_stage_c.sh          # hd
#          sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_stage_c.sh curn

#SBATCH --job-name=mdc2_stage_c
#SBATCH --account=oz022
# The OzSTAR job_submit plugin canonicalizes any GPU request to milan-gpu; if
# milan-gpu is down, GPU jobs queue until it returns (milan-c has no GPU route).
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/mdc2_stage_c_%j.out

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
CONFIG="${ROOT}/configs/mdc2_stage_c_${MODE}.ini"

if [ ! -f "${ROOT}/data/stage_a_empirical_priors.json" ]; then
    echo "ERROR: ${ROOT}/data/stage_a_empirical_priors.json not found." >&2
    echo "Run scripts/extract_stage_a.py after the Stage A array job." >&2
    exit 1
fi

mkdir -p "${ROOT}/outputs/logfiles"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus

# CRITICAL: argus is pip-installed EDITABLE pointing at /fred/oz022/tkimpson/Argus
# (the main checkout, which lacks the empirical-prior mode). Prepend THIS
# worktree's python/ via PYTHONPATH so `import argus` resolves to the branch code.
export PYTHONPATH="${WORKTREE_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import jax, flax; print('jax', jax.__version__, 'flax', flax.__version__)"
python -c "import argus.prior_models as pm; print('argus.prior_models from:', pm.__file__); print('empirical mode available:', hasattr(pm, 'get_empirical_noise_priors'))"
nvidia-smi -L

echo "=== running Stage C (MODE=${MODE}) ==="
time python -u "${DRIVER}" "${CONFIG}"

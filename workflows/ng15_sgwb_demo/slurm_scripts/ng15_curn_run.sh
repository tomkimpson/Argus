#!/bin/bash
# NG15 SGWB Stage-3 CURN run on 4x A100 (workflow task T3.4) -- REAL aligned NG15 data.
#
# The null companion to the HD production run (ng15_production_run.sh / outputs/ng15_real):
# identical config and NUTS settings, but run_curn.py overrides the Hellings-Downs
# inter-pulsar correlation with the IDENTITY at runtime (common UNCORRELATED red noise),
# with NO library edit. Output lands in outputs/ng15_curn/. Its evidence (logZ_CURN),
# combined with logZ_HD, gives the HD-vs-CURN Bayes factor lnB = logZ_HD - logZ_CURN.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/ng15_curn_run.sh
#
# Cloned from ng15_production_run.sh; only the driver (run_curn.py) and config
# (ng15_curn_config.ini, output_id=ng15_curn) differ. num_chains=4 matched to --gres=gpu:4
# so the chains run concurrently (T3.3 finished ~71 min on 4x A100; 8 h is generous).

#SBATCH --job-name=ng15_curn
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:4
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ng15_curn_%j.out

# NB: no `set -e` -- sourcing ~/.bashrc / conda init returns non-zero in a
# non-interactive shell, which under `set -e` aborts the job before any output.

ROOT=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
WORKTREE_PY=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python

CONFIG="${ROOT}/configs/ng15_curn_config.ini"

mkdir -p "${ROOT}/outputs/logfiles"

echo "=== run config ==="
echo "config: ${CONFIG}"
echo "driver: run_curn.py (identity-ORF override)"
grep -E "^data_path|^output_id|^num_samples|^num_warmup|^num_chains|^dense_mass|^target_accept_prob|^log10_ha_min|^log10_ha_max" "${CONFIG}"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus

# CRITICAL: argus is pip-installed EDITABLE against /fred/oz022/tkimpson/Argus (main, no q11
# fix). Prepend THIS worktree's python/ so `import argus` resolves to the patched code.
export PYTHONPATH="${WORKTREE_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import jax, flax; print('jax', jax.__version__, 'flax', flax.__version__)"
python -c "import jax; print('jax devices:', jax.devices())"
python -c "import argus.model as m; print('argus.model from:', m.__file__)"
python -c "import inspect, argus.model as m; q11line=[l for l in inspect.getsource(m.get_Q_block).splitlines() if l.strip().startswith('q11')][0]; print('q11 line:', q11line.strip()); print('q11 fixed (uses gamma**2):', q11line.rstrip().endswith('γ**2'))"
nvidia-smi -L

echo "=== running NG15 CURN recovery (real aligned data, identity ORF) ==="
time python -u "${ROOT}/run_curn.py" "${CONFIG}"

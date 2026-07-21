#!/bin/bash
# NG15 SGWB Stage-3 FULL-ARRAY QUICK-LOOK run on 4x A100 (workflow task T3.5).
#
# A first-look amplitude recovery on the full ~68-pulsar union-aligned array at reduced NUTS
# settings (300 warmup + 500 samples) before committing to a longer run. Sized from the 1-GPU
# probe (job 14173193): ~150-200 s/sample-iteration, ~0.26 s/likelihood-eval, 3.5 GB peak RAM.
# At 300+500 = 800 iters this is ~12-18 h wall on 4x A100 (4 chains run in parallel). Output
# lands in outputs/ng15_full_quicklook/.
#
# Runs configs/ng15_config_full_quicklook.ini (num_warmup=300, num_samples=500, dense_mass=false,
# output_id=ng15_full_quicklook). If the quick look looks healthy, rerun the full-settings
# ng15_full_run.sh for publication-grade ESS.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/ng15_full_quicklook.sh

#SBATCH --job-name=ng15_full_ql
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ng15_full_quicklook_%j.out

ROOT=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
WORKTREE_PY=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python

CONFIG="${ROOT}/configs/ng15_config_full_quicklook.ini"

mkdir -p "${ROOT}/outputs/logfiles"

echo "=== run config ==="
echo "config: ${CONFIG}"
grep -E "^data_path|^output_id|^num_samples|^num_warmup|^num_chains|^dense_mass|^target_accept_prob|^log10_ha_min|^log10_ha_max" "${CONFIG}"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus

# CRITICAL: argus is pip-installed EDITABLE against /fred/oz022/tkimpson/Argus (main, no
# missing-observation mask / HD-diagonal fix). Prepend THIS worktree's python/ so `import argus`
# resolves to the patched code -- without this the GPU run silently uses the main checkout.
export PYTHONPATH="${WORKTREE_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import jax; print('jax devices:', jax.devices())"
python -c "import argus.model as m; print('argus.model from:', m.__file__)"
python -c "import inspect, argus.jax_kalman_filter as k; print('masked filter present:', 'mask_matrices' in inspect.signature(k._run_kalman_filter_scan).parameters)"
python -c "import numpy as np, inspect, argus.gravitational_waves as g; print('HD diag fix present:', 'fill_diagonal' in inspect.getsource(g.pairwise_angular_separation))"
nvidia-smi -L

echo "=== running NG15 FULL-ARRAY QUICK-LOOK recovery (real union-aligned data) ==="
time python -u "${ROOT}/run_analysis.py" "${CONFIG}"

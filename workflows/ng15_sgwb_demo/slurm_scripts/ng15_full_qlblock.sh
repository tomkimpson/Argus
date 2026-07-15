#!/bin/bash
# NG15 SGWB Stage-3 FULL-ARRAY CONVERGENCE CHECK on 4x A100 (workflow task T3.5).
#
# Re-runs the full ~68-pulsar union-aligned array at quick-look settings (500 warmup + 500 samples)
# but with a per-block DENSE mass matrix over the two GW latents (dense_mass_blocks =
# log10_ha_prime, log10_gamma_a_prime) to straighten the log10_ha<->log10_gamma_a ridge that broke
# the diagonal-mass quick-look (job 14195666: max r_hat ~18.6, one outlier chain). Purpose: confirm
# r_hat(log10_ha) -> ~1 BEFORE committing to the ~2.5-3 day production run (ng15_full_run.sh).
# ~half-day wall on 4x A100 (4 chains in parallel). Output lands in outputs/ng15_full_qlblock/.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/ng15_full_qlblock.sh

#SBATCH --job-name=ng15_full_qlb
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ng15_full_qlblock_%j.out

ROOT=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
WORKTREE_PY=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python

CONFIG="${ROOT}/configs/ng15_config_full_qlblock.ini"

mkdir -p "${ROOT}/outputs/logfiles"

echo "=== run config ==="
echo "config: ${CONFIG}"
grep -E "^data_path|^output_id|^num_samples|^num_warmup|^num_chains|^dense_mass|^dense_mass_blocks|^target_accept_prob|^log10_ha_min|^log10_ha_max" "${CONFIG}"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus

# CRITICAL: argus is pip-installed EDITABLE against /fred/oz022/tkimpson/Argus (main, no
# missing-observation mask / HD-diagonal fix / dense_mass_blocks parse). Prepend THIS worktree's
# python/ so `import argus` resolves to the patched code -- without this the GPU run silently uses
# the main checkout.
export PYTHONPATH="${WORKTREE_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import jax; print('jax devices:', jax.devices())"
python -c "import argus.model as m; print('argus.model from:', m.__file__)"
python -c "import inspect, argus.jax_kalman_filter as k; print('masked filter present:', 'mask_matrices' in inspect.signature(k._run_kalman_filter_scan).parameters)"
python -c "import numpy as np, inspect, argus.gravitational_waves as g; print('HD diag fix present:', 'fill_diagonal' in inspect.getsource(g.pairwise_angular_separation))"
python -c "import inspect, argus.bayesian_inference as b; print('dense_mass_blocks parse present:', 'dense_mass_blocks' in inspect.getsource(b.setup_nuts_kernel))"
nvidia-smi -L

echo "=== running NG15 FULL-ARRAY CONVERGENCE CHECK (dense GW block, real union-aligned data) ==="
time python -u "${ROOT}/run_analysis.py" "${CONFIG}"

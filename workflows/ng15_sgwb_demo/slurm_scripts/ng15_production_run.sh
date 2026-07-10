#!/bin/bash
# NG15 SGWB Stage-3 PRODUCTION run on 4x A100 (workflow task T3.3) -- REAL aligned NG15 data.
#
# This is the headline run: the mature GWB + Hellings-Downs + NUTS path on the REAL
# epoch-aligned NG15 wideband feathers (data/aligned/, 6 pulsars x 78 joint epochs), after
# the Stage-2 decision gate passed (T2.4). It runs the committed production config
# configs/ng15_config.ini (NUTS 2000/1000/4, dense_mass, target_accept 0.95, log10_gamma_a
# FREE, log10_ha prior bracketing the published amplitude). Output lands in outputs/ng15_real/.
#
# Unlike the lite/confirm scripts, there is NO MODE argument and NO derived-config sed step:
# the production config is complete and self-contained (data_path -> real aligned feathers,
# output_id = ng15_real baked in) and lives in configs/ (outside the per-run output dir), so
# run_inference's config-copy cannot hit shutil.SameFileError. It is run directly.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/ng15_production_run.sh
#
# NB run_nuts_sampling uses chain_method="parallel" only when n_devices >= num_chains
# (bayesian_inference.py:684); num_chains=4 is matched to --gres=gpu:4 here so the 4 chains
# run concurrently (wall time ~ one chain) rather than sequentially. The confirm run (2000
# total iters) finished in ~24 min on 4x A100; production is 3000 total iters at
# target_accept=0.95 on real data (~1-3 h realistically), so 8 h is generous headroom.

#SBATCH --job-name=ng15_production
#SBATCH --account=oz022
# A100 GPU jobs on oz022 route to milan-gpu regardless of requested partition (job_submit
# plugin canonicalizes it); if milan-gpu is down, GPU jobs queue (Reason=PartitionDown).
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:4
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ng15_real_%j.out

# NB: no `set -e` -- sourcing ~/.bashrc / conda init returns non-zero in a
# non-interactive shell, which under `set -e` aborts the job before any output.

ROOT=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
WORKTREE_PY=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python

# The committed, self-contained production config (T3.1). Run directly -- no derivation.
CONFIG="${ROOT}/configs/ng15_config.ini"

mkdir -p "${ROOT}/outputs/logfiles"

echo "=== run config ==="
echo "config: ${CONFIG}"
grep -E "^data_path|^output_id|^num_samples|^num_warmup|^num_chains|^dense_mass|^target_accept_prob|^log10_ha_min|^log10_ha_max" "${CONFIG}"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus

# CRITICAL: argus is pip-installed EDITABLE against /fred/oz022/tkimpson/Argus (main, no q11
# fix). run_analysis.py only sys.path.append()s the repo python/ (loses to the editable
# install). Prepend THIS worktree's python/ so `import argus` resolves to the patched code.
export PYTHONPATH="${WORKTREE_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import jax, flax; print('jax', jax.__version__, 'flax', flax.__version__)"
python -c "import jax; print('jax devices:', jax.devices())"
python -c "import argus.model as m; print('argus.model from:', m.__file__)"
python -c "import inspect, argus.model as m; q11line=[l for l in inspect.getsource(m.get_Q_block).splitlines() if l.strip().startswith('q11')][0]; print('q11 line:', q11line.strip()); print('q11 fixed (uses gamma**2):', q11line.rstrip().endswith('γ**2'))"
nvidia-smi -L

echo "=== running NG15 PRODUCTION recovery (real aligned data) ==="
time python -u "${ROOT}/run_analysis.py" "${CONFIG}"

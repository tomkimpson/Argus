#!/bin/bash
# NG15 SGWB Stage-3 FULL-ARRAY run on 4x A100 (workflow task T3.5) -- REAL union-aligned data.
#
# Scales the T3.3 subset run to the full ~68-pulsar NG15 array using the missing-observation
# (per-epoch mask) support in the joint GWB Kalman filter. Runs the self-contained config
# configs/ng15_config_full.ini (data_path -> data/aligned_full/, output_id = ng15_full,
# dense_mass=false at ~142-D). Output lands in outputs/ng15_full/.
#
# COST WARNING (size this from the 1-GPU probe first -- ng15_full_probe.sh):
#   The dense Kalman filter is ~O(nx^3) per epoch, nx = 4*Npsr + sum(dim_M). Going from the
#   6-pulsar subset (nx ~ 130) to 68 pulsars (nx ~ 1500) is a ~10x larger state, i.e. a ~1000x+
#   per-likelihood-eval cost, over ~190 (vs 78) union epochs. This can be many hours to a couple
#   of days of wall time and needs substantial memory for the reverse-mode-autodiff trajectory.
#   Run ng15_full_probe.sh (a few warmup iters on 1 GPU) BEFORE this to measure per-iteration
#   time + peak memory, then tune --time/--mem (and, if needed, num_samples) below.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/ng15_full_run.sh

#SBATCH --job-name=ng15_full
#SBATCH --account=oz022
# A100 GPU jobs on oz022 route to milan-gpu regardless of requested partition.
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:4
# Probe (job 14173193) measured ~5 days for 1000+2000 iters; 7-day cap allows it but it is
# fragile without checkpointing. Peak RAM was 3.5 GB, so 32G is ample.
#SBATCH --time=6-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ng15_full_%j.out

# NB: no `set -e` -- sourcing ~/.bashrc / conda init returns non-zero in a
# non-interactive shell, which under `set -e` aborts the job before any output.

ROOT=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
WORKTREE_PY=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python

CONFIG="${ROOT}/configs/ng15_config_full.ini"

mkdir -p "${ROOT}/outputs/logfiles"

echo "=== run config ==="
echo "config: ${CONFIG}"
grep -E "^data_path|^output_id|^num_samples|^num_warmup|^num_chains|^dense_mass|^target_accept_prob|^log10_ha_min|^log10_ha_max" "${CONFIG}"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus

# CRITICAL: argus is pip-installed EDITABLE against /fred/oz022/tkimpson/Argus (main, no
# missing-observation mask). run_analysis.py only sys.path.append()s the repo python/ (loses to
# the editable install). Prepend THIS worktree's python/ so `import argus` resolves to the
# masked-filter code -- without this the GPU run silently uses the main checkout.
export PYTHONPATH="${WORKTREE_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import jax, flax; print('jax', jax.__version__, 'flax', flax.__version__)"
python -c "import jax; print('jax devices:', jax.devices())"
python -c "import argus.model as m; print('argus.model from:', m.__file__)"
python -c "import inspect, argus.model as m; q11line=[l for l in inspect.getsource(m.get_Q_block).splitlines() if l.strip().startswith('q11')][0]; print('q11 fixed (uses gamma**2):', q11line.rstrip().endswith('γ**2'))"
python -c "import inspect, argus.jax_kalman_filter as k; print('masked filter present:', 'mask_matrices' in inspect.signature(k._run_kalman_filter_scan).parameters)"
nvidia-smi -L

echo "=== running NG15 FULL-ARRAY recovery (real union-aligned data) ==="
time python -u "${ROOT}/run_analysis.py" "${CONFIG}"

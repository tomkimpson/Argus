#!/bin/bash
# NG15 SGWB Stage-3 FULL-ARRAY 1-GPU TIMING/MEMORY PROBE (workflow task T3.5).
#
# Runs a tiny NUTS job (num_warmup=10, num_samples=5, 1 chain) on the full ~68-pulsar
# union-aligned array to MEASURE per-iteration wall time and peak GPU/host memory BEFORE
# committing the multi-day 4-GPU production run (ng15_full_run.sh). The full dense filter
# is ~O(nx^3) with nx ~ 1500 at 68 pulsars, so this probe is the go/no-go + resource-sizing
# gate. Read the reported step time and `nvidia-smi`/maxrss to set --time and --mem on the
# production script (and decide whether num_samples / the pulsar set needs trimming).
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/ng15_full_probe.sh

#SBATCH --job-name=ng15_full_probe
#SBATCH --account=oz022
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ng15_full_probe_%j.out

ROOT=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
WORKTREE_PY=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python

CONFIG="${ROOT}/configs/ng15_config_full_probe.ini"

mkdir -p "${ROOT}/outputs/logfiles"

echo "=== probe config ==="
grep -E "^data_path|^output_id|^num_samples|^num_warmup|^num_chains|^dense_mass" "${CONFIG}"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus
export PYTHONPATH="${WORKTREE_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import jax; print('jax devices:', jax.devices())"
python -c "import argus.model as m; print('argus.model from:', m.__file__)"
python -c "import inspect, argus.jax_kalman_filter as k; print('masked filter present:', 'mask_matrices' in inspect.signature(k._run_kalman_filter_scan).parameters)"
nvidia-smi -L

echo "=== running FULL-ARRAY PROBE (few iters, 1 GPU) ==="
# /usr/bin/time -v reports Maximum resident set size (host RAM); the NUTS progress bar reports
# per-iteration wall time (compile is the first iter; steady-state is what to extrapolate from).
/usr/bin/time -v python -u "${ROOT}/run_analysis.py" "${CONFIG}"

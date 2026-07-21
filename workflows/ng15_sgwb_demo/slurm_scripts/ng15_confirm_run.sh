#!/bin/bash
# NG15 SGWB Stage-2 CONFIRMATION run on 4x A100 (workflow task T2.4 hi-res re-run).
#
# The lite runs (ng15_slurm_run.sh) cleared the OU control but were under-converged on the
# power-law gate (max r_hat 1.05, min ESS 34). This re-runs both modes at production-grade
# NUTS resolution (configs/ng15_config_confirm.ini: 1000/1000/4, dense_mass, max_tree_depth 10,
# widened log10_gamma_a) with 4 chains IN PARALLEL across 4 GPUs, so the load-bearing gate
# result is defensible before Stage 3.
#   MODE=ou        -> data/inject_ou/       (control baseline, re-run at the new settings)
#   MODE=powerlaw  -> data/inject_powerlaw/ (the decision gate)  [default]
# Output lands in outputs/ng15_confirm_${MODE}/.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/ng15_confirm_run.sh            # powerlaw (default)
#          sbatch workflows/ng15_sgwb_demo/slurm_scripts/ng15_confirm_run.sh ou         # control
#
# NB run_nuts_sampling uses chain_method="parallel" only when n_devices >= num_chains
# (bayesian_inference.py:684); num_chains=4 is matched to --gres=gpu:4 here so the 4 chains
# run concurrently (wall time ~ one chain) rather than sequentially.

#SBATCH --job-name=ng15_confirm
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
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ng15_confirm_%j.out

# NB: no `set -e` -- sourcing ~/.bashrc / conda init returns non-zero in a
# non-interactive shell, which under `set -e` aborts the job before any output.

ROOT=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
WORKTREE_PY=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python

MODE="${1:-powerlaw}"   # powerlaw (T2.4 gate, default) | ou (control)
DATA_DIR="${ROOT}/data/inject_${MODE}/"
OUTID="ng15_confirm_${MODE}"
# Derived config must live OUTSIDE the per-run output dir (run_inference copies the config
# into that dir; a config already inside it would copy onto itself -> shutil.SameFileError).
RUN="${ROOT}/outputs/derived_configs/run_confirm_${MODE}.ini"

mkdir -p "${ROOT}/outputs/derived_configs" "${ROOT}/outputs/logfiles"

# Line-anchored sed only touches the data_path / output_id lines (comments left intact).
sed -e "s|^data_path = .*|data_path = ${DATA_DIR}|" \
    -e "s|^output_id = .*|output_id = ${OUTID}|" \
    "${ROOT}/configs/ng15_config_confirm.ini" > "${RUN}"

echo "=== run config (MODE=${MODE}) ==="
echo "derived config: ${RUN}"
grep -E "^data_path|^output_id|^num_samples|^num_warmup|^num_chains|^dense_mass|^log10_gamma_a_max" "${RUN}"

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

echo "=== running NG15 confirmation-recovery (MODE=${MODE}) ==="
time python -u "${ROOT}/run_analysis.py" "${RUN}"

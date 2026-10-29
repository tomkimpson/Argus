#!/bin/bash
# T2.6 GATE 3 — blackjax nested-sampling validation on 1x A100.
#
# Runs the blackjax NS evidence engine (sampler = blackjax) on MDC2 dataset_2b across
# several seeds, so we can check (1) the NS posterior reproduces the NUTS baseline
# (outputs/mdc2_smoke_wide/: log10_ha ~ -12.88, log10_gamma_a ~ -8.1) and (2) the
# recovered logZ is reproducible across seeds. The GPU is the natural venue: Argus's joint
# Kalman likelihood is ~seconds/eval on CPU (NS needs thousands of evals) but fast on A100.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/blackjax_ns_run.sh
#          sbatch workflows/ng15_sgwb_demo/slurm_scripts/blackjax_ns_run.sh "42 7 123"   # custom seeds
#
# Uses the `Argus` conda env, which has blackjax installed from blackjax-devs main
# (blackjax.nss; the PyPI wheels lack the NS module) alongside the pinned jax 0.4.38.
# run_blackjax_nested_sampling shims jax.shard_map so `import blackjax` works on 0.4.38.

#SBATCH --job-name=bjax_ns
#SBATCH --account=oz022
# A100 GPU jobs on oz022 route to milan-gpu regardless of requested partition (job_submit
# plugin canonicalizes it); if milan-gpu is down, GPU jobs queue (Reason=PartitionDown).
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/bjax_ns_%j.out

# NB: no `set -e` -- sourcing ~/.bashrc / conda init returns non-zero in a
# non-interactive shell, which under `set -e` aborts the job before any output.

ROOT=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
WORKTREE_PY=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python

SEEDS="${1:-42 7 123}"

mkdir -p "${ROOT}/outputs/derived_configs" "${ROOT}/outputs/logfiles"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus

# CRITICAL: argus is pip-installed EDITABLE pointing at /fred/oz022/tkimpson/Argus (the main
# checkout, which lacks the T2.6 blackjax backend). run_analysis.py only sys.path.append()s
# the repo python/ dir, which loses to the editable install. Prepend THIS worktree's python/
# via PYTHONPATH so `import argus` resolves to the patched code.
export PYTHONPATH="${WORKTREE_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import jax; print('jax', jax.__version__)"
python -c "import argus.bayesian_inference as bi; print('run_blackjax_nested_sampling:', hasattr(bi,'run_blackjax_nested_sampling'))"
python -c "import argus.bayesian_inference as bi; bj=bi._import_blackjax_ns(); print('blackjax', bj.__version__, 'nss:', hasattr(bj,'nss'))"
nvidia-smi -L

for SEED in ${SEEDS}; do
  OUTID="mdc2_blackjax_ns_s${SEED}"
  RUN="${ROOT}/outputs/derived_configs/run_bjax_s${SEED}.ini"
  # Line-anchored sed only touches the seed / output_id lines (comments left intact).
  sed -e "s|^seed = .*|seed = ${SEED}|" \
      -e "s|^output_id = .*|output_id = ${OUTID}|" \
      "${ROOT}/configs/mdc2_blackjax_ns.ini" > "${RUN}"
  echo "=== running blackjax NS (seed=${SEED}) -> ${OUTID} ==="
  grep -E "^seed|^output_id|^sampler|^num_live_points" "${RUN}"
  time python -u "${ROOT}/run_analysis.py" "${RUN}"
done

echo "=== all seeds done ==="

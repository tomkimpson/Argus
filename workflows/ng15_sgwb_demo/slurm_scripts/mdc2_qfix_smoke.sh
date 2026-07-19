#!/bin/bash
# MDC2 GWB+HD+NUTS smoke test (workflow task T0.1) on 1x A100.
#
# Validates that the get_Q_block q11 bugfix (γ**3 -> γ**2) did not break GWB
# recovery on the known-good MDC2 dataset_2b. Runs the shared lite run_analysis.py
# (use_gw=True, x64) against ng15_sgwb_demo/configs/mdc2_smoke_lite.ini (absolute
# paths). Uses the correct `Argus` conda env (the committed example scripts wrongly
# activate `argus-env`, which lacks flax and has an out-of-range jax).
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/mdc2_qfix_smoke.sh

#SBATCH --job-name=mdc2_qfix_smoke
#SBATCH --account=oz022
# milan-gpu preferred; using milan-c here (same gina* A100 nodes) as milan-gpu was down.
#SBATCH --partition=milan-c
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/mdc2_qfix_smoke_%j.out

# NB: no `set -e` -- sourcing ~/.bashrc / conda init returns non-zero in a
# non-interactive shell, which under `set -e` aborts the job before any output
# (the existing example slurm scripts likewise omit it).
mkdir -p outputs/logfiles

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus

# CRITICAL: argus is pip-installed EDITABLE pointing at /fred/oz022/tkimpson/Argus
# (the main checkout, which lacks the q11 fix). run_analysis.py only sys.path.append()s
# the repo python/ dir, which loses to the editable install. Prepend THIS worktree's
# python/ via PYTHONPATH so `import argus` resolves to the patched code.
export PYTHONPATH=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python:$PYTHONPATH

echo "=== env check ==="
which python
python -c "import jax, flax; print('jax', jax.__version__, 'flax', flax.__version__)"
python -c "import argus.model as m; print('argus.model from:', m.__file__)"
python -c "import inspect, argus.model as m; q11line=[l for l in inspect.getsource(m.get_Q_block).splitlines() if l.strip().startswith('q11')][0]; print('q11 line:', q11line.strip()); print('q11 fixed (uses gamma**2):', q11line.rstrip().endswith('γ**2'))"
nvidia-smi -L

echo "=== running MDC2 GWB smoke test ==="
time python -u /home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/example_workflow_lite/run_analysis.py \
    /home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/configs/mdc2_smoke_lite.ini

#!/bin/bash
# NG15 SGWB Stage-2 injection-recovery run on 1x A100 (workflow tasks T2.3 / T2.4).
#
# Runs the lite GWB+HD+NUTS recovery (run_analysis.py, use_gw=True, x64) on the
# synthetic injections built on the *real* NG15 sampling geometry:
#   MODE=ou        (T2.3 control)      -> data/inject_ou/       expect log10_ha ~ -14.35,
#                                          log10_gamma_a ~ -8.5  (harness validation)
#   MODE=powerlaw  (T2.4 decision gate)-> data/inject_powerlaw/ (log10_A_gw=-14.6, gamma=13/3)
#
# One committed config (configs/ng15_config_lite.ini) serves both: this script derives a
# per-mode config (data_path + output_id) into the run's own output dir, so T2.3 and T2.4
# do not clobber. Output lands in outputs/ng15_inject_${MODE}/.
#
# Submit:  sbatch workflows/ng15_sgwb_demo/slurm_scripts/ng15_slurm_run.sh            # ou (default)
#          sbatch workflows/ng15_sgwb_demo/slurm_scripts/ng15_slurm_run.sh powerlaw   # T2.4
#
# Uses the correct `Argus` conda env (the committed example scripts wrongly activate
# `argus-env`, which lacks flax and has an out-of-range jax).

#SBATCH --job-name=ng15_inject
#SBATCH --account=oz022
# A100 GPU jobs on oz022 route here (user standing pref). NB the OzSTAR job_submit plugin
# canonicalizes any GPU request to milan-gpu regardless of the requested partition, so this
# line is effectively documentation; if milan-gpu is administratively down, GPU jobs simply
# queue (Reason=PartitionDown) until it returns -- milan-c does not provide a GPU route.
#SBATCH --partition=milan-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --export=ALL
#SBATCH --chdir=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
#SBATCH --output=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo/outputs/logfiles/ng15_inject_%j.out

# NB: no `set -e` -- sourcing ~/.bashrc / conda init returns non-zero in a
# non-interactive shell, which under `set -e` aborts the job before any output.

ROOT=/home/tkimpson/.treehouse/Argus-891104/1/Argus/workflows/ng15_sgwb_demo
WORKTREE_PY=/home/tkimpson/.treehouse/Argus-891104/1/Argus/python

MODE="${1:-ou}"   # ou (T2.3 control) | powerlaw (T2.4 gate)
DATA_DIR="${ROOT}/data/inject_${MODE}/"
OUTID="ng15_inject_${MODE}"
# The derived config must live OUTSIDE the per-run output dir (outputs/${OUTID}/): run_inference
# copies the config into that dir, and a config already inside it would copy onto itself
# (shutil.SameFileError). Keep it under the workflow tree so io_manager resolves workflow_name.
RUN="${ROOT}/outputs/derived_configs/run_config_${MODE}.ini"

mkdir -p "${ROOT}/outputs/derived_configs" "${ROOT}/outputs/logfiles"

# Derive the per-mode config from the committed base. Line-anchored sed only touches the
# data_path / output_id lines (comments left intact).
sed -e "s|^data_path = .*|data_path = ${DATA_DIR}|" \
    -e "s|^output_id = .*|output_id = ${OUTID}|" \
    "${ROOT}/configs/ng15_config_lite.ini" > "${RUN}"

echo "=== run config (MODE=${MODE}) ==="
echo "derived config: ${RUN}"
grep -E "^data_path|^output_id" "${RUN}"

source ~/.bashrc
conda activate /fred/oz022/tkimpson/conda_envs/Argus

# CRITICAL: argus is pip-installed EDITABLE pointing at /fred/oz022/tkimpson/Argus
# (the main checkout, which lacks the q11 fix). run_analysis.py only sys.path.append()s
# the repo python/ dir, which loses to the editable install. Prepend THIS worktree's
# python/ via PYTHONPATH so `import argus` resolves to the patched code.
export PYTHONPATH="${WORKTREE_PY}:${PYTHONPATH}"

echo "=== env check ==="
which python
python -c "import jax, flax; print('jax', jax.__version__, 'flax', flax.__version__)"
python -c "import argus.model as m; print('argus.model from:', m.__file__)"
python -c "import inspect, argus.model as m; q11line=[l for l in inspect.getsource(m.get_Q_block).splitlines() if l.strip().startswith('q11')][0]; print('q11 line:', q11line.strip()); print('q11 fixed (uses gamma**2):', q11line.rstrip().endswith('γ**2'))"
nvidia-smi -L

echo "=== running NG15 injection-recovery (MODE=${MODE}) ==="
time python -u "${ROOT}/run_analysis.py" "${RUN}"
